"""Unified local and centralized baseline training loop.

Adheres strictly to AGENTS.md Section 2.2, 3.1, 3.2, 3.3:
  - Thermal guard reactive throttling with soft/hard pauses and resume state.
  - Atomic checkpointing (best.pt and last.pt).
  - Mixed precision (AMP), channels_last memory format, effective batch size 16.
  - Masked cross-entropy loss with training class weights.
  - Validation-based early stopping (patience 12 on validation macro-AUC).
  - Saved per-breast test predictions (preds/) and results/ledger.csv append.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import (
    MaskedCrossEntropyLoss,
    compute_class_weights,
    create_dataloader,
)
from src.eval.metrics import (
    aggregate_breast_predictions,
    compute_patient_bootstrap_ci,
    evaluate_predictions,
)
from src.models.backbone import build_model
from src.utils.ckpt import CheckpointManager
from src.utils.provenance import RunLogger
from src.utils.seed import set_seed
from src.utils.thermal import ThermalGuard

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[float, pd.DataFrame, Dict[str, Any]]:
    """Evaluate model on a dataloader and return loss, breast-level DataFrame, and metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    pred_records = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device, memory_format=torch.channels_last)
            labels = batch["label"].to(device)
            allowed_masks = batch["allowed_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = loss_fn(logits, labels, allowed_masks)

            total_loss += float(loss.item())
            n_batches += 1

            logits_np = logits.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            for i in range(len(labels_np)):
                pred_records.append({
                    "site": batch["site"][i],
                    "patient_hash": batch["patient_hash"][i],
                    "laterality": batch["laterality"][i],
                    "view": batch["view"][i],
                    "image_id": batch["image_id"][i],
                    "logits": logits_np[i],
                    "label": int(labels_np[i]),
                    "split": batch["split"][i],
                })

    avg_loss = total_loss / max(1, n_batches)
    breast_df = aggregate_breast_predictions(pred_records)
    metrics = evaluate_predictions(breast_df) if not breast_df.empty else {}
    return avg_loss, breast_df, metrics


def train_local(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    model: nn.Module | None = None,
    thermal_guard: ThermalGuard | None = None,
    run_logger: RunLogger | None = None,
    output_dir: Path | str | None = None,
) -> Dict[str, Any]:
    """Execute unified local training loop.

    Adheres strictly to AGENTS.md run discipline and baseline requirements.
    """
    seed = int(config.get("seed", 42))
    set_seed(seed)

    # Device and precision setup
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    use_amp = (device_type == "cuda")

    # Hyperparameters
    batch_size = int(config.get("batch_size", 16))
    max_epochs = int(config.get("max_epochs", 40))
    patience = int(config.get("patience", 12))
    backbone_lr = float(config.get("backbone_lr", 1e-4))
    head_lr_mult = float(config.get("head_lr_mult", 10.0))
    bn_mode = str(config.get("bn_mode", "frozen"))
    weight_decay = float(config.get("weight_decay", 1e-2))
    lr_schedule = str(config.get("lr_schedule", "cosine"))
    experiment_id = str(config.get("experiment_id", "baseline"))
    site_name = str(config.get("site", "pooled"))

    # Initialize RunLogger
    if run_logger is None:
        run_logger = RunLogger(cfg=config, results_root=output_dir or Path("results"))
    run_dir = run_logger.run_dir
    ckpt_dir = run_dir / "ckpt"

    # Thermal guard setup
    if thermal_guard is None:
        thermal_guard = ThermalGuard(
            soft_c=float(config.get("thermal_soft_c", 70.0)),
            hard_c=float(config.get("thermal_hard_c", 75.0)),
            resume_c=float(config.get("thermal_resume_c", 55.0)),
            check_every_iters=int(config.get("thermal_check_iters", 20)),
            max_wait_s=float(config.get("thermal_max_wait_s", 600.0)),
            pause_sleep_s=float(config.get("thermal_pause_sleep_s", 2.0)),
        )

    # Initialize model
    if model is None:
        model = build_model(num_classes=3, pretrained=True, bn_mode=bn_mode)
    model = model.to(device, memory_format=torch.channels_last)

    # Data loaders
    data_dir = config.get("data_dir", "data")
    site_balanced = bool(config.get("site_balanced", False))
    train_loader = create_dataloader(
        train_df,
        batch_size=batch_size,
        is_train=True,
        data_dir=data_dir,
        site_balanced=site_balanced,
    )
    val_loader = create_dataloader(val_df, batch_size=batch_size, is_train=False, data_dir=data_dir)

    # Loss function with class weights
    class_weights = compute_class_weights(train_df, num_classes=3).to(device)
    loss_fn = MaskedCrossEntropyLoss(weight=class_weights)

    # Optimizer & Scheduler
    param_groups = model.get_parameter_groups(backbone_lr=backbone_lr, head_lr_multiplier=head_lr_mult)
    optimizer = AdamW(param_groups, weight_decay=weight_decay)

    if lr_schedule == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir)

    # Training state tracking
    best_val_auc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    start_time = time.time()
    total_paused_s = 0.0
    start_epoch = 1

    resume_mode = config.get("resume", False)
    if (resume_mode is True or resume_mode == "auto") and ckpt_mgr.last_path.exists():
        try:
            ep, step, best_metric, extra = ckpt_mgr.load(
                ckpt_path=ckpt_mgr.last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
            )
            start_epoch = ep + 1
            if best_metric is not None:
                best_val_auc = float(best_metric)
            best_epoch = int(extra.get("best_epoch", ep))
            epochs_no_improve = int(extra.get("epochs_no_improve", 0))
            logger.info("Resumed from checkpoint %s at epoch %d (best val AUC: %.4f)", ckpt_mgr.last_path, start_epoch, best_val_auc)
        except Exception as e:
            logger.warning("Failed to resume from %s (%s). Starting from scratch.", ckpt_mgr.last_path, e)

    logger.info(
        "Starting training [%s]: site=%s, seed=%d, bn_mode=%s, lr=%.1e, max_epochs=%d (from epoch %d)",
        run_logger.run_id, site_name, seed, bn_mode, backbone_lr, max_epochs, start_epoch
    )

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            # Thermal check inside training loop
            thermal_guard.step(
                on_hard=lambda: ckpt_mgr.save(
                    epoch=epoch,
                    global_step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    metric_value=best_val_auc,
                    is_best=False,
                )
            )
            total_paused_s = thermal_guard.total_paused_seconds

            imgs = batch["image"].to(device, memory_format=torch.channels_last)
            labels = batch["label"].to(device)
            allowed_masks = batch["allowed_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                loss = loss_fn(logits, labels, allowed_masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            train_loss += float(loss.item())
            n_train_batches += 1

        if scheduler is not None:
            scheduler.step()

        epoch_train_loss = train_loss / max(1, n_train_batches)
        epoch_dur = time.time() - epoch_start

        # Evaluate on validation set
        val_loss, val_breast_df, val_metrics = evaluate_model(model, val_loader, device, loss_fn)
        val_macro_auc = val_metrics.get("macro_auc", 0.5)
        val_c2_auc = val_metrics.get("auc_class_2_vs_rest", 0.5)
        val_f1 = val_metrics.get("macro_f1", 0.0)

        # Check for improvement
        is_best = val_macro_auc > best_val_auc
        if is_best:
            best_val_auc = val_macro_auc
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Checkpoint save
        ckpt_mgr.save(
            epoch=epoch,
            global_step=epoch * len(train_loader),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metric_value=best_val_auc,
            is_best=is_best,
        )

        # Log epoch metrics to metrics.jsonl
        current_lr = float(optimizer.param_groups[0]["lr"])
        run_logger.log_metric({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_auc": round(val_macro_auc, 4),
            "val_class_2_auc": round(val_c2_auc, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_macro_precision": round(val_metrics.get("macro_precision", 0.0), 4),
            "val_macro_recall": round(val_metrics.get("macro_recall", 0.0), 4),
            "val_macro_sensitivity": round(val_metrics.get("macro_sensitivity", 0.0), 4),
            "val_macro_specificity": round(val_metrics.get("macro_specificity", 0.0), 4),
            "val_sens_at_90_spec": round(val_metrics.get("sens_at_90_spec", 0.0), 4),
            "lr": current_lr,
            "epoch_duration_s": round(epoch_dur, 2),
            "paused_s": round(total_paused_s, 2),
            "is_best": is_best,
        })

        logger.info(
            "Epoch %d/%d (%.1fs) | Train Loss: %.4f | Val Loss: %.4f | Val Macro-AUC: %.4f (Best: %.4f @ ep%d) | Patience: %d/%d",
            epoch, max_epochs, epoch_dur, epoch_train_loss, val_loss, val_macro_auc, best_val_auc, best_epoch, epochs_no_improve, patience
        )

        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered at epoch %d (patience %d reached)", epoch, patience)
            break

    total_wall_time = time.time() - start_time

    # Load best checkpoint for final evaluation
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        ckpt_mgr.load(best_ckpt, model=model)

    # Final evaluation on validation set with best weights
    _, final_val_breast_df, final_val_metrics = evaluate_model(model, val_loader, device, loss_fn)

    # Final evaluation on held-out test set if provided
    final_test_metrics = {}
    test_ci = {}
    if test_df is not None and not test_df.empty:
        test_loader = create_dataloader(test_df, batch_size=batch_size, is_train=False, data_dir=data_dir)
        _, test_breast_df, final_test_metrics = evaluate_model(model, test_loader, device, loss_fn)

        # 1,000 patient-resampled test bootstrap CI
        test_ci = compute_patient_bootstrap_ci(test_breast_df, n_bootstraps=1000, seed=seed)

        # If pooled dataset, evaluate per-site test metrics
        if test_breast_df["site"].nunique() > 1:
            per_site_metrics = {}
            for s_name, s_grp in test_breast_df.groupby("site"):
                s_metrics = evaluate_predictions(s_grp)
                per_site_metrics[s_name] = s_metrics
                logger.info(
                    "  [Pooled Test] Site [%s]: Macro-AUC=%.4f | Class-2 AUC=%.4f | Macro-F1=%.4f",
                    s_name,
                    s_metrics.get("macro_auc", 0.0),
                    s_metrics.get("auc_class_2_vs_rest", 0.0),
                    s_metrics.get("macro_f1", 0.0),
                )
            final_test_metrics["per_site"] = per_site_metrics

        # Save test breast predictions to preds/
        preds_dir = run_dir / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)
        pred_path = preds_dir / "test_breast_predictions.npz"
        np.savez_compressed(
            pred_path,
            patient_hashes=test_breast_df["patient_hash"].to_numpy(),
            sites=test_breast_df["site"].to_numpy(),
            lateralities=test_breast_df["laterality"].to_numpy(),
            labels=test_breast_df["label"].to_numpy(),
            prob_0=test_breast_df["prob_0"].to_numpy(),
            prob_1=test_breast_df["prob_1"].to_numpy(),
            prob_2=test_breast_df["prob_2"].to_numpy(),
            logits=np.vstack(test_breast_df["logits"].to_numpy()),
        )
        logger.info("Saved test breast predictions to %s", pred_path)

        # Save comprehensive test metrics JSON
        test_metrics_json = preds_dir / "test_metrics.json"
        try:
            with open(test_metrics_json, "w") as fp:
                json.dump({
                    "test_metrics": final_test_metrics,
                    "test_ci": test_ci,
                }, fp, indent=2, default=str)
            logger.info("Saved test metrics (AUC, F1, precision, recall, specificity, sensitivity) to %s", test_metrics_json)
        except Exception as e:
            logger.warning("Failed to dump %s: %s (non-fatal, continuing to ledger append)", test_metrics_json, e)

    # Append row to results/ledger.csv per Rule 3.3
    primary_metric_val = final_test_metrics.get("macro_auc", final_val_metrics.get("macro_auc", 0.5))
    method_name = config.get("method", "local_baseline")
    supersedes = config.get("supersedes", None)
    run_logger.append_ledger(
        experiment_id=experiment_id,
        method=method_name,
        site=site_name,
        seed=seed,
        primary_metric="macro_auc",
        metric_value=float(primary_metric_val) if not np.isnan(primary_metric_val) else 0.5,
        wall_time_s=total_wall_time,
        paused_time_s=total_paused_s,
        status="completed",
        bytes_transferred=0,
        supersedes=supersedes,
    )

    logger.info(
        "Finished run [%s]: Best Val Macro-AUC: %.4f | Test Macro-AUC: %s | Time: %.1fs (Paused: %.1fs)",
        run_logger.run_id, best_val_auc,
        f"{final_test_metrics.get('macro_auc', 0.0):.4f}" if test_df is not None else "N/A",
        total_wall_time, total_paused_s,
    )

    return {
        "run_id": run_logger.run_id,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "val_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
        "test_ci": test_ci,
        "model": model,
    }
