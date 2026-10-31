"""Training loop with early stopping on validation AUC."""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

from src.model import get_trainable_params, freeze_batchnorm
from src.evaluate import predict, compute_metrics
from src.thermal import gpu_thermal_cooldown
from src.dataset import compute_pos_weight


# ────────────────────────────────────────────────────────────────
#  Training loop
# ────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, cfg, device="cpu",
                max_epochs=None, patience=None, verbose=True,
                checkpoint_dir=None, resume=True):
    """Standard training with early stopping on val AUC and GPU-aware tracking.

    Key improvements over vanilla training:
    - Automatic pos_weight for BCEWithLogitsLoss (handles class imbalance)
    - Cosine annealing LR schedule with linear warmup
    - Gradient clipping for stability
    - Supports automatic resumption from last_checkpoint.pt if interrupted.

    Returns
    -------
    best_model_state : state dict of the best checkpoint
    history : list of dicts (one per epoch) with train_loss, val_auc, vram, timing, etc.
    """
    from pathlib import Path
    max_epochs = max_epochs or cfg["max_epochs"]
    patience = patience or cfg["patience"]
    device = torch.device(device)
    is_cuda = (device.type == "cuda")

    if verbose and is_cuda:
        gpu_name = torch.cuda.get_device_name(device)
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"  [GPU] Training on {gpu_name} (Total VRAM: {total_vram:.2f} GB)", flush=True)

    # ── Class-balanced loss ──
    # Compute pos_weight = n_neg / n_pos so the minority (positive) class
    # contributes proportionally more gradient.  This is the #1 fix for
    # BCDR (15% pos) and CBIS (40% pos) where the model previously just
    # learned to predict "negative" for everything.
    pos_weight = compute_pos_weight(train_loader).to(device)
    if verbose:
        print(f"  [Class Balance] pos_weight = {pos_weight.item():.2f} "
              f"(>1 means positives are upweighted)", flush=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer ──
    lr = cfg["lr"]
    optimizer = Adam(get_trainable_params(model),
                     lr=lr, weight_decay=cfg["weight_decay"])

    # ── LR Scheduler: linear warmup (5 epochs) → cosine decay ──
    warmup_epochs = cfg.get("warmup_epochs", 5)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer,
                                          T_max=max(max_epochs - warmup_epochs, 1),
                                          eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer,
                              schedulers=[warmup_scheduler, cosine_scheduler],
                              milestones=[warmup_epochs])

    # Gradient clipping threshold
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0
    history = []
    start_epoch = 1

    # Check for existing checkpoint to resume
    ckpt_path = Path(checkpoint_dir) / "last_checkpoint.pt" if checkpoint_dir else None
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        # Save training images manifest if possible
        manifest_path = Path(checkpoint_dir) / "train_images_manifest.csv"
        if not manifest_path.exists() and hasattr(train_loader.dataset, "df"):
            train_loader.dataset.df.to_csv(manifest_path, index=False)

    if resume and ckpt_path and ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            best_val_auc = ckpt["best_val_auc"]
            best_state = ckpt["best_state"]
            epochs_no_improve = ckpt["epochs_no_improve"]
            history = ckpt["history"]
            start_epoch = ckpt["epoch"] + 1
            # Advance scheduler to match resumed epoch
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            else:
                for _ in range(start_epoch - 1):
                    scheduler.step()
            if verbose:
                print(f"  [Resume] Resuming training from epoch {start_epoch}/{max_epochs} "
                      f"(found checkpoint at epoch {ckpt['epoch']}, best val AUC: {best_val_auc:.4f})")
        except Exception as err:
            if verbose:
                print(f"  [Resume Warning] Could not resume from {ckpt_path} ({err}), starting fresh.")

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        # Re-freeze BN if needed (model.train() un-freezes BN eval mode)
        if cfg.get("freeze_bn", True):
            freeze_batchnorm(model)

        train_loss = 0.0
        n_batches = 0
        train_batch_probs, train_batch_labels = [], []
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(get_trainable_params(model), max_grad_norm)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

            train_batch_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            train_batch_labels.append(labels.detach().cpu().numpy())

        train_loss /= max(n_batches, 1)
        all_train_probs = np.concatenate(train_batch_probs) if train_batch_probs else np.array([])
        all_train_labels = np.concatenate(train_batch_labels) if train_batch_labels else np.array([])
        train_metrics = compute_metrics(all_train_labels, all_train_probs) if len(np.unique(all_train_labels)) > 1 else {"auc": 0.5, "prob_std": 0.0}
        train_auc = train_metrics["auc"]
        train_prob_std = train_metrics.get("prob_std", float(np.std(all_train_probs)) if len(all_train_probs) else 0.0)

        # Step the LR scheduler after each epoch
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # ── Validate ──
        probs, val_labels, _ = predict(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, probs)
        val_auc = val_metrics["auc"]
        val_prob_std = val_metrics.get("prob_std", float(np.std(probs)))
        epoch_time = time.time() - t0

        vram_mb = (torch.cuda.memory_allocated(device) / (1024**2)) if is_cuda else 0.0

        # ── Early stopping ──
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            is_best = True
        else:
            epochs_no_improve += 1
            is_best = False

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_auc": round(train_auc, 5),
            "train_prob_std": round(train_prob_std, 5),
            "val_auc": round(val_auc, 5),
            "val_prob_std": round(val_prob_std, 5),
            "val_sensitivity": round(val_metrics["sensitivity"], 5),
            "val_specificity": round(val_metrics["specificity"], 5),
            "best_val_auc": round(best_val_auc, 5),
            "lr": round(current_lr, 7),
            "time_sec": round(epoch_time, 2),
            "vram_mb": round(vram_mb, 1),
            "improved": is_best,
        }
        history.append(record)

        # Save latest epoch checkpoint for crash resumption
        if checkpoint_dir and ckpt_path:
            ckpt_data = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_auc": best_val_auc,
                "best_state": best_state,
                "epochs_no_improve": epochs_no_improve,
                "history": history,
            }
            torch.save(ckpt_data, ckpt_path)

        if verbose:
            gpu_tag = f" | VRAM {vram_mb:4.0f}MB" if is_cuda else ""
            status_tag = "★ BEST" if is_best else f"no-improve {epochs_no_improve}/{patience}"
            print(f"  Epoch {epoch:2d}/{max_epochs:2d} ({epoch_time:4.1f}s{gpu_tag}) | "
                  f"Loss {train_loss:.4f} | Train AUC {train_auc:.4f} (std {train_prob_std:.3f}) | "
                  f"Val AUC {val_auc:.4f} (std {val_prob_std:.3f}) | "
                  f"LR {current_lr:.1e} | Best {best_val_auc:.4f} [{status_tag}]", flush=True)

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stopping triggered at epoch {epoch} "
                      f"(no improvement for {patience} consecutive epochs)", flush=True)
            break

        # Cooldown pause if training continues to next epoch
        if epoch < max_epochs:
            gpu_thermal_cooldown(cfg, verbose=verbose, epoch=epoch)

    return best_state, history


# ────────────────────────────────────────────────────────────────
#  Federated local training (limited epochs, no early stopping)
# ────────────────────────────────────────────────────────────────

def train_local_epochs(model, train_loader, cfg, device="cpu",
                       n_epochs=None, lr=None, global_params=None, mu=0.0):
    """Train for a fixed number of epochs (used inside each FedAvg / FedProx round).

    Supports:
    - Standard FedAvg (mu=0)
    - FedProx proximal regularization: Loss = BCE + (mu / 2) * ||w - w_global||^2
    - Adaptive round-based learning rate (lr)
    - Gradient clipping for numerical stability
    """
    n_epochs = n_epochs or cfg["fed"]["local_epochs"]
    learning_rate = lr if lr is not None else cfg["lr"]

    # Class-balanced loss for federated local training too
    pos_weight = compute_pos_weight(train_loader).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(get_trainable_params(model),
                     lr=learning_rate, weight_decay=cfg["weight_decay"])

    model.train()
    if cfg.get("freeze_bn", True):
        freeze_batchnorm(model)

    total_loss = 0.0
    n_batches = 0

    for _ in range(n_epochs):
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(-1)
            loss = criterion(logits, labels)

            # FedProx proximal regularization
            if mu > 0.0 and global_params is not None:
                prox = 0.0
                for w, w_t in zip(get_trainable_params(model), global_params):
                    prox = prox + torch.sum((w - w_t) ** 2)
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            torch.nn.utils.clip_grad_norm_(get_trainable_params(model), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return model, avg_loss


def set_seed(seed: int):
    """Set seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
