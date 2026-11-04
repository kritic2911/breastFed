"""Central Federated Learning (FL) server coordinating cross-silo rounds.

Adheres strictly to AGENTS.md Section 2.3, 3.1, 3.2, 3.3, Phase P3:
  - Supports FedAvg (sample-weighted, uniform), FedProx (mu in {0.01, 0.1}), FedBN.
  - Constant LR (Decision D-008: backbone_lr=1e-4, head_lr=1e-3).
  - Explicit BatchNorm handling: averaged in FedAvg, kept local in FedBN.
  - Checkpoint integrity: atomic saves (last.pt, best.pt), resumable mid-run.
  - Telemetry: metrics.jsonl, events.jsonl, bytes transferred, results/ledger.csv.
  - Offline evaluation on pooled validation set to select best global model.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch

from src.data.dataset import MaskedCrossEntropyLoss, create_dataloader
from src.eval.metrics import aggregate_breast_predictions, compute_patient_bootstrap_ci, evaluate_predictions
from src.fl.aggregation import aggregate_fedavg, aggregate_per_family_heads
from src.fl.client import FLClient, compute_state_dict_bytes
from src.models.backbone import build_model
from src.utils.ckpt import CheckpointManager
from src.utils.provenance import RunLogger
from src.utils.thermal import ThermalGuard

logger = logging.getLogger(__name__)


class FLServer:
    """Orchestrates communication rounds, aggregation, validation, and test evaluation."""

    def __init__(
        self,
        clients: List[FLClient],
        val_df: pd.DataFrame,
        test_dfs: Dict[str, pd.DataFrame],
        strategy: Literal["fedavg", "fedprox", "fedbn", "per_family_heads"] = "fedavg",
        aggregation_mode: Literal["sample_weighted", "uniform"] = "sample_weighted",
        mu: float = 0.0,
        num_rounds: int = 20,
        run_dir: Path | str = "results/runs/fl_run",
        experiment_id: str = "fl_simulation",
        seed: int = 42,
        data_dir: Path | str = "data",
        device: torch.device | None = None,
        thermal_guard: ThermalGuard | None = None,
    ):
        self.clients = clients
        self.val_df = val_df.reset_index(drop=True)
        self.test_dfs = {k: v.reset_index(drop=True) for k, v in test_dfs.items()}
        self.strategy = strategy
        self.aggregation_mode = aggregation_mode
        self.mu = mu if strategy == "fedprox" else 0.0
        self.is_fedbn = (strategy == "fedbn")
        self.is_per_family = (strategy == "per_family_heads")
        self.num_rounds = num_rounds
        self.run_dir = Path(run_dir)
        self.experiment_id = experiment_id
        self.seed = seed
        self.data_dir = Path(data_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.thermal_guard = thermal_guard

        # Initialize global model
        self.model = build_model(
            num_classes=3, pretrained=True, bn_mode="trainable", per_family_heads=self.is_per_family
        )
        self.model.to(self.device, memory_format=torch.channels_last)

        # Global loss function for validation
        self.loss_fn = MaskedCrossEntropyLoss()

        # Validation DataLoader
        self.val_loader = create_dataloader(
            self.val_df,
            data_dir=self.data_dir,
            is_train=False,
            batch_size=32,
        )

        # Logging and checkpoint management
        fl_cfg = {
            "experiment_id": self.experiment_id,
            "strategy": self.strategy,
            "aggregation_mode": self.aggregation_mode,
            "mu": self.mu,
            "num_rounds": self.num_rounds,
            "seed": self.seed,
            "clients": [c.site for c in self.clients],
        }
        self.run_logger = RunLogger(cfg=fl_cfg, run_dir=self.run_dir)
        self.ckpt_mgr = CheckpointManager(self.run_logger.ckpt_dir)

        # State tracking
        self.current_round = 0
        self.best_val_macro_auc = 0.0
        self.best_round = 0
        self.total_bytes_transferred = 0

    def evaluate_global_model(self, loader: torch.utils.data.DataLoader) -> Tuple[float, pd.DataFrame, Dict[str, Any]]:
        """Evaluate global model on a DataLoader (validation or test)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        pred_records = []

        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(self.device, memory_format=torch.channels_last)
                labels = batch["label"].to(self.device)
                allowed_masks = batch["allowed_mask"].to(self.device)

                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    if self.is_per_family:
                        site_names = batch["site"]
                        logits = torch.empty((len(imgs), 3), device=self.device)
                        for fam in ["birads", "biopsy"]:
                            fam_sites = ["vindr", "inbreast"] if fam == "birads" else ["cmmd", "cdd"]
                            mask = [s in fam_sites for s in site_names]
                            if any(mask):
                                mask_tensor = torch.tensor(mask, device=self.device)
                                sub_imgs = imgs[mask_tensor]
                                logits[mask_tensor] = self.model(sub_imgs, family=fam).to(dtype=logits.dtype)
                    else:
                        logits = self.model(imgs)
                    loss = self.loss_fn(logits, labels, allowed_masks)

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
        metrics = evaluate_predictions(breast_df)
        return avg_loss, breast_df, metrics

    def run(self, resume: bool = True) -> Dict[str, Any]:
        """Execute federated learning simulation across communication rounds."""
        start_round = 1

        # Check for resumable checkpoint
        if resume and self.ckpt_mgr.last_path.exists():
            try:
                ep, step, best_metric, extra = self.ckpt_mgr.load(
                    ckpt_path=self.ckpt_mgr.last_path,
                    model=self.model,
                    device=self.device,
                )
                start_round = ep + 1
                if best_metric is not None:
                    self.best_val_macro_auc = float(best_metric)
                self.best_round = int(extra.get("best_round", ep))
                self.total_bytes_transferred = int(extra.get("total_bytes_transferred", 0))
                logger.info(
                    "Resumed FL Server from %s at round %d (best: %.4f @ round %d)",
                    self.ckpt_mgr.last_path, start_round - 1, self.best_val_macro_auc, self.best_round
                )
            except Exception as e:
                logger.warning("Failed resuming from last checkpoint: %s. Starting fresh.", e)

        logger.info(
            "Starting FL Simulation [%s]: Strategy=%s, Aggregation=%s, Rounds=%d (from %d), Clients=%d",
            self.experiment_id, self.strategy, self.aggregation_mode, self.num_rounds, start_round, len(self.clients)
        )

        global_start_time = time.time()

        for round_idx in range(start_round, self.num_rounds + 1):
            round_start = time.time()
            self.current_round = round_idx

            if self.thermal_guard is not None:
                self.thermal_guard.step()

            # 1. Distribute global model to participating clients
            global_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            bytes_broadcast = compute_state_dict_bytes(global_state) * len(self.clients)
            self.total_bytes_transferred += bytes_broadcast

            for client in self.clients:
                client.set_global_weights(global_state, is_fedbn=self.is_fedbn)

            # 2. Local client training
            client_updates = []
            client_counts = []
            client_losses = []
            round_uplink_bytes = 0

            for client in self.clients:
                if self.thermal_guard is not None:
                    self.thermal_guard.step()

                upd_state, n_samples, train_loss, bytes_sent = client.train_round(
                    round_idx=round_idx,
                    mu=self.mu,
                    is_fedbn=self.is_fedbn,
                )
                client_updates.append(upd_state)
                client_counts.append(n_samples)
                client_losses.append(train_loss)
                round_uplink_bytes += bytes_sent

            self.total_bytes_transferred += round_uplink_bytes
            round_bytes = bytes_broadcast + round_uplink_bytes
            avg_train_loss = float(np.mean(client_losses))

            # 3. Server Aggregation (FedAvg / FedBN / Per-Family Heads)
            if self.thermal_guard is not None:
                self.thermal_guard.step()

            if self.is_per_family:
                client_families = [
                    getattr(c, "label_family", "birads" if c.site in ["vindr", "inbreast"] else "biopsy")
                    for c in self.clients
                ]
                new_global_state = aggregate_per_family_heads(
                    client_updates=client_updates,
                    client_sample_counts=client_counts,
                    client_families=client_families,
                    mode=self.aggregation_mode,
                    include_bn_stats=(not self.is_fedbn),
                )
            else:
                new_global_state = aggregate_fedavg(
                    client_updates=client_updates,
                    client_sample_counts=client_counts,
                    mode=self.aggregation_mode,
                    include_bn_stats=(not self.is_fedbn),
                )
            self.model.load_state_dict(new_global_state, strict=(not self.is_fedbn))

            # 4. Global Validation (on pooled validation cohort)
            if self.is_per_family:
                val_loss, val_breast_df, _ = self.evaluate_global_model(self.val_loader)
                site_val_aucs = []
                site_c2_aucs = []
                site_f1s = []
                for s_name, s_grp in val_breast_df.groupby("site"):
                    m = evaluate_predictions(s_grp)
                    if "macro_auc" in m:
                        site_val_aucs.append(m["macro_auc"])
                    if "auc_class_2_vs_rest" in m:
                        site_c2_aucs.append(m["auc_class_2_vs_rest"])
                    if "macro_f1" in m:
                        site_f1s.append(m["macro_f1"])
                val_macro_auc = float(np.mean(site_val_aucs)) if site_val_aucs else 0.5
                val_class_2_auc = float(np.mean(site_c2_aucs)) if site_c2_aucs else 0.5
                val_f1 = float(np.mean(site_f1s)) if site_f1s else 0.0
            else:
                val_loss, _, val_metrics = self.evaluate_global_model(self.val_loader)
                val_macro_auc = val_metrics.get("macro_auc", 0.0)
                val_class_2_auc = val_metrics.get("auc_class_2_vs_rest", 0.0)
                val_f1 = val_metrics.get("macro_f1", 0.0)

            is_best = val_macro_auc > self.best_val_macro_auc
            if is_best:
                self.best_val_macro_auc = val_macro_auc
                self.best_round = round_idx

            round_dur = time.time() - round_start
            paused_s = self.thermal_guard.total_paused_seconds if self.thermal_guard else 0.0

            logger.info(
                "Round [%02d/%02d] (%.1fs) | Train Loss: %.4f | Val Loss: %.4f | Val Macro-AUC: %.4f (Best: %.4f @ R%02d) | Bytes: %.2f MB",
                round_idx, self.num_rounds, round_dur, avg_train_loss, val_loss, val_macro_auc, self.best_val_macro_auc, self.best_round, round_bytes / 1e6
            )

            # Log metrics row
            self.run_logger.log_metric({
                "round": round_idx,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_macro_auc": round(val_macro_auc, 4),
                "val_class_2_auc": round(val_class_2_auc, 4),
                "val_macro_f1": round(val_f1, 4),
                "bytes_round": round_bytes,
                "bytes_total": self.total_bytes_transferred,
                "round_duration_s": round(round_dur, 2),
                "paused_s": round(paused_s, 2),
                "is_best": is_best,
            })

            # 5. Atomic Checkpointing
            extra_state = {
                "best_round": self.best_round,
                "total_bytes_transferred": self.total_bytes_transferred,
                "strategy": self.strategy,
                "aggregation_mode": self.aggregation_mode,
            }
            self.ckpt_mgr.save(
                epoch=round_idx,
                global_step=round_idx * len(self.clients),
                model=self.model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                metric_value=self.best_val_macro_auc,
                is_best=is_best,
                extra_state=extra_state,
            )

        total_wall_s = time.time() - global_start_time

        # 6. Final Evaluation: Load best model checkpoint and evaluate on site test splits
        logger.info("FL Simulation completed. Evaluating best model (Round %d) on test splits...", self.best_round)
        if self.ckpt_mgr.best_path.exists():
            self.ckpt_mgr.load(self.ckpt_mgr.best_path, model=self.model, device=self.device)
        elif self.ckpt_mgr.last_path.exists():
            self.ckpt_mgr.load(self.ckpt_mgr.last_path, model=self.model, device=self.device)

        test_results = self.evaluate_test_splits()

        # Append to results/ledger.csv
        paused_total = self.thermal_guard.total_paused_seconds if self.thermal_guard else 0.0
        self.run_logger.append_ledger(
            experiment_id=self.experiment_id,
            method=self.strategy,
            site="federated",
            seed=self.seed,
            primary_metric="macro_auc",
            metric_value=float(test_results.get("overall_macro_auc", 0.0)),
            wall_time_s=total_wall_s,
            paused_time_s=paused_total,
            bytes_transferred=self.total_bytes_transferred,
            status="completed",
        )

        return test_results

    def evaluate_test_splits(self) -> Dict[str, Any]:
        """Evaluate best global model on each hospital site's held-out test split."""
        preds_dir = self.run_dir / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)

        all_breast_dfs = []
        per_site_metrics = {}

        for site_name, test_df in self.test_dfs.items():
            loader = create_dataloader(test_df, data_dir=self.data_dir, is_train=False, batch_size=32)
            _, breast_df, s_metrics = self.evaluate_global_model(loader)
            all_breast_dfs.append(breast_df)
            per_site_metrics[site_name] = s_metrics
            logger.info("  [FL Test] Site [%s]: Macro-AUC=%.4f | Class-2 AUC=%.4f | F1=%.4f",
                        site_name, s_metrics.get("macro_auc", 0.0), s_metrics.get("auc_class_2_vs_rest", 0.0), s_metrics.get("macro_f1", 0.0))

        # Overall pooled test metrics
        combined_test_df = pd.concat(all_breast_dfs, ignore_index=True)
        overall_metrics = evaluate_predictions(combined_test_df)

        # 1,000 patient-resampled test bootstrap CI
        test_ci = compute_patient_bootstrap_ci(combined_test_df, n_bootstraps=1000, seed=self.seed)

        # Worst-site AUC and cross-site gap per AGENTS.md Section 1.4 & Phase P3 Checkpoint
        site_aucs = [m.get("macro_auc", 0.0) for m in per_site_metrics.values() if "macro_auc" in m]
        worst_site_auc = float(min(site_aucs)) if site_aucs else 0.0
        best_site_auc = float(max(site_aucs)) if site_aucs else 0.0
        cross_site_gap = float(best_site_auc - worst_site_auc) if site_aucs else 0.0

        # Per label-family metrics (birads vs biopsy)
        per_family_metrics = {}
        combined_test_df["label_family"] = combined_test_df["site"].map(
            lambda s: "birads" if s in ["vindr", "inbreast"] else "biopsy"
        )
        for fam_name, fam_grp in combined_test_df.groupby("label_family"):
            per_family_metrics[fam_name] = evaluate_predictions(fam_grp)

        # Save test breast predictions
        pred_out_path = preds_dir / "test_breast_predictions.npz"
        np.savez_compressed(
            pred_out_path,
            patient_hashes=combined_test_df["patient_hash"].to_numpy(),
            sites=combined_test_df["site"].to_numpy(),
            lateralities=combined_test_df["laterality"].to_numpy(),
            labels=combined_test_df["label"].to_numpy(),
            prob_0=combined_test_df["prob_0"].to_numpy(),
            prob_1=combined_test_df["prob_1"].to_numpy(),
            prob_2=combined_test_df["prob_2"].to_numpy(),
            logits=np.vstack(combined_test_df["logits"].to_numpy()),
        )

        overall_macro_auc_val = float(np.mean(site_aucs)) if (self.is_per_family and site_aucs) else float(overall_metrics.get("macro_auc", 0.0))

        test_payload = {
            "run_id": self.run_dir.name,
            "experiment_id": self.experiment_id,
            "strategy": self.strategy,
            "aggregation_mode": self.aggregation_mode,
            "seed": self.seed,
            "best_round": self.best_round,
            "total_bytes_transferred": self.total_bytes_transferred,
            "overall_macro_auc": overall_macro_auc_val,
            "worst_site_auc": worst_site_auc,
            "cross_site_gap": cross_site_gap,
            "overall_metrics": overall_metrics,
            "per_site_metrics": per_site_metrics,
            "per_family_metrics": per_family_metrics,
            "test_ci": test_ci,
        }

        with open(preds_dir / "test_metrics.json", "w") as fp:
            json.dump(test_payload, fp, indent=2, default=str)

        return test_payload
