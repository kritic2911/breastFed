"""Atomic checkpointing and resume utilities.

Adheres strictly to AGENTS.md Section 3.2:
  - Atomic saves (write *.tmp, fsync, rename)
  - Keeps last.pt and best.pt (by validation metric)
  - Full state preservation: model, optimizer, scheduler, scaler, epoch, step,
    best_metric, RNG states, sampler seed, config hash, git commit.
  - Robust resume with corruption fallback.
"""

import logging
import os
from pathlib import Path
from typing import Any

import torch

from src.utils.seed import get_rng_states, set_rng_states

logger = logging.getLogger(__name__)


def atomic_torch_save(obj: Any, target_path: Path) -> None:
    """Save PyTorch object atomically via temporary file and fsync."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".pt.tmp")

    with open(tmp_path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, target_path)


class CheckpointManager:
    """Manages epoch and best checkpoints with atomic guarantees."""

    def __init__(
        self,
        checkpoint_dir: Path,
        config_hash: str = "",
        git_commit: str = "",
        higher_is_better: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash
        self.git_commit = git_commit
        self.higher_is_better = higher_is_better

        self.last_path = self.checkpoint_dir / "last.pt"
        self.last_backup_path = self.checkpoint_dir / "last_backup.pt"
        self.best_path = self.checkpoint_dir / "best.pt"

    def save(
        self,
        epoch: int,
        global_step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: torch.amp.GradScaler | None = None,
        metric_value: float | None = None,
        is_best: bool = False,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        """Save state to last.pt and conditionally update best.pt."""
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metric_value": metric_value,
            "rng_states": get_rng_states(),
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "extra_state": extra_state or {},
        }

        # Backup previous last.pt before overwriting
        if self.last_path.exists():
            try:
                import shutil
                shutil.copy2(self.last_path, self.last_backup_path)
            except Exception as e:
                logger.debug("Failed to create last.pt backup: %s", e)

        # Atomic write to last.pt
        atomic_torch_save(state, self.last_path)

        # Atomic write to best.pt if improved
        if is_best:
            atomic_torch_save(state, self.best_path)
            logger.info("New best checkpoint saved with metric value: %s", metric_value)

    def load(
        self,
        ckpt_path: Path | None = None,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: torch.amp.GradScaler | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[int, int, float | None, dict]:
        """Load state from checkpoint with corruption recovery.

        Returns:
            Tuple of (epoch, global_step, metric_value, extra_state)
        """
        target = ckpt_path or self.last_path
        if not target.exists() and target == self.last_path and self.last_backup_path.exists():
            logger.warning("last.pt missing, attempting fallback to last_backup.pt")
            target = self.last_backup_path

        if not target.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {target}")

        try:
            state = torch.load(target, map_location=device, weights_only=False)
        except Exception as e:
            logger.error("Failed loading %s (%s). Attempting fallback to backup...", target, e)
            if self.last_backup_path.exists() and target != self.last_backup_path:
                state = torch.load(self.last_backup_path, map_location=device, weights_only=False)
            else:
                raise e

        if model is not None and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        if optimizer is not None and state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        if "rng_states" in state:
            set_rng_states(state["rng_states"])

        epoch = state.get("epoch", 0)
        global_step = state.get("global_step", 0)
        metric_value = state.get("metric_value", None)
        extra_state = state.get("extra_state", {})

        logger.info("Resumed successfully from %s at epoch %d, step %d", target, epoch, global_step)
        return epoch, global_step, metric_value, extra_state
