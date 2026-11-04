"""Federated client representation and local training loop for mammography hospitals.

Adheres strictly to AGENTS.md Section 2.2, 2.3 & Decision D-008:
  - Constant LR: backbone_lr=1e-4, head_lr=1e-3 (10x multiplier). No restarted cosine schedules.
  - Reuses MammographyDataset, MaskedCrossEntropyLoss, and MammographyResNet18.
  - Supports FedProx proximal loss term (mu > 0.0).
  - Explicit BatchNorm handling (preserves local BN parameters in FedBN mode).
  - Computes byte overhead for telemetry.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from torch.optim import AdamW

from src.data.dataset import (
    MaskedCrossEntropyLoss,
    compute_class_weights,
    create_dataloader,
)
from src.fl.aggregation import is_batchnorm_key
from src.models.backbone import build_model
from src.utils.thermal import ThermalGuard

logger = logging.getLogger(__name__)


def compute_state_dict_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """Compute total payload size in bytes for a PyTorch state dict."""
    total_bytes = 0
    for tensor in state_dict.values():
        total_bytes += tensor.nelement() * tensor.element_size()
    return total_bytes


class FLClient:
    """Hospital silo client executing local epochs and computing parameter updates."""

    def __init__(
        self,
        site: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        data_dir: Path | str = "data",
        device: torch.device | None = None,
        backbone_lr: float = 1e-4,
        local_epochs: int = 1,
        batch_size: int = 16,
        use_amp: bool = True,
        thermal_guard: ThermalGuard | None = None,
        per_family_heads: bool = False,
    ):
        self.site = site
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True) if val_df is not None else None
        self.data_dir = Path(data_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_lr = backbone_lr
        self.head_lr = backbone_lr * 10.0
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.thermal_guard = thermal_guard
        self.per_family_heads = per_family_heads
        self.label_family = "birads" if self.site in ["vindr", "inbreast"] else "biopsy"

        # Local model instance
        self.model = build_model(
            num_classes=3, pretrained=False, bn_mode="trainable", per_family_heads=self.per_family_heads
        )
        self.model.to(self.device, memory_format=torch.channels_last)

        # Local class weights and masked loss
        self.class_weights = compute_class_weights(self.train_df, num_classes=3).to(self.device)
        self.loss_fn = MaskedCrossEntropyLoss(weight=self.class_weights)

        # Data loaders
        self.train_loader = create_dataloader(
            self.train_df,
            data_dir=self.data_dir,
            is_train=True,
            batch_size=self.batch_size,
        )

        # Store persistent local BN state for FedBN
        self.local_bn_state: Dict[str, torch.Tensor] = {}

    @property
    def num_samples(self) -> int:
        """Total local training samples (images)."""
        return len(self.train_df)

    def set_global_weights(self, global_state_dict: Dict[str, torch.Tensor], is_fedbn: bool = False):
        """Load global weights into local model, preserving local BN if in FedBN mode."""
        current_state = self.model.state_dict()
        new_state = {}

        for k, v in global_state_dict.items():
            if is_fedbn and is_batchnorm_key(k):
                # In FedBN, keep client's own BatchNorm parameters
                if k in self.local_bn_state:
                    new_state[k] = self.local_bn_state[k].clone()
                else:
                    new_state[k] = current_state[k].clone()
            else:
                new_state[k] = v.clone()

        self.model.load_state_dict(new_state)

    def train_round(
        self,
        round_idx: int,
        mu: float = 0.0,
        is_fedbn: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], int, float, int]:
        """Execute local training for this round.

        Args:
            round_idx: Current global FL communication round.
            mu: FedProx proximal coefficient (0.0 for FedAvg).
            is_fedbn: If True, preserves local BatchNorm parameters.

        Returns:
            updated_state_dict: Serialized client parameters.
            num_samples: Number of training samples.
            avg_train_loss: Mean training loss over local epochs.
            bytes_transferred: Payload size in bytes transmitted to server.
        """
        self.model.train()

        # Optimizer: Fresh AdamW with CONSTANT LR (Decision D-008 & AGENTS.md Section 2.3)
        if self.per_family_heads:
            param_groups = self.model.get_parameter_groups_for_family(
                self.label_family, backbone_lr=self.backbone_lr, head_lr_multiplier=10.0
            )
        else:
            param_groups = self.model.get_parameter_groups(backbone_lr=self.backbone_lr, head_lr_multiplier=10.0)
        optimizer = AdamW(param_groups, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # If FedProx, snapshot initial global model weights for proximal regularization
        global_params_snapshot = None
        if mu > 0.0:
            global_params_snapshot = [p.detach().clone() for p in self.model.parameters()]

        total_loss = 0.0
        n_batches = 0

        for epoch in range(self.local_epochs):
            for step, batch in enumerate(self.train_loader):
                if self.thermal_guard is not None:
                    self.thermal_guard.step()

                imgs = batch["image"].to(self.device, memory_format=torch.channels_last)
                labels = batch["label"].to(self.device)
                allowed_masks = batch["allowed_mask"].to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    if self.per_family_heads:
                        logits = self.model(imgs, family=self.label_family)
                    else:
                        logits = self.model(imgs)
                    loss = self.loss_fn(logits, labels, allowed_masks)

                    # FedProx Proximal Regularization Term: (mu / 2) * ||theta - theta_t||^2
                    if mu > 0.0 and global_params_snapshot is not None:
                        prox_term = 0.0
                        for p, p_init in zip(self.model.parameters(), global_params_snapshot):
                            prox_term += torch.sum((p - p_init) ** 2)
                        loss += (mu / 2.0) * prox_term

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                total_loss += float(loss.item())
                n_batches += 1

        avg_loss = total_loss / max(1, n_batches)

        # Extract updated state dict (CPU tensors)
        updated_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        # Save local BN parameters if FedBN
        if is_fedbn:
            for k, v in updated_state.items():
                if is_batchnorm_key(k):
                    self.local_bn_state[k] = v.clone()

        bytes_sent = compute_state_dict_bytes(updated_state)
        return updated_state, self.num_samples, avg_loss, bytes_sent
