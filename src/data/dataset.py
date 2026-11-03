"""PyTorch dataset and data loaders for mammography classification.

Adheres strictly to AGENTS.md Section 1.3, 2.2:
  - Loads 512x512 cached float arrays.
  - Light data augmentation for training: random horizontal flip, small rotation/translation/scale,
    and brightness/contrast jitter.
  - Stratified patient-grouped train/val partitioning per seed (fixed held-out test split).
  - Masked cross-entropy loss accounting for site-specific available classes (CMMD classes {1, 2}).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SITE_AVAILABLE_CLASSES: Dict[str, List[int]] = {
    "vindr": [0, 1, 2],
    "inbreast": [0, 1, 2],
    "cdd": [0, 1, 2],
    "cmmd": [1, 2],
}


def get_train_val_test_splits(
    manifest_df: pd.DataFrame,
    seed: int = 42,
    val_size: float = 0.20,
    site: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition manifest into train, validation, and fixed held-out test sets.

    Adheres strictly to AGENTS.md Section 1.3:
      - Fixed held-out test set (~20% of patients per site) is identical across seeds.
      - Train cohort is partitioned into train (80%) and validation (20%) per seed.
      - Zero patient leakage between splits.
      - CC and MLO views of each patient remain together.
    """
    df = manifest_df.copy()
    if site is not None:
        df = df[df["site"] == site].copy()

    test_df = df[df["split"] == "test"].copy()
    train_pool = df[df["split"] == "train"].copy()

    # Stratified patient-grouped split of the training pool into train and val
    val_rows = []
    train_rows = []

    for s, grp in train_pool.groupby("site"):
        pt_labels = grp.groupby("patient_hash")["label"].max()
        pts = pt_labels.index.to_numpy()
        pt_y = pt_labels.to_numpy()

        # Handle sites with very small patient pools gracefully
        unique_classes = np.unique(pt_y)
        stratify = pt_y if len(unique_classes) > 1 and all(np.sum(pt_y == c) >= 2 for c in unique_classes) else None

        tr_pts, val_pts = train_test_split(
            pts,
            test_size=val_size,
            stratify=stratify,
            random_state=seed,
        )
        val_set = set(val_pts)
        val_rows.append(grp[grp["patient_hash"].isin(val_set)])
        train_rows.append(grp[~grp["patient_hash"].isin(val_set)])

    val_df = pd.concat(val_rows, ignore_index=True)
    train_df = pd.concat(train_rows, ignore_index=True)

    # Verification: assert 0 patient overlap
    tr_p = set(train_df["patient_hash"])
    va_p = set(val_df["patient_hash"])
    te_p = set(test_df["patient_hash"])
    assert len(tr_p & va_p) == 0, f"Train-val leakage: {len(tr_p & va_p)} overlapping patients!"
    assert len(tr_p & te_p) == 0, f"Train-test leakage: {len(tr_p & te_p)} overlapping patients!"
    assert len(va_p & te_p) == 0, f"Val-test leakage: {len(va_p & te_p)} overlapping patients!"

    return train_df, val_df, test_df


def compute_class_weights(df: pd.DataFrame, num_classes: int = 3) -> torch.Tensor:
    """Compute inverse frequency class weights from training split."""
    counts = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        counts[c] = float((df["label"] == c).sum())

    total = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (len(np.where(counts > 0)[0]) * counts[c])
        else:
            weights[c] = 0.0

    return torch.tensor(weights, dtype=torch.float32)


class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with logits masked to each site's available classes.

    Adheres strictly to AGENTS.md Section 2.2:
      - CMMD has classes {1, 2} only; class 0 logit is set to -1e9.
      - Per-site / pooled class weights applied.
    """

    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        allowed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked cross entropy loss.

        Args:
            logits: (B, 3) raw model logits.
            targets: (B,) class labels in {0, 1, 2}.
            allowed_mask: (B, 3) binary mask where 1 = available class, 0 = unavailable.
        """
        # Mask unavailable classes with -1e9 so their softmax probability is 0
        masked_logits = logits.clone()
        unavailable = (allowed_mask == 0)
        masked_logits[unavailable] = -1e9

        return F.cross_entropy(masked_logits, targets, weight=self.weight)


class MammographyDataset(Dataset):
    """PyTorch Dataset loading preprocessed 512x512 digital mammograms."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path | str = "data",
        is_train: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """Apply light data augmentation for digital mammography."""
        # 1. Random horizontal flip (nipple orientation variation)
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()

        # 2. Random small affine rotation (+/- 8 deg) and scale (+/- 5%)
        h, w = img.shape
        angle = float(np.random.uniform(-8.0, 8.0))
        scale = float(np.random.uniform(0.95, 1.05))
        tx = float(np.random.uniform(-0.03 * w, 0.03 * w))
        ty = float(np.random.uniform(-0.03 * h, 0.03 * h))

        center = (w / 2.0, h / 2.0)
        m = cv2.getRotationMatrix2D(center, angle, scale)
        m[0, 2] += tx
        m[1, 2] += ty

        img = cv2.warpAffine(
            img,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        # 3. Random contrast and brightness jitter (+/- 8%)
        alpha = float(np.random.uniform(0.92, 1.08))  # contrast
        beta = float(np.random.uniform(-0.05, 0.05))   # brightness
        img = np.clip(img * alpha + beta, 0.0, 1.0)
        return img

    def __getitem__(self, idx: int) -> Dict[str, any]:
        row = self.df.iloc[idx]
        cached_rel = row["cached_path"]
        cached_path = self.data_dir / cached_rel

        if cached_path.exists():
            arr = np.load(cached_path)
        else:
            arr = np.zeros((512, 512), dtype=np.float32)

        if self.is_train:
            arr = self._apply_augmentation(arr)

        tensor_img = torch.from_numpy(arr).unsqueeze(0).float()  # (1, 512, 512)
        label = int(row["label"])
        site = str(row["site"])

        # Mask of allowed classes: [1, 1, 1] for 3-class sites, [0, 1, 1] for CMMD
        allowed = [0.0, 0.0, 0.0]
        for c in SITE_AVAILABLE_CLASSES.get(site, [0, 1, 2]):
            allowed[c] = 1.0
        allowed_mask = torch.tensor(allowed, dtype=torch.float32)

        return {
            "image": tensor_img,
            "label": torch.tensor(label, dtype=torch.long),
            "allowed_mask": allowed_mask,
            "site": site,
            "patient_hash": str(row["patient_hash"]),
            "laterality": str(row["laterality"]),
            "view": str(row["view"]),
            "image_id": str(row["image_id"]),
            "split": str(row["split"]),
        }


def create_dataloader(
    df: pd.DataFrame,
    batch_size: int = 16,
    is_train: bool = False,
    data_dir: Path | str = "data",
    num_workers: int = 4,
    site_balanced: bool = False,
) -> DataLoader:
    """Create PyTorch DataLoader with pinned memory.

    If site_balanced is True (used in centralized site-balanced baseline),
    samples are weighted inversely to site size via WeightedRandomSampler.
    """
    dataset = MammographyDataset(df, data_dir=data_dir, is_train=is_train)
    sampler = None
    shuffle = is_train

    if is_train and site_balanced:
        site_counts = df["site"].value_counts().to_dict()
        weights = df["site"].map(lambda s: 1.0 / max(1, site_counts.get(s, 1))).to_numpy()
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(df),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

