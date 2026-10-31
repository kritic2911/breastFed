"""Mammography dataset that reads from the unified labels_split.csv + proc512/ cache."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ────────────────────────────────────────────────────────────────
#  Transforms
# ────────────────────────────────────────────────────────────────

# Grayscale-symmetric ImageNet normalization: use the luminance-weighted
# average of the ImageNet per-channel stats so all 3 (identical) channels
# of a grayscale→RGB image get the same scaling. This avoids introducing
# an artificial colour bias on single-channel medical images.
GREY_MEAN = [0.449, 0.449, 0.449]   # avg of [0.485, 0.456, 0.406]
GREY_STD  = [0.226, 0.226, 0.226]   # avg of [0.229, 0.224, 0.225]


def build_transforms(cfg, split: str):
    """Return a torchvision transform pipeline from the config dict."""
    img_size = cfg["img_size"]

    if split == "train":
        ops = []
        # Stronger augmentation: RandomResizedCrop instead of plain Resize
        # gives scale+position jitter — crucial for small datasets
        ops.append(transforms.RandomResizedCrop(
            img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)))

        # Config-driven augmentations
        for aug in cfg.get("augmentation", {}).get("train", []):
            if isinstance(aug, str):
                ops.append(getattr(transforms, aug)())
            elif isinstance(aug, dict):
                for name, kwargs in aug.items():
                    ops.append(getattr(transforms, name)(**kwargs))

        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=GREY_MEAN, std=GREY_STD),
            # Random erasing acts as cutout regularisation — very effective
            # on small medical-imaging datasets to reduce overfitting
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
        ]
    else:
        ops = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=GREY_MEAN, std=GREY_STD),
        ]

    return transforms.Compose(ops)


# ────────────────────────────────────────────────────────────────
#  Dataset
# ────────────────────────────────────────────────────────────────

class MammoDataset(Dataset):
    """One site, one split.

    Parameters
    ----------
    data_root : str or Path
        e.g. "data"
    site : str
        e.g. "CMMD"
    split : str
        "train", "val", or "test"
    cfg : dict
        Parsed experiment.yaml
    """

    def __init__(self, data_root: str, site: str, split: str, cfg: dict):
        self.site = site
        self.split = split
        site_dir = Path(data_root) / site
        df = pd.read_csv(site_dir / cfg["labels_file"], dtype={"patient_id": str})
        self.df = df[df[cfg["split_col"]] == split].reset_index(drop=True)
        self.site_dir = site_dir
        self.image_col = cfg["image_col"]
        self.label_col = cfg["label_col"]
        self.patient_col = cfg["patient_col"]
        self.img_size = cfg.get("img_size", 224)
        self.cache_images = cfg.get("cache_images", True)
        self._cache = {}
        self.transform = build_transforms(cfg, split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.cache_images:
            if idx not in self._cache:
                img_path = self.site_dir / row[self.image_col]
                img = Image.open(img_path).convert("RGB")
                if img.size != (self.img_size, self.img_size):
                    img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                self._cache[idx] = img
            img = self._cache[idx]
        else:
            img_path = self.site_dir / row[self.image_col]
            img = Image.open(img_path).convert("RGB")

        img_tensor = self.transform(img)
        label = torch.tensor(row[self.label_col], dtype=torch.float32)
        patient_id = row[self.patient_col]
        return img_tensor, label, patient_id

    @property
    def labels(self):
        return self.df[self.label_col].to_numpy()

    @property
    def patient_ids(self):
        return self.df[self.patient_col].to_numpy()


# ────────────────────────────────────────────────────────────────
#  Class-weight helpers
# ────────────────────────────────────────────────────────────────

def compute_pos_weight(loader) -> torch.Tensor:
    """Compute pos_weight = n_neg / n_pos for BCEWithLogitsLoss.

    This re-weights the loss so that each positive sample contributes
    as much gradient as (n_neg / n_pos) negative samples, effectively
    balancing the class influence.  Critical for imbalanced datasets
    like BCDR (15% positive) and CBIS (40% positive).
    """
    # If the loader is already using a class-balanced sampler, pos_weight should be 1.0
    sampler = getattr(loader, 'sampler', None)
    if sampler is not None and isinstance(sampler, WeightedRandomSampler):
        return torch.tensor([1.0], dtype=torch.float32)

    if hasattr(loader.dataset, 'labels'):
        labels = loader.dataset.labels
    else:
        # ConcatDataset — iterate sub-datasets
        labels = np.concatenate([ds.labels for ds in loader.dataset.datasets])
    n_pos = max(float(labels.sum()), 1.0)
    n_neg = max(float(len(labels) - n_pos), 1.0)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def build_class_balanced_sampler(dataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples the minority class.

    Each sample gets weight 1/n_class, so the expected number of positive
    and negative samples per epoch is equal.
    """
    labels = dataset.labels
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    w_pos = 1.0 / max(n_pos, 1)
    w_neg = 1.0 / max(n_neg, 1)
    weights = [w_pos if lbl == 1 else w_neg for lbl in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


# ────────────────────────────────────────────────────────────────
#  DataLoader helpers
# ────────────────────────────────────────────────────────────────

def build_loader(dataset, cfg: dict, shuffle: bool = False,
                 sampler=None) -> DataLoader:
    num_workers = cfg.get("num_workers", 4)
    loader_kwargs = {
        "batch_size": cfg["batch_size"],
        "shuffle": (shuffle and sampler is None),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = cfg.get("prefetch_factor", 2)
    return DataLoader(dataset, **loader_kwargs)


def build_site_loader(data_root, site, split, cfg, shuffle=False,
                      class_balanced=False):
    """Convenience: dataset + loader in one call.

    If class_balanced=True and split=='train', uses WeightedRandomSampler
    to oversample the minority class.
    """
    ds = MammoDataset(data_root, site, split, cfg)
    sampler = None
    if class_balanced and split == "train":
        sampler = build_class_balanced_sampler(ds)
        shuffle = False  # sampler handles shuffling
    return ds, build_loader(ds, cfg, shuffle=shuffle, sampler=sampler)


def build_pooled_loader(data_root, sites, split, cfg, site_balanced=False):
    """Pool multiple sites into one loader.

    If site_balanced, use WeightedRandomSampler so each site contributes
    equally per epoch (oversamples small sites).
    """
    datasets = [MammoDataset(data_root, s, split, cfg) for s in sites]
    pooled = torch.utils.data.ConcatDataset(datasets)

    sampler = None
    if site_balanced and split == "train":
        weights = []
        for ds in datasets:
            w = 1.0 / len(ds)
            weights.extend([w] * len(ds))
        sampler = WeightedRandomSampler(
            weights, num_samples=len(pooled), replacement=True
        )

    return pooled, build_loader(pooled, cfg, shuffle=(not site_balanced and split == "train"),
                                sampler=sampler)
