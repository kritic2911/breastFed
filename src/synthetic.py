"""Synthetic mammography dataset generator for testing, smoke runs, and CI.

Generates synthetic noisy images with a planted signal across four simulated
clinical sites with different intensity statistics. Guarantees patient-level
stratification with zero leakage between train/val/test splits.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SITE_STATS = {
    "CMMD": {"mean": 0.50, "std": 0.15},
    "CBIS": {"mean": 0.35, "std": 0.10},
    "BCDR": {"mean": 0.65, "std": 0.18},
    "INbreast": {"mean": 0.40, "std": 0.12},
}


def _create_planted_signal_image(
    site_mean: float,
    site_std: float,
    label: int,
    img_size: int = 224,
    rng: np.random.RandomState = None,
) -> Image.Image:
    """Generate a single synthetic mammography image.

    - Benign (label 0): background Gaussian noise with site-specific mean and std.
    - Malignant (label 1): background noise + a planted 2D Gaussian lesion signal.
    """
    if rng is None:
        rng = np.random.RandomState()

    # Background tissue noise
    base = rng.normal(loc=site_mean, scale=site_std, size=(img_size, img_size))

    if label == 1:
        # Plant a synthetic lesion (bright Gaussian bump)
        yy, xx = np.mgrid[0:img_size, 0:img_size]
        center_y = rng.randint(int(img_size * 0.25), int(img_size * 0.75))
        center_x = rng.randint(int(img_size * 0.25), int(img_size * 0.75))
        sigma = rng.uniform(8.0, 16.0)
        signal = 0.40 * np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
        base = base + signal

    # Clip to [0, 1] and map to 8-bit grayscale
    clipped = np.clip(base, 0.0, 1.0)
    arr_uint8 = (clipped * 255.0).astype(np.uint8)
    return Image.fromarray(arr_uint8, mode="L")


def generate_synthetic_dataset(
    data_root: str | Path = "synthetic_data",
    sites: list[str] = None,
    patients_per_site: int = 12,
    images_per_patient: int = 2,
    img_size: int = 224,
    seed: int = 42,
) -> Path:
    """Generate a full synthetic dataset with patient-grouped splits.

    Parameters
    ----------
    data_root : Path or str
        Destination directory for synthetic dataset.
    sites : list of str
        Site names (defaults to CMMD, CBIS, BCDR, INbreast).
    patients_per_site : int
        Number of unique patients per site.
    images_per_patient : int
        Number of images per patient.
    img_size : int
        Width and height of generated images.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Path : The data_root path.
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    sites = sites or list(SITE_STATS.keys())
    rng = np.random.RandomState(seed)

    for site in sites:
        site_dir = root / site
        proc_dir = site_dir / "proc512"
        proc_dir.mkdir(parents=True, exist_ok=True)

        stats = SITE_STATS.get(site, {"mean": 0.50, "std": 0.15})
        site_mean = stats["mean"]
        site_std = stats["std"]

        rows = []
        # Stratified patient split: guarantee positive and negative representation in each split
        # with zero patient leakage across train, val, test
        effective_n_patients = max(6, patients_per_site)
        all_patients = list(range(effective_n_patients))
        pos_pids = [p for p in all_patients if p % 2 == 1]
        neg_pids = [p for p in all_patients if p % 2 == 0]

        rng.shuffle(pos_pids)
        rng.shuffle(neg_pids)

        n_val_pos = max(1, int(len(pos_pids) * 0.2))
        n_test_pos = max(1, int(len(pos_pids) * 0.2))
        val_pos = set(pos_pids[:n_val_pos])
        test_pos = set(pos_pids[n_val_pos:n_val_pos + n_test_pos])
        train_pos = set(pos_pids[n_val_pos + n_test_pos:])

        n_val_neg = max(1, int(len(neg_pids) * 0.2))
        n_test_neg = max(1, int(len(neg_pids) * 0.2))
        val_neg = set(neg_pids[:n_val_neg])
        test_neg = set(neg_pids[n_val_neg:n_val_neg + n_test_neg])
        train_neg = set(neg_pids[n_val_neg + n_test_neg:])

        train_pids = train_pos | train_neg
        val_pids = val_pos | val_neg
        _test_pids = test_pos | test_neg

        for pid_idx in range(effective_n_patients):
            patient_id = f"{site}_P{pid_idx:04d}"
            if pid_idx in train_pids:
                split = "train"
            elif pid_idx in val_pids:
                split = "val"
            else:
                split = "test"

            # Patient-level ground truth label: 50% positive
            patient_label = 1 if (pid_idx % 2 == 1) else 0

            for img_idx in range(images_per_patient):
                image_id = f"{patient_id}_img{img_idx}"
                rel_path = f"proc512/{image_id}.png"
                img_path = site_dir / rel_path

                img = _create_planted_signal_image(
                    site_mean=site_mean,
                    site_std=site_std,
                    label=patient_label,
                    img_size=img_size,
                    rng=rng,
                )
                img.save(img_path)

                rows.append({
                    "patient_id": patient_id,
                    "image_id": image_id,
                    "proc_path": rel_path,
                    "label": int(patient_label),
                    "split": split,
                })

        df = pd.DataFrame(rows)
        df.to_csv(site_dir / "labels_split.csv", index=False)

    return root
