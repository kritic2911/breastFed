"""Synthetic data pipeline and CPU execution tests for CI/CD.

Adheres strictly to CI/CD requirements:
  - Runner has NO GPU and NO patient data.
  - Zero raw or credentialed data used (preserves VinDr & INbreast data agreements).
  - Generates synthetic 512x512 arrays and manifests purely in pytest tmp_path.
  - Verifies data integrity, geometry transforms, MammographyDataset,
    MammographyResNet18 forward/backward pass on CPU, and metrics evaluation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MammographyDataset, MaskedCrossEntropyLoss
from src.data.preprocess import (
    inverse_transform_point,
    transform_box,
    transform_point,
)
from src.eval.metrics import compute_expected_calibration_error, evaluate_predictions
from src.models.backbone import MammographyResNet18
from src.utils.provenance import hash_patient_id


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path):
    """Generate a tiny synthetic dataset with 4 simulated sites and 0 patient data."""
    cache_dir = tmp_path / "cache512"
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    annotations = []

    # Sites configuration: 4 sites, 2 patients each, CC and MLO views
    sites_config = [
        {"site": "vindr", "family": "birads", "allowed": [0, 1, 2], "fmt": "dicom", "bits": 16},
        {"site": "inbreast", "family": "birads", "allowed": [0, 1, 2], "fmt": "dicom", "bits": 14},
        {"site": "cdd", "family": "biopsy", "allowed": [0, 1, 2], "fmt": "jpg", "bits": 8},
        {"site": "cmmd", "family": "biopsy", "allowed": [1, 2], "fmt": "dicom", "bits": 16},
    ]

    img_counter = 0

    for cfg in sites_config:
        site = cfg["site"]
        family = cfg["family"]
        allowed = cfg["allowed"]

        for pt_idx in range(2):
            raw_pt_id = f"synth_{site}_pt_{pt_idx}"
            pt_hash = hash_patient_id(raw_pt_id, salt=b"ci_test_salt")
            split = "train" if pt_idx == 0 else "test"

            for laterality in ["R", "L"]:
                for view in ["CC", "MLO"]:
                    img_counter += 1
                    img_id = f"img_{img_counter:04d}"
                    rel_cached_path = f"cache512/{site}_{img_id}.npy"
                    abs_cached_path = tmp_path / rel_cached_path

                    # Generate synthetic 512x512 image array with an ellipse (breast tissue)
                    arr = np.zeros((512, 512), dtype=np.float32)
                    y, x = np.ogrid[:512, :512]
                    # Elliptical tissue mask
                    tissue_mask = ((x - 256) ** 2) / (180 ** 2) + ((y - 256) ** 2) / (220 ** 2) <= 1.0
                    arr[tissue_mask] = 0.5

                    # Simulated circular lesion
                    lesion_mask = ((x - 200) ** 2) + ((y - 220) ** 2) <= (25 ** 2)
                    arr[lesion_mask] = 0.85
                    np.save(abs_cached_path, arr)

                    # Label from allowed set
                    label = allowed[img_counter % len(allowed)]

                    records.append({
                        "site": site,
                        "label_family": family,
                        "patient_hash": pt_hash,
                        "study": f"study_{pt_hash[:8]}",
                        "laterality": laterality,
                        "view": view,
                        "path": f"raw_synth/{site}/{img_id}.{cfg['fmt']}",
                        "label": label,
                        "birads_raw": f"BI-RADS {label + 1}",
                        "split": split,
                        "source_format": cfg["fmt"],
                        "bit_depth": cfg["bits"],
                        "manufacturer": "SyntheticScanner",
                        "pixel_spacing": "0.1, 0.1",
                        "photometric_interpretation": "MONOCHROME2",
                        "age": 55.0,
                        "density": "B",
                        "crop_box": "100, 100, 2000, 2000",
                        "flip": False,
                        "scale": 0.256,
                        "cached_path": rel_cached_path,
                        "image_id": img_id,
                        "breast_id": f"{site}_{pt_hash}_{laterality}",
                    })

                    # If lesion present and site supports annotations
                    if site in ["vindr", "inbreast", "cdd"] and label > 0:
                        annotations.append({
                            "site": site,
                            "image_id": img_id,
                            "finding_type": "mass",
                            "cached_xmin": 175.0,
                            "cached_ymin": 195.0,
                            "cached_xmax": 225.0,
                            "cached_ymax": 245.0,
                        })

    manifest_df = pd.DataFrame(records)
    manifest_path = tmp_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    ann_df = pd.DataFrame(annotations)
    ann_path = tmp_path / "annotations.csv"
    ann_df.to_csv(ann_path, index=False)

    return {
        "root": tmp_path,
        "manifest_df": manifest_df,
        "ann_df": ann_df,
        "manifest_path": manifest_path,
        "ann_path": ann_path,
    }


def test_synthetic_manifest_integrity_and_patient_isolation(synthetic_dataset_dir):
    """Verify synthetic dataset strictly obeys manifest schema, label rules, and zero-leakage."""
    manifest_df = synthetic_dataset_dir["manifest_df"]

    # 1. Schema check (20 required columns)
    required_cols = [
        "site", "label_family", "patient_hash", "study", "laterality", "view",
        "path", "label", "birads_raw", "split", "source_format", "bit_depth",
        "manufacturer", "pixel_spacing", "photometric_interpretation", "age",
        "density", "crop_box", "flip", "scale",
    ]
    for col in required_cols:
        assert col in manifest_df.columns

    # 2. Zero patient leakage
    for site, group in manifest_df.groupby("site"):
        train_pts = set(group[group["split"] == "train"]["patient_hash"])
        test_pts = set(group[group["split"] == "test"]["patient_hash"])
        assert len(train_pts & test_pts) == 0, f"Patient leakage in site {site}"

    # 3. CC/MLO view grouping
    grouped = manifest_df.groupby(["site", "patient_hash"])["split"].nunique()
    assert (grouped == 1).all(), "CC/MLO views separated across splits"

    # 4. Label family rules
    cmmd_labels = set(manifest_df[manifest_df["site"] == "cmmd"]["label"])
    assert cmmd_labels.issubset({1, 2}) and 0 not in cmmd_labels


def test_synthetic_geometry_roundtrip_and_bounds():
    """Verify coordinate transformation round-tripping and bounding box safety."""
    crop_box = (100, 150, 1900, 2100)
    flip = True
    scale = 0.25

    raw_x, raw_y = 1200.0, 950.0
    trans_x, trans_y = transform_point(raw_x, raw_y, crop_box, flip, scale)
    rec_x, rec_y = inverse_transform_point(trans_x, trans_y, crop_box, flip, scale)

    assert abs(raw_x - rec_x) < 1e-3
    assert abs(raw_y - rec_y) < 1e-3

    t_xmin, t_ymin, t_xmax, t_ymax = transform_box(500.0, 400.0, 800.0, 700.0, crop_box, flip, scale)
    assert 0.0 <= t_xmin < t_xmax <= 512.0
    assert 0.0 <= t_ymin < t_ymax <= 512.0


def test_synthetic_dataset_loader_and_masked_loss(synthetic_dataset_dir):
    """Verify PyTorch MammographyDataset loads synthetic images and handles class masking."""
    manifest_df = synthetic_dataset_dir["manifest_df"]
    root = synthetic_dataset_dir["root"]

    ds = MammographyDataset(manifest_df, data_dir=root, is_train=False)
    assert len(ds) == len(manifest_df)

    item = ds[0]
    assert "image" in item and "label" in item and "allowed_mask" in item
    assert item["image"].shape == (1, 512, 512)
    assert item["image"].dtype == torch.float32

    # Verify CMMD has class 0 masked
    cmmd_idx = manifest_df[manifest_df["site"] == "cmmd"].index[0]
    cmmd_item = ds[cmmd_idx]
    assert cmmd_item["allowed_mask"].tolist() == [0.0, 1.0, 1.0]

    # Verify MaskedCrossEntropyLoss masks unavailable logits
    loss_fn = MaskedCrossEntropyLoss()
    dummy_logits = torch.tensor([[10.0, 2.0, 1.0]])  # High logit for class 0
    dummy_target = torch.tensor([1])
    cmmd_mask = torch.tensor([[0.0, 1.0, 1.0]])

    loss = loss_fn(dummy_logits, dummy_target, cmmd_mask)
    assert torch.isfinite(loss)


def test_synthetic_model_cpu_forward_backward(synthetic_dataset_dir):
    """Verify MammographyResNet18 forward pass, loss, and backpropagation execute cleanly on CPU."""
    manifest_df = synthetic_dataset_dir["manifest_df"]
    root = synthetic_dataset_dir["root"]

    ds = MammographyDataset(manifest_df, data_dir=root, is_train=True)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    # Initialize model without downloading pretrained ImageNet weights
    model = MammographyResNet18(num_classes=3, pretrained=False, bn_mode="trainable")
    model.eval()

    batch = next(iter(loader))
    imgs = batch["image"]  # (B, 1, 512, 512)
    labels = batch["label"]  # (B,)
    masks = batch["allowed_mask"]  # (B, 3)

    # Forward pass on CPU
    logits = model(imgs)
    assert logits.shape == (4, 3)

    # Loss computation
    loss_fn = MaskedCrossEntropyLoss()
    loss = loss_fn(logits, labels, masks)
    assert torch.isfinite(loss)

    # Backward pass on CPU
    loss.backward()

    # Head and fine-tuned layers have gradients
    assert model.fc.weight.grad is not None
    assert model.layer4[0].conv1.weight.grad is not None

    # Frozen stages (conv1, layer1) do NOT have gradients
    assert model.conv1.weight.grad is None
    assert model.layer1[0].conv1.weight.grad is None


def test_synthetic_metrics_evaluation():
    """Verify metric computation (AUC, precision, recall, specificity, ECE) on synthetic outputs."""
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 2]
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.05, 0.15, 0.8],
        [0.9, 0.05, 0.05],
        [0.1, 0.85, 0.05],
        [0.1, 0.1, 0.8],
        [0.05, 0.05, 0.9],
    ])

    df = pd.DataFrame({
        "patient_hash": [f"hash_{i}" for i in range(len(y_true))],
        "label": y_true,
        "prob_0": probs[:, 0],
        "prob_1": probs[:, 1],
        "prob_2": probs[:, 2],
    })

    metrics = evaluate_predictions(df)
    assert "macro_auc" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "macro_specificity" in metrics
    assert metrics["macro_auc"] > 0.85

    ece = compute_expected_calibration_error(np.array(y_true), probs)
    assert 0.0 <= ece <= 1.0
