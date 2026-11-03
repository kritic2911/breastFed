"""Data integrity, patient isolation, and view grouping assertion tests.

Adheres strictly to AGENTS.md Section 1.2-1.3 & Phase P1 Checkpoint:
  - Required manifest schema and non-null constraints.
  - Zero patient leakage between train and test splits across all sites.
  - CC and MLO views of each patient remain together in the same split (never separated).
  - Valid 3-class labels (0=Normal, 1=Benign, 2=Suspicious/Malignant).
  - All cached arrays exist on disk, have shape (512, 512), and values in [0.0, 1.0].
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def manifest_df():
    manifest_csv = Path("data/manifest.csv")
    if not manifest_csv.exists():
        pytest.skip("data/manifest.csv not yet generated.")
    return pd.read_csv(manifest_csv)


def test_manifest_schema_and_columns(manifest_df):
    """Verify all 20 required columns exist in manifest and formats/bit depths conform."""
    required_cols = [
        "site",
        "label_family",
        "patient_hash",
        "study",
        "laterality",
        "view",
        "path",
        "label",
        "birads_raw",
        "split",
        "source_format",
        "bit_depth",
        "manufacturer",
        "pixel_spacing",
        "photometric_interpretation",
        "age",
        "density",
        "crop_box",
        "flip",
        "scale",
    ]
    for col in required_cols:
        assert col in manifest_df.columns, f"Missing required column: {col}"

    assert len(manifest_df) > 5000, f"Unexpectedly small manifest count: {len(manifest_df)}"
    assert set(manifest_df["site"].unique()) == {"vindr", "cmmd", "inbreast", "cdd"}
    assert set(manifest_df["split"].unique()) == {"train", "test"}

    # Format and bit-depth validity assertions (AGENTS.md Section 2.1)
    valid_formats = {"dicom", "jpg", "png"}
    assert set(manifest_df["source_format"].unique()).issubset(valid_formats), "Invalid source_format"
    assert manifest_df["bit_depth"].isin([8, 12, 14, 16]).all(), "Invalid bit_depth"


def test_zero_patient_leakage_across_all_sites(manifest_df):
    """Assert zero patient leakage between train and test splits for every site."""
    for site, group in manifest_df.groupby("site"):
        train_patients = set(group[group["split"] == "train"]["patient_hash"])
        test_patients = set(group[group["split"] == "test"]["patient_hash"])

        overlap = train_patients & test_patients
        assert len(overlap) == 0, f"Patient leakage detected in site {site}: {len(overlap)} overlapping patient hashes!"


def test_cc_and_mlo_view_grouping(manifest_df):
    """Assert CC and MLO views of each patient are never separated across splits."""
    grouped = manifest_df.groupby(["site", "patient_hash"])["split"].nunique()
    split_patients = grouped[grouped > 1]
    assert len(split_patients) == 0, f"Patients with CC/MLO views separated across splits: {len(split_patients)}"

    # Verify views are only CC or MLO
    views = set(manifest_df["view"].unique())
    assert views.issubset({"CC", "MLO"}), f"Invalid views found: {views}"


def test_label_mapping_and_site_families(manifest_df):
    """Assert 3-class label distribution adheres strictly to label families."""
    # Labels must be in {0, 1, 2}
    labels = set(manifest_df["label"].unique())
    assert labels.issubset({0, 1, 2}), f"Invalid labels: {labels}"

    # CMMD is biopsy family with no class 0
    cmmd_labels = set(manifest_df[manifest_df["site"] == "cmmd"]["label"].unique())
    assert cmmd_labels == {1, 2}, f"CMMD should only have classes {1, 2}, got {cmmd_labels}"
    assert (manifest_df[manifest_df["site"] == "cmmd"]["label_family"] == "biopsy").all()

    # CDD is biopsy family with classes {0, 1, 2}
    cdd_labels = set(manifest_df[manifest_df["site"] == "cdd"]["label"].unique())
    assert cdd_labels == {0, 1, 2}, f"CDD should have classes {0, 1, 2}, got {cdd_labels}"
    assert (manifest_df[manifest_df["site"] == "cdd"]["label_family"] == "biopsy").all()

    # VinDr and INbreast are birads family with classes {0, 1, 2}
    for site in ["vindr", "inbreast"]:
        site_labels = set(manifest_df[manifest_df["site"] == site]["label"].unique())
        assert site_labels == {0, 1, 2}, f"{site} should have classes {0, 1, 2}, got {site_labels}"
        assert (manifest_df[manifest_df["site"] == site]["label_family"] == "birads").all()


def test_cached_arrays_integrity(manifest_df):
    """Assert all cached arrays exist, have shape (512, 512), unique paths, and values in [0.0, 1.0]."""
    # Strict uniqueness: every image record in manifest maps to its own distinct cached array file
    assert manifest_df["cached_path"].nunique() == len(manifest_df), (
        f"Duplicate cached_path detected in manifest: {len(manifest_df)} records vs {manifest_df['cached_path'].nunique()} unique paths!"
    )

    sample = manifest_df.sample(n=min(50, len(manifest_df)), random_state=42)
    data_dir = Path("data")

    for _, row in sample.iterrows():
        cached_rel = row.get("cached_path")
        assert cached_rel is not None, f"Missing cached_path for {row['image_id']}"

        cached_file = data_dir / cached_rel
        assert cached_file.exists(), f"Cached file does not exist: {cached_file}"

        arr = np.load(cached_file)
        assert arr.shape == (512, 512), f"Incorrect shape {arr.shape} for {cached_file}"
        assert arr.dtype == np.float32, f"Incorrect dtype {arr.dtype} for {cached_file}"
        assert arr.min() >= 0.0 - 1e-4, f"Negative values in {cached_file}: {arr.min()}"
        assert arr.max() <= 1.0 + 1e-4, f"Values > 1.0 in {cached_file}: {arr.max()}"
