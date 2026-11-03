#!/usr/bin/env python3
"""Site-ID Probe: Domain Shift & Scanner Fingerprint Quantification.

Adheres strictly to AGENTS.md Phase P1 (Task 5) & Checkpoint P1:
  - Trains a lightweight linear classifier to predict the hospital site and scanner
    from preprocessed mammograms.
  - Evaluates on held-out test split images.
  - Compares test accuracy and macro-F1 against the majority-class baseline.
  - Quantifies cross-silo domain shift and physical scanner heterogeneity.
"""

import json
import logging
import sys
import time
from pathlib import Path

_repo_dir = Path(__file__).resolve().parent.parent
_venv_dir = _repo_dir / ".venv"
_venv_py = _venv_dir / "bin" / "python3"
if _venv_py.exists() and sys.prefix != str(_venv_dir):
    try:
        import pandas  # noqa: F401
    except ImportError:
        import os
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

if str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_probe_features(img_path: Path) -> np.ndarray:
    """Extract fast spatial-intensity feature vector from cached 512x512 array."""
    arr = np.load(img_path)
    # 1. 32x32 thumbnail
    thumb = cv2.resize(arr, (32, 32), interpolation=cv2.INTER_AREA).flatten()

    # 2. Intensity statistics
    tissue = arr[arr > 0.01]
    if len(tissue) > 0:
        mean_val = float(np.mean(tissue))
        std_val = float(np.std(tissue))
        p10 = float(np.percentile(tissue, 10))
        p50 = float(np.percentile(tissue, 50))
        p90 = float(np.percentile(tissue, 90))
    else:
        mean_val, std_val, p10, p50, p90 = 0.0, 0.0, 0.0, 0.0, 0.0

    stats = np.array([mean_val, std_val, p10, p50, p90, float(np.mean(arr > 0.01))])
    return np.concatenate([thumb, stats])


def run_site_probe(manifest_df: pd.DataFrame, target_col: str = "site"):
    """Train and evaluate linear classifier on train split and evaluate on test split."""
    data_dir = Path("data")
    logger.info("Extracting features for target '%s' ...", target_col)

    train_df = manifest_df[manifest_df["split"] == "train"].copy()
    test_df = manifest_df[manifest_df["split"] == "test"].copy()

    # Feature extraction
    X_train, y_train = [], []
    for _, row in train_df.iterrows():
        p = data_dir / row["cached_path"]
        if p.exists():
            X_train.append(extract_probe_features(p))
            y_train.append(str(row[target_col]))

    X_test, y_test = [], []
    for _, row in test_df.iterrows():
        p = data_dir / row["cached_path"]
        if p.exists():
            X_test.append(extract_probe_features(p))
            y_test.append(str(row[target_col]))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Majority baseline
    classes, counts = np.unique(y_test, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    majority_acc = float(np.max(counts) / len(y_test))
    majority_pred = np.full_like(y_test, majority_class)
    majority_macro_f1 = float(f1_score(y_test, majority_pred, average="macro", zero_division=0))

    # Train linear classifier
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    probe_acc = float(accuracy_score(y_test, y_pred))
    probe_macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "target": target_col,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "majority_class": str(majority_class),
        "majority_accuracy": round(majority_acc, 4),
        "majority_macro_f1": round(majority_macro_f1, 4),
        "probe_accuracy": round(probe_acc, 4),
        "probe_macro_f1": round(probe_macro_f1, 4),
        "accuracy_gain_over_majority": round(probe_acc - majority_acc, 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1-score": round(report[cls]["f1-score"], 4),
                "support": report[cls]["support"],
            }
            for cls in classes
            if cls in report
        },
    }


def main():
    start_t = time.time()
    manifest_csv = Path("data/manifest.csv")
    if not manifest_csv.exists():
        logger.error("data/manifest.csv not found! Run scripts/build_manifest_and_cache.py first.")
        sys.exit(1)

    manifest_df = pd.read_csv(manifest_csv)
    logger.info("Loaded manifest with %d rows", len(manifest_df))

    # Run Site-ID probe
    site_results = run_site_probe(manifest_df, target_col="site")

    # Run Scanner / Manufacturer probe
    mfg_results = run_site_probe(manifest_df, target_col="manufacturer")

    probe_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "site_id_probe": site_results,
        "scanner_probe": mfg_results,
        "elapsed_seconds": round(time.time() - start_t, 2),
    }

    out_file = Path("paper/tables/site_id_probe_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(probe_summary, fp, indent=2)

    logger.info("Saved Site-ID probe results to %s", out_file)

    print("\n" + "=" * 65)
    print("SITE-ID PROBE RESULTS (Test Split Evaluation)")
    print("=" * 65)
    print("Site Prediction:")
    print(f"  Majority Baseline Accuracy : {site_results['majority_accuracy']:.2%} (Class: {site_results['majority_class']})")
    print(f"  Probe Accuracy             : {site_results['probe_accuracy']:.2%} (Macro-F1: {site_results['probe_macro_f1']:.4f})")
    print(f"  Gain over Majority         : +{site_results['accuracy_gain_over_majority']:.2%}")
    print("\nScanner Prediction:")
    print(f"  Majority Baseline Accuracy : {mfg_results['majority_accuracy']:.2%} (Class: {mfg_results['majority_class']})")
    print(f"  Probe Accuracy             : {mfg_results['probe_accuracy']:.2%} (Macro-F1: {mfg_results['probe_macro_f1']:.4f})")
    print(f"  Gain over Majority         : +{mfg_results['accuracy_gain_over_majority']:.2%}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
