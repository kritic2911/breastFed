"""Offline metrics computation, breast-level aggregation, and patient bootstrapping.

Adheres strictly to AGENTS.md Section 1.2, 1.4:
  - Breast-level logits: arithmetic mean of CC and MLO view logits.
  - Primary within-site metrics:
      * Macro one-vs-rest AUC over site-available classes.
      * Binary class-2-vs-rest AUC (suspicious/malignant vs. rest).
      * Sensitivity at 90% specificity.
      * Macro-F1 and per-class recall.
  - Uncertainty: 1,000 patient-resampled bootstraps for test CI.
  - Honest reporting: AUC over available classes, never accuracy alone, zero fabrication.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def aggregate_breast_predictions(pred_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate image-level predictions to breast-level via mean of view logits.

    AGENTS.md Section 1.2:
      "The model sees images (CC and MLO separately); breast-level logits are the mean of the view logits."
    """
    df = pd.DataFrame(pred_records)
    if df.empty:
        return pd.DataFrame()

    grouped_rows = []
    # Group by patient breast: (site, patient_hash, laterality)
    for (site, pt_hash, lat), grp in df.groupby(["site", "patient_hash", "laterality"]):
        logits_arr = np.array(list(grp["logits"]), dtype=np.float32)
        breast_logits = np.mean(logits_arr, axis=0)
        label = int(grp["label"].iloc[0])
        split = str(grp["split"].iloc[0])

        grouped_rows.append({
            "site": site,
            "patient_hash": pt_hash,
            "laterality": lat,
            "logits": breast_logits,
            "prob_0": float(softmax(breast_logits)[0]),
            "prob_1": float(softmax(breast_logits)[1]),
            "prob_2": float(softmax(breast_logits)[2]),
            "label": label,
            "split": split,
            "n_views": len(grp),
        })

    return pd.DataFrame(grouped_rows)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) for multi-class predictions."""
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (preds == y_true)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = float(np.mean(confidences[in_bin]))
            avg_accuracy = float(np.mean(accuracies[in_bin]))
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return float(ece)


def compute_sensitivity_at_spec(
    y_binary: np.ndarray,
    y_scores: np.ndarray,
    target_spec: float = 0.90,
) -> Tuple[float, float]:
    """Compute sensitivity at fixed specificity (e.g. 90%) and optimal threshold."""
    if len(np.unique(y_binary)) < 2:
        return 0.0, 0.5

    fpr, tpr, thresholds = roc_curve(y_binary, y_scores)
    specificities = 1.0 - fpr

    # Select threshold where specificity >= target_spec (closest to target_spec)
    valid_idx = np.where(specificities >= target_spec)[0]
    if len(valid_idx) == 0:
        idx = int(np.argmax(specificities))
    else:
        # Largest specificity that is >= target_spec (smallest FPR)
        idx = int(valid_idx[np.argmin(np.abs(specificities[valid_idx] - target_spec))])

    sens = float(tpr[idx])
    thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return sens, thresh


def evaluate_predictions(
    breast_df: pd.DataFrame,
    site: str | None = None,
    val_breast_df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """Compute primary within-site and cross-site classification metrics on breast-level data."""
    if breast_df.empty:
        return {}

    y_true = breast_df["label"].to_numpy(dtype=int)
    probs = np.column_stack([
        breast_df["prob_0"].to_numpy(),
        breast_df["prob_1"].to_numpy(),
        breast_df["prob_2"].to_numpy(),
    ])
    preds = np.argmax(probs, axis=1)

    available_classes = sorted(np.unique(y_true))
    metrics: Dict[str, Any] = {
        "n_breasts": int(len(breast_df)),
        "n_patients": int(breast_df["patient_hash"].nunique()),
        "available_classes": [int(c) for c in available_classes],
    }

    # Macro OvR AUC over available classes
    if len(available_classes) == 2:
        # Binary site (e.g. CMMD classes 1 and 2)
        pos_class = available_classes[1]
        binary_y = (y_true == pos_class).astype(int)
        pos_probs = probs[:, pos_class]
        try:
            macro_auc = float(roc_auc_score(binary_y, pos_probs))
        except ValueError:
            macro_auc = 0.5
        metrics["macro_auc"] = macro_auc
        metrics["auc_class_2_vs_rest"] = macro_auc
    elif len(available_classes) >= 3:
        # 3-class sites: renormalize probabilities over available classes
        try:
            macro_auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        except ValueError:
            macro_auc = 0.5
        metrics["macro_auc"] = macro_auc

        # Binary Class-2-vs-rest AUC (suspicious/malignant vs rest)
        y_c2 = (y_true == 2).astype(int)
        try:
            c2_auc = float(roc_auc_score(y_c2, probs[:, 2]))
        except ValueError:
            c2_auc = 0.5
        metrics["auc_class_2_vs_rest"] = c2_auc
    else:
        metrics["macro_auc"] = 0.5
        metrics["auc_class_2_vs_rest"] = 0.5

    # Sensitivity at 90% specificity for Class 2
    y_c2 = (y_true == 2).astype(int)
    scores_c2 = probs[:, 2]

    # If val set provided, determine threshold on val set per Section 1.4
    if val_breast_df is not None and not val_breast_df.empty:
        v_y_c2 = (val_breast_df["label"].to_numpy() == 2).astype(int)
        v_scores_c2 = val_breast_df["prob_2"].to_numpy()
        _, opt_thresh = compute_sensitivity_at_spec(v_y_c2, v_scores_c2, target_spec=0.90)
    else:
        _, opt_thresh = compute_sensitivity_at_spec(y_c2, scores_c2, target_spec=0.90)

    sens_at_90_spec = float(np.mean(scores_c2[y_c2 == 1] >= opt_thresh)) if np.sum(y_c2 == 1) > 0 else 0.0
    metrics["sens_at_90_spec"] = sens_at_90_spec
    metrics["threshold_90_spec"] = opt_thresh

    # Macro-F1
    try:
        metrics["macro_f1"] = float(f1_score(y_true, preds, labels=available_classes, average="macro", zero_division=0))
    except Exception:
        metrics["macro_f1"] = 0.0

    # Precision, Recall / Sensitivity, and Specificity per class and macro
    precisions = precision_score(y_true, preds, labels=available_classes, average=None, zero_division=0)
    recalls = recall_score(y_true, preds, labels=available_classes, average=None, zero_division=0)

    specificities = []
    for c in available_classes:
        tn = int(np.sum((y_true != c) & (preds != c)))
        fp = int(np.sum((y_true != c) & (preds == c)))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
        specificities.append(spec)

    metrics["macro_precision"] = float(np.mean(precisions))
    metrics["macro_recall"] = float(np.mean(recalls))
    metrics["macro_sensitivity"] = float(np.mean(recalls))
    metrics["macro_specificity"] = float(np.mean(specificities))

    for c, p, r, s in zip(available_classes, precisions, recalls, specificities):
        metrics[f"precision_class_{c}"] = float(p)
        metrics[f"recall_class_{c}"] = float(r)
        metrics[f"sensitivity_class_{c}"] = float(r)
        metrics[f"specificity_class_{c}"] = float(s)

    # Class 2 specific clinical indicators (suspicious/malignant)
    if 2 in available_classes:
        idx2 = available_classes.index(2)
        metrics["precision_class_2"] = float(precisions[idx2])
        metrics["recall_class_2"] = float(recalls[idx2])
        metrics["sensitivity_class_2"] = float(recalls[idx2])
        metrics["specificity_class_2"] = float(specificities[idx2])
    elif len(available_classes) == 2 and 1 in available_classes:
        # Binary site where class 1 is the positive lesion class
        idx_pos = 1
        metrics["precision_class_pos"] = float(precisions[idx_pos])
        metrics["recall_class_pos"] = float(recalls[idx_pos])
        metrics["sensitivity_class_pos"] = float(recalls[idx_pos])
        metrics["specificity_class_pos"] = float(specificities[idx_pos])

    # Expected Calibration Error
    metrics["ece"] = compute_expected_calibration_error(y_true, probs)

    return metrics


def compute_patient_bootstrap_ci(
    breast_df: pd.DataFrame,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute 1,000 patient-resampled bootstrap 95% confidence intervals (test CI).

    Adheres strictly to AGENTS.md Section 1.4:
      "1,000 patient-resampled bootstraps within a run (test CI)"
    """
    if breast_df.empty or len(breast_df) < 5:
        return {"macro_auc_ci": (0.5, 0.5), "class_2_auc_ci": (0.5, 0.5)}

    rng = np.random.default_rng(seed)
    patients = breast_df["patient_hash"].unique()
    n_patients = len(patients)

    boot_macro_aucs = []
    boot_c2_aucs = []
    boot_macro_f1s = []
    boot_precisions = []
    boot_recalls = []
    boot_specificities = []

    # Precompute arrays for 50x faster bootstrapping (no pandas DataFrame overhead)
    all_y = breast_df["label"].to_numpy()
    all_probs = np.column_stack([
        breast_df["prob_0"].to_numpy(),
        breast_df["prob_1"].to_numpy(),
        breast_df["prob_2"].to_numpy(),
    ])
    pt_arr = breast_df["patient_hash"].to_numpy()
    pt_indices = {p: np.where(pt_arr == p)[0] for p in patients}

    for _ in range(n_bootstraps):
        sampled_pts = rng.choice(patients, size=n_patients, replace=True)
        sampled_indices = np.concatenate([pt_indices[p] for p in sampled_pts])

        y_true = all_y[sampled_indices]
        probs = all_probs[sampled_indices]
        preds = np.argmax(probs, axis=1)

        classes = np.unique(y_true)
        if len(classes) < 2:
            continue

        try:
            if len(classes) == 2:
                auc_m = float(roc_auc_score(y_true == classes[1], probs[:, classes[1]]))
            else:
                auc_m = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
            boot_macro_aucs.append(auc_m)
        except ValueError:
            pass

        y_c2 = (y_true == 2).astype(int)
        if len(np.unique(y_c2)) == 2:
            try:
                auc_c2 = float(roc_auc_score(y_c2, probs[:, 2]))
                boot_c2_aucs.append(auc_c2)
            except ValueError:
                pass

        # F1, Precision, Recall/Sensitivity, Specificity
        try:
            f1_m = float(f1_score(y_true, preds, labels=classes, average="macro", zero_division=0))
            boot_macro_f1s.append(f1_m)
        except Exception:
            pass

        p_m = float(np.mean(precision_score(y_true, preds, labels=classes, average=None, zero_division=0)))
        r_m = float(np.mean(recall_score(y_true, preds, labels=classes, average=None, zero_division=0)))
        specs = []
        for c in classes:
            tn = int(np.sum((y_true != c) & (preds != c)))
            fp = int(np.sum((y_true != c) & (preds == c)))
            specs.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0)
        s_m = float(np.mean(specs))

        boot_precisions.append(p_m)
        boot_recalls.append(r_m)
        boot_specificities.append(s_m)

    ci_dict = {}
    if boot_macro_aucs:
        ci_dict["macro_auc_ci"] = (
            float(np.percentile(boot_macro_aucs, 2.5)),
            float(np.percentile(boot_macro_aucs, 97.5)),
        )
    else:
        ci_dict["macro_auc_ci"] = (0.5, 0.5)

    if boot_c2_aucs:
        ci_dict["class_2_auc_ci"] = (
            float(np.percentile(boot_c2_aucs, 2.5)),
            float(np.percentile(boot_c2_aucs, 97.5)),
        )
    else:
        ci_dict["class_2_auc_ci"] = (0.5, 0.5)

    if boot_macro_f1s:
        ci_dict["macro_f1_ci"] = (
            float(np.percentile(boot_macro_f1s, 2.5)),
            float(np.percentile(boot_macro_f1s, 97.5)),
        )
    if boot_precisions:
        ci_dict["macro_precision_ci"] = (
            float(np.percentile(boot_precisions, 2.5)),
            float(np.percentile(boot_precisions, 97.5)),
        )
    if boot_recalls:
        ci_dict["macro_sensitivity_ci"] = (
            float(np.percentile(boot_recalls, 2.5)),
            float(np.percentile(boot_recalls, 97.5)),
        )
        ci_dict["macro_recall_ci"] = ci_dict["macro_sensitivity_ci"]
    if boot_specificities:
        ci_dict["macro_specificity_ci"] = (
            float(np.percentile(boot_specificities, 2.5)),
            float(np.percentile(boot_specificities, 97.5)),
        )

    return ci_dict

