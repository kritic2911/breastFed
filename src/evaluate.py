"""Evaluation: AUC, sensitivity@specificity, patient-level bootstrap CIs."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

# ────────────────────────────────────────────────────────────────
#  Inference
# ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: nn.Module, loader, device: str = "cpu"):
    """Run inference, return arrays of (probabilities, labels, patient_ids)."""
    model.eval()
    all_probs, all_labels, all_pids = [], [], []
    for imgs, labels, pids in loader:
        logits = model(imgs.to(device, non_blocking=True)).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
        all_pids.extend(pids)
    return (np.concatenate(all_probs),
            np.concatenate(all_labels),
            np.array(all_pids))


# ────────────────────────────────────────────────────────────────
#  Threshold selection (on val only)
# ────────────────────────────────────────────────────────────────

def find_threshold_at_specificity(labels, probs, target_specificity=0.90):
    """Find the decision threshold that achieves a target specificity.

    Specificity = TNR = 1 - FPR.
    We pick the threshold where FPR <= (1 - target_specificity).
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    # We want FPR <= (1 - target_spec)
    target_fpr = 1.0 - target_specificity
    # Find the largest threshold where fpr <= target_fpr
    valid = fpr <= target_fpr
    if not valid.any():
        # Fall back to the highest threshold
        return thresholds[0], tpr[0]
    idx = np.where(valid)[0][-1]  # last index where fpr is still acceptable
    return thresholds[idx], tpr[idx]


# ────────────────────────────────────────────────────────────────
#  Core metrics
# ────────────────────────────────────────────────────────────────

def compute_metrics(labels, probs, threshold=None, target_specificity=0.90):
    """Compute AUC and sensitivity at fixed specificity.

    Parameters
    ----------
    labels : array of 0/1
    probs : array of predicted probabilities
    threshold : float or None
        If None, find threshold on this data (use only for val).
        If given, apply it directly (use for test).
    target_specificity : float

    Returns
    -------
    dict with auc, sensitivity, specificity, threshold
    """
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")

    if threshold is None:
        threshold, _ = find_threshold_at_specificity(labels, probs, target_specificity)

    preds = (probs >= threshold).astype(int)
    pos = labels == 1
    neg = labels == 0

    sensitivity = preds[pos].mean() if pos.sum() > 0 else float("nan")
    specificity = 1.0 - preds[neg].mean() if neg.sum() > 0 else float("nan")

    return {
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "threshold": float(threshold),
        "prob_std": float(np.std(probs)),
        "prob_mean": float(np.mean(probs)),
        "n_pos": int(pos.sum()),
        "n_neg": int(neg.sum()),
        "n_total": len(labels),
    }


# ────────────────────────────────────────────────────────────────
#  Patient-level aggregation
# ────────────────────────────────────────────────────────────────

def aggregate_to_patient(probs, labels, patient_ids):
    """Aggregate image-level predictions to patient level (max probability).

    For patients with multiple images, take the max predicted probability
    and the max label (any positive image → positive patient).
    """
    import pandas as pd
    df = pd.DataFrame({"prob": probs, "label": labels, "pid": patient_ids})
    agg = df.groupby("pid").agg(prob=("prob", "max"), label=("label", "max")).reset_index()
    return agg["prob"].values, agg["label"].values, agg["pid"].values


# ────────────────────────────────────────────────────────────────
#  Bootstrap confidence intervals
# ────────────────────────────────────────────────────────────────

def bootstrap_ci(probs, labels, patient_ids, metric_fn=roc_auc_score,
                  n_bootstrap=1000, ci=0.95, seed=42):
    """Patient-level bootstrap CI.

    Resamples patients (with replacement), then computes the metric on
    all images belonging to the sampled patients.

    Returns
    -------
    dict with mean, lower, upper, std
    """
    rng = np.random.RandomState(seed)
    unique_pids = np.unique(patient_ids)
    scores = []

    for _ in range(n_bootstrap):
        sampled_pids = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        # Build mask: include all images for each sampled patient
        # (a patient sampled k times → its images appear k times)
        indices = []
        for pid in sampled_pids:
            indices.extend(np.where(patient_ids == pid)[0])
        indices = np.array(indices)

        boot_labels = labels[indices]
        boot_probs = probs[indices]

        # Skip if only one class present in bootstrap sample
        if len(np.unique(boot_labels)) < 2:
            continue
        scores.append(metric_fn(boot_labels, boot_probs))

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(scores)),
        "lower": float(np.percentile(scores, 100 * alpha)),
        "upper": float(np.percentile(scores, 100 * (1 - alpha))),
        "std": float(np.std(scores)),
        "n_valid": len(scores),
    }


# ────────────────────────────────────────────────────────────────
#  Full evaluation pipeline
# ────────────────────────────────────────────────────────────────

def evaluate_on_site(model, loader, device, threshold=None,
                     n_bootstrap=1000, seed=42):
    """Full evaluation: predict → metrics → bootstrap CI.

    Parameters
    ----------
    threshold : float or None
        Pass the val-derived threshold when evaluating on test.

    Returns
    -------
    dict with image_level and patient_level metrics + CIs
    """
    probs, labels, pids = predict(model, loader, device)

    # Image-level metrics
    img_metrics = compute_metrics(labels, probs, threshold=threshold)

    # Patient-level aggregation
    pat_probs, pat_labels, pat_pids = aggregate_to_patient(probs, labels, pids)
    pat_metrics = compute_metrics(pat_labels, pat_probs, threshold=threshold)

    # Bootstrap CI on image-level AUC (patient-level resampling)
    ci = bootstrap_ci(probs, labels, pids, n_bootstrap=n_bootstrap, seed=seed)

    return {
        "image_level": img_metrics,
        "patient_level": pat_metrics,
        "bootstrap_auc_ci": ci,
        "predictions": {
            "patient_ids": pids.tolist(),
            "labels": labels.tolist(),
            "probs": probs.tolist(),
            "preds": (probs >= (threshold if threshold is not None else 0.5)).astype(int).tolist(),
        },
    }
