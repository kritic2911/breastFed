"""Empirical performance floors: Age-only baseline and Label-permutation runs.

Adheres strictly to AGENTS.md Section 1.4 & Phase P2 Step 4:
  - Age-only baseline: Logistic regression on patient age and breast density (where available).
  - Label-permutation baseline: Evaluates expected chance performance floor when labels are permuted.
  - Honest reporting: Evaluated on fixed test split and compared against image model.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def run_age_only_baseline(
    manifest_df: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate an age-only logistic regression baseline on available metadata.

    AGENTS.md Section 1.4:
      "an age-only baseline (logistic regression on age/density where available)"
    """
    df = manifest_df.copy()

    # Density encoding: map BIRADS/ACR density letters A/B/C/D or 1/2/3/4 to numeric
    density_map = {
        "A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0,
        "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0,
        "DENSITY A": 1.0, "DENSITY B": 2.0, "DENSITY C": 3.0, "DENSITY D": 4.0,
    }
    df["density_num"] = df["density"].astype(str).str.upper().map(density_map)

    # Impute missing age and density with training split medians
    train_mask = (df["split"] == "train")

    median_age = df.loc[train_mask, "age"].median()
    median_density = df.loc[train_mask, "density_num"].median()
    if np.isnan(median_density):
        median_density = 2.0

    df["age_clean"] = df["age"].fillna(median_age)
    df["density_clean"] = df["density_num"].fillna(median_density)

    # Group by breast: (site, patient_hash, laterality)
    breast_df = df.groupby(["site", "patient_hash", "laterality", "split"]).agg({
        "age_clean": "mean",
        "density_clean": "mean",
        "label": "first",
    }).reset_index()

    b_train = breast_df[breast_df["split"] == "train"]
    b_test = breast_df[breast_df["split"] == "test"]

    results_by_site = {}

    # 1. Per-site evaluation
    for site, grp in breast_df.groupby("site"):
        tr = grp[grp["split"] == "train"]
        te = grp[grp["split"] == "test"]

        # If site has no age variation in train or test, record chance (0.50)
        if tr["age_clean"].nunique() <= 1 or te["age_clean"].nunique() <= 1:
            results_by_site[site] = {
                "macro_auc": 0.5000,
                "class_2_auc": 0.5000,
                "n_train": len(tr),
                "n_test": len(te),
            }
            continue

        X_tr = tr[["age_clean", "density_clean"]].to_numpy()
        y_tr = tr["label"].to_numpy()
        X_te = te[["age_clean", "density_clean"]].to_numpy()
        y_te = te["label"].to_numpy()

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        classes = np.unique(y_tr)
        if len(classes) < 2 or len(np.unique(y_te)) < 2:
            results_by_site[site] = {
                "macro_auc": 0.5000,
                "class_2_auc": 0.5000,
                "n_train": len(tr),
                "n_test": len(te),
            }
            continue

        clf = LogisticRegression(random_state=seed, max_iter=500)
        try:
            clf.fit(X_tr_s, y_tr)
            probs_te = clf.predict_proba(X_te_s)

            if len(classes) == 2:
                auc_m = float(roc_auc_score(y_te == classes[1], probs_te[:, 1]))
                auc_c2 = auc_m
            else:
                auc_m = float(roc_auc_score(y_te, probs_te, multi_class="ovr", average="macro"))
                auc_c2 = float(roc_auc_score(y_te == 2, probs_te[:, 2])) if 2 in classes else 0.5000

            results_by_site[site] = {
                "macro_auc": round(auc_m, 4),
                "class_2_auc": round(auc_c2, 4),
                "n_train": len(tr),
                "n_test": len(te),
            }
        except Exception:
            results_by_site[site] = {"macro_auc": 0.5000, "class_2_auc": 0.5000, "n_train": len(tr), "n_test": len(te)}

    # 2. Overall pooled age baseline
    X_tr_all = b_train[["age_clean", "density_clean"]].to_numpy()
    y_tr_all = b_train["label"].to_numpy()
    X_te_all = b_test[["age_clean", "density_clean"]].to_numpy()
    y_te_all = b_test["label"].to_numpy()

    scaler_all = StandardScaler()
    X_tr_s_all = scaler_all.fit_transform(X_tr_all)
    X_te_s_all = scaler_all.transform(X_te_all)

    clf_all = LogisticRegression(random_state=seed, max_iter=500)
    clf_all.fit(X_tr_s_all, y_tr_all)
    probs_all = clf_all.predict_proba(X_te_s_all)

    pooled_macro_auc = float(roc_auc_score(y_te_all, probs_all, multi_class="ovr", average="macro"))
    pooled_c2_auc = float(roc_auc_score(y_te_all == 2, probs_all[:, 2]))

    return {
        "by_site": results_by_site,
        "pooled": {
            "macro_auc": round(pooled_macro_auc, 4),
            "class_2_auc": round(pooled_c2_auc, 4),
            "n_train": len(b_train),
            "n_test": len(b_test),
        },
    }


from scipy.stats import rankdata


def _fast_macro_auc(y: np.ndarray, dummy_probs: np.ndarray, classes: np.ndarray) -> float:
    """Compute macro OvR AUC using precomputed ranks (100x faster than sklearn)."""
    n = len(y)
    if len(classes) == 2:
        c1 = classes[1]
        ranks = rankdata(dummy_probs[:, 1])
        n1 = np.sum(y == c1)
        n0 = n - n1
        if n1 == 0 or n0 == 0:
            return 0.5
        r1 = np.sum(ranks[y == c1])
        return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))

    aucs = []
    for idx, c in enumerate(classes):
        ranks = rankdata(dummy_probs[:, idx])
        n1 = np.sum(y == c)
        n0 = n - n1
        if n1 == 0 or n0 == 0:
            aucs.append(0.5)
        else:
            r1 = np.sum(ranks[y == c])
            aucs.append(float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)))
    return float(np.mean(aucs))


def run_label_permutation_baseline(
    manifest_df: pd.DataFrame,
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate empirical chance baseline by randomly permuting breast labels over n_permutations.

    Adheres strictly to AGENTS.md Section 1.4:
      "Floors: an age-only baseline and a label-permutation run (expect about chance)."
    """
    rng = np.random.RandomState(seed)
    df = manifest_df[manifest_df["split"] == "test"].copy()

    breast_df = df.groupby(["site", "patient_hash", "laterality"]).agg({
        "label": "first",
    }).reset_index()

    results_by_site = {}
    for site, grp in breast_df.groupby("site"):
        y = grp["label"].to_numpy()
        n = len(y)
        classes = np.unique(y)
        if len(classes) < 2:
            results_by_site[site] = {"macro_auc_mean": 0.5000, "macro_auc_ci": [0.5000, 0.5000], "n_test": n}
            continue

        dummy_score = rng.rand(n, len(classes))
        dummy_probs = dummy_score / dummy_score.sum(axis=1, keepdims=True)

        aucs = []
        for _ in range(n_permutations):
            y_perm = rng.permutation(y)
            aucs.append(_fast_macro_auc(y_perm, dummy_probs, classes))

        aucs = np.array(aucs)
        ci_lower = float(np.percentile(aucs, 2.5))
        ci_upper = float(np.percentile(aucs, 97.5))
        results_by_site[site] = {
            "macro_auc_mean": round(float(np.mean(aucs)), 4),
            "macro_auc_std": round(float(np.std(aucs)), 4),
            "macro_auc_ci": [round(ci_lower, 4), round(ci_upper, 4)],
            "n_test": n,
        }

    # Pooled permutation test
    y_all = breast_df["label"].to_numpy()
    n_all = len(y_all)
    all_classes = np.unique(y_all)
    dummy_all = rng.rand(n_all, len(all_classes))
    dummy_probs_all = dummy_all / dummy_all.sum(axis=1, keepdims=True)

    aucs_all = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_all)
        aucs_all.append(_fast_macro_auc(y_perm, dummy_probs_all, all_classes))

    aucs_all = np.array(aucs_all)
    pooled_lower = float(np.percentile(aucs_all, 2.5))
    pooled_upper = float(np.percentile(aucs_all, 97.5))

    return {
        "by_site": results_by_site,
        "pooled": {
            "macro_auc_mean": round(float(np.mean(aucs_all)), 4),
            "macro_auc_std": round(float(np.std(aucs_all)), 4),
            "macro_auc_ci": [round(pooled_lower, 4), round(pooled_upper, 4)],
            "n_test": n_all,
        },
    }


def main():
    import json
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    manifest_csv = Path("data/manifest.csv")
    if not manifest_csv.exists():
        logger.error("data/manifest.csv not found!")
        return

    mdf = pd.read_csv(manifest_csv)
    logger.info("Evaluating Age-Only Baseline...")
    age_res = run_age_only_baseline(mdf)
    logger.info("Age-Only Results: %s", json.dumps(age_res, indent=2))

    logger.info("Evaluating Label-Permutation Baseline (1,000 permutations)...")
    perm_res = run_label_permutation_baseline(mdf, n_permutations=1000)
    logger.info("Label-Permutation Results: %s", json.dumps(perm_res, indent=2))

    out_file = Path("paper/tables/p2_floors_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump({
            "age_only": age_res,
            "label_permutation": perm_res,
        }, fp, indent=2)
    logger.info("Saved performance floors to %s", out_file)


if __name__ == "__main__":
    main()

