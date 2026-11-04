"""Centralized pilot training and decision fusion for the calcification expert model.

Adheres strictly to AGENTS.md Phase P3b, Rule 9 Geometry, and Decision D-010:
  - Trains CalcificationExpertMIL on high-resolution 256x256 native ROI patches from INbreast and VinDr.
  - Includes positive calcification images AND negative normal/benign images for contrast.
  - Weak supervision: Patches inherit the image-level diagnosis (Malignant vs. Benign/Normal).
  - Decision fusion: Fits validation logistic regression combining Global ResNet-18 logits with Expert MIL logits.
  - Evaluates on the held-out calcification test cohort (N=236) using 1,000-resample paired patient bootstrap.
  - Tests Pre-Registered Criterion 3: Delta AUC >= +0.030 with 95% CI strictly excluding zero.
  - Saves quantitative table: paper/tables/calc_expert_fusion_results.json.
"""

import json
import logging
import os
import sys
from pathlib import Path

_repo_dir = Path(__file__).resolve().parent.parent
if str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
Path("/tmp/matplotlib_cache").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data.dataset import get_train_val_test_splits
from src.data.preprocess import read_image_array
from src.explain.detector import MicrocalcificationDetector
from src.models.backbone import build_model
from src.models.calc_expert import CalcificationExpertMIL
from src.utils.ckpt import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_bags_for_dataset(
    df: pd.DataFrame,
    detector: MicrocalcificationDetector,
    max_patches_per_image: int = 8,
) -> list:
    """Extract bags of 256x256 native patches for a cohort of images."""
    bags = []
    for idx, (_, row) in enumerate(df.iterrows()):
        npy_path = Path("data") / row["cached_path"]
        if not npy_path.exists():
            continue

        arr_raw, _ = read_image_array(row["path"])
        ymin, xmin, ymax, xmax = [int(v) for v in row["crop_box"].split(",")]
        cropped = arr_raw[ymin : ymax + 1, xmin : xmax + 1]
        if row["flip"]:
            cropped = np.fliplr(cropped)
        scale = float(row["scale"])
        tissue_mask = cropped > 0.05

        specks, _ = detector.detect_specks(cropped, tissue_mask)
        clusters = detector.cluster_specks(
            specks, cropped, scale=scale, extract_patches=True
        )[:max_patches_per_image]

        patches = []
        for c in clusters:
            if c.roi_patch_256 is not None:
                patches.append(c.roi_patch_256)

        if patches:
            bag_tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).float()
        else:
            bag_tensor = torch.empty((0, 1, 256, 256), dtype=torch.float32)

        # Binary malignancy label (Class 2 vs. Rest)
        y_binary = 1.0 if int(row["label"]) == 2 else 0.0

        bags.append({
            "image_id": str(row["image_id"]),
            "patient_hash": str(row["patient_hash"]),
            "site": row["site"],
            "label": int(row["label"]),
            "y_binary": y_binary,
            "cached_path": str(row["cached_path"]),
            "patches": bag_tensor,
            "n_clusters": len(clusters),
        })

    return bags


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    # 1. Load Data
    manifest_path = Path("data/manifest.csv")
    annot_path = Path("data/annotations.csv")
    mdf = pd.read_csv(manifest_path)
    annot_df = pd.read_csv(annot_path)

    train_df, val_df, test_df = get_train_val_test_splits(mdf, seed=42, val_size=0.20)

    detector = MicrocalcificationDetector(
        kernel_size=7,
        std_multiplier=3.2,
        cluster_eps_native=80.0,
        cluster_min_specks=3,
        max_clusters_per_image=8,
        patch_size=256,
    )

    # Filter training and validation cohort to balanced calcification-focused subset from INbreast and VinDr
    calc_annot_ids = set(annot_df[annot_df["finding_type"] == "calcification"]["image_id"].astype(str).unique())

    train_iv = train_df[train_df["site"].isin(["inbreast", "vindr"])].copy()
    train_pos = train_iv[train_iv["image_id"].astype(str).isin(calc_annot_ids)]
    train_neg = train_iv[~train_iv["image_id"].astype(str).isin(calc_annot_ids)].sample(
        n=min(len(train_pos), len(train_iv) - len(train_pos)), random_state=42
    )
    train_subset = pd.concat([train_pos, train_neg]).sample(frac=1.0, random_state=42)

    val_iv = val_df[val_df["site"].isin(["inbreast", "vindr"])].copy()
    val_pos = val_iv[val_iv["image_id"].astype(str).isin(calc_annot_ids)]
    val_neg = val_iv[~val_iv["image_id"].astype(str).isin(calc_annot_ids)].sample(
        n=min(len(val_pos), len(val_iv) - len(val_pos)), random_state=42
    )
    val_subset = pd.concat([val_pos, val_neg]).sample(frac=1.0, random_state=42)

    logger.info("Extracting patch bags for Training Cohort (%d images)...", len(train_subset))
    train_bags = extract_bags_for_dataset(train_subset, detector)

    logger.info("Extracting patch bags for Validation Cohort (%d images)...", len(val_subset))
    val_bags = extract_bags_for_dataset(val_subset, detector)

    # 2. Build and Train CalcificationExpertMIL Model
    expert_model = CalcificationExpertMIL(in_channels=1, embed_dim=128, pooling_mode="max").to(device)
    optimizer = torch.optim.AdamW(expert_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    logger.info("Training CalcificationExpertMIL on centralized patch bags (15 epochs)...")
    expert_model.train()
    for epoch in range(15):
        total_loss = 0.0
        np.random.shuffle(train_bags)
        for bag in train_bags:
            patches = bag["patches"].to(device)
            target = torch.tensor([bag["y_binary"]], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            bag_logit, _ = expert_model.forward_bag(patches)
            loss = criterion(bag_logit.view(1), target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        if (epoch + 1) % 5 == 0:
            logger.info("Epoch %d/15: Mean Bag Loss = %.4f", epoch + 1, total_loss / max(1, len(train_bags)))

    expert_model.eval()

    # 3. Load Global ResNet-18 Model
    global_ckpt = Path("results/runs/20261103_222946-d91a0e4a-e899c89/ckpt/best.pt")
    global_model = build_model(num_classes=3, pretrained=False, bn_mode="trainable")
    ckpt_mgr = CheckpointManager(global_ckpt.parent)
    ckpt_mgr.load(global_ckpt, model=global_model, device=device)
    global_model.to(device)
    global_model.eval()

    # 4. Generate Validation Features for Logistic Fusion
    logger.info("Generating validation predictions for decision fusion...")
    val_global_logits = []
    val_expert_logits = []
    val_y = []

    with torch.no_grad():
        for bag in val_bags:
            # Global score
            arr_512 = np.load(Path("data") / bag["cached_path"])
            img_t = torch.from_numpy(arr_512).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
            g_logits = global_model(img_t).squeeze().cpu().numpy()
            g_c2_logit = float(g_logits[2] - np.max(g_logits[:2]))

            # Expert score
            patches = bag["patches"].to(device)
            exp_logit, _ = expert_model.forward_bag(patches)
            exp_logit_val = float(exp_logit.item())

            val_global_logits.append(g_c2_logit)
            val_expert_logits.append(exp_logit_val)
            val_y.append(int(bag["y_binary"]))

    X_val = np.column_stack([val_global_logits, val_expert_logits])
    y_val = np.array(val_y)

    fusion_clf = LogisticRegression()
    fusion_clf.fit(X_val, y_val)
    logger.info("Fitted validation logistic fusion weights: Global = %.4f | Expert = %.4f | Intercept = %.4f",
                fusion_clf.coef_[0, 0], fusion_clf.coef_[0, 1], fusion_clf.intercept_[0])

    # 5. Evaluate on Held-Out Calcification Test Cohort (N=236)
    # Map test cohort into findings
    cmmd_path = Path("data/cmmd/CMMD/CMMD_clinicaldata_revision (1).xlsx")
    cdd_path = Path("data/cdd/radiology_manual_ann.xlsx")
    cmmd_clinical = pd.read_excel(cmmd_path)
    cdd_clinical = pd.read_excel(cdd_path)

    from scripts.run_mass_vs_calc_analysis import annotate_test_df_with_findings
    test_annotated = annotate_test_df_with_findings(test_df, annot_df, cmmd_clinical, cdd_clinical)
    calc_test_df = test_annotated[test_annotated["finding_category"] == "calcification"].copy()

    logger.info("\nEvaluating on Held-Out Calcification Test Cohort: %d images across all 4 sites...", len(calc_test_df))
    test_bags = extract_bags_for_dataset(calc_test_df, detector)

    test_global_probs = []
    test_expert_probs = []
    test_fused_probs = []
    test_y = []
    test_patients = []

    with torch.no_grad():
        for bag in test_bags:
            # Global score
            arr_512 = np.load(Path("data") / bag["cached_path"])
            img_t = torch.from_numpy(arr_512).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
            g_logits = global_model(img_t).squeeze().cpu().numpy()
            g_prob_c2 = float(torch.softmax(torch.from_numpy(g_logits), dim=-1)[2])

            g_c2_logit = float(g_logits[2] - np.max(g_logits[:2]))

            # Expert score
            patches = bag["patches"].to(device)
            exp_logit, _ = expert_model.forward_bag(patches)
            exp_logit_val = float(exp_logit.item())
            exp_prob = float(torch.sigmoid(exp_logit).item())

            # Fused score
            X_sample = np.array([[g_c2_logit, exp_logit_val]])
            fused_prob = float(fusion_clf.predict_proba(X_sample)[0, 1])

            test_global_probs.append(g_prob_c2)
            test_expert_probs.append(exp_prob)
            test_fused_probs.append(fused_prob)
            test_y.append(int(bag["y_binary"]))
            test_patients.append(bag["patient_hash"])

    y_test = np.array(test_y)
    p_global = np.array(test_global_probs)
    p_fused = np.array(test_fused_probs)
    patients = np.array(test_patients)

    auc_global = float(roc_auc_score(y_test, p_global))
    auc_fused = float(roc_auc_score(y_test, p_fused))
    delta_auc = auc_fused - auc_global

    # 6. Paired Patient Bootstrap (1,000 resamples)
    logger.info("Executing 1,000-resample paired patient bootstrap...")
    unique_patients = np.unique(patients)
    n_pts = len(unique_patients)
    rng = np.random.RandomState(42)

    boot_deltas = []
    for _ in range(1000):
        sampled_pts = rng.choice(unique_patients, size=n_pts, replace=True)
        # Gather samples
        idx_list = []
        for p in sampled_pts:
            idx_list.extend(np.where(patients == p)[0])
        idx_arr = np.array(idx_list)

        y_b = y_test[idx_arr]
        if len(np.unique(y_b)) < 2:
            continue
        auc_g_b = roc_auc_score(y_b, p_global[idx_arr])
        auc_f_b = roc_auc_score(y_b, p_fused[idx_arr])
        boot_deltas.append(auc_f_b - auc_g_b)

    ci_low = float(np.percentile(boot_deltas, 2.5))
    ci_high = float(np.percentile(boot_deltas, 97.5))
    criterion_met = (delta_auc >= 0.030) and (ci_low > 0.0)

    logger.info("\n=== Calcification Expert Decision Fusion Results (Held-Out Test N=%d) ===", len(y_test))
    logger.info("1. Global ResNet-18 Malignancy AUC: %.4f", auc_global)
    logger.info("2. Global + Routed Expert AUC:     %.4f", auc_fused)
    logger.info("3. Delta AUC:                     %+.4f (95%% Bootstrap CI: [%+.4f, %+.4f])",
                delta_auc, ci_low, ci_high)
    logger.info("Pre-Registered Criterion 3 Status (Delta >= +0.030, CI > 0): %s",
                "PASSED" if criterion_met else "FAILED / NULL RESULT")

    out_summary = {
        "cohort": f"Calcification Test Cohort (N={len(y_test)} images across all 4 sites)",
        "global_resnet18_auc": auc_global,
        "routed_expert_fused_auc": auc_fused,
        "delta_auc": delta_auc,
        "bootstrap_95ci_low": ci_low,
        "bootstrap_95ci_high": ci_high,
        "pre_registered_criterion_3": {
            "target_delta_auc": 0.030,
            "measured_delta_auc": delta_auc,
            "ci_strictly_positive": bool(ci_low > 0.0),
            "status": "PASSED" if criterion_met else "FAILED / NULL RESULT",
        },
        "scientific_interpretation": (
            "Hypothesis validated: High-resolution patch expert fusion overcomes 512 px pooling loss."
            if criterion_met else
            "Hypothesis rejected (Null Result): High-resolution patch expert fusion on weak labels does not "
            "provide statistically significant lift over the global ResNet-18 model (Delta AUC < +0.030 or CI crosses zero). "
            "Per Decision D-010 and AGENTS.md Rule 5, the detector is retained strictly as an explainability and reporting tool."
        ),
    }

    out_json = Path("paper/tables/calc_expert_fusion_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fp:
        json.dump(out_summary, fp, indent=2)
    logger.info("Saved calcification expert fusion results to %s", out_json)


if __name__ == "__main__":
    main()
