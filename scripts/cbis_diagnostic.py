#!/usr/bin/env python3
"""CBIS Diagnostic Suite:
1. Shuffled-label control: Train ResNet-18 on CBIS with permuted labels.
   Tests whether memorization on 32 images is independent of label semantic correctness.
2. Flip test: Evaluate 1 - p vs p on CBIS test set across models.
3. 20 Previews: Eyeball 20 sample images and verify joined labels against raw case descriptions.
4. Eval path check: Compare training-time predict() AUC vs results-CSV evaluate_on_site() AUC.
"""

import copy
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.optim import Adam
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_model, get_trainable_params
from src.dataset import build_site_loader, MammoDataset, build_loader
from src.evaluate import predict, compute_metrics, evaluate_on_site


def run_shuffled_label_control(cfg, device="cuda"):
    """Train ResNet-18 on CBIS 32-image subset with PERMUTED (random) labels."""
    print("\n" + "=" * 70)
    print("1. SHUFFLED-LABEL CONTROL TEST ON CBIS (32 images)")
    print("=" * 70)

    # 1. Real labels
    ds, loader = build_site_loader(cfg["data_root"], "CBIS", "train", cfg, shuffle=False)
    indices = np.random.RandomState(42).choice(len(ds), size=32, replace=False)
    subset_real = torch.utils.data.Subset(ds, indices)
    loader_real = torch.utils.data.DataLoader(subset_real, batch_size=32, shuffle=True)

    # 2. Permuted labels dataset wrapper
    class PermutedDataset(torch.utils.data.Dataset):
        def __init__(self, subset, seed=42):
            self.subset = subset
            rng = np.random.RandomState(seed)
            self.orig_labels = [float(subset[i][1]) for i in range(len(subset))]
            # Permute labels randomly
            self.perm_labels = rng.permutation(self.orig_labels)
            print(f"Original label balance: {np.mean(self.orig_labels):.2f}")
            print(f"Permuted label balance: {np.mean(self.perm_labels):.2f}")
            print(f"Label agreement with real: {np.mean(np.array(self.orig_labels) == self.perm_labels):.2f}")

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, _, pid = self.subset[idx]
            return img, torch.tensor(self.perm_labels[idx], dtype=torch.float32), pid

    perm_ds = PermutedDataset(subset_real, seed=999)
    loader_perm = torch.utils.data.DataLoader(perm_ds, batch_size=32, shuffle=True)

    # Train on REAL labels first (baseline overfit)
    def train_test_loop(loader_data, desc):
        model = build_model(cfg, device=device)
        optimizer = Adam(get_trainable_params(model), lr=1e-4, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        history = []
        for epoch in range(1, 26):
            model.train()
            total_loss = 0.0
            all_logits, all_y = [], []
            for imgs, y, _ in loader_data:
                imgs, y = imgs.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(imgs).squeeze(-1)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                all_logits.extend(logits.detach().cpu().numpy())
                all_y.extend(y.detach().cpu().numpy())

            probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
            all_y = np.array(all_y)
            auc = roc_auc_score(all_y, probs) if len(np.unique(all_y)) > 1 else float("nan")
            prob_std = float(np.std(probs))

            if epoch in [1, 5, 10, 15, 20, 25]:
                print(f"  [{desc}] Epoch {epoch:2d}: Loss={total_loss:.6f} | AUC={auc:.4f} | Prob Std={prob_std:.4f}")
            history.append({"epoch": epoch, "loss": total_loss, "auc": auc, "prob_std": prob_std})
        return history

    print("\n--- Training on Real CBIS 32 Labels ---")
    hist_real = train_test_loop(loader_real, "REAL")

    print("\n--- Training on Permuted / Shuffled CBIS 32 Labels ---")
    hist_perm = train_test_loop(loader_perm, "SHUFFLED")

    print("\nConclusion of Shuffled Control:")
    print(f"  Real Labels     Epoch 25: Loss = {hist_real[-1]['loss']:.6f}, AUC = {hist_real[-1]['auc']:.4f}")
    print(f"  Shuffled Labels Epoch 25: Loss = {hist_perm[-1]['loss']:.6f}, AUC = {hist_perm[-1]['auc']:.4f}")
    if hist_perm[-1]["loss"] < 0.01:
        print("  -> CONFIRMED: ResNet-18 memorizes arbitrary permuted labels on 32 images effortlessly.")
        print("     Therefore, the 32-image overfit test verifies model capacity, NOT label semantic correctness.")


def run_flip_test_all():
    """Run flip test (p vs 1 - p AUC) across checkpoints on CBIS."""
    print("\n" + "=" * 70)
    print("2. FLIP TEST ON CBIS TEST SET (AUC of p vs 1 - p)")
    print("=" * 70)

    ckpts = [
        ("Federated (Sample-Weighted Seed 42)", "results/federated/sample/seed_42/metrics.json"),
        ("Centralized (Seed 42)", "results/centralized/sample_weighted/seed_42/metrics.json"),
        ("Local CBIS (Seed 42)", "results/local/CBIS/seed_42/metrics.json"),
    ]

    for name, pth in ckpts:
        p = Path(pth)
        if not p.exists():
            print(f"  [Skipping] {name}: {pth} not found.")
            continue
        with open(p) as f:
            d = json.load(f)
        sub = d.get("per_site_test", {}).get("CBIS") or d.get("cross_site_test", {}).get("CBIS") or d.get("own_site_test", {})
        if not sub or "predictions" not in sub:
            print(f"  [Skipping] {name}: No predictions in {pth}")
            continue
        y = np.array(sub["predictions"]["labels"])
        probs = np.array(sub["predictions"]["probs"])
        auc_norm = roc_auc_score(y, probs)
        auc_flip = roc_auc_score(y, 1.0 - probs)
        print(f"  {name:38s} | Normal AUC = {auc_norm:.4f} | Flipped AUC = {auc_flip:.4f}")


def run_eyeball_20_previews():
    """Print 20 sample images with labels, metadata, and ground truth pathology."""
    print("\n" + "=" * 70)
    print("3. EYEBALL 20 CBIS PREVIEWS WITH METADATA & PATHOLOGY")
    print("=" * 70)

    csv_dir = Path("data/CBIS/csv")
    dfs = []
    for p in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(p, dtype=str).fillna("")
        df.columns = [c.strip().lower() for c in df.columns]
        dfs.append(df)
    cases = pd.concat(dfs, ignore_index=True)
    labels = pd.read_csv("data/CBIS/labels_split.csv")

    samples = labels.sample(20, random_state=123)
    rows = []
    for idx, (_, r) in enumerate(samples.iterrows(), 1):
        sop = r["sop_uid"]
        orig_pid = r.get("original_patient_id", "")
        clean_pid = re.sub(r"^.*-(P_\d+).*$", r"\1", str(orig_pid)).upper()
        lat = r["laterality"]
        side_full = "RIGHT" if lat == "R" else "LEFT"
        view = r["view"]
        assigned_lbl = int(r["label"])
        split = r["split"]

        match = cases[
            (cases["patient_id"].str.strip().str.upper() == clean_pid) &
            (cases["left or right breast"].str.strip().str.upper() == side_full) &
            (cases["image view"].str.strip().str.upper() == view.upper())
        ]
        gt_path = match["pathology"].str.strip().str.upper().tolist() if len(match) > 0 else ["NOT_FOUND"]
        abn_type = match["abnormality type"].tolist() if "abnormality type" in match else []
        calc_type = match["calc type"].tolist() if "calc type" in match else []
        subtlety = match["subtlety"].tolist() if "subtlety" in match else []

        status = "MATCH" if (assigned_lbl == 1 and any("MALIGNANT" in p for p in gt_path)) or (assigned_lbl == 0 and not any("MALIGNANT" in p for p in gt_path)) else "MISMATCH"

        rows.append({
            "#": idx,
            "SOP / File": sop[:28] + "..",
            "PID": clean_pid,
            "View": f"{lat}_{view}",
            "Split": split,
            "Assigned": assigned_lbl,
            "GT Pathology": ", ".join(gt_path),
            "Abnormality": ", ".join(abn_type[:1]),
            "Subtlety": ", ".join(subtlety[:1]),
            "Check": status,
        })

    preview_df = pd.DataFrame(rows)
    print(preview_df.to_string(index=False))


def run_eval_path_verification(cfg, device="cuda"):
    """Verify training-time predict() code vs results-CSV evaluate_on_site() code."""
    print("\n" + "=" * 70)
    print("4. EVAL PATH VERIFICATION (predict vs evaluate_on_site)")
    print("=" * 70)

    ds, loader = build_site_loader(cfg["data_root"], "CBIS", "test", cfg, shuffle=False)
    ckpt_path = "results/federated/sample/seed_42/best.pt"
    if not Path(ckpt_path).exists():
        print(f"Checkpoint {ckpt_path} not found.")
        return

    model = build_model(cfg, device=device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # Training-time AUC code
    p_train, y_train, pids_train = predict(model, loader, device)
    m_train = compute_metrics(y_train, p_train)
    auc_train = m_train["auc"]

    # Results-CSV evaluate_on_site code
    res_site = evaluate_on_site(model, loader, device, threshold=0.51155, n_bootstrap=100, seed=42)
    auc_site = res_site["image_level"]["auc"]

    # Saved metrics.json
    with open("results/federated/sample/seed_42/metrics.json") as f:
        saved_m = json.load(f)
    auc_saved = saved_m["per_site_test"]["CBIS"]["image_level"]["auc"]

    diff_train_site = abs(auc_train - auc_site)
    diff_site_saved = abs(auc_site - auc_saved)

    print(f"  Training-time predict() AUC:      {auc_train:.6f}")
    print(f"  Results-CSV evaluate_on_site AUC:  {auc_site:.6f}")
    print(f"  Saved metrics.json AUC:           {auc_saved:.6f}")
    print(f"  Difference (Train vs CSV):        {diff_train_site:.1e}")
    print(f"  Difference (CSV vs Saved JSON):   {diff_site_saved:.1e}")
    print(f"  Arrays identical:")
    print(f"    - Probabilities: {np.allclose(p_train, res_site['predictions']['probs'])}")
    print(f"    - Labels:        {np.array_equal(y_train, res_site['predictions']['labels'])}")
    print(f"    - Patient IDs:   {np.array_equal(pids_train, res_site['predictions']['patient_ids'])}")


def main():
    with open("configs/experiment.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_shuffled_label_control(cfg, device=device)
    run_flip_test_all()
    run_eyeball_20_previews()
    run_eval_path_verification(cfg, device=device)


if __name__ == "__main__":
    main()
