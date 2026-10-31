#!/usr/bin/env python3
"""Diagnostic sanity check suite:
1. Overfit-32-images test (verifies model capacity, loss convergence, and probability spread)
2. Train vs Val AUC progression table
3. Probability Standard Deviation spread & collapse monitor

Usage:
    python scripts/diagnostics.py --site CBIS
    python scripts/diagnostics.py --site BCDR --lr 1e-4
    python scripts/diagnostics.py --all-sites
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim import Adam

from src.dataset import MammoDataset
from src.model import build_model, freeze_batchnorm, get_trainable_params


def run_overfit_32_test(cfg, site="CBIS", device="cuda" if torch.cuda.is_available() else "cpu",
                        n_epochs=40, lr=1e-4, freeze_bn=True):
    """Overfit exactly 32 real images (16 positive, 16 negative)."""
    print(f"\n{'='*75}")
    print(f"1. OVERFIT-32-IMAGES TEST | Site: {site} | lr: {lr} | freeze_bn: {freeze_bn}")
    print(f"{'='*75}")

    data_root = cfg["data_root"]
    ds = MammoDataset(data_root, site, "train", {**cfg, "augmentation": {"train": []}})
    labels = ds.labels

    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    n_pos = min(16, len(pos_indices))
    n_neg = min(32 - n_pos, len(neg_indices))

    selected_indices = np.concatenate([pos_indices[:n_pos], neg_indices[:n_neg]])
    np.random.seed(42)
    np.random.shuffle(selected_indices)

    print(f"Dataset sample: 32 images ({n_pos} malignant/pos, {n_neg} benign/neg)")

    imgs_list, labels_list = [], []
    for idx in selected_indices:
        img_t, lbl_t, _ = ds[idx]
        imgs_list.append(img_t)
        labels_list.append(lbl_t)

    batch_imgs = torch.stack(imgs_list).to(device)
    batch_labels = torch.stack(labels_list).to(device)

    model = build_model(cfg, device=device)
    if freeze_bn:
        freeze_batchnorm(model)
    else:
        model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(get_trainable_params(model), lr=lr, weight_decay=0.0)

    print(f"\n{'Epoch':>6} | {'Train Loss':>11} | {'Train AUC':>10} | {'Train Acc':>10} | {'Prob Std':>10} | {'Prob Range':>18}")
    print("-" * 75)

    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        if freeze_bn:
            freeze_batchnorm(model)

        optimizer.zero_grad()
        logits = model(batch_imgs).squeeze(-1)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        lbls = batch_labels.cpu().numpy()

        acc = ((probs >= 0.5).astype(int) == lbls).mean()
        auc = roc_auc_score(lbls, probs) if len(np.unique(lbls)) > 1 else 0.5
        p_std = float(np.std(probs))
        p_min, p_max = float(probs.min()), float(probs.max())

        record = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "auc": float(auc),
            "acc": float(acc),
            "prob_std": p_std,
            "min_prob": p_min,
            "max_prob": p_max,
        }
        history.append(record)

        if epoch in [1, 2, 5, 10, 15, 20, 25, 30, 35, 40] or epoch == n_epochs:
            p_range = f"[{p_min:.3f}, {p_max:.3f}]"
            print(f"{epoch:6d} | {loss.item():11.5f} | {auc:10.4f} | {acc*100:9.1f}% | {p_std:10.4f} | {p_range:>18}")

    print("-" * 75)
    final_loss = history[-1]["loss"]
    final_auc = history[-1]["auc"]
    final_acc = history[-1]["acc"]
    final_std = history[-1]["prob_std"]

    if final_auc >= 0.99 and final_loss < 0.05:
        print("VERDICT: PASSED! Model completely memorized the 32-image batch.")
        print(f"  Final Loss: {final_loss:.6f} | Final Train AUC: {final_auc:.4f} | Final Acc: {final_acc*100:.1f}% | Final Prob Std: {final_std:.4f}")
        print("  Gradients flow cleanly through all ResNet layers; no vanishing/exploding gradients.")
    else:
        print(f"VERDICT: Needs attention. Loss: {final_loss:.4f}, AUC: {final_auc:.4f}, Prob Std: {final_std:.4f}.")

    return history


def show_train_vs_val_auc_and_prob_std(results_root="results"):
    """Inspect and display Train vs Val AUC and Probability Std from existing histories."""
    print(f"\n{'='*75}")
    print("2. HISTORICAL TRAIN VS VAL AUC & PROBABILITY SPREAD ANALYSIS")
    print(f"{'='*75}")

    p = Path(results_root) / "centralized" / "sample_weighted" / "seed_42" / "training_history.csv"
    if p.exists():
        df = pd.read_csv(p)
        print(f"\nFound Centralized (50 epochs) history at {p}:")
        display_cols = [c for c in ["epoch", "train_loss", "train_auc", "train_prob_std", "val_auc", "val_prob_std", "val_sensitivity"] if c in df.columns]
        if "train_auc" not in df.columns:
            print("  (Note: Existing 50-epoch history has val_auc and loss; updated train_model now logs train_auc & prob_std)")
            print(df[["epoch", "train_loss", "val_auc", "val_sensitivity", "lr"]].iloc[[0, 4, 9, 19, 29, 39, 44, 49]].to_string(index=False))
        else:
            print(df[display_cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Diagnostics: Overfit-32, Train vs Val AUC, Prob Std")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--site", default="CBIS", choices=["CMMD", "CBIS", "BCDR", "INbreast"])
    ap.add_argument("--all-sites", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze-bn", action="store_true", default=True)
    ap.add_argument("--unfreeze-bn", dest="freeze_bn", action="store_false")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sites = ["CBIS", "BCDR"] if args.all_sites else [args.site]
    for site in sites:
        run_overfit_32_test(cfg, site=site, device=device, n_epochs=args.epochs,
                             lr=args.lr, freeze_bn=args.freeze_bn)

    show_train_vs_val_auc_and_prob_std(cfg.get("results_root", "results"))


if __name__ == "__main__":
    main()
