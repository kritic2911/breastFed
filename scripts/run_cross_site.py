#!/usr/bin/env python3
"""Cross-site gap: leave-one-site-out (LOSO) evaluation.

Train centralized and federated on 3 sites, test on the held-out 4th.

Usage:
    # Run all held-out sites (or single held-out site)
    python scripts/run_cross_site.py --seed 42
    python scripts/run_cross_site.py --seed 42 --held-out BCDR

    # Force retraining models from scratch
    python scripts/run_cross_site.py --seed 42 --held-out BCDR --force-train

    # Force re-evaluation only (re-evaluates test sets using trained models in seconds)
    python scripts/run_cross_site.py --seed 42 --held-out BCDR --force-eval
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml
import torch

from src.dataset import build_site_loader, build_pooled_loader
from src.model import build_model
from src.train import train_model, set_seed
from src.evaluate import evaluate_on_site, predict, compute_metrics
from src.federated import run_federated


def run_centralized_loso(cfg, held_out, seed, device, results_root,
                         force_train=False, force_eval=False):
    """Train centralized on 3 sites (or load existing best.pt), test on held-out and in-distribution."""
    set_seed(seed)
    train_sites = [s for s in cfg["sites"] if s.lower() != held_out.lower()]
    data_root = cfg["data_root"]

    out_dir = Path(results_root) / "cross_site" / held_out / "centralized" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.json"

    if metrics_path.exists() and not force_train and not force_eval:
        print(f"  [Skipping] Centralized LOSO held_out={held_out} already completed at {metrics_path}.")
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except Exception:
            pass

    print(f"\n  CENTRALIZED LOSO | held_out={held_out} | train_on={train_sites}")
    model = build_model(cfg, device=device)
    history = []

    if best_model_path.exists() and not force_train:
        print(f"    [Reusing Trained Model] Loading centralized checkpoint: {best_model_path}")
        best_state = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(best_state)
        if (out_dir / "training_history.csv").exists():
            try:
                history = pd.read_csv(out_dir / "training_history.csv").to_dict(orient="records")
            except Exception:
                pass
    else:
        print(f"    Training centralized model on 3 pooled sites: {train_sites}")
        _, train_loader = build_pooled_loader(data_root, train_sites, "train", cfg)
        _, val_loader = build_pooled_loader(data_root, train_sites, "val", cfg)

        if force_train:
            ckpt_p = out_dir / "last_checkpoint.pt"
            if ckpt_p.exists():
                try:
                    ckpt_p.unlink()
                except Exception:
                    pass

        best_state, history = train_model(model, train_loader, val_loader, cfg,
                                           device=device, verbose=True,
                                           checkpoint_dir=out_dir,
                                           resume=not force_train)
        torch.save(best_state, best_model_path)
        model.load_state_dict(best_state)

    model.eval()

    # Threshold from 3-site pooled validation set
    _, val_loader = build_pooled_loader(data_root, train_sites, "val", cfg)
    val_probs, val_labels, _ = predict(model, val_loader, device)
    threshold = compute_metrics(val_labels, val_probs)["threshold"]

    # Evaluate on held-out site
    print(f"    Evaluating on held-out test site: {held_out}...")
    _, held_out_loader = build_site_loader(data_root, held_out, "test", cfg)
    held_out_results = evaluate_on_site(model, held_out_loader, device,
                                         threshold=threshold,
                                         n_bootstrap=cfg["eval"]["bootstrap_patients"],
                                         seed=seed)
    held_auc = held_out_results["image_level"]["auc"]
    held_ci = held_out_results["bootstrap_auc_ci"]
    print(f"    Held-out ({held_out}): AUC {held_auc:.4f} "
          f"[{held_ci['lower']:.4f}, {held_ci['upper']:.4f}]")

    # Evaluate on in-distribution sites
    in_dist_results = {}
    in_dist_aucs = []
    for site in train_sites:
        _, test_loader = build_site_loader(data_root, site, "test", cfg)
        site_results = evaluate_on_site(model, test_loader, device,
                                         threshold=threshold,
                                         n_bootstrap=cfg["eval"]["bootstrap_patients"],
                                         seed=seed)
        in_dist_results[site] = site_results
        in_dist_aucs.append(site_results["image_level"]["auc"])
        print(f"    In-dist  ({site}): AUC {site_results['image_level']['auc']:.4f}")

    gap = np.mean(in_dist_aucs) - held_auc
    print(f"    Centralized Gap (avg in-dist - held-out): {gap:.4f}")

    results = {
        "method": "centralized_loso",
        "held_out": held_out,
        "train_sites": train_sites,
        "seed": seed,
        "threshold": float(threshold),
        "held_out_test": held_out_results,
        "in_dist_test": in_dist_results,
        "gap": float(gap),
        "history": history,
    }
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def run_federated_loso(cfg, held_out, seed, device, results_root,
                       force_train=False, force_eval=False):
    """Train federated on 3 sites (or load existing best.pt), test on held-out."""
    loso_cfg = dict(cfg)
    loso_cfg["sites"] = [s for s in cfg["sites"] if s.lower() != held_out.lower()]

    output_dir = Path(results_root) / "cross_site" / held_out / "federated" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    if metrics_path.exists() and not force_train and not force_eval:
        print(f"  [Skipping] Federated LOSO held_out={held_out} already completed at {metrics_path}.")
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except Exception:
            pass

    print(f"\n  FEDERATED LOSO | held_out={held_out} | train_on={loso_cfg['sites']}")
    fed_results = run_federated(loso_cfg, seed, device=device,
                                 output_dir=str(output_dir),
                                 resume=not force_train,
                                 force_train=force_train,
                                 force_eval=force_eval)

    # Evaluate on held-out site with the best federated model
    best_state = fed_results.get("best_state")
    if best_state is None:
        ckpt_path = output_dir / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = output_dir / "best_checkpoint.pt"
        best_state = torch.load(ckpt_path, map_location=device, weights_only=True)

    model = build_model(cfg, device=device)
    model.load_state_dict(best_state)
    model.eval()

    threshold = fed_results["threshold"]
    _, held_out_loader = build_site_loader(cfg["data_root"], held_out, "test", cfg)
    held_out_results = evaluate_on_site(model, held_out_loader, device,
                                         threshold=threshold,
                                         n_bootstrap=cfg["eval"]["bootstrap_patients"],
                                         seed=seed)
    held_auc = held_out_results["image_level"]["auc"]
    held_ci = held_out_results["bootstrap_auc_ci"]
    print(f"    Held-out ({held_out}): AUC {held_auc:.4f} "
          f"[{held_ci['lower']:.4f}, {held_ci['upper']:.4f}]")

    # Compute gap
    in_dist_aucs = [fed_results["per_site_test"][s]["image_level"]["auc"]
                    for s in loso_cfg["sites"]]
    gap = np.mean(in_dist_aucs) - held_auc
    print(f"    Federated Gap (avg in-dist - held-out): {gap:.4f}")

    # Update results
    fed_results["held_out"] = held_out
    fed_results["held_out_test"] = held_out_results
    fed_results["gap"] = float(gap)

    with open(metrics_path, "w") as f:
        json.dump(fed_results, f, indent=2, default=str)

    return fed_results


def main():
    ap = argparse.ArgumentParser(description="Leave-One-Site-Out (LOSO) cross-site evaluation")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--held-out", default=None,
                    help="Single site to hold out (default: all 4)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true",
                    help="Force retraining and re-evaluation even if results already exist")
    ap.add_argument("--force-train", action="store_true",
                    help="Force retraining models from scratch")
    ap.add_argument("--force-eval", action="store_true",
                    help="Force re-evaluation using already trained models without retraining")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    held_outs = [args.held_out] if args.held_out else cfg["sites"]
    force_train = args.force or args.force_train
    force_eval = args.force or args.force_eval

    for held_out in held_outs:
        print(f"\n{'='*60}")
        print(f"LEAVE-ONE-SITE-OUT: {held_out}")
        print(f"{'='*60}")

        run_centralized_loso(cfg, held_out, args.seed, device, cfg["results_root"],
                             force_train=force_train, force_eval=force_eval)

        run_federated_loso(cfg, held_out, args.seed, device, cfg["results_root"],
                           force_train=force_train, force_eval=force_eval)


if __name__ == "__main__":
    main()
