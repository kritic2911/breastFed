#!/usr/bin/env python3
"""Local-only baseline: train one model per site, evaluate on own + all other sites.

    python scripts/run_local.py --seed 42
    python scripts/run_local.py --seed 42 --site CMMD   # single site
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.dataset import build_site_loader
from src.evaluate import compute_metrics, evaluate_on_site, predict
from src.model import build_model
from src.storage import save_run_artifacts
from src.train import set_seed, train_model


def run_one_site(site, cfg, seed, device, results_root):
    """Train on `site`, evaluate on self and all others."""
    set_seed(seed)
    print(f"\n{'='*65}")
    print(f"LOCAL BASELINE | Site: {site} | Seed: {seed}")
    if torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)}) | "
              f"VRAM Allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")
    print(f"{'='*65}")

    data_root = cfg["data_root"]

    out_dir = Path(results_root) / "local" / site / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "metrics.json").exists() and not cfg.get("force", False):
        print(f"  [Skipping] Site '{site}' (seed {seed}) already completed at {out_dir}. Use --force to retrain.")
        try:
            with open(out_dir / "metrics.json") as f:
                return json.load(f).get("cross_site_test", {})
        except Exception:
            pass

    # Loaders for this site
    _, train_loader = build_site_loader(data_root, site, "train", cfg, shuffle=True)
    _, val_loader = build_site_loader(data_root, site, "val", cfg)
    _, test_loader = build_site_loader(data_root, site, "test", cfg)

    print(f"Dataset: Train={len(train_loader.dataset)} | Val={len(val_loader.dataset)} | "
          f"Test={len(test_loader.dataset)}")

    force = cfg.get("force", False)
    if force:
        ckpt_p = out_dir / "last_checkpoint.pt"
        if ckpt_p.exists():
            try:
                ckpt_p.unlink()
            except Exception:
                pass

    # Train (with automatic checkpointing and epoch-level resumption)
    model = build_model(cfg, device=device)
    best_state, history = train_model(model, train_loader, val_loader, cfg,
                                       device=device, checkpoint_dir=out_dir,
                                       resume=not force)
    model.load_state_dict(best_state)
    model.eval()

    # Get threshold from val
    val_probs, val_labels, _ = predict(model, val_loader, device)
    val_metrics = compute_metrics(val_labels, val_probs)
    threshold = val_metrics["threshold"]
    print(f"\nDecision threshold selected from validation: {threshold:.4f} "
          f"(Sens@90%Spec: {val_metrics['sensitivity']:.4f})")

    # Evaluate on own test set
    own_results = evaluate_on_site(model, test_loader, device,
                                    threshold=threshold,
                                    n_bootstrap=cfg["eval"]["bootstrap_patients"],
                                    seed=seed)
    auc = own_results["image_level"]["auc"]
    ci = own_results["bootstrap_auc_ci"]
    print(f"\n  [Self-Evaluation] {site} Test AUC: {auc:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    # Cross-site evaluation (test on all other sites)
    cross_results = {}
    for other_site in cfg["sites"]:
        if other_site == site:
            cross_results[other_site] = own_results
            continue
        _, other_loader = build_site_loader(data_root, other_site, "test", cfg)
        other_results = evaluate_on_site(model, other_loader, device,
                                          threshold=threshold,
                                          n_bootstrap=cfg["eval"]["bootstrap_patients"],
                                          seed=seed)
        cross_auc = other_results["image_level"]["auc"]
        cross_ci = other_results["bootstrap_auc_ci"]
        print(f"  [Cross-Site Evaluation] -> {other_site} Test AUC: {cross_auc:.4f} "
              f"[{cross_ci['lower']:.4f}, {cross_ci['upper']:.4f}]")
        cross_results[other_site] = other_results

    # Persist all run artifacts and update central summaries
    out_dir = save_run_artifacts(
        method="local",
        train_site=site,
        seed=seed,
        threshold=threshold,
        best_state=best_state,
        history=history,
        own_results=own_results,
        cross_results=cross_results,
        results_root=results_root,
    )

    return cross_results


def main():
    ap = argparse.ArgumentParser(description="Run local-only mammography baseline")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="Batch run multiple seeds, e.g. --seeds 42 43 44")
    ap.add_argument("--all-seeds", action="store_true", help="Batch run all seeds defined in config")
    ap.add_argument("--site", default=None, help="Run single site (default: all)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="Force retrain even if results already exist")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["force"] = args.force

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware execution device: {device}")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Determine seeds to run (batch processing)
    if args.all_seeds:
        seeds = cfg["eval"].get("seeds", [42, 43, 44])
    elif args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]

    sites = [args.site] if args.site else cfg["sites"]
    print(f"Sites to run: {sites} | Seeds to run: {seeds}")

    for seed in seeds:
        for site in sites:
            run_one_site(site, cfg, seed, device, cfg["results_root"])


if __name__ == "__main__":
    main()
