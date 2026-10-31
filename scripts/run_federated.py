#!/usr/bin/env python3
"""Federated (FedAvg) training — thin wrapper around src.federated.

    python scripts/run_federated.py --seed 42
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.federated import run_federated


def main():
    ap = argparse.ArgumentParser(description="Run federated learning (FedAvg) training")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="Batch run multiple seeds, e.g. --seeds 42 43 44")
    ap.add_argument("--all-seeds", action="store_true", help="Batch run all seeds defined in config")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="Force retrain and re-evaluate even if results already exist")
    ap.add_argument("--force-train", action="store_true", help="Force retrain federated model from round 1")
    ap.add_argument("--force-eval", action="store_true", help="Force re-evaluation of test sites using existing best.pt without retraining")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware execution device: {device}")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    if args.all_seeds:
        seeds = cfg["eval"].get("seeds", [42, 43, 44])
    elif args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42]

    force_train = args.force or args.force_train
    force_eval = args.force or args.force_eval

    for seed in seeds:
        output_dir = Path(cfg['results_root']) / "federated" / cfg['fed']['weighting'] / f"seed_{seed}"
        run_federated(cfg, seed, device=device, output_dir=str(output_dir),
                      resume=not force_train, force_train=force_train,
                      force_eval=force_eval)


if __name__ == "__main__":
    main()
