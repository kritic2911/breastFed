#!/usr/bin/env python3
"""Centralized baseline: train ONCE on all pooled sites, test on individual datasets.

Usage:
    # Test on a specific dataset (reusing the already-trained centralized model)
    python scripts/run_centralized.py --site INbreast --seed 42
    python scripts/run_centralized.py --site CBIS --seed 42
    python scripts/run_centralized.py --site BCDR --seed 42
    python scripts/run_centralized.py --site CMMD --seed 42

    # Batch test across all datasets individually (trains once, evaluates each)
    python scripts/run_centralized.py --all-sites --seed 42

    # Force retrain the centralized model (trains once, then evaluates)
    python scripts/run_centralized.py --all-sites --seed 42 --force-train

    # Force re-evaluation only (re-evaluates test sets without retraining the model)
    python scripts/run_centralized.py --all-sites --seed 42 --force-eval
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import yaml

from src.calibration import TemperatureScaler
from src.dataset import build_pooled_loader, build_site_loader
from src.evaluate import compute_metrics, evaluate_on_site, predict
from src.model import build_model, verify_bn_state
from src.storage import save_run_artifacts
from src.train import set_seed, train_model


def get_or_train_centralized_model(cfg, seed, site_balanced, device, results_root, force_train=False):
    """Train the centralized model ONCE on all pooled sites, or load if already trained.

    Returns
    -------
    model : trained torch.nn.Module in eval mode
    threshold : float decision threshold calibrated on pooled val
    best_state : dict state_dict
    history : list of epoch metric dicts
    out_dir : Path where artifacts reside
    variant : "site_balanced" or "sample_weighted"
    """
    set_seed(seed)
    variant = "site_balanced" if site_balanced else "sample_weighted"
    data_root = cfg["data_root"]
    all_sites = cfg["sites"]

    out_dir = Path(results_root) / "centralized" / variant / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = out_dir / "best.pt"
    calib_file = out_dir / "calibration.json"
    model = build_model(cfg, device=device)
    verify_bn_state(model, verbose=True)
    history = []
    threshold = None

    if best_model_path.exists() and not force_train:
        print(f"\n{'='*65}")
        print(f"CENTRALIZED BASELINE | Variant: {variant} | Seed: {seed}")
        print(f"  [Reusing Trained Model] Loading centralized checkpoint: {best_model_path}")
        if torch.cuda.is_available():
            print(f"  Device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"{'='*65}")

        best_state = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(best_state)
        if (out_dir / "training_history.csv").exists():
            try:
                history = pd.read_csv(out_dir / "training_history.csv").to_dict(orient="records")
            except Exception:
                pass

        # Try to load calibrated threshold from calibration.json or metrics.json
        if calib_file.exists():
            try:
                with open(calib_file) as f:
                    calib_data = json.load(f)
                    threshold = float(calib_data["threshold"])
            except Exception:
                pass
        if threshold is None and (out_dir / "metrics.json").exists():
            try:
                with open(out_dir / "metrics.json") as f:
                    m_data = json.load(f)
                    threshold = float(m_data.get("threshold"))
            except Exception:
                pass
    else:
        print(f"\n{'='*65}")
        print(f"CENTRALIZED BASELINE (TRAINING ONCE) | Variant: {variant} | Seed: {seed}")
        if torch.cuda.is_available():
            print(f"Device: {device} ({torch.cuda.get_device_name(0)}) | "
                  f"VRAM Allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")
        print(f"{'='*65}")

        print(f"  Training centralized model on all pooled sites: {all_sites}")
        _, train_loader = build_pooled_loader(data_root, all_sites, "train", cfg,
                                               site_balanced=site_balanced)
        _, val_loader = build_pooled_loader(data_root, all_sites, "val", cfg,
                                             site_balanced=False)
        print(f"  Dataset sizes -> Pooled Train: {len(train_loader.dataset)} images | "
              f"Pooled Val: {len(val_loader.dataset)} images")

        # Clean old checkpoints if retraining
        if force_train:
            ckpt_p = out_dir / "last_checkpoint.pt"
            if ckpt_p.exists():
                try:
                    ckpt_p.unlink()
                except Exception:
                    pass

        best_state, history = train_model(model, train_loader, val_loader, cfg,
                                           device=device, checkpoint_dir=out_dir,
                                           resume=not force_train)
        torch.save(best_state, best_model_path)
        model.load_state_dict(best_state)

    model.eval()

    # Calibrate decision threshold and temperature scaling on validation set
    scaler = TemperatureScaler()
    scaler_file = out_dir / "temperature_scaler.json"
    if threshold is None or not scaler_file.exists():
        print("  Calibrating decision threshold and temperature scaling on pooled validation set...")
        _, pooled_val_loader = build_pooled_loader(data_root, all_sites, "val", cfg,
                                                    site_balanced=False)
        val_probs, val_labels, _, val_logits = predict(model, pooled_val_loader, device, return_logits=True)
        val_metrics = compute_metrics(val_labels, val_probs)
        threshold = float(val_metrics["threshold"])
        scaler.fit(val_logits, val_labels)
        scaler.save(scaler_file)
        with open(calib_file, "w") as f:
            json.dump({
                "threshold": threshold,
                "val_auc": float(val_metrics["auc"]),
                "val_sensitivity": float(val_metrics["sensitivity"]),
                "val_specificity": float(val_metrics["specificity"]),
                "temperature": float(scaler.temperature),
                "val_ece": val_metrics.get("ece"),
            }, f, indent=2)
    else:
        scaler.load(scaler_file)

    print(f"  Calibrated Decision Threshold (Pooled Val): {threshold:.4f} | Temperature: {scaler.temperature:.3f}")
    return model, threshold, scaler, best_state, history, out_dir, variant


def evaluate_centralized_on_site(model, threshold, scaler, test_site, cfg, seed, device,
                                  results_root, out_dir, variant, best_state, history,
                                  force_eval=False):
    """Evaluate the already-trained centralized model on a specific test dataset."""
    data_root = cfg["data_root"]
    site_metric_file = out_dir / f"metrics_{test_site}.json"
    pred_file = out_dir / f"predictions_{test_site}.csv"

    if site_metric_file.exists() and pred_file.exists() and not force_eval:
        print(f"\n  [Skipping] Centralized {variant} (seed {seed}) test on '{test_site}' already completed at {site_metric_file}.")
        print("  Use --force or --force-eval to re-evaluate.")
        try:
            with open(site_metric_file) as f:
                return json.load(f)
        except Exception:
            pass

    print(f"\n  --> Evaluating Centralized Model on '{test_site}' Test Set...")
    _, test_loader = build_site_loader(data_root, test_site, "test", cfg)
    site_results = evaluate_on_site(model, test_loader, device,
                                     threshold=threshold,
                                     n_bootstrap=cfg["eval"]["bootstrap_patients"],
                                     seed=seed,
                                     scaler=scaler)

    auc = site_results["image_level"]["auc"]
    ci = site_results["bootstrap_auc_ci"]
    sens = site_results["image_level"]["sensitivity"]
    spec = site_results["image_level"]["specificity"]
    pat_auc = site_results["patient_level"]["auc"]

    print(f"  [Centralized Result on {test_site}]")
    print(f"    Image-level AUC:        {auc:.4f}  [95% CI: {ci['lower']:.4f}, {ci['upper']:.4f}]")
    print(f"    Sensitivity @ 90% Spec: {sens:.4f}  (Observed Specificity: {spec:.4f})")
    print(f"    Patient-level AUC:      {pat_auc:.4f}")

    # Persist run artifacts for this site (merges into metrics.json and all_runs_summary.csv)
    save_run_artifacts(
        method="centralized",
        train_site=variant,
        seed=seed,
        threshold=threshold,
        best_state=best_state,
        history=history,
        own_results={},
        cross_results={test_site: site_results},
        results_root=results_root,
        output_dir=out_dir,
        cfg=cfg,
        extra_meta={
            "variant": variant,
            "test_site": test_site,
        }
    )

    return site_results


def main():
    ap = argparse.ArgumentParser(description="Centralized baseline: train ONCE on all pooled sites, test on individual datasets")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="Batch run multiple seeds, e.g. --seeds 42 43 44")
    ap.add_argument("--all-seeds", action="store_true", help="Batch run all seeds defined in config")
    ap.add_argument("--test-site", "--site", dest="test_site", default=None,
                    choices=["CMMD", "CBIS", "BCDR", "INbreast"],
                    help="Particular dataset to test against (e.g. CMMD, CBIS, BCDR, INbreast)")
    ap.add_argument("--all-sites", action="store_true",
                    help="Evaluate against each dataset individually in sequence")
    ap.add_argument("--site-balanced", action="store_true",
                    help="Use site-balanced sampling (uniform across sites)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true",
                    help="Force retraining of centralized model and re-evaluation of test sites")
    ap.add_argument("--force-train", action="store_true",
                    help="Force retraining of centralized model from scratch")
    ap.add_argument("--force-eval", action="store_true",
                    help="Force re-evaluation of test sites without retraining the model")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.test_site is None and not args.all_sites:
        ap.error("Centralized model trains on all datasets pooled, but must be tested against a particular dataset only.\n"
                 "Please specify --site/--test-site <SITE> (e.g. --site INbreast, --site CMMD, --site CBIS, --site BCDR)\n"
                 "or pass --all-sites to evaluate each dataset individually.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
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

    test_sites = [args.test_site] if args.test_site else cfg["sites"]
    force_train = args.force or args.force_train
    force_eval = args.force or args.force_eval or force_train

    for seed in seeds:
        # STEP 1: Train (or load) the centralized model ONCE for this seed
        model, threshold, scaler, best_state, history, out_dir, variant = get_or_train_centralized_model(
            cfg, seed, args.site_balanced, device, cfg["results_root"],
            force_train=force_train
        )

        # STEP 2: Evaluate on the requested test site(s) using that single trained model
        for t_site in test_sites:
            evaluate_centralized_on_site(
                model, threshold, scaler, t_site, cfg, seed, device, cfg["results_root"],
                out_dir, variant, best_state, history,
                force_eval=force_eval
            )


if __name__ == "__main__":
    main()
