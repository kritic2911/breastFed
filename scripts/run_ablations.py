#!/usr/bin/env python3
"""Ablation and experimental campaign driver.

Executes and evaluates federated learning ablations:
  1. Harmonization ablation: proc512 vs raw512 (3 seeds each)
  2. Aggregation weighting: sample-weighted vs uniform (3 seeds each)
  3. FedProx regularization: mu in {0.001, 0.01, 0.1} (3 seeds each)
  4. Normalization strategy: Frozen BN vs FedBN (3 seeds each)
  5. Privacy & Attacks: DP-SGD noise sweep, SecAgg, Gradient Inversion, MIA

Reports Mean ± 95% Confidence Interval across seeds for all sites.
"""

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.federated import run_federated


def compute_mean_ci(values, confidence=0.95):
    """Compute mean, sample std, and 95% confidence interval using Student's t-distribution."""
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(np.mean(vals))
    if n < 2:
        return mean, 0.0, mean, mean
    std = float(np.std(vals, ddof=1))
    std_err = std / np.sqrt(n)
    t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    margin = t_crit * std_err
    return mean, margin, mean - margin, mean + margin


def format_mean_ci(mean, margin, ci_low=None, ci_high=None):
    """Format as 'Mean ± Margin' and optional bounds."""
    return f"{mean:.4f} ± {margin:.4f}"


def load_variant_results(variant_dir: Path, seeds: list, sites: list):
    """Load per-seed metrics from variant directory."""
    records = {site: {"image_auc": [], "patient_auc": [], "sensitivity": []} for site in sites}
    records["Macro"] = {"image_auc": [], "patient_auc": [], "sensitivity": []}
    seed_details = []

    for seed in seeds:
        seed_dir = variant_dir / f"seed_{seed}"
        metrics_file = seed_dir / "metrics.json"
        if not metrics_file.exists():
            print(f"  [Warning] Missing {metrics_file}")
            continue

        with open(metrics_file) as f:
            data = json.load(f)

        # per_site_test or cross_site_test
        test_data = data.get("per_site_test") or data.get("cross_site_test", {})
        macro_img_auc = []
        macro_pat_auc = []
        macro_sens = []

        row_detail = {"seed": seed}
        for site in sites:
            site_res = test_data.get(site, {})
            img_auc = site_res.get("image_level", {}).get("auc", np.nan)
            pat_auc = site_res.get("patient_level", {}).get("auc", np.nan)
            sens = site_res.get("image_level", {}).get("sensitivity", np.nan)

            records[site]["image_auc"].append(img_auc)
            records[site]["patient_auc"].append(pat_auc)
            records[site]["sensitivity"].append(sens)

            row_detail[f"{site}_image_auc"] = img_auc
            row_detail[f"{site}_pat_auc"] = pat_auc
            row_detail[f"{site}_sens"] = sens

            if not np.isnan(img_auc):
                macro_img_auc.append(img_auc)
            if not np.isnan(pat_auc):
                macro_pat_auc.append(pat_auc)
            if not np.isnan(sens):
                macro_sens.append(sens)

        m_img = np.mean(macro_img_auc) if macro_img_auc else np.nan
        m_pat = np.mean(macro_pat_auc) if macro_pat_auc else np.nan
        m_sens = np.mean(macro_sens) if macro_sens else np.nan

        records["Macro"]["image_auc"].append(m_img)
        records["Macro"]["patient_auc"].append(m_pat)
        records["Macro"]["sensitivity"].append(m_sens)

        row_detail["Macro_image_auc"] = m_img
        row_detail["Macro_pat_auc"] = m_pat
        row_detail["Macro_sens"] = m_sens
        seed_details.append(row_detail)

    return records, seed_details


def summarize_ablation(ablation_dir: Path, variants: list, seeds: list, sites: list):
    """Generate summary table of Mean ± 95% CI across seeds for all variants and sites."""
    summary_rows = []
    print("\n" + "=" * 90)
    print(f"ABLATION SUMMARY: {ablation_dir.name}")
    print(f"Seeds: {seeds} | Sites: {sites}")
    print("=" * 90)

    for var_name in variants:
        var_dir = ablation_dir / var_name
        if not var_dir.exists():
            print(f"  [Notice] Variant directory {var_dir} does not exist yet.")
            continue

        records, _ = load_variant_results(var_dir, seeds, sites)

        for site in sites + ["Macro"]:
            img_m, img_err, img_l, img_u = compute_mean_ci(records[site]["image_auc"])
            pat_m, pat_err, pat_l, pat_u = compute_mean_ci(records[site]["patient_auc"])
            sen_m, sen_err, sen_l, sen_u = compute_mean_ci(records[site]["sensitivity"])

            summary_rows.append({
                "Variant": var_name,
                "Site": site,
                "Image_AUC_Mean": img_m,
                "Image_AUC_CI_Half": img_err,
                "Image_AUC_95CI": format_mean_ci(img_m, img_err),
                "Patient_AUC_Mean": pat_m,
                "Patient_AUC_CI_Half": pat_err,
                "Patient_AUC_95CI": format_mean_ci(pat_m, pat_err),
                "Sensitivity_Mean": sen_m,
                "Sensitivity_CI_Half": sen_err,
                "Sensitivity_95CI": format_mean_ci(sen_m, sen_err),
                "N_Seeds": len(records[site]["image_auc"]),
            })

    df_summary = pd.DataFrame(summary_rows)
    if df_summary.empty:
        print("No results found to summarize.")
        return df_summary

    # Display Pivot for Image AUC (Site vs Variant)
    pivot_img = df_summary.pivot(index="Site", columns="Variant", values="Image_AUC_95CI")
    # Order index: sites then Macro
    site_order = [s for s in sites if s in pivot_img.index] + (["Macro"] if "Macro" in pivot_img.index else [])
    pivot_img = pivot_img.reindex(site_order)

    print("\n[Image-Level AUC (Mean ± 95% CI across 3 seeds)]")
    print(pivot_img.to_markdown())

    pivot_pat = df_summary.pivot(index="Site", columns="Variant", values="Patient_AUC_95CI")
    pivot_pat = pivot_pat.reindex(site_order)
    print("\n[Patient-Level AUC (Mean ± 95% CI across 3 seeds)]")
    print(pivot_pat.to_markdown())

    # Save summary files
    out_csv = ablation_dir / "summary.csv"
    df_summary.to_csv(out_csv, index=False)
    out_md = ablation_dir / "summary.md"
    with open(out_md, "w") as f:
        f.write(f"# Ablation Results: {ablation_dir.name}\n\n")
        f.write("### Image-Level AUC (Mean ± 95% CI across seeds)\n\n")
        f.write(pivot_img.to_markdown())
        f.write("\n\n### Patient-Level AUC (Mean ± 95% CI across seeds)\n\n")
        f.write(pivot_pat.to_markdown())
        f.write("\n\n### Full Numerical Summary\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n")

    print(f"\nSaved summary to:\n  - {out_csv}\n  - {out_md}\n" + "=" * 90)
    return df_summary


def run_harmonization_ablation(cfg, seeds, device="cpu", force_train=False, force_eval=False):
    """Step 1: Harmonization ablation — proc512 vs raw512."""
    ablation_root = Path(cfg.get("results_root", "results")) / "ablation_harmonization"
    variants = ["proc512", "raw512"]
    sites = cfg["sites"]

    print("\n" + "#" * 70)
    print("STEP 1: HARMONIZATION ABLATION (proc512 vs raw512)")
    print(f"Seeds: {seeds} | Device: {device}")
    print("#" * 70)

    # 1. Reuse existing federated sample baseline runs for proc512 if available
    for s in seeds:
        proc_s_dir = ablation_root / "proc512" / f"seed_{s}"
        fed_s_dir = Path(cfg.get("results_root", "results")) / "federated" / "sample" / f"seed_{s}"
        if fed_s_dir.exists() and (fed_s_dir / "metrics.json").exists() and not proc_s_dir.exists():
            print(f"  [Reusing Run] Copying pre-computed federated baseline for proc512 seed {s} from {fed_s_dir}")
            shutil.copytree(fed_s_dir, proc_s_dir)

    for var_name, img_col in [("proc512", "proc_path"), ("raw512", "raw_path")]:
        var_cfg = copy.deepcopy(cfg)
        var_cfg["image_col"] = img_col
        print(f"\n{'='*60}\nRunning Variant: {var_name} (image_col = {img_col})\n{'='*60}")

        for seed in seeds:
            out_dir = ablation_root / var_name / f"seed_{seed}"
            print(f"\n--> {var_name} | Seed {seed} | Out: {out_dir}")
            run_federated(
                cfg=var_cfg,
                seed=seed,
                device=device,
                output_dir=str(out_dir),
                resume=not force_train,
                force_train=force_train,
                force_eval=force_eval,
                method_name=f"ablation_harmonization_{var_name}",
            )

    summarize_ablation(ablation_root, variants, seeds, sites)


def run_weighting_ablation(cfg, seeds, device="cpu", force_train=False, force_eval=False):
    """Step 2: Aggregation weighting — sample-weighted vs uniform."""
    ablation_root = Path(cfg.get("results_root", "results")) / "ablation_weighting"
    variants = ["sample", "uniform"]
    sites = cfg["sites"]

    print("\n" + "#" * 70)
    print("STEP 2: AGGREGATION WEIGHTING ABLATION (sample-weighted vs uniform)")
    print(f"Seeds: {seeds} | Device: {device}")
    print("#" * 70)

    # Reuse proc512 / federated baseline runs for sample weighting if available
    sample_dir = ablation_root / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        target_s = sample_dir / f"seed_{s}"
        fed_s = Path(cfg.get("results_root", "results")) / "federated" / "sample" / f"seed_{s}"
        proc_s = Path(cfg.get("results_root", "results")) / "ablation_harmonization" / "proc512" / f"seed_{s}"
        src = fed_s if (fed_s / "metrics.json").exists() else (proc_s if (proc_s / "metrics.json").exists() else None)
        if src and not target_s.exists():
            print(f"  [Reusing Runs] Copying run for seed {s} to sample weighting from {src}...")
            shutil.copytree(src, target_s)

    for weighting in ["sample", "uniform"]:
        var_cfg = copy.deepcopy(cfg)
        var_cfg["fed"]["weighting"] = weighting
        print(f"\n{'='*60}\nRunning Variant: {weighting}-weighted FedAvg\n{'='*60}")

        for seed in seeds:
            out_dir = ablation_root / weighting / f"seed_{seed}"
            print(f"\n--> Weighting: {weighting} | Seed {seed} | Out: {out_dir}")
            run_federated(
                cfg=var_cfg,
                seed=seed,
                device=device,
                output_dir=str(out_dir),
                resume=not force_train,
                force_train=force_train,
                force_eval=force_eval,
                method_name=f"ablation_weighting_{weighting}",
            )

    summarize_ablation(ablation_root, variants, seeds, sites)


def run_fedprox_ablation(cfg, seeds, mu_list=None, device="cpu", force_train=False, force_eval=False):
    """Step 3: FedProx regularization sweep — mu in {0.001, 0.01, 0.1}."""
    mu_list = mu_list or [0.001, 0.01, 0.1]
    ablation_root = Path(cfg.get("results_root", "results")) / "ablation_fedprox"
    variants = [f"mu_{mu}" for mu in mu_list]
    sites = cfg["sites"]

    print("\n" + "#" * 70)
    print(f"STEP 3: FEDPROX REGULARIZATION SWEEP (mu in {mu_list})")
    print(f"Seeds: {seeds} | Device: {device}")
    print("#" * 70)

    for mu in mu_list:
        var_name = f"mu_{mu}"
        var_cfg = copy.deepcopy(cfg)
        var_cfg["fed"]["mu"] = mu
        print(f"\n{'='*60}\nRunning Variant: FedProx (mu = {mu})\n{'='*60}")

        for seed in seeds:
            out_dir = ablation_root / var_name / f"seed_{seed}"
            print(f"\n--> FedProx mu={mu} | Seed {seed} | Out: {out_dir}")
            run_federated(
                cfg=var_cfg,
                seed=seed,
                device=device,
                output_dir=str(out_dir),
                resume=not force_train,
                force_train=force_train,
                force_eval=force_eval,
                method_name=f"ablation_fedprox_{var_name}",
            )

    summarize_ablation(ablation_root, variants, seeds, sites)


def run_fedbn_ablation(cfg, seeds, device="cpu", force_train=False, force_eval=False):
    """Step 4: Normalization ablation — Frozen BN vs FedBN (per-site BN)."""
    ablation_root = Path(cfg.get("results_root", "results")) / "ablation_fedbn"
    variants = ["frozen_bn", "fedbn"]
    sites = cfg["sites"]

    print("\n" + "#" * 70)
    print("STEP 4: NORMALIZATION STRATEGY (Frozen BN vs FedBN)")
    print(f"Seeds: {seeds} | Device: {device}")
    print("#" * 70)

    # Reuse proc512 / federated baseline runs for frozen_bn if available
    frozen_dir = ablation_root / "frozen_bn"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        target_s = frozen_dir / f"seed_{s}"
        fed_s = Path(cfg.get("results_root", "results")) / "federated" / "sample" / f"seed_{s}"
        proc_s = Path(cfg.get("results_root", "results")) / "ablation_harmonization" / "proc512" / f"seed_{s}"
        src = fed_s if (fed_s / "metrics.json").exists() else (proc_s if (proc_s / "metrics.json").exists() else None)
        if src and not target_s.exists():
            print(f"  [Reusing Runs] Copying baseline run for frozen_bn seed {s} from {src}...")
            shutil.copytree(src, target_s)

    for var_name, is_fedbn in [("frozen_bn", False), ("fedbn", True)]:
        var_cfg = copy.deepcopy(cfg)
        var_cfg["fed"]["fedbn"] = is_fedbn
        var_cfg["freeze_bn"] = not is_fedbn  # FedBN unfreezes BN to learn domain stats
        print(f"\n{'='*60}\nRunning Variant: {var_name} (fedbn={is_fedbn})\n{'='*60}")

        for seed in seeds:
            out_dir = ablation_root / var_name / f"seed_{seed}"
            print(f"\n--> {var_name} | Seed {seed} | Out: {out_dir}")
            run_federated(
                cfg=var_cfg,
                seed=seed,
                device=device,
                output_dir=str(out_dir),
                resume=not force_train,
                force_train=force_train,
                force_eval=force_eval,
                method_name=f"ablation_{var_name}",
            )

    summarize_ablation(ablation_root, variants, seeds, sites)


def main():
    ap = argparse.ArgumentParser(description="FL Ablation and Experimental Campaign Driver")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--ablation", required=True,
                    choices=["harmonization", "weighting", "fedprox", "fedbn", "privacy", "summary"],
                    help="Which ablation step to run")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="Seeds to run, default: 0 1 2")
    ap.add_argument("--target-dir", default=None,
                    help="For --ablation summary: specify the ablation directory to summarize")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="Force retrain from round 1")
    ap.add_argument("--force-train", action="store_true", help="Force retrain from round 1")
    ap.add_argument("--force-eval", action="store_true", help="Re-evaluate existing checkpoints")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware execution device: {device}")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    force_train = args.force or args.force_train
    force_eval = args.force or args.force_eval

    if args.ablation == "harmonization":
        run_harmonization_ablation(cfg, args.seeds, device=device,
                                  force_train=force_train, force_eval=force_eval)
    elif args.ablation == "weighting":
        run_weighting_ablation(cfg, args.seeds, device=device,
                               force_train=force_train, force_eval=force_eval)
    elif args.ablation == "fedprox":
        run_fedprox_ablation(cfg, args.seeds, device=device,
                             force_train=force_train, force_eval=force_eval)
    elif args.ablation == "fedbn":
        run_fedbn_ablation(cfg, args.seeds, device=device,
                           force_train=force_train, force_eval=force_eval)
    elif args.ablation == "summary":
        if not args.target_dir:
            print("Error: --target-dir must be provided for --ablation summary")
            sys.exit(1)
        target = Path(args.target_dir)
        variants = [p.name for p in target.iterdir() if p.is_dir() and not p.name.startswith(".")]
        summarize_ablation(target, sorted(variants), args.seeds, cfg["sites"])


if __name__ == "__main__":
    main()
