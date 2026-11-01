#!/usr/bin/env python3
"""Resolution test: Local CBIS at 224 vs 384 px across seeds 42, 53, 64.

Determines whether higher input resolution improves classification of
calcifications and micro-abnormalities on CBIS-DDSM beyond the 95% CI.
"""

import argparse
import copy
import sys
from pathlib import Path

# Ensure root in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import yaml

from scripts.run_local import run_one_site
from src.ledger import compute_seed_statistics


def main():
    parser = argparse.ArgumentParser(description="CBIS Resolution Test: 224 px vs 384 px")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 53, 64])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[224, 384])
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("PHASE 1 RESOLUTION TEST: CBIS-DDSM (224 px vs 384 px)")
    print(f"Seeds: {args.seeds} | Device: {device}")
    print("=" * 75)

    results_by_res = {r: [] for r in args.resolutions}

    for res in args.resolutions:
        print(f"\n>>> Running Resolution: {res}×{res} <<<")
        res_cfg = copy.deepcopy(base_cfg)
        res_cfg["img_size"] = res
        res_cfg["sites"] = ["CBIS"]
        res_cfg["force"] = args.force

        for seed in args.seeds:
            res_root = Path("results/resolution_test") / f"res_{res}"
            site_results = run_one_site(
                site="CBIS",
                cfg=res_cfg,
                seed=seed,
                device=device,
                results_root=str(res_root),
            )
            auc_val = site_results["CBIS"]["image_level"]["auc"]
            ci = site_results["CBIS"]["bootstrap_auc_ci"]
            results_by_res[res].append({
                "resolution": res,
                "seed": seed,
                "auc": auc_val,
                "ci_lower": ci["lower"],
                "ci_upper": ci["upper"],
            })

    print("\n" + "=" * 75)
    print("RESOLUTION TEST COMPARISON SUMMARY")
    print("=" * 75)

    rows = []
    for res in args.resolutions:
        aucs = [r["auc"] for r in results_by_res[res]]
        st = compute_seed_statistics(aucs)
        rows.append({
            "Resolution": f"{res}×{res}",
            "Seeds": args.seeds,
            "Mean AUC": round(st["mean"], 4),
            "Std": round(st["std"], 4),
            "95% CI": f"[{st['ci_low']:.4f}, {st['ci_high']:.4f}]",
        })

    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))

    # Evaluate decision rule
    if len(args.resolutions) >= 2:
        res1, res2 = args.resolutions[0], args.resolutions[1]
        m1 = df_summary.loc[df_summary["Resolution"] == f"{res1}×{res1}", "Mean AUC"].values[0]
        m2 = df_summary.loc[df_summary["Resolution"] == f"{res2}×{res2}", "Mean AUC"].values[0]

        diff = m2 - m1
        print(f"\nDelta AUC ({res2}px vs {res1}px): {diff:+.4f}")
        if diff > 0.03:
            print(f"-> RESOLUTION DECISION: Meaningful gain at {res2}px. Adopt {res2}px.")
        else:
            print(f"-> RESOLUTION DECISION: Gain is marginal or within CI. Retain standard {res1}px for efficiency.")
    print("=" * 75)


if __name__ == "__main__":
    main()
