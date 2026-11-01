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

    # Evaluate decision rule under Rule 5 (paired difference 95% CI)
    if len(args.resolutions) >= 2:
        res1, res2 = args.resolutions[0], args.resolutions[1]
        aucs1 = [r["auc"] for r in results_by_res[res1]]
        aucs2 = [r["auc"] for r in results_by_res[res2]]

        paired_diffs = [a2 - a1 for a1, a2 in zip(aucs1, aucs2)]
        diff_stats = compute_seed_statistics(paired_diffs)
        mean_diff = diff_stats["mean"]
        ci_low = diff_stats["ci_low"]
        ci_high = diff_stats["ci_high"]

        print(f"\nPaired Delta AUC ({res2}px - {res1}px) over seeds {args.seeds}:")
        print(f"  Per-seed deltas: {[round(d, 4) for d in paired_diffs]}")
        print(f"  Mean paired diff: {mean_diff:+.4f} | 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")

        if ci_low > 0.0 and mean_diff > 0.03:
            print(f"-> DECISION: Significant gain outside 95% CI (excludes zero). Adopt {res2}px.")
        else:
            print(f"-> DECISION (Rule 5 & Phase 1.2): 95% CI of paired difference includes zero [{ci_low:+.4f}, {ci_high:+.4f}].")
            print(f"   No statistically significant improvement. Retain standard {res1}px for computational efficiency.")
            print("   Document this negative result in the paper as planned.")
    print("=" * 75)


if __name__ == "__main__":
    main()
