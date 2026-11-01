#!/usr/bin/env python3
"""CLI utility to aggregate multi-seed experiment ledger into summary tables."""

import argparse
import sys
from pathlib import Path

# Ensure root in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ledger import aggregate_ledger


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experiment ledger.")
    parser.add_argument("--ledger", default="results/ledger.csv", help="Path to ledger.csv")
    parser.add_argument("--phase", default=None, help="Filter by phase (e.g. phase_1)")
    parser.add_argument("--metric", default="auc", help="Metric to aggregate (default: auc)")
    parser.add_argument("--out-csv", default="results/aggregated_summary.csv", help="Output summary CSV")
    args = parser.parse_args()

    df_agg = aggregate_ledger(ledger_path=args.ledger, phase=args.phase, metric=args.metric)
    if df_agg.empty:
        print(f"No entries found in ledger at {args.ledger}")
        return

    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(out_p, index=False)
    print(f"\n[Ledger Aggregation] Processed {len(df_agg)} method-site groups -> {out_p}")

    print("\n" + "=" * 80)
    print(f"EXPERIMENT LEDGER SUMMARY TABLE (Metric: {args.metric.upper()})")
    print("=" * 80)
    display_cols = ["method", "variant", "site", "n_seeds", "display"]
    print(df_agg[display_cols].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

