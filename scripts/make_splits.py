#!/usr/bin/env python3
"""Patient-level, label-stratified train/val/test split (~64/16/20) for one site.

    python scripts/make_splits.py --site cmmd --seed 0

Reads data/<site>/labels.csv, writes data/<site>/labels_split.csv with a 'split' column.
No patient ever appears in more than one split.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = Path(a.out_root) / a.site
    df = pd.read_csv(d / "labels.csv", dtype={"patient_id": str})
    y, g = df["label"].to_numpy(), df["patient_id"].to_numpy()
    idx = np.arange(len(df))

    rest, test = next(StratifiedGroupKFold(5, shuffle=True, random_state=a.seed).split(idx, y, g))
    tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=a.seed + 1).split(rest, y[rest], g[rest]))
    train, val = rest[tr], rest[va]

    df["split"] = ""
    df.loc[train, "split"] = "train"
    df.loc[val, "split"] = "val"
    df.loc[test, "split"] = "test"

    pats = {s: set(df.loc[df.split == s, "patient_id"]) for s in ("train", "val", "test")}
    assert not (pats["train"] & pats["val"] or pats["train"] & pats["test"] or pats["val"] & pats["test"]), \
        "patient leakage between splits"

    df.to_csv(d / "labels_split.csv", index=False)
    summary = df.groupby("split").agg(images=("label", "size"), patients=("patient_id", "nunique"),
                                      positive_rate=("label", "mean")).round(3)
    print(summary)
    if df["patient_id"].nunique() == len(df):
        print("WARNING: every image has a unique patient_id, so this is an image-level split (leakage risk).")
    print(f"wrote {d/'labels_split.csv'}")


if __name__ == "__main__":
    main()
