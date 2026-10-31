#!/usr/bin/env python3
"""Make a montage to eyeball preprocessing: top row = raw512, bottom row = proc512.

    python scripts/preview.py --site cmmd --n 8
Opens nothing; writes data/<site>/preview.png. Check that crops contain the whole breast,
orientation is consistent (chest wall on the left) and labels/markers are not dominating.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = Path(a.out_root) / a.site
    csv = d / "labels.csv" if (d / "labels.csv").exists() else d / "meta.csv"
    df = pd.read_csv(csv, dtype=str)
    rows = df.sample(min(a.n, len(df)), random_state=a.seed)
    cols = []
    for sop in rows["sop_uid"]:
        raw = cv2.imread(str(d / "raw512" / f"{sop}.png"), 0)
        proc = cv2.imread(str(d / "proc512" / f"{sop}.png"), 0)
        cols.append(np.vstack([cv2.resize(raw, (256, 256)), cv2.resize(proc, (256, 256))]))
    cv2.imwrite(str(d / "preview.png"), np.hstack(cols))
    print(f"wrote {d/'preview.png'}")


if __name__ == "__main__":
    main()
