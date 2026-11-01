"""Experiment ledger and multi-seed aggregation for FL-Mammo.

Maintains an append-only ledger at results/ledger.csv with schema:
run_id, phase, hypothesis_id, method, variant, site, seed, metric, value, ci_low, ci_high, git_sha, split_hash, status, notes

Provides aggregation functions to compute:
- Mean and 95% confidence intervals across independent seeds (42, 53, 64)
- Summary tables (CSV / Markdown) for paper and milestone checkpoints
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LEDGER_COLUMNS = [
    "run_id",
    "phase",
    "hypothesis_id",
    "method",
    "variant",
    "site",
    "seed",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "git_sha",
    "split_hash",
    "status",
    "notes",
]


def append_to_ledger(
    run_id: str,
    method: str,
    site: str,
    seed: int,
    metric: str,
    value: float,
    phase: str = "phase_1",
    hypothesis_id: str = "H_baseline",
    variant: str = "standard",
    ci_low: float | None = None,
    ci_high: float | None = None,
    git_sha: str = "unknown",
    split_hash: str = "patient_grouped",
    status: str = "completed",
    notes: str = "",
    ledger_path: str | Path = "results/ledger.csv",
):
    """Append a single verified metric entry to the append-only ledger."""
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "run_id": run_id,
        "phase": phase,
        "hypothesis_id": hypothesis_id,
        "method": method,
        "variant": variant,
        "site": site,
        "seed": seed,
        "metric": metric,
        "value": float(value) if value is not None else np.nan,
        "ci_low": float(ci_low) if ci_low is not None else np.nan,
        "ci_high": float(ci_high) if ci_high is not None else np.nan,
        "git_sha": git_sha,
        "split_hash": split_hash,
        "status": status,
        "notes": notes,
    }

    df_row = pd.DataFrame([row], columns=LEDGER_COLUMNS)
    if not p.exists():
        df_row.to_csv(p, index=False, columns=LEDGER_COLUMNS)
    else:
        df_row.to_csv(p, mode="a", header=False, index=False, columns=LEDGER_COLUMNS)


def compute_seed_statistics(values: list[float] | np.ndarray) -> dict:
    """Compute mean and 95% confidence interval over seeds.

    Uses Student's t distribution for small sample sizes (e.g. n=3).
    """
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    mean = float(np.mean(arr))
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean, "n": 1}

    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(n)
    # 95% CI with degrees of freedom df = n - 1
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean - t_crit * sem
    ci_high = mean + t_crit * sem
    return {
        "mean": mean,
        "std": std,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n": n,
    }


def aggregate_ledger(
    ledger_path: str | Path = "results/ledger.csv",
    phase: str = None,
    metric: str = "auc",
) -> pd.DataFrame:
    """Read ledger and compute mean ± 95% CI across seeds per (method, variant, site)."""
    p = Path(ledger_path)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)
    if phase:
        df = df[df["phase"] == phase]
    if metric:
        df = df[df["metric"] == metric]

    records = []
    groups = df.groupby(["method", "variant", "site"])
    for (mth, var, site), grp in groups:
        vals = grp["value"].values
        stats_dict = compute_seed_statistics(vals)
        records.append({
            "method": mth,
            "variant": var,
            "site": site,
            "metric": metric,
            "seeds": sorted(grp["seed"].unique().tolist()),
            "mean": stats_dict["mean"],
            "std": stats_dict["std"],
            "ci_low": stats_dict["ci_low"],
            "ci_high": stats_dict["ci_high"],
            "display": f"{stats_dict['mean']:.4f} ± {stats_dict['std']:.4f} [{stats_dict['ci_low']:.4f}, {stats_dict['ci_high']:.4f}]" if stats_dict["n"] > 1 else f"{stats_dict['mean']:.4f}",
            "n_seeds": stats_dict["n"],
        })

    return pd.DataFrame(records)

