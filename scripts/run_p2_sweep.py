#!/usr/bin/env python3
"""Phase P2 Pre-Registered Mini-Sweep on CMMD.

Adheres strictly to AGENTS.md Phase P2 Step 2:
  - Sweep grid: BN {frozen, trainable} x backbone LR {1e-4, 3e-4}
  - Site: CMMD (biopsy family, classes {1, 2})
  - 1 seed (seed 42), evaluated strictly on validation set
  - Locks winning configuration for all subsequent baseline and FL runs
  - Records winner in docs/DECISIONS.md
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

_repo_dir = Path(__file__).resolve().parent.parent
_venv_dir = _repo_dir / ".venv"
_venv_py = _venv_dir / "bin" / "python3"
if _venv_py.exists() and sys.prefix != str(_venv_dir):
    try:
        import pandas  # noqa: F401
    except ImportError:
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

if str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
Path("/tmp/matplotlib_cache").mkdir(parents=True, exist_ok=True)

import pandas as pd

from src.data.dataset import get_train_val_test_splits
from src.train.local import train_local
from src.utils.thermal import ThermalGuard

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_cmmd_sweep():
    manifest_csv = Path("data/manifest.csv")
    if not manifest_csv.exists():
        logger.error("data/manifest.csv not found!")
        sys.exit(1)

    mdf = pd.read_csv(manifest_csv)
    cmmd_df = mdf[mdf["site"] == "cmmd"].copy()
    logger.info("Loaded CMMD cohort: %d images across %d patients", len(cmmd_df), cmmd_df["patient_hash"].nunique())

    # Get stratified patient-grouped train/val/test splits for seed 42
    train_df, val_df, test_df = get_train_val_test_splits(cmmd_df, seed=42, val_size=0.20)
    logger.info("CMMD splits: Train=%d, Val=%d, Test=%d (held out)", len(train_df), len(val_df), len(test_df))

    # Pre-registered 2x2 grid
    grid = [
        {"bn_mode": "frozen", "backbone_lr": 1e-4},
        {"bn_mode": "frozen", "backbone_lr": 3e-4},
        {"bn_mode": "trainable", "backbone_lr": 1e-4},
        {"bn_mode": "trainable", "backbone_lr": 3e-4},
    ]

    thermal_guard = ThermalGuard(soft_c=70.0, hard_c=75.0, resume_c=55.0, check_every_iters=20)
    results = []

    for idx, params in enumerate(grid, 1):
        bn_mode = params["bn_mode"]
        lr = params["backbone_lr"]
        logger.info("\n" + "=" * 60)
        logger.info("=== Sweep Run %d/4: BN=%s | LR=%.1e ===", idx, bn_mode, lr)
        logger.info("=" * 60)

        # Check if completed run on the corrected manifest already exists in ledger.csv
        cached_result = None
        ledger_path = Path("results/ledger.csv")
        completed_run_ids = set()
        if ledger_path.exists():
            try:
                ldf = pd.read_csv(ledger_path)
                completed_run_ids = set(ldf[ldf["status"] == "completed"]["run_id"])
            except Exception:
                pass

        for r_dir in sorted(Path("results/runs").glob("*")):
            if r_dir.name < "20261102_192000" or r_dir.name not in completed_run_ids:
                continue
            cfg_p = r_dir / "config.yaml"
            ckpt_p = r_dir / "ckpt" / "best.pt"
            metrics_p = r_dir / "metrics.jsonl"
            if cfg_p.exists() and ckpt_p.exists() and metrics_p.exists():
                try:
                    import yaml
                    with open(cfg_p) as fp:
                        c_data = yaml.safe_load(fp)
                    if (
                        c_data.get("experiment_id") == "p2_sweep_cmmd"
                        and c_data.get("bn_mode") == bn_mode
                        and abs(float(c_data.get("backbone_lr", 0)) - lr) < 1e-9
                        and c_data.get("seed") == 42
                    ):
                        with open(metrics_p) as fp:
                            lines = [json.loads(line) for line in fp]
                        if lines:
                            best = max(lines, key=lambda x: x.get("val_macro_auc", 0))
                            cached_result = {
                                "run_id": r_dir.name,
                                "bn_mode": bn_mode,
                                "backbone_lr": lr,
                                "best_epoch": best["epoch"],
                                "val_macro_auc": best["val_macro_auc"],
                                "val_macro_f1": best.get("val_macro_f1", 0.0),
                                "duration_s": sum(item.get("epoch_duration_s", 0) for item in lines),
                            }
                            logger.info(
                                "Found completed run %s: Best Epoch=%d, Val Macro-AUC=%.4f (reusing)",
                                r_dir.name, best["epoch"], best["val_macro_auc"]
                            )
                            break
                except Exception:
                    pass

        if cached_result is not None:
            results.append(cached_result)
            continue

        config = {
            "experiment_id": "p2_sweep_cmmd",
            "site": "cmmd",
            "seed": 42,
            "bn_mode": bn_mode,
            "backbone_lr": lr,
            "head_lr_mult": 10.0,
            "batch_size": 16,
            "max_epochs": 25,     # Sufficient for fast convergence with early stopping
            "patience": 6,
            "weight_decay": 1e-2,
            "lr_schedule": "cosine",
            "data_dir": "data",
        }

        t_start = time.time()
        # Train and evaluate strictly on validation (test_df omitted for sweep per AGENTS.md)
        res = train_local(
            config=config,
            train_df=train_df,
            val_df=val_df,
            test_df=None,
            thermal_guard=thermal_guard,
        )
        elapsed = time.time() - t_start

        val_auc = res["val_metrics"].get("macro_auc", 0.5)
        val_f1 = res["val_metrics"].get("macro_f1", 0.0)
        results.append({
            "run_id": res["run_id"],
            "bn_mode": bn_mode,
            "backbone_lr": lr,
            "best_epoch": res["best_epoch"],
            "val_macro_auc": val_auc,
            "val_macro_f1": val_f1,
            "duration_s": round(elapsed, 1),
        })

    # Summary table
    res_df = pd.DataFrame(results)
    logger.info("\n" + "=" * 60)
    logger.info("=== P2 CMMD Mini-Sweep Results ===")
    logger.info("=" * 60)
    logger.info("\n%s", res_df.to_string(index=False))

    # Determine winner based on validation macro-AUC
    winner = res_df.sort_values(by="val_macro_auc", ascending=False).iloc[0]
    logger.info(
        "\nWINNER: BN=%s, Backbone LR=%.1e | Val Macro-AUC=%.4f (Run: %s)",
        winner["bn_mode"], winner["backbone_lr"], winner["val_macro_auc"], winner["run_id"]
    )

    out_json = Path("paper/tables/p2_cmmd_sweep_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fp:
        json.dump({
            "sweep_grid": grid,
            "results": results,
            "winner": winner.to_dict(),
        }, fp, indent=2)
    logger.info("Saved sweep results to %s", out_json)


if __name__ == "__main__":
    run_cmmd_sweep()

