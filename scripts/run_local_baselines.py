#!/usr/bin/env python3
"""Run Local Baselines for each site across 3 seeds.

Adheres strictly to AGENTS.md Phase P2 Step 3 & Section 1.3:
  - Sites: vindr, cmmd, inbreast, cdd
  - Seeds: [42, 53, 64] (different train/val partition and initialization)
  - Evaluated on fixed held-out test set once after training completes
  - Logs metrics, 1,000 patient-resampled test bootstrap CIs, and appends to ledger.csv
"""

import argparse
import json
import logging
import os
import sys
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
from src.utils.config import create_thermal_guard, get_local_training_config, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run local baselines per site across 3 seeds.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to master config file.")
    parser.add_argument("--sites", nargs="+", default=None, help="Sites to train on (defaults to config).")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Random seeds (defaults to config).")
    parser.add_argument("--bn_mode", type=str, default=None, help="BatchNorm mode (defaults to config/sweep).")
    parser.add_argument("--lr", type=float, default=None, help="Backbone learning rate (defaults to config/sweep).")
    parser.add_argument("--epochs", type=int, default=None, help="Maximum epochs (defaults to config).")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (defaults to config).")
    parser.add_argument("--patience", type=int, default=None, help="Validation early stopping patience (defaults to config).")
    parser.add_argument("--force", action="store_true", help="Force rerun even if completed.")
    args = parser.parse_args()

    # Load master configuration
    master_cfg = load_config(args.config)
    local_cfg = master_cfg.get("local_baseline", {})
    model_cfg = master_cfg.get("model", {})
    data_cfg = master_cfg.get("data", {})

    sites = args.sites or local_cfg.get("sites", ["vindr", "cmmd", "inbreast", "cdd"])
    seeds = args.seeds or local_cfg.get("seeds", [42, 53, 64])
    bn_mode = args.bn_mode or model_cfg.get("bn_mode", "trainable")
    lr = args.lr or float(local_cfg.get("backbone_lr", 3e-4))
    max_epochs = args.epochs or int(local_cfg.get("max_epochs", 40))
    batch_size = args.batch_size or int(local_cfg.get("batch_size", 16))
    patience = args.patience or int(local_cfg.get("patience", 12))

    # If sweep results exist and user did not explicitly override, load locked winner parameters
    sweep_json = Path("paper/tables/p2_cmmd_sweep_results.json")
    if sweep_json.exists() and args.bn_mode is None and args.lr is None:
        with open(sweep_json) as fp:
            sweep_data = json.load(fp)
            winner = sweep_data.get("winner", {})
            if "bn_mode" in winner:
                bn_mode = winner["bn_mode"]
                lr = float(winner["backbone_lr"])
                logger.info("Loaded locked sweep winner hyperparameters: BN=%s, LR=%.1e", bn_mode, lr)

    manifest_csv = Path(data_cfg.get("manifest_csv", "data/manifest.csv"))
    if not manifest_csv.exists():
        logger.error("%s not found!", manifest_csv)
        sys.exit(1)

    mdf = pd.read_csv(manifest_csv)
    thermal_guard = create_thermal_guard(master_cfg)

    all_results = []

    for site in sites:
        site_df = mdf[mdf["site"] == site].copy()
        if site_df.empty:
            logger.warning("No records for site %s, skipping", site)
            continue

        for seed in seeds:
            logger.info("\n" + "=" * 65)
            logger.info("=== Running Local Baseline: Site=%s | Seed=%d | BN=%s | LR=%.1e | Epochs=%d | BS=%d ===", site, seed, bn_mode, lr, max_epochs, batch_size)
            logger.info("=" * 65)

            # Check if already completed
            ledger_path = Path("results/ledger.csv")
            cached_result = None
            supersede_run_id = None
            if ledger_path.exists():
                try:
                    ldf = pd.read_csv(ledger_path)
                    match = ldf[
                        (ldf["experiment_id"] == f"local_baseline_{site}")
                        & (ldf["seed"] == seed)
                        & (ldf["status"] == "completed")
                    ]
                    if not match.empty:
                        run_id = match.iloc[-1]["run_id"]
                        run_dir = Path("results/runs") / run_id
                        preds_p = run_dir / "preds" / "test_breast_predictions.npz"
                        metrics_p = run_dir / "metrics.jsonl"
                        cfg_p = run_dir / "config.yaml"

                        import yaml
                        run_patience = None
                        if cfg_p.exists():
                            with open(cfg_p) as f:
                                c_data = yaml.safe_load(f) or {}
                                run_patience = c_data.get("patience")

                        if not args.force and preds_p.exists() and metrics_p.exists() and run_patience == patience:
                            logger.info(
                                "Found completed run %s for Site=%s, Seed=%d with patience=%d (reusing)",
                                run_id, site, seed, patience
                            )
                            with open(metrics_p) as fp:
                                m_lines = [json.loads(line) for line in fp]
                            val_best = max([item.get("val_macro_auc", 0.0) for item in m_lines]) if m_lines else 0.0
                            cached_result = {
                                "run_id": run_id,
                                "site": site,
                                "seed": seed,
                                "val_macro_auc": val_best,
                                "test_macro_auc": float(match.iloc[-1]["metric_value"]),
                            }
                        else:
                            supersede_run_id = run_id
                            logger.info(
                                "Previous run %s had patience=%s (current requested=%d, force=%s). Rerunning to supersede %s.",
                                run_id, run_patience, patience, args.force, supersede_run_id
                            )
                except Exception:
                    pass

            if cached_result is not None:
                all_results.append(cached_result)
                continue

            train_df, val_df, test_df = get_train_val_test_splits(site_df, seed=seed, val_size=0.20)
            logger.info("[%s - Seed %d] Splits: Train=%d, Val=%d, Test=%d", site, seed, len(train_df), len(val_df), len(test_df))

            config = get_local_training_config(
                master_cfg,
                site=site,
                seed=seed,
                supersedes=supersede_run_id,
                bn_mode=bn_mode,
                backbone_lr=lr,
                max_epochs=max_epochs,
                batch_size=batch_size,
                patience=patience,
            )

            res = train_local(
                config=config,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                thermal_guard=thermal_guard,
            )

            test_auc = res["test_metrics"].get("macro_auc", 0.0)
            ci = res["test_ci"].get("macro_auc_ci", (0.0, 0.0))
            logger.info(
                "[%s - Seed %d] Complete: Val Macro-AUC=%.4f | Test Macro-AUC=%.4f (95%% CI: [%.4f, %.4f])",
                site, seed, res["val_metrics"].get("macro_auc", 0.0), test_auc, ci[0], ci[1]
            )
            all_results.append({
                "run_id": res["run_id"],
                "site": site,
                "seed": seed,
                "val_macro_auc": res["val_metrics"].get("macro_auc", 0.0),
                "test_macro_auc": test_auc,
                "test_ci": ci,
                "test_metrics": res["test_metrics"],
            })

    out_json = Path("paper/tables/p2_local_baselines.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fp:
        json.dump(all_results, fp, indent=2)
    logger.info("Saved local baselines summary to %s", out_json)


if __name__ == "__main__":
    main()

