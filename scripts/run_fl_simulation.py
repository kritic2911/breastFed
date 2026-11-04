#!/usr/bin/env python3
"""Run Federated Learning (FL) cross-silo simulations.

Adheres strictly to AGENTS.md Phase P3, Section 2.3, 3.1, 3.2, 3.3:
  - Strategies: FedAvg (sample-weighted, uniform), FedProx (mu in {0.01, 0.1}), FedBN.
  - Constant LR (Decision D-008: backbone_lr=1e-4, head_lr=1e-3).
  - 4 hospital sites: vindr, cmmd, inbreast, cdd.
  - Experiment E6 (Heterogeneity): FL on all 4 sites vs birads-only vs biopsy-only vs leave-one-site-out.
  - Thermal guard active between rounds and clients.
  - Checkpoint integrity: atomic last.pt, best.pt, resumable mid-round.
  - Ledger tracking with byte tracking and wall/paused time.
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
import torch

from src.data.dataset import get_train_val_test_splits
from src.fl.client import FLClient
from src.fl.server import FLServer
from src.utils.provenance import generate_run_id
from src.utils.thermal import ThermalGuard

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run cross-silo FL simulation.")
    parser.add_argument("--strategy", type=str, choices=["fedavg", "fedprox", "fedbn", "per_family_heads"], default="fedavg", help="FL strategy.")
    parser.add_argument("--mode", type=str, choices=["sample_weighted", "uniform"], default="sample_weighted", help="FedAvg aggregation mode.")
    parser.add_argument("--mu", type=float, default=0.0, help="FedProx proximal regularizer parameter mu.")
    parser.add_argument("--rounds", type=int, default=20, help="Total communication rounds.")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round per client.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Constant backbone learning rate (Decision D-008).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 53, 64], help="Random seeds.")
    parser.add_argument("--sites", nargs="+", default=["vindr", "cmmd", "inbreast", "cdd"], help="Participating sites.")
    parser.add_argument("--experiment-id", type=str, default="", help="Custom experiment ID (auto-generated if empty).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per client.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root dir containing manifest.csv and cache512/.")
    parser.add_argument("--resume", type=str, default="auto", help="Resume mode (auto / none).")
    parser.add_argument("--force", action="store_true", help="Force rerun even if already in ledger.")
    args = parser.parse_args()

    # Determine experiment ID
    if not args.experiment_id:
        if args.strategy == "fedprox":
            args.experiment_id = f"fl_fedprox_mu{args.mu}"
        elif args.strategy == "fedbn":
            args.experiment_id = "fl_fedbn"
        elif args.strategy == "per_family_heads":
            args.experiment_id = f"fl_per_family_{args.mode}"
        else:
            args.experiment_id = f"fl_fedavg_{args.mode}"

        # If site subset (e.g. birads-only, biopsy-only, leave-one-out)
        all_sites = ["vindr", "cmmd", "inbreast", "cdd"]
        if set(args.sites) != set(all_sites):
            site_tag = "_".join(sorted(args.sites))
            args.experiment_id += f"_{site_tag}"

    manifest_csv = args.data_dir / "manifest.csv"
    if not manifest_csv.exists():
        logger.error("Manifest not found at %s", manifest_csv)
        sys.exit(1)

    mdf = pd.read_csv(manifest_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    thermal_guard = ThermalGuard(soft_c=70.0, hard_c=75.0, resume_c=55.0, check_every_iters=20)

    all_seed_results = []

    for seed in args.seeds:
        logger.info("\n" + "=" * 70)
        logger.info("=== FL Simulation: Exp=%s | Strategy=%s | Mode=%s | Seed=%d | Sites=%s ===",
                    args.experiment_id, args.strategy, args.mode, seed, args.sites)
        logger.info("=" * 70)

        # Check ledger for completed run
        ledger_path = Path("results/ledger.csv")
        cached_result = None
        if ledger_path.exists() and not args.force:
            try:
                ldf = pd.read_csv(ledger_path)
                match = ldf[
                    (ldf["experiment_id"] == args.experiment_id)
                    & (ldf["seed"] == seed)
                    & (ldf["status"] == "completed")
                ]
                if not match.empty:
                    run_id = match.iloc[-1]["run_id"]
                    run_dir = Path("results/runs") / run_id
                    metrics_p = run_dir / "preds" / "test_metrics.json"
                    if metrics_p.exists():
                        logger.info("Found completed run %s for Exp=%s, Seed=%d (reusing)", run_id, args.experiment_id, seed)
                        with open(metrics_p) as fp:
                            m_data = json.load(fp)
                        cached_result = m_data
            except Exception as e:
                logger.debug("Ledger lookup check failed: %s", e)

        if cached_result is not None:
            all_seed_results.append(cached_result)
            continue

        # Get splits
        train_df, val_df, test_df = get_train_val_test_splits(mdf, seed=seed, val_size=0.20)

        # Build run directory
        run_id = generate_run_id({"exp": args.experiment_id, "seed": seed, "lr": args.lr, "rounds": args.rounds})
        run_dir = Path("results/runs") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build clients
        clients = []
        for site_name in args.sites:
            site_train = train_df[train_df["site"] == site_name].copy()
            logger.info("  Site [%s] training set: %d images", site_name, len(site_train))
            client = FLClient(
                site=site_name,
                train_df=site_train,
                data_dir=args.data_dir,
                device=device,
                backbone_lr=args.lr,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                thermal_guard=thermal_guard,
                per_family_heads=(args.strategy == "per_family_heads"),
            )
            clients.append(client)

        # Prepare participating validation split (honest model selection without peeking at held-out sites)
        participating_val_df = val_df[val_df["site"].isin(args.sites)].copy()

        # Prepare test splits for evaluation across all sites (measuring in-domain and out-of-domain transfer)
        all_sites = ["vindr", "cmmd", "inbreast", "cdd"]
        test_dfs = {}
        for site_name in all_sites:
            site_test = test_df[test_df["site"] == site_name].copy()
            if not site_test.empty:
                test_dfs[site_name] = site_test

        # Instantiate FL Server
        server = FLServer(
            clients=clients,
            val_df=participating_val_df,
            test_dfs=test_dfs,
            strategy=args.strategy,
            aggregation_mode=args.mode,
            mu=args.mu,
            num_rounds=args.rounds,
            run_dir=run_dir,
            device=device,
            data_dir=args.data_dir,
            seed=seed,
            experiment_id=args.experiment_id,
            thermal_guard=thermal_guard,
        )

        res = server.run(resume=True)
        all_seed_results.append(res)
        logger.info("Completed FL Simulation Seed=%d: Overall Macro-AUC = %.4f", seed, res.get("overall_macro_auc", 0.0))

    # Save summary
    summary_path = Path("paper/tables") / f"{args.experiment_id}_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fp:
        json.dump(all_seed_results, fp, indent=2, default=str)
    logger.info("Saved experiment summary to %s", summary_path)


if __name__ == "__main__":
    main()
