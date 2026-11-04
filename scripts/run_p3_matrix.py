"""Orchestrator script for Phase P3 FedProx sweeps and Experiment E6 Heterogeneity matrix.

Adheres strictly to AGENTS.md Phase P3:
  - Strategies: FedAvg (sample-weighted, uniform), FedProx (mu=0.01, 0.1, n=1 seed).
  - Experiment E6 (heterogeneity): all 4 sites vs. birads-only vs. biopsy-only
    plus leave-one-site-out (n=1 seed: 42).
  - Evaluates test performance on all 4 sites to capture in-domain and cross-silo generalization.
  - Skips runs already recorded as completed in results/ledger.csv.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Run matrix definitions
RUN_SPECS = [
    # 1. FedProx sweeps (mu=0.01, mu=0.1 on uniform aggregation)
    {
        "experiment_id": "fl_fedprox_mu0.01",
        "strategy": "fedprox",
        "mode": "uniform",
        "mu": 0.01,
        "sites": ["vindr", "cmmd", "inbreast", "cdd"],
        "seeds": [42],
    },
    {
        "experiment_id": "fl_fedprox_mu0.1",
        "strategy": "fedprox",
        "mode": "uniform",
        "mu": 0.1,
        "sites": ["vindr", "cmmd", "inbreast", "cdd"],
        "seeds": [42],
    },
    # 2. Experiment E6: Label-family isolations
    {
        "experiment_id": "fl_e6_birads_only",
        "strategy": "fedavg",
        "mode": "uniform",
        "mu": 0.0,
        "sites": ["vindr", "inbreast"],
        "seeds": [42],
    },
    {
        "experiment_id": "fl_e6_biopsy_only",
        "strategy": "fedavg",
        "mode": "uniform",
        "mu": 0.0,
        "sites": ["cmmd", "cdd"],
        "seeds": [42],
    },
    # 3. Experiment E6: Leave-one-site-out isolations
    {
        "experiment_id": "fl_e6_leave_out_vindr",
        "strategy": "fedavg",
        "mode": "uniform",
        "mu": 0.0,
        "sites": ["cmmd", "inbreast", "cdd"],
        "seeds": [42],
    },
    {
        "experiment_id": "fl_e6_leave_out_cmmd",
        "strategy": "fedavg",
        "mode": "uniform",
        "mu": 0.0,
        "sites": ["vindr", "inbreast", "cdd"],
        "seeds": [42],
    },
    {
        "experiment_id": "fl_e6_leave_out_inbreast",
        "strategy": "fedavg",
        "mode": "uniform",
        "mu": 0.0,
        "sites": ["vindr", "cmmd", "cdd"],
        "seeds": [42],
    },
    {
        "experiment_id": "fl_e6_leave_out_cdd",
        "strategy": "fedavg",
        "mode": "uniform",
        "mu": 0.0,
        "sites": ["vindr", "cmmd", "inbreast"],
        "seeds": [42],
    },
]


def is_already_completed(experiment_id: str, seed: int) -> bool:
    ledger_path = Path("results/ledger.csv")
    if not ledger_path.exists():
        return False
    try:
        ldf = pd.read_csv(ledger_path)
        match = ldf[(ldf["experiment_id"] == experiment_id) & (ldf["seed"] == seed) & (ldf["status"] == "completed")]
        return len(match) > 0
    except Exception as e:
        logger.warning("Error reading ledger: %s", e)
        return False


def main():
    python_bin = sys.executable

    for spec in RUN_SPECS:
        exp_id = spec["experiment_id"]
        strategy = spec["strategy"]
        mode = spec["mode"]
        mu = spec["mu"]
        sites = spec["sites"]
        seeds = spec["seeds"]

        for seed in seeds:
            if is_already_completed(exp_id, seed):
                logger.info(">>> Skipping already completed run: Exp=%s | Seed=%d", exp_id, seed)
                continue

            logger.info("\n" + "=" * 70)
            logger.info(">>> Launching P3 Run: Exp=%s | Strategy=%s | Mode=%s | Mu=%.2f | Seed=%d | Sites=%s",
                        exp_id, strategy, mode, mu, seed, sites)
            logger.info("=" * 70)

            cmd = [
                python_bin,
                "scripts/run_fl_simulation.py",
                "--experiment-id", exp_id,
                "--strategy", strategy,
                "--mode", mode,
                "--mu", str(mu),
                "--rounds", "20",
                "--seeds", str(seed),
                "--lr", "1e-4",
                "--batch-size", "8",
                "--sites", *sites,
            ]

            res = subprocess.run(cmd)
            if res.returncode != 0:
                logger.error("Run failed with exit code %d: %s", res.returncode, exp_id)
                sys.exit(res.returncode)

    logger.info("\nAll Phase P3 FedProx and Experiment E6 runs completed successfully!")


if __name__ == "__main__":
    main()
