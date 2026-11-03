"""Provenance, structured logging, environment capture, and experiment ledger.

Adheres strictly to AGENTS.md Section 3.3 & Rules 5-6:
  - run_id: YYYYMMDD_HHMMSS-<confighash8>-<gitsha7>
  - Strict run directory layout: config.yaml, env.json, metrics.jsonl, events.jsonl, preds/, ckpt/
  - Append-only results/ledger.csv tracking wall time, paused time, and metrics with n.
  - Salted SHA-256 patient ID hashing.
"""

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)

# Fixed salt for patient de-identification hashing
SALT = b"btp_fl_mammo_2026_salt"


def hash_patient_id(raw_id: str, salt: bytes = SALT) -> str:
    """Compute deterministic salted SHA-256 hash for patient de-identification."""
    h = hashlib.sha256(salt + str(raw_id).encode("utf-8"))
    return h.hexdigest()[:16]


def get_git_info() -> dict[str, Any]:
    """Capture git commit SHA and dirty working tree status."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown_commit"

    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        dirty = len(status) > 0
    except Exception:
        dirty = False

    return {"commit": commit, "short_sha": commit[:7], "dirty": dirty}


def compute_config_hash(cfg: dict[str, Any]) -> str:
    """Compute deterministic 8-char SHA-256 hash of configuration dictionary."""
    serialized = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]


def generate_run_id(cfg: dict[str, Any]) -> str:
    """Format canonical run ID: YYYYMMDD_HHMMSS-<confighash8>-<gitsha7>."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_hash = compute_config_hash(cfg)
    git_sha = get_git_info()["short_sha"]
    return f"{timestamp}-{cfg_hash}-{git_sha}"


def capture_environment_info(thermal_telemetry: bool = True) -> dict[str, Any]:
    """Capture full hardware, software, and dependency environment info."""
    git_info = get_git_info()

    # Capture pip freeze
    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).strip().split("\n")
    except Exception:
        pip_freeze = []

    # GPU info
    gpu_name = "CPU"
    cuda_avail = torch.cuda.is_available()
    cuda_ver = torch.version.cuda if cuda_avail else None
    if cuda_avail:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA_GPU"

    return {
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver,
        "gpu_name": gpu_name,
        "thermal_telemetry": thermal_telemetry,
        "pip_freeze": pip_freeze,
    }


class RunLogger:
    """Structured Run Logger managing per-run directories, events, metrics, and ledger."""

    def __init__(
        self,
        cfg: dict[str, Any],
        results_root: Path = Path("results"),
        run_id: str | None = None,
        thermal_telemetry: bool = True,
    ):
        self.cfg = cfg
        self.results_root = Path(results_root)
        self.run_id = run_id or generate_run_id(cfg)
        self.run_dir = self.results_root / "runs" / self.run_id

        # Subdirectories
        self.preds_dir = self.run_dir / "preds"
        self.ckpt_dir = self.run_dir / "ckpt"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.preds_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.events_path = self.run_dir / "events.jsonl"
        self.ledger_path = self.results_root / "ledger.csv"

        # Initialize config.yaml and env.json
        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        env_info = capture_environment_info(thermal_telemetry=thermal_telemetry)
        with open(self.run_dir / "env.json", "w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=2)

    def log_metric(self, record: dict[str, Any]) -> None:
        """Append one metrics row to metrics.jsonl."""
        record["ts"] = datetime.now().isoformat()
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_event(self, event_name: str, details: dict[str, Any]) -> None:
        """Append one structured lifecycle event to events.jsonl."""
        entry = {
            "ts": datetime.now().isoformat(),
            "event": event_name,
            **details,
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def append_ledger(
        self,
        experiment_id: str,
        method: str,
        site: str,
        seed: int,
        primary_metric: str,
        metric_value: float,
        wall_time_s: float,
        paused_time_s: float,
        status: str = "completed",
        bytes_transferred: int = 0,
        supersedes: str | None = None,
    ) -> None:
        """Append a completed run record to results/ledger.csv."""
        row = {
            "run_id": self.run_id,
            "experiment_id": experiment_id,
            "method": method,
            "site": site,
            "seed": seed,
            "primary_metric": primary_metric,
            "metric_value": metric_value,
            "wall_time_s": round(wall_time_s, 2),
            "paused_time_s": round(paused_time_s, 2),
            "bytes_transferred": bytes_transferred,
            "status": status,
            "supersedes": supersedes or "",
            "timestamp": datetime.now().isoformat(),
        }

        df_row = pd.DataFrame([row])
        if self.ledger_path.exists():
            df_row.to_csv(self.ledger_path, mode="a", header=False, index=False)
        else:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            df_row.to_csv(self.ledger_path, mode="w", header=True, index=False)
        logger.info("Run recorded in ledger: %s -> %s = %.4f", self.run_id, primary_metric, metric_value)
