"""Provenance, configuration hashing, run ID generation, and structured logging."""

import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml


def get_git_sha() -> str:
    """Return short git commit SHA or 'unknown' if not in git repo."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        # Fallback: attempt reading from .git directly if git executable fails
        git_head = Path(".git/HEAD")
        if git_head.exists():
            try:
                ref = git_head.read_text().strip()
                if ref.startswith("ref: "):
                    ref_path = Path(".git") / ref[5:]
                    if ref_path.exists():
                        return ref_path.read_text().strip()[:7]
                elif len(ref) >= 7:
                    return ref[:7]
            except Exception:
                pass
        return "unknown"


def compute_config_hash(cfg: dict, length: int = 8) -> str:
    """Compute deterministic short hash of configuration dict."""
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def get_environment_info() -> dict:
    """Capture runtime environment details for experiment provenance."""
    cuda_avail = torch.cuda.is_available()
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "gpu_device": torch.cuda.get_device_name(0) if cuda_avail else None,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }


def generate_run_id(cfg: dict, timestamp: str = None) -> tuple[str, str, str]:
    """Generate run_id = <timestamp>-<config hash>-<git sha>.

    Returns
    -------
    (run_id, config_hash, git_sha)
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_hash = compute_config_hash(cfg)
    git_sha = get_git_sha()
    run_id = f"{ts}-{cfg_hash}-{git_sha}"
    return run_id, cfg_hash, git_sha


class StructuredLogger:
    """JSONL structured logger adhering to the schema:
    ts, run_id, event, round, site, split, metric, value
    """

    def __init__(self, log_path: str | Path, run_id: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def log(
        self,
        event: str,
        round: int | None = None,
        site: str | None = None,
        split: str | None = None,
        metric: str | None = None,
        value: float | int | str | None = None,
        ts: str | None = None,
    ):
        """Append one JSON record to logs.jsonl."""
        record = {
            "ts": ts or datetime.now().isoformat(),
            "run_id": self.run_id,
            "event": event,
            "round": round,
            "site": site,
            "split": split,
            "metric": metric,
            "value": value,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def save_resolved_config(cfg: dict, output_dir: Path) -> Path:
    """Save resolved configuration dict as YAML."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = output_dir / "resolved_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path
