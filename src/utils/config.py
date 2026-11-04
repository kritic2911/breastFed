"""Centralized Configuration Manager for Cross-Silo Mammography BTP.

Provides unified loading, validation, and parameter extraction from configs/config.yaml.
Allows changing all hyperparameters across local baselines, centralized runs,
federated learning simulations, and hardware thermal guards from a single file.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.thermal import ThermalGuard

logger = logging.getLogger(__name__)

# Default configuration path relative to repository root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses configs/config.yaml.

    Returns:
        Dict containing all configuration sections.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning("Config file not found at %s. Returning default fallback config.", path)
        return get_default_config()

    try:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        logger.info("Loaded master configuration from %s", path)
        return cfg
    except Exception as e:
        logger.error("Failed to parse config file %s: %s. Using default fallback.", path, e)
        return get_default_config()


def save_config(cfg: Dict[str, Any], save_path: str | Path) -> None:
    """Save configuration dictionary to a YAML file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved configuration to %s", path)


def create_thermal_guard(
    cfg: Dict[str, Any],
    event_logger=None,
    sensor=None,
    exit_on_timeout: bool = True,
) -> ThermalGuard:
    """Instantiate a ThermalGuard configured directly from the configuration dictionary.

    Extracts soft_c, hard_c, resume_c, max_wait_s, pause_sleep_s, check_every_iters
    from cfg['hardware']['thermal'].
    """
    thermal_cfg = cfg.get("hardware", {}).get("thermal", {})
    if not thermal_cfg:
        # Fallback if flat config or alternative key
        thermal_cfg = cfg.get("thermal", {})

    soft_c = float(thermal_cfg.get("soft_c", 70.0))
    hard_c = float(thermal_cfg.get("hard_c", 75.0))
    resume_c = float(thermal_cfg.get("resume_c", 55.0))
    check_every_iters = int(thermal_cfg.get("check_every_iters", 20))
    max_wait_s = float(thermal_cfg.get("max_wait_s", 600.0))
    pause_sleep_s = float(thermal_cfg.get("pause_sleep_s", 2.0))

    return ThermalGuard(
        sensor=sensor,
        event_logger=event_logger,
        soft_c=soft_c,
        hard_c=hard_c,
        resume_c=resume_c,
        check_every_iters=check_every_iters,
        max_wait_s=max_wait_s,
        pause_sleep_s=pause_sleep_s,
        exit_on_timeout=exit_on_timeout,
    )


def get_local_training_config(
    master_cfg: Dict[str, Any],
    site: str,
    seed: int,
    supersedes: str | None = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Generate the exact config dictionary expected by train_local().

    Merges model defaults, local_baseline parameters, thermal parameters, and any overrides.
    """
    model_cfg = master_cfg.get("model", {})
    local_cfg = master_cfg.get("local_baseline", {})
    thermal_cfg = master_cfg.get("hardware", {}).get("thermal", {})
    data_cfg = master_cfg.get("data", {})

    cfg: Dict[str, Any] = {
        "experiment_id": f"local_baseline_{site}",
        "method": "local_baseline",
        "site": site,
        "seed": int(seed),
        "bn_mode": str(model_cfg.get("bn_mode", "trainable")),
        "backbone_lr": float(local_cfg.get("backbone_lr", 3e-4)),
        "head_lr_mult": float(model_cfg.get("head_lr_mult", 10.0)),
        "batch_size": int(local_cfg.get("batch_size", 16)),
        "max_epochs": int(local_cfg.get("max_epochs", 40)),
        "patience": int(local_cfg.get("patience", 12)),
        "weight_decay": float(local_cfg.get("weight_decay", 1e-2)),
        "lr_schedule": str(local_cfg.get("lr_schedule", "cosine")),
        "data_dir": str(data_cfg.get("data_dir", "data")),
        "supersedes": supersedes,
        # Hardware & Thermal Guard parameters for training loop
        "thermal_soft_c": float(thermal_cfg.get("soft_c", 70.0)),
        "thermal_hard_c": float(thermal_cfg.get("hard_c", 75.0)),
        "thermal_resume_c": float(thermal_cfg.get("resume_c", 55.0)),
        "thermal_check_iters": int(thermal_cfg.get("check_every_iters", 20)),
        "thermal_max_wait_s": float(thermal_cfg.get("max_wait_s", 600.0)),
    }

    # Apply explicit keyword overrides
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    return cfg


def get_centralized_training_config(
    master_cfg: Dict[str, Any],
    mode: str,
    seed: int,
    supersedes: str | None = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Generate the config dictionary expected by train_local() for centralized training."""
    model_cfg = master_cfg.get("model", {})
    cent_cfg = master_cfg.get("centralized", {})
    thermal_cfg = master_cfg.get("hardware", {}).get("thermal", {})
    data_cfg = master_cfg.get("data", {})

    exp_id = f"centralized_{mode}"
    is_site_balanced = (mode == "site_balanced")

    cfg: Dict[str, Any] = {
        "experiment_id": exp_id,
        "method": exp_id,
        "site": "pooled",
        "seed": int(seed),
        "bn_mode": str(model_cfg.get("bn_mode", "trainable")),
        "backbone_lr": float(cent_cfg.get("backbone_lr", 3e-4)),
        "head_lr_mult": float(model_cfg.get("head_lr_mult", 10.0)),
        "batch_size": int(cent_cfg.get("batch_size", 16)),
        "max_epochs": int(cent_cfg.get("max_epochs", 40)),
        "patience": int(cent_cfg.get("patience", 12)),
        "weight_decay": float(cent_cfg.get("weight_decay", 1e-2)),
        "lr_schedule": str(cent_cfg.get("lr_schedule", "cosine")),
        "site_balanced": is_site_balanced,
        "data_dir": str(data_cfg.get("data_dir", "data")),
        "supersedes": supersedes,
        # Hardware & Thermal Guard parameters for training loop
        "thermal_soft_c": float(thermal_cfg.get("soft_c", 70.0)),
        "thermal_hard_c": float(thermal_cfg.get("hard_c", 75.0)),
        "thermal_resume_c": float(thermal_cfg.get("resume_c", 55.0)),
        "thermal_check_iters": int(thermal_cfg.get("check_every_iters", 20)),
        "thermal_max_wait_s": float(thermal_cfg.get("max_wait_s", 600.0)),
    }

    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    return cfg


def get_default_config() -> Dict[str, Any]:
    """Provide hardcoded default configuration dictionary as fallback."""
    return {
        "hardware": {
            "device": "cuda",
            "thermal": {
                "soft_c": 70.0,
                "hard_c": 75.0,
                "resume_c": 55.0,
                "check_every_iters": 20,
                "max_wait_s": 600.0,
                "pause_sleep_s": 2.0,
            },
        },
        "data": {
            "data_dir": "data",
            "cache_dir": "data/cache512",
            "manifest_csv": "data/manifest.csv",
            "image_size": 512,
            "seeds": [42, 53, 64],
            "val_size": 0.20,
            "all_sites": ["vindr", "cmmd", "inbreast", "cdd"],
        },
        "model": {
            "backbone": "resnet18",
            "num_classes": 3,
            "pretrained": True,
            "head_lr_mult": 10.0,
            "bn_mode": "trainable",
        },
        "local_baseline": {
            "max_epochs": 40,
            "patience": 12,
            "batch_size": 16,
            "backbone_lr": 3e-4,
            "weight_decay": 1e-2,
            "lr_schedule": "cosine",
            "sites": ["vindr", "cmmd", "inbreast", "cdd"],
            "seeds": [42, 53, 64],
        },
        "centralized": {
            "max_epochs": 40,
            "patience": 12,
            "batch_size": 16,
            "backbone_lr": 3e-4,
            "weight_decay": 1e-2,
            "lr_schedule": "cosine",
            "modes": ["pooled", "site_balanced"],
            "seeds": [42, 53, 64],
        },
        "federated": {
            "rounds": 20,
            "local_epochs": 1,
            "batch_size": 8,
            "backbone_lr": 1e-4,
            "strategy": "fedavg",
            "aggregation_mode": "uniform",
            "mu": 0.01,
            "sites": ["vindr", "cmmd", "inbreast", "cdd"],
            "seeds": [42, 53, 64],
        },
        "explainability": {
            "target_layer": "layer4",
            "grid_rows": 3,
            "grid_cols": 3,
            "risk_threshold_low": 0.15,
            "risk_threshold_high": 0.50,
        },
    }
