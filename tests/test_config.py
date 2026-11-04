"""Tests for master configuration loading and thermal/training helpers."""

import tempfile
from pathlib import Path

from src.utils.config import (
    create_thermal_guard,
    get_centralized_training_config,
    get_local_training_config,
    load_config,
    save_config,
)


class MockSensor:
    def __init__(self, slowdown_temp=87.0):
        self.slowdown_temp = slowdown_temp

    def query(self):
        return {"temperature": 45.0, "slowdown_temp": self.slowdown_temp}


def test_master_config_exists_and_loads():
    """Verify configs/config.yaml exists and contains all required sections."""
    cfg = load_config("configs/config.yaml")
    assert isinstance(cfg, dict)

    required_sections = [
        "hardware",
        "data",
        "model",
        "local_baseline",
        "centralized",
        "federated",
        "explainability",
        "systems",
    ]
    for section in required_sections:
        assert section in cfg, f"Missing section '{section}' in configs/config.yaml"


def test_thermal_config_extraction():
    """Verify create_thermal_guard accurately pulls cooldown thresholds and timings."""
    custom_cfg = {
        "hardware": {
            "thermal": {
                "soft_c": 68.0,
                "hard_c": 74.0,
                "resume_c": 52.0,
                "check_every_iters": 15,
                "max_wait_s": 300.0,
                "pause_sleep_s": 1.0,
            }
        }
    }
    sensor = MockSensor(slowdown_temp=87.0)
    guard = create_thermal_guard(custom_cfg, sensor=sensor)

    assert guard.soft_c == 68.0
    assert guard.hard_c == 74.0
    assert guard.resume_c == 52.0
    assert guard.check_every_iters == 15
    assert guard.max_wait_s == 300.0
    assert guard.pause_sleep_s == 1.0


def test_local_training_config_generation():
    """Verify get_local_training_config merges master config with overrides."""
    master = load_config("configs/config.yaml")
    cfg = get_local_training_config(
        master,
        site="vindr",
        seed=42,
        max_epochs=25,
        backbone_lr=1.5e-4,
    )

    assert cfg["site"] == "vindr"
    assert cfg["seed"] == 42
    assert cfg["max_epochs"] == 25  # Overridden
    assert cfg["backbone_lr"] == 1.5e-4  # Overridden
    assert cfg["patience"] == master["local_baseline"]["patience"]  # From file
    assert cfg["thermal_soft_c"] == master["hardware"]["thermal"]["soft_c"]
    assert cfg["thermal_resume_c"] == master["hardware"]["thermal"]["resume_c"]


def test_centralized_training_config_generation():
    """Verify get_centralized_training_config sets site_balanced and pooled modes."""
    master = load_config("configs/config.yaml")

    pooled_cfg = get_centralized_training_config(master, mode="pooled", seed=53)
    assert pooled_cfg["site_balanced"] is False
    assert pooled_cfg["experiment_id"] == "centralized_pooled"
    assert pooled_cfg["seed"] == 53

    bal_cfg = get_centralized_training_config(master, mode="site_balanced", seed=64)
    assert bal_cfg["site_balanced"] is True
    assert bal_cfg["experiment_id"] == "centralized_site_balanced"
    assert bal_cfg["seed"] == 64


def test_save_and_reload_config():
    """Verify roundtrip saving and reloading of configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_cfg_path = Path(tmpdir) / "test_config.yaml"
        sample_cfg = {
            "hardware": {"thermal": {"resume_c": 50.0}},
            "federated": {"rounds": 35, "local_epochs": 2},
        }
        save_config(sample_cfg, tmp_cfg_path)
        assert tmp_cfg_path.exists()

        reloaded = load_config(tmp_cfg_path)
        assert reloaded["federated"]["rounds"] == 35
        assert reloaded["federated"]["local_epochs"] == 2
        assert reloaded["hardware"]["thermal"]["resume_c"] == 50.0
