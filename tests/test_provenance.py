"""Unit tests for provenance, patient de-identification, RunLogger, and ledger."""

import json

from src.utils.provenance import (
    RunLogger,
    compute_config_hash,
    generate_run_id,
    hash_patient_id,
)


def test_patient_id_hashing():
    """Verify salted SHA-256 patient ID hashing is deterministic and non-trivial."""
    h1 = hash_patient_id("patient_1001")
    h2 = hash_patient_id("patient_1001")
    h3 = hash_patient_id("patient_1002")

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16
    assert "patient" not in h1


def test_config_hash_and_run_id():
    """Verify config hash and run ID format."""
    cfg1 = {"lr": 1e-4, "batch_size": 16, "site": "CMMD"}
    cfg2 = {"site": "CMMD", "batch_size": 16, "lr": 1e-4}
    cfg3 = {"lr": 3e-4, "batch_size": 16, "site": "CMMD"}

    # Key order independent
    assert compute_config_hash(cfg1) == compute_config_hash(cfg2)
    assert compute_config_hash(cfg1) != compute_config_hash(cfg3)

    run_id = generate_run_id(cfg1)
    parts = run_id.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 15  # YYYYMMDD_HHMMSS
    assert len(parts[1]) == 8   # 8-char config hash
    assert len(parts[2]) == 7   # 7-char git SHA


def test_run_logger_and_ledger(tmp_path):
    """Verify RunLogger writes full directory schema and appends to ledger."""
    cfg = {"experiment_id": "TEST", "method": "test_method", "site": "test_site", "seed": 42}
    logger = RunLogger(cfg=cfg, results_root=tmp_path)

    assert (logger.run_dir / "config.yaml").exists()
    assert (logger.run_dir / "env.json").exists()
    assert logger.preds_dir.exists()
    assert logger.ckpt_dir.exists()

    logger.log_metric({"epoch": 1, "loss": 0.5, "auc": 0.85})
    logger.log_event("test_event", {"status": "ok"})

    assert logger.metrics_path.exists()
    assert logger.events_path.exists()

    # Read back metrics
    with open(logger.metrics_path) as f:
        metrics = [json.loads(line) for line in f]
    assert len(metrics) == 1
    assert metrics[0]["auc"] == 0.85

    # Append to ledger
    logger.append_ledger(
        experiment_id="TEST",
        method="test_method",
        site="test_site",
        seed=42,
        primary_metric="auc",
        metric_value=0.85,
        wall_time_s=10.5,
        paused_time_s=0.0,
    )

    assert logger.ledger_path.exists()
    with open(logger.ledger_path) as f:
        lines = f.readlines()
    assert len(lines) == 2  # Header + 1 row
    assert "test_site" in lines[1]
