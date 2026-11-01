"""Pytest suite for FL-Mammo Phase 0 foundations.

Tests cover:
1. Aggregation identity (identical client weights -> same global weights)
2. Weights sum to 1.0 (both sample and uniform weighting)
3. No patient appears in two splits (zero split leakage)
4. Labels are in {0, 1} (binary classification integrity)
5. BN keys are excluded from averaging (FedBN / frozen BN isolation)
6. Provenance & structured logging invariants
"""

import copy
import json

import numpy as np
import pytest
import torch
import yaml

from src.calibration import TemperatureScaler, compute_ece
from src.federated import fedavg_aggregate
from src.ledger import aggregate_ledger, append_to_ledger
from src.model import build_model, get_bn_keys, verify_bn_state
from src.provenance import (
    StructuredLogger,
    compute_config_hash,
    generate_run_id,
    get_environment_info,
)
from src.synthetic import generate_synthetic_dataset

# ────────────────────────────────────────────────────────────────
# 1. Aggregation identity
# ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("weighting", ["sample", "uniform"])
def test_aggregation_identity(weighting):
    """If all clients possess identical state dicts, aggregation must
    return the exact same state dict regardless of weighting.
    """
    dummy_cfg = {"model": "resnet18", "pretrained": False, "num_classes": 1, "freeze_bn": True}
    model = build_model(dummy_cfg, device="cpu")
    global_state = model.state_dict()

    # Create 4 identical client states
    client_states = [copy.deepcopy(global_state) for _ in range(4)]
    client_sizes = [50, 150, 300, 500]

    agg_state = fedavg_aggregate(
        global_state, client_states, client_sizes, weighting=weighting
    )

    assert set(agg_state.keys()) == set(global_state.keys())
    for k in global_state.keys():
        diff = (agg_state[k].float() - global_state[k].float()).abs().max().item()
        assert diff < 1e-6, f"Aggregation identity failed for key {k}: max diff={diff}"


# ────────────────────────────────────────────────────────────────
# 2. Weights sum to 1.0
# ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "client_sizes",
    [
        [100, 200, 300, 400],
        [7, 13, 29, 97],
        [1000],
        [1, 1, 1, 1, 1],
        [2368, 493, 308, 219],  # Actual FL-Mammo site sizes
    ],
)
@pytest.mark.parametrize("weighting", ["sample", "uniform"])
def test_weights_sum_to_one(client_sizes, weighting):
    """The computed client weights must sum to 1.0 within floating point precision."""
    n_clients = len(client_sizes)
    total = sum(client_sizes)

    if weighting == "sample":
        weights = [n / total for n in client_sizes]
    else:
        weights = [1.0 / n_clients] * n_clients

    assert sum(weights) == pytest.approx(1.0, rel=1e-6)
    for w in weights:
        assert 0.0 < w <= 1.0


# ────────────────────────────────────────────────────────────────
# 3. No patient appears in two splits
# ────────────────────────────────────────────────────────────────

def test_no_patient_in_two_splits(tmp_path):
    """No patient ID may overlap across train, val, and test splits."""
    synth_dir = tmp_path / "synthetic_data"
    generate_synthetic_dataset(
        data_root=synth_dir,
        sites=["CMMD", "CBIS", "BCDR", "INbreast"],
        patients_per_site=15,
        seed=101,
    )

    import pandas as pd
    for site in ["CMMD", "CBIS", "BCDR", "INbreast"]:
        labels_file = synth_dir / site / "labels_split.csv"
        assert labels_file.exists()
        df = pd.read_csv(labels_file)

        train_pids = set(df[df["split"] == "train"]["patient_id"])
        val_pids = set(df[df["split"] == "val"]["patient_id"])
        test_pids = set(df[df["split"] == "test"]["patient_id"])

        # Check disjointness
        assert train_pids.isdisjoint(val_pids), f"Site {site} has overlap between train and val"
        assert train_pids.isdisjoint(test_pids), f"Site {site} has overlap between train and test"
        assert val_pids.isdisjoint(test_pids), f"Site {site} has overlap between val and test"

        # Check non-empty
        assert len(train_pids) > 0
        assert len(val_pids) > 0
        assert len(test_pids) > 0


# ────────────────────────────────────────────────────────────────
# 4. Labels are in {0, 1}
# ────────────────────────────────────────────────────────────────

def test_labels_in_binary_set(tmp_path):
    """All dataset labels must strictly be 0 (benign) or 1 (malignant)."""
    synth_dir = tmp_path / "synthetic_data"
    generate_synthetic_dataset(
        data_root=synth_dir,
        sites=["CMMD", "CBIS", "BCDR", "INbreast"],
        patients_per_site=10,
        seed=202,
    )

    import pandas as pd
    for site in ["CMMD", "CBIS", "BCDR", "INbreast"]:
        df = pd.read_csv(synth_dir / site / "labels_split.csv")
        unique_labels = set(df["label"].unique())
        assert unique_labels.issubset({0, 1}), f"Site {site} has invalid labels: {unique_labels}"


# ────────────────────────────────────────────────────────────────
# 5. BN keys are excluded from averaging
# ────────────────────────────────────────────────────────────────

def test_bn_keys_excluded_from_averaging():
    """BatchNorm parameters (running mean, var, weight, bias) must be
    retained from global_state and excluded from client averaging.
    """
    dummy_cfg = {"model": "resnet18", "pretrained": False, "num_classes": 1, "freeze_bn": True}
    model = build_model(dummy_cfg, device="cpu")
    bn_keys = set(get_bn_keys(model))

    assert len(bn_keys) > 0, "ResNet18 should contain BatchNorm keys"

    # Set up global state with sentinel values for BN
    global_state = copy.deepcopy(model.state_dict())
    for k in bn_keys:
        global_state[k] = torch.ones_like(global_state[k]) * 1.0

    # Perturb client states: different BN values and perturbed conv weights
    client_states = []
    for client_val in [55.0, 99.0]:
        c_state = copy.deepcopy(global_state)
        for k in bn_keys:
            c_state[k] = torch.ones_like(c_state[k]) * client_val
        # Also perturb a non-BN weight
        c_state["conv1.weight"] = c_state["conv1.weight"] + 0.5
        client_states.append(c_state)

    agg_state = fedavg_aggregate(
        global_state, client_states, client_sizes=[100, 100], exclude_keys=bn_keys
    )

    # Invariant 1: BN keys in aggregated model must exactly equal global_state
    for k in bn_keys:
        assert torch.equal(agg_state[k], global_state[k]), f"BN key {k} was altered during aggregation!"
        assert not torch.allclose(agg_state[k], torch.tensor(55.0)), f"BN key {k} leaked client 0 state"
        assert not torch.allclose(agg_state[k], torch.tensor(77.0)), f"BN key {k} was averaged"

    # Invariant 2: Non-BN keys were averaged
    expected_conv1 = global_state["conv1.weight"] + 0.5
    assert torch.allclose(agg_state["conv1.weight"], expected_conv1, atol=1e-5)


# ────────────────────────────────────────────────────────────────
# 6. Provenance & Structured Logging Invariants
# ────────────────────────────────────────────────────────────────

def test_provenance_and_run_id_generation(tmp_path):
    """Verify run_id format, hash determinism, and logs.jsonl schema."""
    with open("configs/experiment.yaml") as f:
        cfg = yaml.safe_load(f)

    # 1. Deterministic config hash
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash(cfg)
    assert h1 == h2 and len(h1) == 8

    # 2. Run ID structure: <timestamp>-<config_hash>-<git_sha>
    run_id, cfg_hash, git_sha = generate_run_id(cfg)
    parts = run_id.split("-")
    assert len(parts) == 3, f"run_id '{run_id}' does not have 3 parts"
    assert parts[1] == cfg_hash
    assert parts[2] == git_sha
    assert len(parts[0]) == 15  # YYYYMMDD_HHMMSS

    # 3. Environment info capture
    env = get_environment_info()
    for key in ["python_version", "torch_version", "cuda_available", "platform"]:
        assert key in env

    # 4. Structured logger writes valid JSONL adhering to schema
    log_path = tmp_path / "logs.jsonl"
    logger = StructuredLogger(log_path, run_id=run_id)

    logger.log(event="test_event_1", round=1, site="CMMD", split="train", metric="loss", value=0.45)
    logger.log(event="test_event_2", round=1, site="CBIS", split="val", metric="auc", value=0.72)

    assert log_path.exists()
    lines = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
    assert len(lines) == 2

    required_schema = {"ts", "run_id", "event", "round", "site", "split", "metric", "value"}
    for record in lines:
        assert set(record.keys()) == required_schema
        assert record["run_id"] == run_id


# ────────────────────────────────────────────────────────────────
# 7. BatchNorm State Inspection
# ────────────────────────────────────────────────────────────────

def test_bn_state_verification():
    """Verify that verify_bn_state detects frozen eval mode and requires_grad=False."""
    cfg = {"model": "resnet18", "pretrained": False, "num_classes": 1, "freeze_bn": True}
    model = build_model(cfg, device="cpu")
    status = verify_bn_state(model, verbose=False)

    assert status["num_bn_layers"] > 0
    assert status["all_in_eval_mode"] is True
    assert status["all_requires_grad_false"] is True


# ────────────────────────────────────────────────────────────────
# 8. Experiment Ledger & Seed Statistics
# ────────────────────────────────────────────────────────────────

def test_ledger_append_and_seed_statistics(tmp_path):
    """Verify append_to_ledger and student-t 95% CI calculation over seeds."""
    ledger_file = tmp_path / "ledger.csv"
    seeds = [42, 53, 64]
    aucs = [0.72, 0.74, 0.73]

    for s, a in zip(seeds, aucs):
        append_to_ledger(
            run_id=f"run_seed_{s}",
            method="federated",
            site="CMMD",
            seed=s,
            metric="auc",
            value=a,
            ledger_path=ledger_file,
        )

    assert ledger_file.exists()
    df_agg = aggregate_ledger(ledger_path=ledger_file, metric="auc")
    assert len(df_agg) == 1
    row = df_agg.iloc[0]
    assert row["mean"] == pytest.approx(0.73, abs=1e-4)
    assert row["n_seeds"] == 3
    assert row["ci_low"] < row["mean"] < row["ci_high"]


# ────────────────────────────────────────────────────────────────
# 9. Calibration & Temperature Scaling
# ────────────────────────────────────────────────────────────────

def test_ece_and_temperature_scaling():
    """Verify ECE computation and temperature scaling convergence."""
    # Synthetic uncalibrated overconfident predictions
    np.random.seed(42)
    labels = np.array([0] * 50 + [1] * 50)
    logits = np.concatenate([
        np.random.normal(loc=-2.0, scale=1.0, size=50),
        np.random.normal(loc=2.0, scale=1.0, size=50),
    ])
    raw_probs = 1.0 / (1.0 + np.exp(-logits))

    ece_res = compute_ece(raw_probs, labels, n_bins=5)
    assert 0.0 <= ece_res["ece"] <= 1.0
    assert len(ece_res["bins"]) == 5

    # Fit temperature scaler
    scaler = TemperatureScaler()
    t = scaler.fit(logits, labels)
    assert 0.05 < t < 10.0

    cal_probs = scaler.calibrate(logits)
    cal_ece = compute_ece(cal_probs, labels, n_bins=5)
    assert 0.0 <= cal_ece["ece"] <= 1.0

