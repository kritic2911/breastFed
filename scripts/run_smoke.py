#!/usr/bin/env python3
"""Smoke run: runs 2 federated rounds on synthetic data on CPU in under 2 minutes.

Verifies that the federated pipeline runs end-to-end and writes all required
artifacts (resolved config, config hash, git SHA, seed, environment, logs.jsonl,
metrics.json, preds.csv) into the run directory.
"""

import copy
import json
import sys
import time
from pathlib import Path

# Ensure root in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.federated import run_federated
from src.provenance import generate_run_id
from src.synthetic import generate_synthetic_dataset


def main():
    t0 = time.time()
    print("=" * 70)
    print("FL-Mammo Phase 0: Synthetic Smoke Test (2 FedAvg rounds on CPU)")
    print("=" * 70)

    # 1. Generate or verify synthetic data
    synth_dir = Path("synthetic_data")
    print("1. Generating synthetic dataset with planted signal...")
    generate_synthetic_dataset(
        data_root=synth_dir,
        sites=["CMMD", "CBIS", "BCDR", "INbreast"],
        patients_per_site=12,
        images_per_patient=2,
        img_size=224,
        seed=42,
    )

    # 2. Load base config and adjust for smoke test
    with open("configs/experiment.yaml") as f:
        cfg = yaml.safe_load(f)

    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["data_root"] = str(synth_dir)
    smoke_cfg["sites"] = ["CMMD", "CBIS", "BCDR", "INbreast"]
    smoke_cfg["pretrained"] = False        # No external downloads, clean CPU run
    smoke_cfg["num_workers"] = 0          # Fast single-thread dataloading
    smoke_cfg["batch_size"] = 8
    smoke_cfg["thermal"] = {"enabled": False}
    smoke_cfg["fed"]["rounds"] = 2        # Exactly 2 communication rounds
    smoke_cfg["fed"]["local_epochs"] = 1
    smoke_cfg["fed"]["weighting"] = "sample"
    smoke_cfg["eval"]["bootstrap_patients"] = 25
    smoke_cfg["eval"]["seeds"] = [0]

    # Generate unique run ID and output directory
    run_id, cfg_hash, git_sha = generate_run_id(smoke_cfg)
    out_dir = Path("results/smoke") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_cfg["results_root"] = "results"

    print(f"2. Launching smoke run: run_id={run_id}")
    print(f"   Config Hash: {cfg_hash} | Git SHA: {git_sha} | Output: {out_dir}")

    # 3. Execute federated run
    _results = run_federated(
        cfg=smoke_cfg,
        seed=0,
        device="cpu",
        output_dir=str(out_dir),
        resume=False,
        force_train=True,
        method_name="smoke_fedavg",
    )

    elapsed = time.time() - t0
    print(f"\n3. Smoke run completed in {elapsed:.2f} seconds.")

    # 4. Verify Gate requirements
    print("\n4. Verifying Gate invariants & artifacts (Rule 3)...")
    assert elapsed < 120.0, f"Smoke run exceeded 2 minutes ({elapsed:.1f}s)"

    expected_files = [
        "resolved_config.yaml",
        "logs.jsonl",
        "metrics.json",
        "preds.csv",
        "best.pt",
        "training_history.csv",
    ]

    for fname in expected_files:
        fpath = out_dir / fname
        assert fpath.exists(), f"Missing expected artifact: {fpath}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fpath}"
        print(f"   [OK] {fname} ({fpath.stat().st_size} bytes)")

    # Validate metrics.json contents
    with open(out_dir / "metrics.json") as f:
        metrics_data = json.load(f)

    for field in ["run_id", "config_hash", "git_sha", "seed", "environment"]:
        assert field in metrics_data, f"metrics.json missing provenance field '{field}'"
        print(f"   [OK] metrics.json contains '{field}': {metrics_data[field]}")

    # Validate logs.jsonl schema
    with open(out_dir / "logs.jsonl") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) >= 2, f"logs.jsonl should have multiple event records, got {len(lines)}"

    required_keys = {"ts", "run_id", "event", "round", "site", "split", "metric", "value"}
    for idx, line in enumerate(lines):
        record = json.loads(line)
        missing = required_keys - set(record.keys())
        assert not missing, f"Line {idx} in logs.jsonl missing keys: {missing}"
    print(f"   [OK] logs.jsonl contains {len(lines)} correctly structured records")

    print("\n" + "=" * 70)
    print(f"GATE PASS: Smoke run passed all checks in {elapsed:.2f}s (< 120s)")
    print(f"Run directory: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
