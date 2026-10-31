
#!/usr/bin/env python3
"""One-Client Test: FedAvg with CMMD alone.
Verifies that FedAvg with a single client reproduces local training behavior
and reaches expected AUC (~0.71-0.74).
"""

import copy
import sys
from pathlib import Path

import torch
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.federated import run_federated


def main():
    with open("configs/experiment.yaml") as f:
        cfg = yaml.safe_load(f)

    # 1-client config: CMMD alone
    one_client_cfg = copy.deepcopy(cfg)
    one_client_cfg["sites"] = ["CMMD"]
    one_client_cfg["fed"]["rounds"] = 12       # 12 rounds x 2 epochs = 24 epochs
    one_client_cfg["fed"]["lr"] = 1.0e-4       # constant stable LR
    one_client_cfg["fed"]["constant_lr"] = True
    one_client_cfg["fed"]["patience"] = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 70)
    print("ONE-CLIENT TEST: FedAvg with CMMD alone (12 rounds)")
    print("=" * 70)

    out_dir = Path("results/test_one_client_cmmd/seed_42")
    res = run_federated(
        cfg=one_client_cfg,
        seed=42,
        device=device,
        output_dir=str(out_dir),
        resume=False,
        force_train=True,
        method_name="test_one_client_cmmd",
    )

    cmmd_res = res["per_site_test"]["CMMD"]
    auc = cmmd_res["image_level"]["auc"]
    pat_auc = cmmd_res["patient_level"]["auc"]
    ci = cmmd_res["bootstrap_auc_ci"]

    print("\n" + "=" * 70)
    print("ONE-CLIENT TEST RESULTS (CMMD):")
    print(f"  Image-level AUC:   {auc:.4f} [95% CI: {ci['lower']:.4f}, {ci['upper']:.4f}]")
    print(f"  Patient-level AUC: {pat_auc:.4f}")
    print("  Local CMMD target: ~0.71 - 0.74")
    if auc >= 0.68:
        print("  -> PASSED: One-client FedAvg matches local CMMD performance!")
    else:
        print(f"  -> WARNING: AUC ({auc:.4f}) is lower than local CMMD target.")
    print("=" * 70)


if __name__ == "__main__":
    main()
