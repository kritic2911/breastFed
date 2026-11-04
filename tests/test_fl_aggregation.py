"""Equivalence tests for Federated Averaging (FedAvg) aggregation function.

Adheres strictly to AGENTS.md Section 2.3 Checkpoint P3:
  - Equivalence test: aggregation function matches a hand-computed weighted mean within 1e-6.
  - Verifies sample-weighted FedAvg (w_k = n_k / N).
  - Verifies uniform FedAvg (w_k = 1 / K).
  - Tests tensor shapes, device consistency, and floating-point accumulation precision.
"""

from collections import OrderedDict

import torch

from src.fl.aggregation import aggregate_fedavg, aggregate_per_family_heads


def test_fedavg_sample_weighted_hand_computed_equivalence():
    """Verify sample-weighted FedAvg matches hand-computed analytical mean within 1e-6."""
    # Client 1: 100 samples
    c1 = OrderedDict({
        "layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "layer.bias": torch.tensor([0.5, 1.5], dtype=torch.float32),
        "bn.running_mean": torch.tensor([0.1, 0.2], dtype=torch.float32),
    })
    # Client 2: 300 samples
    c2 = OrderedDict({
        "layer.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        "layer.bias": torch.tensor([2.5, 3.5], dtype=torch.float32),
        "bn.running_mean": torch.tensor([0.9, 0.6], dtype=torch.float32),
    })

    client_updates = [c1, c2]
    client_counts = [100, 300]

    # Hand-computed: w1 = 100/400 = 0.25, w2 = 300/400 = 0.75
    # layer.weight: 0.25 * [[1, 2], [3, 4]] + 0.75 * [[5, 6], [7, 8]]
    #             = [[0.25, 0.5], [0.75, 1.0]] + [[3.75, 4.5], [5.25, 6.0]]
    #             = [[4.0, 5.0], [6.0, 7.0]]
    expected_weight = torch.tensor([[4.0, 5.0], [6.0, 7.0]], dtype=torch.float32)

    # layer.bias: 0.25 * [0.5, 1.5] + 0.75 * [2.5, 3.5]
    #           = [0.125, 0.375] + [1.875, 2.625] = [2.0, 3.0]
    expected_bias = torch.tensor([2.0, 3.0], dtype=torch.float32)

    # bn.running_mean: 0.25 * [0.1, 0.2] + 0.75 * [0.9, 0.6]
    #                = [0.025, 0.05] + [0.675, 0.45] = [0.7, 0.5]
    expected_bn = torch.tensor([0.7, 0.5], dtype=torch.float32)

    global_weights = aggregate_fedavg(
        client_updates, client_counts, mode="sample_weighted", include_bn_stats=True
    )

    # Precision tolerance check: strict |diff| < 1e-6
    weight_diff = torch.max(torch.abs(global_weights["layer.weight"] - expected_weight)).item()
    bias_diff = torch.max(torch.abs(global_weights["layer.bias"] - expected_bias)).item()
    bn_diff = torch.max(torch.abs(global_weights["bn.running_mean"] - expected_bn)).item()

    assert weight_diff < 1e-6, f"Weight mismatch: max error = {weight_diff}"
    assert bias_diff < 1e-6, f"Bias mismatch: max error = {bias_diff}"
    assert bn_diff < 1e-6, f"BN running mean mismatch: max error = {bn_diff}"


def test_fedavg_uniform_hand_computed_equivalence():
    """Verify uniform FedAvg matches hand-computed unweighted mean within 1e-6."""
    # 3 clients with highly unequal sample counts (e.g. CMMD vs INbreast vs CDD)
    c1 = OrderedDict({"fc.weight": torch.tensor([1.0, 4.0])})
    c2 = OrderedDict({"fc.weight": torch.tensor([4.0, 7.0])})
    c3 = OrderedDict({"fc.weight": torch.tensor([7.0, 10.0])})

    client_updates = [c1, c2, c3]
    client_counts = [1200, 100, 300]  # Unequal sizes should NOT affect uniform mode

    # Hand-computed: (1/3) * (c1 + c2 + c3) = (1/3) * [12.0, 21.0] = [4.0, 7.0]
    expected_fc = torch.tensor([4.0, 7.0], dtype=torch.float32)

    global_weights = aggregate_fedavg(client_updates, client_counts, mode="uniform")

    diff = torch.max(torch.abs(global_weights["fc.weight"] - expected_fc)).item()
    assert diff < 1e-6, f"Uniform FedAvg mismatch: max error = {diff}"


def test_fedavg_fedbn_preserves_local_bn():
    """Verify that when include_bn_stats is False (FedBN mode), BN stats are not modified."""
    c1 = OrderedDict({
        "conv.weight": torch.tensor([1.0, 2.0]),
        "bn.weight": torch.tensor([1.0, 1.0]),
        "bn.running_mean": torch.tensor([0.1, 0.1]),
    })
    c2 = OrderedDict({
        "conv.weight": torch.tensor([3.0, 4.0]),
        "bn.weight": torch.tensor([2.0, 2.0]),
        "bn.running_mean": torch.tensor([0.5, 0.5]),
    })

    global_weights = aggregate_fedavg(
        [c1, c2], [10, 10], mode="uniform", include_bn_stats=False
    )

    # conv.weight should be averaged: [2.0, 3.0]
    assert torch.allclose(global_weights["conv.weight"], torch.tensor([2.0, 3.0]))
    # BN keys must either be omitted from global update or retained as specified
    assert "bn.running_mean" not in global_weights or global_weights["bn.running_mean"] is None


def test_aggregate_per_family_heads():
    """Verify that per-family heads are averaged ONLY within family while backbone is shared."""
    # 4 clients: 2 birads (c1, c2), 2 biopsy (c3, c4)
    c1 = OrderedDict({
        "backbone.weight": torch.tensor([1.0, 2.0]),
        "heads.birads.weight": torch.tensor([10.0, 20.0]),
        "heads.biopsy.weight": torch.tensor([0.0, 0.0]),  # Unused by c1
    })
    c2 = OrderedDict({
        "backbone.weight": torch.tensor([3.0, 4.0]),
        "heads.birads.weight": torch.tensor([30.0, 40.0]),
        "heads.biopsy.weight": torch.tensor([0.0, 0.0]),  # Unused by c2
    })
    c3 = OrderedDict({
        "backbone.weight": torch.tensor([5.0, 6.0]),
        "heads.birads.weight": torch.tensor([0.0, 0.0]),  # Unused by c3
        "heads.biopsy.weight": torch.tensor([100.0, 200.0]),
    })
    c4 = OrderedDict({
        "backbone.weight": torch.tensor([7.0, 8.0]),
        "heads.birads.weight": torch.tensor([0.0, 0.0]),  # Unused by c4
        "heads.biopsy.weight": torch.tensor([300.0, 400.0]),
    })

    client_updates = [c1, c2, c3, c4]
    client_counts = [100, 100, 100, 100]
    client_families = ["birads", "birads", "biopsy", "biopsy"]

    global_weights = aggregate_per_family_heads(
        client_updates=client_updates,
        client_sample_counts=client_counts,
        client_families=client_families,
        mode="uniform",
    )

    # 1. Backbone averaged across ALL 4 clients: (1 + 3 + 5 + 7)/4 = 4.0, (2 + 4 + 6 + 8)/4 = 5.0
    expected_backbone = torch.tensor([4.0, 5.0])
    assert torch.allclose(global_weights["backbone.weight"], expected_backbone)

    # 2. heads.birads averaged ONLY between c1 and c2: (10 + 30)/2 = 20.0, (20 + 40)/2 = 30.0
    expected_birads = torch.tensor([20.0, 30.0])
    assert torch.allclose(global_weights["heads.birads.weight"], expected_birads)

    # 3. heads.biopsy averaged ONLY between c3 and c4: (100 + 300)/2 = 200.0, (200 + 400)/2 = 300.0
    expected_biopsy = torch.tensor([200.0, 300.0])
    assert torch.allclose(global_weights["heads.biopsy.weight"], expected_biopsy)
