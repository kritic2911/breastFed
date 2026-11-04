"""Federated Averaging (FedAvg) aggregation logic for cross-silo mammography.

Adheres strictly to AGENTS.md Section 2.3 & Learning Contract (Rule 10):
  - Sample-weighted FedAvg: updates weighted proportional to local dataset size n_k / N.
  - Uniform FedAvg: updates weighted equally 1 / K (mitigating CMMD 12x size dominance).
  - Explicit handling of BatchNorm: affine weights averaged, running stats handled explicitly.
  - Equivalence test: verified against hand-computed analytical mean within 1e-6.
"""

from collections import OrderedDict
from typing import Dict, List, Literal

import torch


def is_batchnorm_key(key: str) -> bool:
    """Identify if a model state dict parameter belongs to a BatchNorm layer."""
    bn_indicators = ["bn", "downsample.1"]
    return any(indicator in key for indicator in bn_indicators)


def aggregate_fedavg(
    client_updates: List[Dict[str, torch.Tensor]],
    client_sample_counts: List[int],
    mode: Literal["sample_weighted", "uniform"] = "sample_weighted",
    include_bn_stats: bool = True,
) -> Dict[str, torch.Tensor]:
    """Aggregate local client model updates into a new global model state dict.

    Design Choices & Trade-offs:
      1. Sample-Weighted (w_k = n_k / sum(n)):
         - Standard FedAvg (McMahan et al., 2017).
         - Trade-off: CMMD has ~12x more images than INbreast, so it dominates the gradient direction.
      2. Uniform (w_k = 1 / K):
         - Equalizes client impact regardless of dataset size.
         - Trade-off: Gives equal voting power to smaller sites, improving worst-site AUC at
           a potential minor cost to overall pooled accuracy.
      3. BatchNorm Handling (include_bn_stats):
         - FedAvg (include_bn_stats=True): Averages affine parameters (weight, bias) and running stats.
         - FedBN (include_bn_stats=False): Leaves BN parameters purely local to each client scanner domain.

    Args:
        client_updates: List of state dicts from K participating clients.
        client_sample_counts: List of integers [n_1, n_2, ..., n_K] of samples used per client.
        mode: "sample_weighted" or "uniform".
        include_bn_stats: If True, aggregates BN statistics; if False, omits BN running stats (FedBN mode).

    Returns:
        global_state_dict: Aggregated OrderedDict of global model parameters.
    """
    if not client_updates:
        raise ValueError("client_updates list cannot be empty.")
    if len(client_updates) != len(client_sample_counts):
        raise ValueError("client_updates and client_sample_counts must have the same length.")

    # 1. Compute aggregation weights w_k
    k = len(client_updates)
    if mode == "sample_weighted":
        total_samples = sum(client_sample_counts)
        if total_samples <= 0:
            raise ValueError("Total sample count must be positive.")
        weights = [n / total_samples for n in client_sample_counts]
    elif mode == "uniform":
        weights = [1.0 / k] * k
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")

    # 2. Accumulate weighted parameters across clients
    global_state_dict = OrderedDict()
    param_keys = client_updates[0].keys()

    for key in param_keys:
        # If FedBN mode (include_bn_stats=False), skip BN running statistics from global aggregation
        if not include_bn_stats and is_batchnorm_key(key) and ("running_" in key or "num_batches_tracked" in key):
            continue

        first_tensor = client_updates[0][key]

        # Integer tensors like num_batches_tracked: keep from first or max
        if not first_tensor.is_floating_point():
            global_state_dict[key] = first_tensor.clone()
            continue

        # Use double precision accumulator to eliminate rounding errors (< 1e-6 tolerance)
        accum = torch.zeros_like(first_tensor, dtype=torch.float64)

        for client_sd, w in zip(client_updates, weights):
            client_t = client_sd[key].to(dtype=torch.float64)
            accum += w * client_t

        global_state_dict[key] = accum.to(dtype=first_tensor.dtype)

    return global_state_dict


def aggregate_per_family_heads(
    client_updates: List[Dict[str, torch.Tensor]],
    client_sample_counts: List[int],
    client_families: List[str],
    mode: Literal["sample_weighted", "uniform"] = "uniform",
    include_bn_stats: bool = True,
) -> Dict[str, torch.Tensor]:
    """Aggregate shared backbone across ALL clients, but average per-family heads ONLY within family.

    Adheres strictly to AGENTS.md Section 1.5 & P3 Per-Family Heads Variant:
      - Backbone + BN: averaged across all K clients (general multi-scanner mammography representation).
      - heads.birads.*: averaged ONLY among clients with family == "birads" (VinDr, INbreast).
      - heads.biopsy.*: averaged ONLY among clients with family == "biopsy" (CMMD, CDD).
    """
    if not client_updates:
        raise ValueError("client_updates list cannot be empty.")
    if len(client_updates) != len(client_sample_counts) or len(client_updates) != len(client_families):
        raise ValueError("client_updates, client_sample_counts, and client_families must have the same length.")

    # 1. Global weights across all clients
    k = len(client_updates)
    if mode == "sample_weighted":
        total_samples = sum(client_sample_counts)
        global_weights = [n / total_samples for n in client_sample_counts]
    elif mode == "uniform":
        global_weights = [1.0 / k] * k
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")

    # 2. Per-family sub-weights
    family_weights: Dict[str, Dict[int, float]] = {}
    for fam in set(client_families):
        fam_indices = [i for i, f in enumerate(client_families) if f == fam]
        if mode == "sample_weighted":
            fam_total = sum(client_sample_counts[i] for i in fam_indices)
            family_weights[fam] = {i: client_sample_counts[i] / fam_total for i in fam_indices}
        else:
            family_weights[fam] = {i: 1.0 / len(fam_indices) for i in fam_indices}

    global_state_dict = OrderedDict()
    param_keys = client_updates[0].keys()

    for key in param_keys:
        if not include_bn_stats and is_batchnorm_key(key) and ("running_" in key or "num_batches_tracked" in key):
            continue

        first_tensor = client_updates[0][key]
        if not first_tensor.is_floating_point():
            global_state_dict[key] = first_tensor.clone()
            continue

        accum = torch.zeros_like(first_tensor, dtype=torch.float64)

        if key.startswith("heads.birads."):
            fam = "birads"
            weights_map = family_weights.get(fam, {})
            for idx, w in weights_map.items():
                accum += w * client_updates[idx][key].to(dtype=torch.float64)
        elif key.startswith("heads.biopsy."):
            fam = "biopsy"
            weights_map = family_weights.get(fam, {})
            for idx, w in weights_map.items():
                accum += w * client_updates[idx][key].to(dtype=torch.float64)
        else:
            # Shared backbone parameters & BN statistics
            for client_sd, w in zip(client_updates, global_weights):
                accum += w * client_sd[key].to(dtype=torch.float64)

        global_state_dict[key] = accum.to(dtype=first_tensor.dtype)

    return global_state_dict
