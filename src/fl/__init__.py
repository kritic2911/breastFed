"""Federated Learning (FL) simulation module for Digital Mammography BTP.

Implements cross-silo FL across heterogeneous digital mammography hospitals:
  - Federated Averaging (FedAvg: sample-weighted and uniform)
  - FedProx (proximal term for domain drift)
  - FedBN (local batch normalization parameters)
  - Systems metrics: byte counters, latency, atomic checkpointing
"""

