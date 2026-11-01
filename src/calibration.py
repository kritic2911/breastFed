"""Probability calibration and reliability evaluation (Guo et al. 2017).

Provides:
- Expected Calibration Error (ECE) and reliability diagram bins
- Temperature scaling optimizer fitted on validation logits
- Per-site calibration reporting before and after temperature scaling
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute Expected Calibration Error (ECE) and bin statistics.

    Parameters
    ----------
    probs : array-like of predicted probabilities in [0, 1]
    labels : array-like of binary ground-truth labels {0, 1}
    n_bins : int, number of calibration bins (default: 10)

    Returns
    -------
    dict with:
      - ece: float Expected Calibration Error
      - max_error: float Maximum Calibration Error (MCE)
      - bins: list of dicts with (bin_idx, count, conf, acc)
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(probs)
    if n == 0:
        return {"ece": np.nan, "max_error": np.nan, "bins": []}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = []
    ece = 0.0
    max_err = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (probs >= low) & (probs <= high)
        else:
            in_bin = (probs >= low) & (probs < high)

        count = int(np.sum(in_bin))
        if count > 0:
            bin_conf = float(np.mean(probs[in_bin]))
            bin_acc = float(np.mean(labels[in_bin]))
            err = abs(bin_acc - bin_conf)
            ece += (count / n) * err
            max_err = max(max_err, err)
            bin_data.append({
                "bin": i,
                "range": [float(low), float(high)],
                "count": count,
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "error": float(err),
            })
        else:
            bin_data.append({
                "bin": i,
                "range": [float(low), float(high)],
                "count": 0,
                "confidence": float((low + high) / 2.0),
                "accuracy": 0.0,
                "error": 0.0,
            })

    return {
        "ece": float(ece),
        "max_error": float(max_err),
        "bins": bin_data,
    }


class TemperatureScaler:
    """Learns a scalar temperature T > 0 to calibrate logits: p = sigmoid(logits / T)."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor) -> float:
        """Fit temperature T on validation logits and labels by minimizing NLL / BCE."""
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        logits = np.asarray(logits, dtype=float).flatten()
        labels = np.asarray(labels, dtype=float).flatten()

        def bce_loss(temp_arr):
            t = temp_arr[0]
            scaled = logits / t
            # Stable log-sigmoid computation
            log_p = -np.logaddexp(0.0, -scaled)
            log_1_p = -np.logaddexp(0.0, scaled)
            loss = -(labels * log_p + (1.0 - labels) * log_1_p).mean()
            return loss

        res = minimize(
            bce_loss,
            x0=[1.0],
            bounds=[(0.05, 10.0)],
            method="L-BFGS-B",
        )
        self.temperature = float(res.x[0])
        return self.temperature

    def calibrate(self, logits: np.ndarray | torch.Tensor) -> np.ndarray:
        """Apply learned temperature scaling to convert logits to calibrated probabilities."""
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        logits = np.asarray(logits, dtype=float)
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))

    def save(self, path: str | Path):
        """Save fitted temperature to JSON."""
        with open(path, "w") as f:
            json.dump({"temperature": self.temperature}, f, indent=2)

    def load(self, path: str | Path):
        """Load temperature from JSON."""
        with open(path) as f:
            data = json.load(f)
            self.temperature = float(data["temperature"])

