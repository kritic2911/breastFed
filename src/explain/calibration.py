"""Post-hoc calibration, temperature scaling, and clinical risk banding for mammography predictions.

Adheres strictly to AGENTS.md Phase P3b:
  - Temperature scaling: fitted strictly on validation split via NLL minimization.
  - Per-site vector calibration: temperature scalar T_s plus per-class logit bias b_s.
  - Probabilities clipped to [1%, 99%] (never claim certainty).
  - Risk banding: low / intermediate / high with thresholds selected on validation.
  - ECE evaluation across reliability bins.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import roc_curve


@dataclass
class CalibratorState:
    """Fitted calibration parameters for a specific site or global cohort."""
    site: str
    temperature: float
    logit_bias: np.ndarray  # Shape: (n_classes,)
    tau_low: float  # Low risk threshold for Class 2
    tau_high: float  # High risk threshold for Class 2
    ece_before: float
    ece_after: float


class MammographyCalibrator:
    """Manages post-hoc logit calibration and clinical risk stratification."""

    def __init__(self, clip_min: float = 0.01, clip_max: float = 0.99):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.site_calibrators: Dict[str, CalibratorState] = {}
        self.global_calibrator: CalibratorState | None = None

    @staticmethod
    def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE) across confidence bins."""
        preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        accuracies = (preds == labels)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_samples = len(labels)
        if n_samples == 0:
            return 0.0

        for i in range(n_bins):
            b_low = bin_boundaries[i]
            b_high = bin_boundaries[i + 1]
            in_bin = (confidences > b_low) & (confidences <= b_high)
            n_in_bin = np.sum(in_bin)
            if n_in_bin > 0:
                avg_conf = float(np.mean(confidences[in_bin]))
                avg_acc = float(np.mean(accuracies[in_bin]))
                ece += (n_in_bin / n_samples) * abs(avg_acc - avg_conf)

        return float(ece)

    def _clip_and_normalize(self, probs: np.ndarray) -> np.ndarray:
        """Clip probabilities to [1%, 99%] and re-normalize across classes."""
        clipped = np.clip(probs, self.clip_min, self.clip_max)
        sums = np.sum(clipped, axis=1, keepdims=True)
        return clipped / sums

    def fit_calibrator(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        site: str = "global",
        fit_bias: bool = True,
        target_sens: float = 0.95,
        target_spec: float = 0.90,
    ) -> CalibratorState:
        """Fit temperature and class bias on validation logits to minimize negative log-likelihood."""
        n_samples, n_classes = val_logits.shape
        raw_probs = softmax(val_logits, axis=1)
        ece_before = self.compute_ece(raw_probs, val_labels)

        # Objective: NLL loss with temperature scaling and optional logit bias
        def nll_objective(params):
            temp = max(1e-3, float(params[0]))
            bias = params[1:] if fit_bias else np.zeros(n_classes)
            scaled = (val_logits / temp) + bias
            p = softmax(scaled, axis=1)
            # Clip for numerical stability in log
            eps = 1e-12
            p_correct = p[np.arange(n_samples), val_labels]
            return -np.mean(np.log(np.clip(p_correct, eps, 1.0)))

        init_params = [1.0]
        if fit_bias:
            init_params.extend([0.0] * n_classes)

        bounds = [(0.05, 10.0)]
        if fit_bias:
            bounds.extend([(-5.0, 5.0)] * n_classes)

        res = minimize(
            nll_objective,
            x0=init_params,
            method="L-BFGS-B",
            bounds=bounds,
        )

        opt_temp = float(res.x[0])
        opt_bias = np.array(res.x[1:]) if fit_bias else np.zeros(n_classes)

        # Calibrated validation probabilities
        cal_logits = (val_logits / opt_temp) + opt_bias
        cal_probs = self._clip_and_normalize(softmax(cal_logits, axis=1))
        ece_after = self.compute_ece(cal_probs, val_labels)

        # Calibrate risk thresholds on Class 2 (Suspicious/Malignant) probabilities
        p_class2 = cal_probs[:, 2] if n_classes > 2 else cal_probs[:, -1]
        y_binary = (val_labels == (2 if n_classes > 2 else val_labels.max())).astype(int)

        tau_low = 0.15
        tau_high = 0.60
        if len(np.unique(y_binary)) > 1:
            fpr, tpr, thresholds = roc_curve(y_binary, p_class2)
            # tau_low: threshold achieving >= target_sens (e.g. 95% sensitivity)
            sens_idx = np.where(tpr >= target_sens)[0]
            if len(sens_idx) > 0:
                tau_low = float(thresholds[sens_idx[0]])
            # tau_high: threshold achieving >= target_spec (e.g. 90% specificity, smallest FPR)
            spec = 1.0 - fpr
            spec_idx = np.where(spec >= target_spec)[0]
            if len(spec_idx) > 0:
                tau_high = float(thresholds[spec_idx[-1]])

        # Clamp thresholds into plausible range
        tau_low = float(np.clip(tau_low, 0.05, 0.35))
        tau_high = float(np.clip(tau_high, max(tau_low + 0.1, 0.40), 0.85))

        state = CalibratorState(
            site=site,
            temperature=opt_temp,
            logit_bias=opt_bias,
            tau_low=tau_low,
            tau_high=tau_high,
            ece_before=ece_before,
            ece_after=ece_after,
        )

        if site == "global":
            self.global_calibrator = state
        else:
            self.site_calibrators[site] = state

        return state

    def calibrate(self, logits: np.ndarray, site: str | None = None) -> Tuple[np.ndarray, List[str]]:
        """Apply calibration and return clipped probabilities with risk bands."""
        calibrator = self.site_calibrators.get(site, self.global_calibrator)
        if calibrator is None:
            # Fallback to uncalibrated
            probs = self._clip_and_normalize(softmax(logits, axis=1))
            tau_l, tau_h = 0.15, 0.60
        else:
            scaled = (logits / calibrator.temperature) + calibrator.logit_bias
            probs = self._clip_and_normalize(softmax(scaled, axis=1))
            tau_l, tau_h = calibrator.tau_low, calibrator.tau_high

        # Risk band for Class 2
        p_c2 = probs[:, 2] if probs.shape[1] > 2 else probs[:, -1]
        risk_bands = []
        for p in p_c2:
            if p <= tau_l:
                risk_bands.append("low")
            elif p >= tau_h:
                risk_bands.append("high")
            else:
                risk_bands.append("intermediate")

        return probs, risk_bands

