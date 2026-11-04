"""Unit tests for Phase P3b explainability, calibration, and anatomical grid partitioning."""

import numpy as np
import pytest
import torch

from src.explain.calibration import MammographyCalibrator
from src.explain.eval_localization import (
    ImageAnnotation,
    check_peak_inside_lesion,
    compute_total_variation_distance,
)
from src.explain.gradcam import GradCAM
from src.explain.grid import fit_anatomical_grid
from src.explain.report import generate_structured_report
from src.models.backbone import build_model


def test_segment_and_grid_fitting():
    """Verify tissue segmentation and 3x3 anatomical grid properties on synthetic image."""
    img = np.zeros((512, 512), dtype=np.float32)
    # Synthetic breast ellipse
    y, x = np.ogrid[:512, :512]
    mask = ((y - 256) ** 2 / 180 ** 2) + ((x - 100) ** 2 / 200 ** 2) <= 1.0
    img[mask] = 0.5

    grid_mlo = fit_anatomical_grid(img, view="MLO")
    assert len(grid_mlo.cells) == 9
    assert grid_mlo.total_tissue_pixels > 0
    assert 0.0 < grid_mlo.breast_fraction_of_image <= 1.0

    # Verify cell coverage and naming
    cell_names = [c.cell_name for c in grid_mlo.cells]
    assert "upper-posterior" in cell_names
    assert "central-middle" in cell_names
    assert "lower-anterior" in cell_names

    total_frac = sum(c.fraction_of_breast for c in grid_mlo.cells)
    assert pytest.approx(total_frac, abs=1e-3) == 1.0

    # CC view naming check
    grid_cc = fit_anatomical_grid(img, view="CC")
    cc_names = [c.cell_name for c in grid_cc.cells]
    assert "lateral-posterior" in cc_names
    assert "medial-anterior" in cc_names


def test_calibrator_optimization():
    """Verify temperature scaling reduces NLL and ECE, clipping bounds honored."""
    np.random.seed(42)
    n_samples = 200
    # Synthetic uncalibrated overconfident logits
    logits = np.random.randn(n_samples, 3) * 3.0
    labels = np.random.choice([0, 1, 2], size=n_samples)

    calibrator = MammographyCalibrator(clip_min=0.01, clip_max=0.99)
    state = calibrator.fit_calibrator(logits, labels, site="test_site")

    assert state.temperature > 0.0
    assert 0.0 <= state.ece_after <= 1.0

    test_logits = np.random.randn(10, 3)
    probs, risk_bands = calibrator.calibrate(test_logits, site="test_site")

    assert probs.shape == (10, 3)
    assert np.all(probs >= 0.01)
    assert np.all(probs <= 0.99)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    assert len(risk_bands) == 10
    for r in risk_bands:
        assert r in ["low", "intermediate", "high"]


def test_gradcam_and_report_generation():
    """Verify Grad-CAM forward/backward hooks, heatmap extraction, and report text generation."""
    model = build_model(num_classes=3, pretrained=False)
    model.eval()

    gradcam = GradCAM(model)

    # Synthetic image tensor (1, 3, 512, 512)
    img_t = torch.ones((1, 3, 512, 512), dtype=torch.float32) * 0.2
    grid = fit_anatomical_grid(img_t[0, 0].numpy(), view="MLO")

    loc_res = gradcam.analyze_localization(img_t, grid, class_idx=2)

    assert loc_res.heatmap.shape == (512, 512)
    assert 0.0 <= np.min(loc_res.heatmap)
    assert np.max(loc_res.heatmap) <= 1.0
    assert len(loc_res.cell_heat_fractions) == 9
    assert loc_res.peak_cell_name in loc_res.cell_heat_fractions
    assert 0.0 <= loc_res.highlighted_area_pct_breast <= 100.0

    # Test report generator
    report = generate_structured_report(
        site="vindr",
        patient_hash="test_patient_hash_12345",
        laterality="R",
        view="MLO",
        calibrated_probs=[0.10, 0.25, 0.65],
        risk_band="high",
        grid=grid,
        loc_res=loc_res,
    )

    assert "MAMMOGRAPHY EXPLAINABILITY REPORT" in report
    assert "VINDR" in report
    assert "HIGH RISK" in report
    assert "Peak Activation Cell" in report
    assert "CLINICAL NOTE & RESOLUTION LIMITATIONS" in report


def test_evaluation_metrics():
    """Verify peak-in-lesion detection and total variation distance."""
    annot = ImageAnnotation(
        image_id="img_123",
        boxes=[(100.0, 100.0, 200.0, 200.0)],
        finding_types=["mass"],
    )

    # Inside box
    assert check_peak_inside_lesion((150, 150), annot, margin=0.0)
    # Outside box
    assert not check_peak_inside_lesion((300, 300), annot, margin=0.0)
    # Near margin
    assert check_peak_inside_lesion((205, 150), annot, margin=10.0)

    # TV distance identical
    p = {"cell_1": 0.5, "cell_2": 0.5}
    assert compute_total_variation_distance(p, p) == 0.0

    # TV distance completely disjoint
    q = {"cell_1": 1.0, "cell_2": 0.0}
    r = {"cell_1": 0.0, "cell_2": 1.0}
    assert compute_total_variation_distance(q, r) == 1.0
