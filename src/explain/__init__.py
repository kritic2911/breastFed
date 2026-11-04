"""Explainability module for federated mammography predictions."""

from src.explain.calibration import CalibratorState, MammographyCalibrator
from src.explain.eval_localization import (
    ImageAnnotation,
    check_peak_inside_lesion,
    compute_annotation_cell_distribution,
    compute_total_variation_distance,
    parse_image_annotations,
    run_adebayo_sanity_check,
    run_faithfulness_test,
)
from src.explain.gradcam import GradCAM, LocalizationResult
from src.explain.grid import BreastGrid, GridCell, fit_anatomical_grid, segment_breast_tissue
from src.explain.report import generate_structured_report

__all__ = [
    "CalibratorState",
    "MammographyCalibrator",
    "GridCell",
    "BreastGrid",
    "segment_breast_tissue",
    "fit_anatomical_grid",
    "GradCAM",
    "LocalizationResult",
    "generate_structured_report",
    "ImageAnnotation",
    "parse_image_annotations",
    "check_peak_inside_lesion",
    "compute_annotation_cell_distribution",
    "compute_total_variation_distance",
    "run_adebayo_sanity_check",
    "run_faithfulness_test",
]

