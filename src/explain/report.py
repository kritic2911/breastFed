"""Deterministic, rule-based structured report generator for explainable mammography findings.

Adheres strictly to AGENTS.md Phase P3b:
  - Template report: fixed sentences filled strictly from quantified measurements.
  - Zero free-text / LLM generation.
  - Explicitly reports risk band, anatomical quadrant, cell heat mass, and lesion area fraction.
  - Notes resolution limitations for microcalcifications at 512 px.
"""

from typing import List

from src.explain.gradcam import LocalizationResult
from src.explain.grid import BreastGrid


def generate_structured_report(
    site: str,
    patient_hash: str,
    laterality: str,
    view: str,
    calibrated_probs: List[float],
    risk_band: str,
    grid: BreastGrid,
    loc_res: LocalizationResult,
    available_classes: List[int] | None = None,
) -> str:
    """Generate deterministic structured findings report from calibrated predictions and localization."""
    # Class names mapping
    is_birads = (site in ["vindr", "inbreast"])
    class_names = {
        0: "Normal (BI-RADS 1)" if is_birads else "Normal (Negative)",
        1: "Benign (BI-RADS 2-3)" if is_birads else "Biopsy-proven Benign",
        2: "Suspicious (BI-RADS 4-5)" if is_birads else "Biopsy-proven Malignant",
    }

    avail = available_classes if available_classes is not None else [0, 1, 2]
    # Argmax restricted to site available classes
    avail_probs = [calibrated_probs[c] for c in avail]
    best_c = avail[int(avail_probs.index(max(avail_probs)))]
    pred_label_str = class_names[best_c]

    p_norm = calibrated_probs[0] if 0 in avail else 0.0
    p_benign = calibrated_probs[1] if 1 in avail else 0.0
    p_susp = calibrated_probs[2] if 2 in avail else 0.0

    # Anatomical description
    peak_cell = loc_res.peak_cell_name
    dominant_cell = loc_res.dominant_cell_name
    dom_pct = loc_res.dominant_cell_fraction * 100.0
    area_pct = loc_res.highlighted_area_pct_breast
    tissue_pct = grid.breast_fraction_of_image * 100.0

    lines = [
        f"MAMMOGRAPHY EXPLAINABILITY REPORT: {laterality} {view} VIEW",
        "=" * 58,
        f"Patient ID (Salted Hash): {patient_hash[:12]}...",
        f"Clinical Site / Scanner:  {site.upper()} ({'BI-RADS' if is_birads else 'Biopsy'} label family)",
        f"Anatomical View:          {laterality} Breast, {view} projection",
        "-" * 58,
        "PREDICTION & RISK ASSESSMENT:",
        f"  - Primary Finding:      {pred_label_str}",
        f"  - Calibrated Risk Band: {risk_band.upper()} RISK (evaluated on validation cohort)",
        f"  - Model Probabilities:  Normal: {p_norm*100:.1f}%, Benign: {p_benign*100:.1f}%, Suspicious/Malignant: {p_susp*100:.1f}%",
        "-" * 58,
        "ANATOMICAL LOCALIZATION & HEAT ATTRIBUTION (3x3 Grid):",
        f"  - Peak Activation Cell: {peak_cell.replace('-', ' ').title()}",
        f"  - Dominant Heat Mass:   {dominant_cell.replace('-', ' ').title()} ({dom_pct:.1f}% of total activation)",
        f"  - Highlighted Area:     {area_pct:.1f}% of breast tissue (total tissue occupies {tissue_pct:.1f}% of frame)",
    ]

    # Detailed 3x3 breakdown
    lines.append("  - Spatial Cell Breakdown:")
    for cell in grid.cells:
        frac = loc_res.cell_heat_fractions.get(cell.cell_name, 0.0) * 100.0
        lines.append(f"      * {cell.cell_name.replace('-', ' ').title():20s}: {frac:5.1f}% heat mass | {cell.fraction_of_breast*100:4.1f}% tissue area")

    lines.extend([
        "-" * 58,
        "CLINICAL NOTE & RESOLUTION LIMITATIONS:",
        "  - Feature maps extracted at 512x512 resolution (layer4 receptive field: 16x16 grid).",
        "  - Coarse resolution may under-resolve subtle clustered microcalcifications.",
        "  - Simulated research model outputs; findings require full-resolution radiological review.",
        "=" * 58,
    ])

    return "\n".join(lines)
