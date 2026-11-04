# Cross-Silo Federated Learning for Explainable Mammography
## Progress Update to Faculty Advisor | 04 November 2026

**Student:** BTP Project Team  
**Sprint Window:** 02 Nov 2026 – 20 Nov 2026 (18-day hard freeze)  
**Current Status:** Day 2.5 of 18 — **Ahead of Schedule** (Phases P0–P3b Completed)

---

## 1. Executive Status at a Glance

| Phase | Milestone Description | Target Window | Status | Key Deliverable / Metric |
|---|---|---|---|---|
| **P0** | Environment, Safety Rails & GPU Calibration | Day 0–1 | **PASS** | RTX 2050 (4 GB) benchmarked at **86.7 img/s**; batch 16 locked |
| **P1** | Data Audit, Manifest & Preprocessing Cache | Day 1–3 | **PASS (Gate 1)** | 6,406 images across 2,481 patients cached at 512×512 |
| **P2** | Local Baselines & Centralized Upper Bound | Day 3–6 | **PASS (Gate 2)** | CMMD AUC **0.804** ($\ge 0.75$ gate); INbreast +9.8% in centralized |
| **P3** | Federated Learning Simulation | Day 6–9 | **PASS** | Uniform FedAvg achieves **0.8577 global AUC** (beats centralized) |
| **P3b**| Calibration, Quadrant Reports & Localization | Day 9–10 | **PASS** | ECE dropped from 20% to **5%**; 59.4% hit rate on 345 lesions |
| **P4** | Systems Build (Networking, Netem, TLS, SecAgg)| Day 8–12 | **IN PROGRESS** | Multi-client topology, `tc netem` profiles, minimum-cohort guard |

---

## 2. Clinical Data Setup & The Heterogeneity Problem

We evaluate across **4 distinct digital mammography hospital silos** (digitized film sets like CBIS-DDSM excluded per Decision [D-001](file:///home/kc/breastFed/docs/DECISIONS.md#L8-L18) to avoid historical scanning artifacts):

```
                   Hospital Silos (6,406 Images / 2,481 Patients)
   ┌───────────────────────┬────────────────────────┬───────────────────────┬──────────────────────┐
   │      VinDr-Mammo      │          CMMD          │       INbreast        │       CDD-CESM       │
   │ 1,497 imgs (375 pts)  │ 3,534 imgs (1,673 pts) │  409 imgs (108 pts)   │  966 imgs (326 pts)  │
   │ Siemens / Planmed     │ GE Senographe          │ Siemens Novation      │ GE Senographe        │
   │ Family: 'birads'      │ Family: 'biopsy'       │ Family: 'birads'      │ Family: 'biopsy'     │
   │ Classes: {0, 1, 2}    │ Classes: {1, 2}        │ Classes: {0, 1, 2}    │ Classes: {0, 1, 2}   │
   └───────────────────────┴────────────────────────┴───────────────────────┴──────────────────────┘
```

- **3-Class Breast-Level Mapping**: Class 0 = Normal, Class 1 = Benign, Class 2 = Suspicious/Malignant.
- **Visual Proof of Domain Shift**: A simple Site-ID classifier achieved **95.30% accuracy** (vs. 54.59% majority baseline), proving severe scanner and pixel distribution divergence between hospitals.
- **Data Quality**: Zero patient leakage across splits; no view separation; all 9,948 lesion bounding boxes geometrically verified on 512×512 grids.
- [View Site QC Preview](file:///home/kc/breastFed/paper/figures/qc_grids/intensity_histograms.png)

---

## 3. Phase P2: Local & Centralized Baselines (Gate 2 Passed)

All pre-registered Gate 2 criteria were exceeded across $n=3$ seeds (`[42, 53, 64]`):

| Strategy | Cohort / Site | Test Macro-AUC | Malignancy AUC | Sens @ 90% Spec | Macro F1 |
|---|---|---|---|---|---|
| **Local (Isolated)** | CDD-CESM (`biopsy`) | $0.7681 \pm 0.0186$ | $0.8679 \pm 0.0253$ | $0.5185 \pm 0.0425$ | $0.5239$ |
| **Local (Isolated)** | CMMD (`biopsy`) | $0.8040 \pm 0.0164$ | $0.8040 \pm 0.0164$ | $0.5981 \pm 0.0346$ | $0.7018$ |
| **Local (Isolated)** | INbreast (`birads`) | $0.7348 \pm 0.0631$ | $0.8630 \pm 0.0196$ | $0.4815 \pm 0.2313$ | $0.4670$ |
| **Local (Isolated)** | VinDr-Mammo (`birads`)| $0.6790 \pm 0.0336$ | $0.7516 \pm 0.0496$ | $0.3571 \pm 0.0859$ | $0.3969$ |
| **Centralized (Pooled)** | All Sites | $0.8462 \pm 0.0028$ | $0.8689 \pm 0.0063$ | $0.6202 \pm 0.0122$ | $0.6197$ |
| **Centralized (Balanced)**| All Sites | $0.8467 \pm 0.0023$ | $0.8715 \pm 0.0048$ | $0.6121 \pm 0.0150$ | $0.6154$ |
| *Chance Floor (Permutation)*| All Sites | $0.4991 \pm 0.0012$ | $0.5000$ | — | — |
| *Clinical Feature Floor* | Age-Only Model | $0.5985$ | — | — | — |

**Takeaway:** Small hospitals gain dramatically from collaboration: INbreast jumps **+9.8% AUC** (0.735 $\to$ 0.833) and CDD jumps **+10.0% AUC** (0.768 $\to$ 0.868) when pooled with larger cohorts.

---

## 4. Phase P3: Federated Learning — Key Algorithmic Discovery

![FL Macro-AUC Convergence](file:///home/kc/breastFed/paper/figures/fl_convergence_macro_auc.png)

### The Dominance Trap: Uniform vs. Sample-Weighted FedAvg
- **Problem with standard FedAvg**: CMMD has ~12× more images than INbreast. In standard sample-weighted FedAvg ($w_k = \frac{n_k}{N}$), CMMD dictates 65% of every global update, dragging worst-site performance down to 0.694.
- **Solution (Uniform FedAvg, $w_k = \frac{1}{K}$)**: Giving equal voting power to each hospital site dramatically improves generalization across heterogeneous scanners.

```
| Federated Strategy           | Global Macro-AUC     | Worst-Site AUC       | Cross-Site Gap | Communication |
|------------------------------|----------------------|----------------------|----------------|---------------|
| FedAvg (Sample-Weighted)     | 0.8407 ± 0.0029      | 0.7064 ± 0.0212      | 0.1250 ± 0.017 | 7.16 GB (20 r)|
| **FedAvg (Uniform Weighting)**| **0.8577 ± 0.0056**  | **0.7556 ± 0.0094**  | **0.1145 ± 0.012** | 7.16 GB (20 r)|
| FedProx (μ = 0.01)           | 0.8499 (n=1)         | 0.7411               | 0.1383         | 7.16 GB (20 r)|
| FedProx (μ = 0.10)           | 0.8489 (n=1)         | 0.7211               | 0.1410         | 7.16 GB (20 r)|
```
**Takeaway:** Uniform FedAvg outperforms pooled centralized training (**0.8577 vs. 0.8462 AUC**) while keeping raw mammograms entirely on local client hardware.

---

## 5. Experiment E6: Label-Family Heterogeneity & Slicing

Can hospitals with biopsy-proven truth collaborate with screening BI-RADS sites?

![E6 Heterogeneity Chart](file:///home/kc/breastFed/paper/figures/fl_e6_heterogeneity.png)

```
| Coalition Setup                      | Global Macro-AUC | Worst-Site AUC | Key Clinical Observation                |
|--------------------------------------|------------------|----------------|-----------------------------------------|
| **All 4 Hospitals (Joint)**          | **0.8577**       | **0.7556**     | Best overall performance & equity       |
| Biopsy-Only Sites (CMMD + CDD)       | 0.8415           | 0.6967         | High specificity, but misses BI-RADS    |
| BI-RADS-Only Sites (VinDr + INbreast)| 0.7713           | 0.7018         | Lacks biopsy diagnostic gradient        |
| Leave-Out VinDr                      | 0.8390           | 0.7126         | Model loses multi-scanner robustness    |
| Leave-Out CMMD                       | 0.8372           | 0.7399         | Loss of high-volume benign/mal cases    |
```
**Takeaway:** Despite the label semantics discrepancy (biopsied benign vs. unbiopsied BI-RADS 2), cross-family federated training is mutually beneficial.

---

## 6. Phase P3b: Explainability & Automated Clinical Reporting

![Dual-View Explainability](file:///home/kc/breastFed/paper/figures/p3b_dual_view_explainability.png)

### Three-Component Explainability Framework
1. **Validation Temperature Scaling**: Fixes neural network overconfidence. Expected Calibration Error (ECE) dropped from 20.1% $\to$ **5.0%** on VinDr and 25.5% $\to$ **11.5%** on INbreast.
2. **Aspect-Preserving 3×3 Grid**: Quantifies heat mass into standard anatomical quadrants (Lateral/Central/Medial for CC; Upper/Central/Lower for MLO).
3. **Structured Rule-Based Reporting**: Automatically fills deterministic clinical templates (zero LLM hallucination risk).

```
Verbatim Clinical Report Output (VinDr Test Case e3a4d422...):
----------------------------------------------------------------------
FINDING:          Benign (BI-RADS 2-3) | Calibrated Risk: INTERMEDIATE
PROBABILITIES:    Normal: 37.2% | Benign: 42.5% | Suspicious/Malignant: 20.4%
LOCALIZATION:     Peak activation in Central Anterior (71.1% of heat mass)
HIGHLIGHTED AREA: 3.8% of breast tissue (tissue occupies 34.0% of frame)
RESOLUTION NOTE:  512x512 resolution (16x16 layer4 grid); microcalcifications require full-res review.
----------------------------------------------------------------------
```

---

## 7. Localization vs. 345 Ground-Truth Radiologist Annotations

We evaluated Grad-CAM activations against **345 patient images with ground-truth lesion geometry**:

```
| Metric                               | Measured Score | Interpretation                                        |
|--------------------------------------|----------------|-------------------------------------------------------|
| **Overall Peak-in-Lesion Hit Rate**  | **59.42%**     | Max activation lands strictly inside lesion contour   |
| - CDD-CESM (Focal Masses)            | **77.34%**     | High contrast focal lesions accurately localized      |
| - VinDr-Mammo (Finding Boxes)        | **54.30%**     | Good spatial alignment across diverse scanners        |
| - INbreast (Microcalcifications)     | **36.36%**     | Challenged by resolution downsampling                 |
| **Total-Variation (TV) Distance**    | **0.4913**     | 3x3 quadrant distribution divergence from ground truth|
| **Adebayo Parameter Sanity Check**   | **r = 0.1319** | Passes check: heatmaps collapse when weights scrambled|
| **Faithfulness Occlusion Rate**      | **96.88%**     | Occluding peak cell drops malignant score vs. random  |
```

![Multisite Gallery](file:///home/kc/breastFed/paper/figures/p3b_multisite_localization_gallery.png)

---

## 8. Resolution Bottleneck & Pre-Registered Experiment (D-010)

![Mass vs Calc GradCAM](file:///home/kc/breastFed/paper/figures/p3b_mass_vs_calc_gradcam.png)

### The Structural Dilemma: Masses vs. Microcalcifications
- **Focal Masses ($N=524$)**: Peak Hit Rate = **61.02%** | Malignancy AUC = **0.7794**
- **Microcalcifications ($N=236$)**: Peak Hit Rate = **48.18%** | Malignancy AUC = **0.8199**
- *The Paradox*: The model learns strong global textural malignancy features for calcifications (0.82 AUC), but 512 px downsampling blurs $<0.5\,\text{mm}$ specks, reducing peak spatial hit rate.

### Pre-Registered Hypothesis Testing (Decision [D-010](file:///home/kc/breastFed/docs/DECISIONS.md#L131-L188))
We tested if a classical high-frequency morphological filter could sharpen CAM activations ($\text{CAM} \times (\epsilon + \text{Detector})$) with a pre-registered success bar of $\Delta \text{Hit Rate} \ge +5.0\%$:

```
| Method                         | Peak Hit Rate | Mean TV Distance | Outcome vs. Pre-Registered Criteria |
|--------------------------------|---------------|------------------|-------------------------------------|
| Pure Grad-CAM                  | 46.23%        | 0.5136           | Baseline                            |
| Classical Detector Alone       | 29.25%        | 0.6231           | Many false-positive normal specks   |
| Multiplicative Combined Map    | 44.34%        | 0.5139           | FAILED / NULL (-1.89% Δ Hit Rate)   |
```
- **Research Integrity**: Following Rule 5, we report the null result honestly. Morphological filtering picks up benign parenchymal textures that deflect CAM peaks.
- **Silver Lining (The Router)**: While sharpening failed, the detector proved to be an exceptional **Calcification Router** (**F1 = 0.8665**, 100% calc-image detection rate on validation).

---

---

## 10. Roadmap: Remaining Deliverables to Day 18 (Nov 20)

```
[Day 0 - 2.5: COMPLETED]
  ✔ P0: Environment, safety rails, GPU throughput calibration
  ✔ P1: Data audit, manifest, 512px cache, site-ID domain probe
  ✔ P2: Local & centralized baselines (all Gate 2 targets passed)
  ✔ P3: FL simulation (FedAvg uniform vs weighted, FedProx, E6)
  ✔ P3b: Explainability, calibration, 3x3 grid, mass vs calc deep-dive

[Day 3 - 6: CURRENT FOCUS - Phase P4 Systems Build]
  ▶ Multi-client network execution (Docker / multi-process topology)
  ▶ Mutual TLS certificate authentication
  ▶ Network emulation via 'tc netem' (Metro: 5ms vs. District: 120ms/35ms jitter)
  ▶ Failure injection & Minimum-Cohort Guard (k)

[Day 7 - 10: Phase P5 Systems Experiments]
  ▶ Measure round completion time, network bytes, and dropout aborts
  ▶ Privacy-reliability frontier plot (k vs. time-to-target)

[Day 11 - 14: Phase P6 Privacy & Phase P7 Artifacts]
  ▶ Loss-threshold membership inference attack
  ▶ Automated paper tables & vector PDF generation (make paper-artifacts)

[Day 15 - 18: Final Delivery & Buffer]
  ▶ Presentation slides, demonstration rehearsal, final thesis submission
```

---

### Reference Artifact Index
- **Full Architecture & Run Guide**: [README.md](file:///home/kc/breastFed/README.md)
- **Immutable Decision Records**: [docs/DECISIONS.md](file:///home/kc/breastFed/docs/DECISIONS.md)
- **Detailed Step-by-Step Progress Log**: [docs/PROGRESS.md](file:///home/kc/breastFed/docs/PROGRESS.md)
- **Gate 2 Baseline Table**: [paper/tables/p2_results_compiled.json](file:///home/kc/breastFed/paper/tables/p2_results_compiled.json)
- **Federated Uniform FedAvg Metrics**: [paper/tables/fl_fedavg_uniform_results.json](file:///home/kc/breastFed/paper/tables/fl_fedavg_uniform_results.json)
- **Quantitative Localization Metrics**: [paper/tables/p3b_localization_metrics.json](file:///home/kc/breastFed/paper/tables/p3b_localization_metrics.json)
