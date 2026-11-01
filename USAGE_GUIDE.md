# FL-Mammo: Pipeline Usage Guide & Operations Manual

This guide documents how to run the cross-site mammography benchmark, how to interpret experimental results, and the clinical/technical limitations of the system.

---

## 1. System Overview & Experimental Controls

FL-Mammo evaluates collaborative breast-cancer detection across four heterogeneous mammography datasets representing simulated clinical hospital sites:

| Site | Modality | Ground Truth Source | Labeled Images | Patients | Malignant Prevalence |
|---|---|---|---|---|---|
| **CMMD** | Digital (FFDM) | Pathology diagnosis | 3,702 | 1,757 | ~70% |
| **CBIS-DDSM** | Digitized Film | Pathology diagnosis | 770 | 461 | ~40% |
| **BCDR** | Film & Digital | Pathology diagnosis | 485 | 193 | ~15% |
| **INbreast** | Digital (FFDM) | BI-RADS assessment | 343 | 99 | ~29% |

### Core Methodological Controls
1. **Model Backbone**: ResNet-18 initialized from ImageNet weights (`IMAGENET1K_V1`).
2. **Frozen BatchNorm**: All BatchNorm running statistics and affine parameters remain frozen at ImageNet defaults to eliminate cross-scanner batch-norm feature contamination.
3. **Constant Learning Rate**: Fixed Adam optimizer learning rate $1.0 \times 10^{-4}$ (`constant_lr: true`) with warmup disabled across all paradigms to eliminate warmup spikes and LR schedule divergence.
4. **Validation-Derived Thresholds**: Operating thresholds are strictly selected on the validation set at target 90% specificity ($FPR \le 0.10$). Test sets are touched exactly once per final model checkpoint.
5. **Evaluation Seeds**: Strictly evaluated across three independent seeds: `[42, 53, 64]`.
6. **Patient-Level Bootstrap**: 1,000 bootstrap iterations resampled strictly by `patient_id` to compute 95% Confidence Intervals $[2.5\%, 97.5\%]$.
7. **Rule 3 Run Provenance**: Every run saves `resolved_config.yaml`, `config_hash`, `git_sha`, `seed`, `environment`, `logs.jsonl` (schema: `ts, run_id, event, round, site, split, metric, value`), `metrics.json`, and unified `preds.csv`.
8. **Append-Only Ledger**: All experiment metrics are appended to `results/ledger.csv`.

---

## 2. Environment Activation & Verification

The project runs on Python 3.12 with PyTorch and CUDA inside WSL2:

```bash
# Verify unit tests and aggregation identity (runs in sandbox)
make test

# Run synthetic smoke pipeline (CPU, 2 rounds, under 30s)
make smoke

# Verify code formatting and linting
make lint
```

---

## 3. How to Run the System (Phase 1 Baseline Repair)

### 3.1 CBIS Resolution Test (224 px vs 384 px)
Compares single-image classification performance between standard downsampled resolution (224 px) and high resolution (384 px) on CBIS-DDSM to test whether higher resolution captures microcalcifications beyond the 95% CI.

```bash
.venv/bin/python scripts/run_resolution_test.py --seeds 42 53 64 --resolutions 224 384 --device cuda
```

### 3.2 Local-Only Baseline
Trains independent models on each individual site and evaluates each model on its own test set as well as cross-site across all other sites.

```bash
# Run individual site across all three seeds
.venv/bin/python scripts/run_local.py --site CMMD --seeds 42 53 64 --device cuda
.venv/bin/python scripts/run_local.py --site CBIS --seeds 42 53 64 --device cuda
.venv/bin/python scripts/run_local.py --site BCDR --seeds 42 53 64 --device cuda
.venv/bin/python scripts/run_local.py --site INbreast --seeds 42 53 64 --device cuda

# Or run all sites in sequence:
.venv/bin/python scripts/run_local.py --all-sites --seeds 42 53 64 --device cuda
```
*Note on Small Sites*: For BCDR and INbreast, training uses a fixed 20 epochs (`small_site_epochs: 20`) without early stopping to prevent premature termination on noisy validation subsets (~40 images).

### 3.3 Centralized Baseline (Pooled & Site-Balanced)
Pools training data across all 4 sites to establish upper-bound performance benchmarks.

```bash
# Standard sample-weighted pooling across all sites
.venv/bin/python scripts/run_centralized.py --all-sites --seeds 42 53 64 --device cuda

# Site-balanced sampling (uniform representation per site per batch)
.venv/bin/python scripts/run_centralized.py --all-sites --seeds 42 53 64 --site-balanced --device cuda
```

### 3.4 Federated Baseline (FedAvg)
Simulates collaborative multi-site training with frozen BatchNorm and sample-weighted aggregation across 35 rounds (2 local epochs per round).

```bash
.venv/bin/python scripts/run_federated.py --seeds 42 53 64 --device cuda
```

### 3.5 Results Aggregation & Ledger Generation
Aggregates ledger rows across seeds, calculates mean $\pm$ 95% Student-$t$ confidence intervals, and formats summary tables.

```bash
.venv/bin/python scripts/aggregate_results.py --ledger results/ledger.csv --phase phase_1 --out-csv results/phase1_summary_table.csv
```

---

## 4. How to Interpret the Results

### 4.1 Primary Metric: Area Under the ROC Curve (Test AUC)
- **Point Estimate**: Reported as the mean test AUC over the three evaluation seeds (`42, 53, 64`).
- **Confidence Intervals**: 
  - Per-run: 95% patient-level bootstrap CI $[2.5\%, 97.5\%]$ accounting for clustering within patients.
  - Across-seed: 95% Student-$t$ CI calculated via `scripts/aggregate_results.py`.
- **Claim Rule (Rule 5)**: We only claim that method $A$ improves over method $B$ if the **95% CI of the paired difference excludes zero**. If the confidence intervals overlap, results must be reported as inconclusive.

### 4.2 Operating Sensitivity at 90% Specificity (`sens_at_90_spec`)
- Measures clinically actionable sensitivity when false positive rate is constrained to $\le 10\%$.
- Threshold is chosen strictly on the validation split and frozen before evaluating test images.

### 4.3 Model Health & Collapse Detection (`prob_std`)
- Every evaluation and training round logs the standard deviation of predicted probabilities: `fl_prob_std`.
- **Healthy Range**: $\text{std}(p) \ge 0.05$.
- **Collapse Red Flag**: $\text{std}(p) < 0.01$ indicates the model has collapsed to outputting constant site prevalence, failing to distinguish lesions from background.

### 4.4 Calibration & Expected Calibration Error (ECE)
- Uncalibrated neural networks often produce overconfident probabilities.
- We report **ECE** before and after temperature scaling:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
- Calibration parameter $T$ is fitted strictly on validation logits; test probabilities are then scaled via $p = \sigma(z / T)$. A successful calibration reduces ECE on at least 3 of 4 sites.

---

## 5. System Limitations & Clinical Nuance

1. **Label Provenance Heterogeneity (Rule 7)**:
   - **Pathology Confirmation**: CMMD, CBIS-DDSM, and BCDR ground truth labels represent histopathological biopsy confirmation (benign vs malignant tissue).
   - **Radiological Assessment**: INbreast labels are derived from BI-RADS radiologist visual assessment (BI-RADS 1-3 classified as benign, 4-6 classified as malignant).
   - *Limitation*: Never refer to INbreast predictions as a "biopsy diagnosis"; it reflects suspicious radiologist assessment.
2. **Sample Size & Wide Confidence Intervals**:
   - Small sites have limited test patients (BCDR: 38 test patients / ~50 images; INbreast: 20 test patients / ~35 images).
   - Resulting CI half-widths are $\pm 0.16$ to $\pm 0.18$. Apparent large point-estimate swings are often within random noise; statistical conclusions must be bounded by CIs.
3. **Spatial Resolution Constraints**:
   - Whole-mammogram images ($2000 \times 3000$ to $4000 \times 5000$ original pixels) are downsampled to $224 \times 224$ (or $384 \times 384$).
   - Subtle microcalcification clusters (prominent in CBIS-DDSM) lose sub-millimeter morphology at 224 px. Expect full-image CBIS AUCs in the 0.55–0.68 range; near 0.60 is expected and does not represent a software bug.
4. **Simulated Federated Environment**:
   - Federated learning is executed sequentially client-by-client on a single NVIDIA GeForce RTX 2050 (4 GB VRAM) GPU under WSL2.
   - Network communication, socket latency, and asynchronous TLS transport are simulated in Phase 1-3, and containerized in Flower/Docker in Phase 4-5.
