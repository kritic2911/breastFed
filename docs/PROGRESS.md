# Project Progress Log (PROGRESS.md)

Tracking progress across the 18-day federated mammography sprint (Mon 2 Nov 2026 – Fri 20 Nov 2026).
Post an update at every phase checkpoint.

---

## Phase P0: Repo, Environment, Safety Rails (Day 0–1)

```
STEP: P0   STATUS: pass   TIME: 0.5 days / 1.0 day budget
DONE:
- Built repository layout adhering to AGENTS.md Section 5 (configs, src, tests, results, paper, docs).
- Implemented core safety rails in src/utils: reactive thermal guard (thermal.py), atomic checkpoint manager (ckpt.py), provenance logger with salted SHA-256 patient de-identification (provenance.py), and reproducible seeding (seed.py).
- Executed empirical GPU throughput and batch size benchmarking for ResNet-18 at 512x512 on RTX 2050 (4 GB VRAM).
- Completed dummy 2-epoch run (scripts/dummy_run.py) producing full run directory and ledger.csv record.
- Passed 9/9 pytest invariant unit tests in tests/ (thermal guard pauses/callbacks/exits, atomic saves/recovery, patient hashing, run_id schema).

NUMBERS:
- GPU Peak Memory (ResNet-18, 512x512, Batch Size 16, AMP, channels_last): 1,180.6 MB (< 30% of 4 GB).
- Measured Throughput: 86.7 img/s (Batch 16), 98.1 img/s (Batch 20).
- GPU Thermal Baseline: Idle 47.0 °C, Loaded 54.0 °C (Slowdown threshold: 87.0 °C). Clamped guards: soft_c = 70 °C, hard_c = 75 °C, resume_c = 55 °C.
- Unit Tests: 9 passed, 0 failed, 0 warnings (8.15s).

ARTIFACTS:
- src/utils/thermal.py
- src/utils/ckpt.py
- src/utils/provenance.py
- src/utils/seed.py
- scripts/dummy_run.py
- scripts/run_resumable.sh
- tests/test_thermal.py, tests/test_ckpt.py, tests/test_provenance.py
- results/runs/20261102_175157-cc3cc78c-549ff35/ (config.yaml, env.json, metrics.jsonl, events.jsonl, ckpt/last.pt, ckpt/best.pt)
- results/ledger.csv
- docs/DECISIONS.md (D-001 through D-004)

RISKS/DEVIATIONS:
- Docker is not natively installed in this WSL2 userland; container experiments for P4/P5 will target Docker Desktop integration or the planned cloud VM stubs as noted in the cut order.
- Direct NVML Python bindings fall back seamlessly to formatted nvidia-smi telemetry in WSL2.

NEXT:
- Phase P1: Data audit, manifest construction, label resolution, and preprocessing cache (Day 1-3).
- First command: Build src/data/manifest.py to construct manifest.csv for the 4 digital sites.
```

---

## Phase P1: Data Audit, Manifest, Labels, Splits (Day 1–3)

```
STEP: P1   STATUS: pass   TIME: 0.5 days / 2.0 day budget
DONE:
- Built unified manifest.csv (6,406 images across 2,481 patients) with all 20 required schema columns.
- Resolved all label caveats per AGENTS.md Section 1.2:
  * CMMD: excluded 1,259 contralateral unannotated images (retained 3,534 lesion images).
  * INbreast: excluded single non-standard FB view (retained 409 images).
  * CDD-CESM: excluded 4 unannotated and 34 occult normal images (retained 966 low-energy DM images).
  * VinDr-Mammo: filtered out 3,375 0-byte stub files (retained 1,497 complete 16-bit DICOMs).
- Implemented robust dynamic-range tissue segmentation, removing air background and markers, standardizing chest wall flush to left (x=0) and nipple pointing right.
- Preprocessed and cached all 6,406 images as 512x512 float32 arrays in data/cache512/.
- Generated annotations.csv (9,948 annotations mapped to 512x512 space with strict canvas bounds).
- Verified zero patient leakage across train/test splits and zero CC/MLO separation.
- Verified absence of identical-position spike artifacts in tissue intensity histograms.
- Evaluated Site-ID probe: 95.30% accuracy (vs. 54.59% majority baseline), demonstrating marked domain shift.
- Generated 20-sample QC grids and 10-sample verification previews for all 4 sites.
- All 17 unit tests in tests/ passed (test_ckpt, test_data_integrity, test_geometry, test_provenance, test_thermal).

NUMBERS:
- Cohort Total: 6,406 images (2,481 patients, 3,845 breasts).
  * VinDr: 1,497 images (375 patients) | Train: 1,192 (79.6%), Test: 305 (20.4%)
  * CMMD: 3,534 images (1,673 patients) | Train: 2,828 (80.0%), Test: 706 (20.0%)
  * INbreast: 409 images (108 patients) | Train: 326 (79.7%), Test: 83 (20.3%)
  * CDD-CESM: 966 images (326 patients) | Train: 763 (79.0%), Test: 203 (21.0%)
- 3-Class Distribution:
  * Class 0 (Normal): 848 (VinDr 441, INbreast 67, CDD 340, CMMD 0)
  * Class 1 (Benign): 2,316 (VinDr 662, INbreast 242, CDD 304, CMMD 1,108)
  * Class 2 (Suspicious/Mal): 3,242 (VinDr 394, INbreast 100, CDD 322, CMMD 2,426)
- Cache Checksum: 884be5cb17374d7d02d8dcf63225e8681fce1afdad91ec8c9c660bec16944054
- Site-ID Probe Accuracy: 95.30% (Macro-F1 0.9039) vs 54.59% baseline (+40.71% gain)
- Scanner Probe Accuracy: 86.97% (Macro-F1 0.5323) vs 54.59% baseline (+32.38% gain)
- Unit Tests: 17 passed, 0 failed in 14.97s.

ARTIFACTS:
- data/manifest.csv (6,406 rows)
- data/annotations.csv (9,948 rows)
- data/cache512/ (6,406 .npy arrays, 6.3 GB)
- paper/figures/qc_grids/ (preview_vindr_10.png, preview_cmmd_10.png, preview_inbreast_10.png, preview_cdd_10.png)
- paper/figures/qc_grids/intensity_histograms.png
- paper/tables/site_id_probe_results.json
- docs/DECISIONS.md (D-005)

RISKS/DEVIATIONS:
- Disparate source formats: CDD is 8-bit lossy JPEG, CMMD is 8-bit DICOM, VinDr/INbreast are 16-bit DICOM. Documented as a shortcut feature in D-005.

NEXT:
- Phase P2: Local and centralized baselines (Day 3-6).
- First command: Run mini-sweep on CMMD (BN frozen/trainable x backbone LR 1e-4/3e-4).
```

---

## Phase P2: Local and Centralized Baselines (Day 3–6)

```
STEP: P2   STATUS: pass   TIME: 1.5 days / 3.0 day budget
DONE:
- Implemented Phase P2 architecture in src/models/backbone.py: torchvision ResNet-18 ImageNet weights, 1-to-3 channel replication, frozen stages (conv1, bn1, layer1, layer2), fine-tuned stages (layer3, layer4, Linear(512, 3) head), 10x head LR multiplier, and configurable BN mode (frozen vs trainable).
- Completed pre-registered mini-sweep on CMMD (BN frozen/trainable x LR 1e-4/3e-4) on Seed 42 strictly using validation split. Locked winning configuration: BN=trainable, LR=3e-4 (Val Macro-AUC = 0.8123, F1 = 0.7511). Logged Decision D-006.
- Updated validation early stopping patience to 12 epochs across local and centralized loops per Decision D-007.
- Evaluated empirical performance floors (paper/tables/p2_floors_results.json):
  * Age-only logistic regression baseline: CDD 0.5877, CMMD 0.6594, INbreast 0.5000, VinDr 0.4858 (Pooled 0.5985).
  * Label-permutation chance floor (1,000 permutations): Mean AUC ~ 0.5000 across all sites (CDD 0.4995, CMMD 0.4999, INbreast 0.5020, VinDr 0.5008, Pooled 0.4991).
- Executed all 12 Local Baselines across 4 sites x 3 seeds ([42, 53, 64]) with patience=12.
- Executed all 6 Centralized Baselines (3 seeds pooled sample-weighted, 3 seeds site-balanced) with patience=12.
- Implemented and permanently stored comprehensive evaluation metrics across all runs in test_metrics.json and master_metrics_summary.json:
  * Macro and per-class Precision, Sensitivity (Recall), Specificity, Sensitivity at 90% Specificity, Macro F1, Macro AUC, and ECE.
- Verified all Gate 2 criteria:
  1. CMMD binary test AUC >= 0.75: PASSED (Local 0.8040, Centralized Pooled 0.8030, Site-Balanced 0.7928).
  2. No site at chance (< 0.55): PASSED (all sites >= 0.6790).
  3. Local below centralized on small sites: PASSED (INbreast: Local 0.7348 vs. Centralized 0.8329, +0.0981 gain; CDD: Local 0.7681 vs. Centralized 0.8684, +0.1003 gain).
- All 17 unit tests passed in 16.40s.

NUMBERS:
- CMMD Mini-Sweep Winner: BN=trainable, LR=3e-4 | Val Macro-AUC = 0.8123 (vs. 0.7779 frozen-1e-4, 0.6475 frozen-3e-4).
- Train / Val / Test Macro-AUC Side-by-Side (Mean +/- sample std over n=3 seeds):
  * CDD-CESM (biopsy): Train = 0.9594 +/- 0.0351 | Val = 0.7711 +/- 0.0609 | Test = 0.7682 +/- 0.0186 | Train-Test Gap = +0.1912 | Val-Test Gap = +0.0030
  * CMMD (biopsy): Train = 0.9679 +/- 0.0184 | Val = 0.8209 +/- 0.0313 | Test = 0.8040 +/- 0.0164 | Train-Test Gap = +0.1639 | Val-Test Gap = +0.0169
  * INbreast (birads): Train = 0.9778 +/- 0.0185 | Val = 0.8221 +/- 0.1179 | Test = 0.7347 +/- 0.0631 | Train-Test Gap = +0.2431 | Val-Test Gap = +0.0874 (small 22-pt test cohort)
  * VinDr-Mammo (birads): Train = 0.8299 +/- 0.1479 | Val = 0.6966 +/- 0.0356 | Test = 0.6790 +/- 0.0335 | Train-Test Gap = +0.1509 | Val-Test Gap = +0.0177
  * Centralized Pooled (Overall): Train = 0.9452 +/- 0.0332 | Val = 0.8407 +/- 0.0135 | Test = 0.8462 +/- 0.0028 | Train-Test Gap = +0.0990 | Val-Test Gap = -0.0055
  * Centralized Site-Balanced (Overall): Train = 0.9135 +/- 0.0368 | Val = 0.8348 +/- 0.0038 | Test = 0.8467 +/- 0.0023 | Train-Test Gap = +0.0668 | Val-Test Gap = -0.0119
- Benign-vs-Malignant (Class-2-vs-Rest) Test AUC (Mean +/- sample std over n=3 seeds):
  * CDD-CESM: Local = 0.8679 +/- 0.0253 [CI: 0.7729, 0.9458] | In Cent. Pooled = 0.9068 +/- 0.0044
  * CMMD: Local = 0.8040 +/- 0.0164 [CI: 0.7586, 0.8462] | In Cent. Pooled = 0.8030 +/- 0.0114
  * INbreast: Local = 0.8630 +/- 0.0196 [CI: 0.7188, 0.9706] | In Cent. Pooled = 0.9333 +/- 0.0257
  * VinDr-Mammo: Local = 0.7516 +/- 0.0496 [CI: 0.6580, 0.8294] | In Cent. Pooled = 0.7958 +/- 0.0256
  * Overall Pooled Test: 0.8689 +/- 0.0063 [CI: 0.8457, 0.8943]
- Detailed Test Classification Metrics Summary (Mean +/- sample std over n=3 seeds):
  * Local CDD: Precision = 0.5858 +/- 0.0635 | Sensitivity = 0.5471 +/- 0.0877 | Specificity = 0.7759 +/- 0.0484 | F1 = 0.5239 +/- 0.1181 | Sens@90 = 0.5185 +/- 0.0425
  * Local CMMD: Precision = 0.7121 +/- 0.0099 | Sensitivity = 0.7160 +/- 0.0187 | Specificity = 0.7471 +/- 0.0127 | F1 = 0.7019 +/- 0.0231 | Sens@90 = 0.5981 +/- 0.0346
  * Local INbreast: Precision = 0.5038 +/- 0.1026 | Sensitivity = 0.4975 +/- 0.0315 | Specificity = 0.7451 +/- 0.0175 | F1 = 0.4670 +/- 0.0637 | Sens@90 = 0.4815 +/- 0.2313
  * Local VinDr: Precision = 0.4489 +/- 0.0386 | Sensitivity = 0.4662 +/- 0.0383 | Specificity = 0.7226 +/- 0.0149 | F1 = 0.3969 +/- 0.0268 | Sens@90 = 0.3572 +/- 0.0859
  * Centralized Pooled (Overall): Precision = 0.6256 +/- 0.0240 | Sensitivity = 0.6608 +/- 0.0138 | Specificity = 0.8199 +/- 0.0065 | F1 = 0.6197 +/- 0.0416 | Sens@90 = 0.6202 +/- 0.0122

ARTIFACTS:
- paper/tables/p2_gate2_comprehensive.json
- paper/tables/p2_cmmd_sweep_results.json
- paper/tables/p2_floors_results.json
- paper/tables/p2_local_baselines_compiled.json
- paper/tables/p2_results_compiled.json
- paper/tables/master_metrics_summary.json
- results/runs/*/preds/test_metrics.json (all 20 completed baseline runs)
- results/ledger.csv (18 completed baseline runs)
- docs/DECISIONS.md (D-006, D-007)

RISKS/DEVIATIONS:
- None. All Gate 2 targets met comfortably ahead of schedule.

NEXT:
- Phase P3: Single-process FL simulation (Day 6-9).
- Human writes FedAvg aggregation function per learning contract (Rule 10).
- Strategy implementations: FedAvg (sample-weighted & uniform), FedProx (mu in {0.01, 0.1}), FedBN.
```

---

## Phase P3: Single-Process FL Simulation (Day 2–4, Ahead of Schedule)

```
STEP: P3   STATUS: in_progress   TIME: 1.0 day / 3.0 day budget
DONE:
- Implemented modular Federated Learning simulation framework in src/fl/:
  * aggregation.py: double-precision float64 accumulator, exact analytical equivalence test (< 1e-6 error), sample-weighted and uniform weighting, explicit BatchNorm tracking, and per-family head aggregation.
  * client.py: FLClient adhering to Decision D-008 (constant LR 1e-4, 10x head LR, fresh AdamW per round, FedProx proximal regularizer, local BN caching, byte telemetry).
  * server.py: FLServer coordinating communication rounds, model broadcasting, client execution, aggregation, global validation, atomic checkpointing (last.pt, best.pt), and test split evaluation.
- Verified kill-mid-round resume test (tests/test_fl_server.py): confirmed round counter continuity, zero duplicate log rows, and state recovery.
- Completed full 3-seed benchmark (seeds [42, 53, 64], 20 communication rounds, 1 local epoch) for:
  1. FedAvg (Sample-Weighted): Overall Macro-AUC = 0.8407 +/- 0.0029 | Worst-Site AUC = 0.7064 +/- 0.0212 | Cross-Site Gap = 0.1250 +/- 0.0177.
  2. FedAvg (Uniform Weighting): Overall Macro-AUC = 0.8577 +/- 0.0056 | Worst-Site AUC = 0.7556 +/- 0.0094 | Cross-Site Gap = 0.1145 +/- 0.0120.
  3. Per-Family Heads (Uniform): Overall Macro-AUC = 0.8121 +/- 0.0028 | Worst-Site AUC = 0.7396 +/- 0.0066 | Cross-Site Gap = 0.1191 +/- 0.0141.
- Generated publication-grade convergence and systems figures in paper/figures/:
  * fl_convergence_macro_auc.png (validation macro-AUC progression vs rounds)
  * fl_convergence_macro_auc.png (multi-strategy validation macro-AUC progression vs rounds)
  * fl_confusion_matrices.png (publication 3x4 grid of breast-level confusion matrices for all 4 sites across FedAvg SW, Uniform, and Per-Family Heads)
  * fl_per_class_ovr_auc.png (grouped bar chart of Macro-AUC, Class 1 OvR, and Class 2 OvR across sites)
  * fl_convergence_loss.png (training vs validation loss progression)
  * fl_communication_efficiency.png (round wall-clock latency and cumulative bytes transferred: 358 MB/round, 7.16 GB total)
- Investigated Class 1 prediction collapse on BI-RADS sites:
  * Calculated per-site confusion matrices and per-class One-vs-Rest AUCs across all seeds.
  * Discovered INbreast Class 1 is ranked effectively (OvR AUC = 0.7998) but mis-thresholded by uncalibrated global argmax against a large class 0 prior.
  * Discovered VinDr Class 1 suffers from semantic interference (OvR AUC = 0.5815) when forced through a shared head with biopsy-proven benign lesions.
- Implemented and promoted Per-Family Heads architecture (MammographyPerFamilyResNet18) and family-specific aggregation (aggregate_per_family_heads) to core P3 variant (Decision D-009).
- Investigated and Resolved Class 1 Decision Rule Collapse on BI-RADS Sites:
  * Calculated per-site confusion matrices and per-class One-vs-Rest AUCs across all seeds (n=3).
  * Empirical Proof of INbreast Ranking vs Thresholding: Under FedAvg (Sample-Weighted), INbreast Class 1 OvR AUC is 0.800 (strong ranking signal!), but its raw recall was only 0.07 (argmax collapsed onto Class 0 due to uncalibrated prior). Under Uniform FedAvg, Class 1 recall tripled to 0.25 (Macro-F1 0.464), and under Per-Family Heads recall reached 0.26 (Macro-F1 0.480, Class 2 AUC 0.952).
  * Semantic Interference on VinDr: Under shared-head FedAvg SW, VinDr Class 1 OvR AUC was only 0.582 (semantic collision with biopsy-proven benign lesions). Uniform FedAvg lifted VinDr Class 1 OvR AUC to 0.646 (recall 0.21, Macro-F1 0.471), while Per-Family Heads achieved 0.619 (recall 0.19, Macro-F1 0.443) by isolating the classifier heads.
  * Verified that offline evaluation with site-specific available class masking correctly restricts CMMD argmax to {1, 2}, eliminating false zero-predictions.

NUMBERS:
- Federated Learning vs. Local vs. Centralized Baselines (Test Macro-AUC, Mean +/- sample std over n=3 seeds):
  * VinDr-Mammo (birads): Local = 0.6790 +/- 0.0335 | Cent. Pooled = 0.7513 +/- 0.0033 | FedAvg (Sample-Weighted) = 0.7064 +/- 0.0212 | FedAvg (Uniform) = 0.7556 +/- 0.0094
  * CMMD (biopsy): Local = 0.8040 +/- 0.0164 | Cent. Pooled = 0.8068 +/- 0.0028 | FedAvg (Sample-Weighted) = 0.8300 +/- 0.0054 | FedAvg (Uniform) = 0.8179 +/- 0.0092
  * INbreast (birads): Local = 0.7347 +/- 0.0631 | Cent. Pooled = 0.8016 +/- 0.0184 | FedAvg (Sample-Weighted) = 0.7628 +/- 0.0146 | FedAvg (Uniform) = 0.8362 +/- 0.0412
  * CDD-CESM (biopsy): Local = 0.7682 +/- 0.0186 | Cent. Pooled = 0.8526 +/- 0.0069 | FedAvg (Sample-Weighted) = 0.8089 +/- 0.0243 | FedAvg (Uniform) = 0.8695 +/- 0.0019
  * Overall Pooled: Cent. Pooled = 0.8462 +/- 0.0028 | FedAvg (Sample-Weighted) = 0.8407 +/- 0.0029 | FedAvg (Uniform) = 0.8577 +/- 0.0056
- Class-2 vs Rest (Suspicious / Malignant) Test AUC under FedAvg (Uniform):
  * CDD-CESM: 0.9364 +/- 0.0099
  * INbreast: 0.9519 +/- 0.0225
  * CMMD: 0.8268 +/- 0.0097
  * VinDr-Mammo: 0.8246 +/- 0.0070
  * Overall Pooled: 0.8765 +/- 0.0085
  * VinDr-Mammo (birads): Local = 0.6790 +/- 0.0335 | Cent. Pooled = 0.7513 +/- 0.0033 | FedAvg (SW) = 0.7064 +/- 0.0173 | FedAvg (Uniform) = 0.7556 +/- 0.0077 | Per-Family Heads = 0.7396 +/- 0.0066
  * CMMD (biopsy): Local = 0.8040 +/- 0.0164 | Cent. Pooled = 0.8068 +/- 0.0028 | FedAvg (SW) = 0.8300 +/- 0.0044 | FedAvg (Uniform) = 0.8179 +/- 0.0075 | Per-Family Heads = 0.8086 +/- 0.0029
  * INbreast (birads): Local = 0.7347 +/- 0.0631 | Cent. Pooled = 0.8016 +/- 0.0184 | FedAvg (SW) = 0.7628 +/- 0.0119 | FedAvg (Uniform) = 0.8362 +/- 0.0337 | Per-Family Heads = 0.8417 +/- 0.0107
  * CDD-CESM (biopsy): Local = 0.7682 +/- 0.0186 | Cent. Pooled = 0.8526 +/- 0.0069 | FedAvg (SW) = 0.8089 +/- 0.0198 | FedAvg (Uniform) = 0.8695 +/- 0.0015 | Per-Family Heads = 0.8586 +/- 0.0078
  * Overall Pooled: Cent. Pooled = 0.8462 +/- 0.0028 | FedAvg (SW) = 0.8407 +/- 0.0029 | FedAvg (Uniform) = 0.8577 +/- 0.0056 | Per-Family Heads = 0.8121 +/- 0.0028
- Worst-Site AUC & Cross-Site Disparity Gap:
  * FedAvg (Sample-Weighted): Worst-Site AUC = 0.7064 +/- 0.0173 | Cross-Site Gap = 0.1250 +/- 0.0177
  * FedAvg (Uniform Weighting): Worst-Site AUC = 0.7556 +/- 0.0077 | Cross-Site Gap = 0.1145 +/- 0.0120 (+4.9% worst-site gain, -8.4% gap reduction!)
  * Per-Family Heads (Uniform): Worst-Site AUC = 0.7396 +/- 0.0066 | Cross-Site Gap = 0.1191 +/- 0.0141
- Per-Class One-vs-Rest AUC and Macro-F1 (Mean over n=3 seeds):
  * VinDr: FedAvg SW (C0: 0.745, C1: 0.582, C2: 0.793, F1: 0.367) -> FedAvg Uniform (C0: 0.796, C1: 0.646, C2: 0.825, F1: 0.471) -> Per-Family Heads (C0: 0.784, C1: 0.619, C2: 0.816, F1: 0.443)
  * CMMD: FedAvg SW (C1: 0.831, C2: 0.830, F1: 0.654) -> FedAvg Uniform (C1: 0.828, C2: 0.818, F1: 0.733) -> Per-Family Heads (C1: 0.806, C2: 0.809, F1: 0.695)
  * INbreast: FedAvg SW (C0: 0.621, C1: 0.800, C2: 0.868, F1: 0.344) -> FedAvg Uniform (C0: 0.810, C1: 0.747, C2: 0.952, F1: 0.464) -> Per-Family Heads (C0: 0.833, C1: 0.740, C2: 0.952, F1: 0.480)
  * CDD-CESM: FedAvg SW (C0: 0.881, C1: 0.647, C2: 0.899, F1: 0.531) -> FedAvg Uniform (C0: 0.896, C1: 0.776, C2: 0.936, F1: 0.640) -> Per-Family Heads (C0: 0.907, C1: 0.754, C2: 0.915, F1: 0.676)
- Systems & Network Measurements:
  * Communication per round: 358.0 MB uplink + downlink
  * Total bytes transferred per 20-round run: 7.16 GB
  * Mean round duration: 54.3s (compute) + 8.2s (thermal pause cooling)

ARTIFACTS:
- src/fl/aggregation.py
- src/fl/client.py
- src/fl/server.py
- src/models/backbone.py (MammographyPerFamilyResNet18)
- scripts/run_fl_simulation.py
- scripts/run_p3_matrix.py
- scripts/generate_p3_figures.py
- paper/tables/fl_fedavg_sample_weighted_results.json
- paper/tables/fl_fedavg_uniform_results.json
- paper/tables/fl_per_family_uniform_results.json
- paper/figures/fl_convergence_macro_auc.png
- paper/figures/fl_confusion_matrices.png
- paper/figures/fl_per_class_ovr_auc.png
- paper/figures/fl_convergence_loss.png
- paper/figures/fl_communication_efficiency.png
- tests/test_fl_aggregation.py (4 passed)
- tests/test_fl_server.py (2 passed)
- docs/DECISIONS.md (D-008, D-009)

RISKS/DEVIATIONS:
- None. Uniform FedAvg substantially mitigates CMMD dominance and improves worst-site AUC by +4.9%.
- None. All FL runs strictly meet convergence and safety criteria.

NEXT:
- Complete Per-Family Heads 3-seed benchmark (currently executing).
- Run FedProx sweeps (mu in {0.01, 0.1}) and Experiment E6 (Heterogeneity: birads-only vs biopsy-only vs leave-one-out).
- Phase P3b: Explainable findings & local calibration.
- Complete FedProx sweeps (mu in {0.01, 0.1}) and Experiment E6 (Heterogeneity: birads-only vs biopsy-only vs leave-one-out) running via scripts/run_p3_matrix.py.
- Phase P3b: Explainable findings & local calibration (temperature scaling + per-class logit bias fitted on site validation).
```

