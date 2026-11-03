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

