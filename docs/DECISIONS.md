# Project Decision Log (DECISIONS.md)

This log records all foundational architectural, algorithmic, clinical, and systems decisions.
Every entry is immutable once signed off; deviations and waivers must be logged here with date and justification.

---

### D-001: Dataset Selection and Heterogeneity Boundary
- **Date**: 2026-11-02 (Day 0)
- **Status**: Accepted
- **Decision**: Restrict benchmark sites to modern digital mammography scanners:
  1. **mini VinDr-Mammo** (`birads` family, 4,872 images, 1,218 patients, Siemens Mammomat Inspiration, Planmed Nuance, Giotto)
  2. **CMMD** (`biopsy` family, 4,793 images, 1,673 patients, GE Senographe)
  3. **INbreast** (`birads` family, 410 images, 108 patients, Siemens Mammomat Novation)
  4. **CDD-CESM** (`biopsy` family, 1,004 low-energy DM images, 326 patients, GE Senographe)
- **Exclusion**: CBIS-DDSM and BCDR (scanned film digitizer collections) are excluded. Heterogeneity claims are strictly limited to digital full-field scanners and clinical label families (`birads` vs. `biopsy`).
- **Rationale**: Digitized film exhibits distinct scanning artifacts (grain, dust, non-linear digitization curves) that conflate scanner shift with historical digitization shift.

---

### D-002: Hardware Calibration, Batch Size, and Run Matrix
- **Date**: 2026-11-02 (Day 0, Phase P0)
- **Status**: Measured & Accepted
- **Hardware Profile**:
  - GPU: NVIDIA GeForce RTX 2050 (4.0 GB VRAM, Compute Capability 8.6)
  - Platform: WSL2 (Ubuntu 24.04, Python 3.12, PyTorch 2.14.1 with CUDA 13.0)
- **Empirical Throughput Benchmark (ResNet-18, 512×512, AMP, channels_last)**:
  - Batch size 4: 66.3 img/s | Peak VRAM: 437.8 MB
  - Batch size 8: 76.8 img/s | Peak VRAM: 685.8 MB
  - Batch size 12: 81.9 img/s | Peak VRAM: 931.2 MB
  - **Batch size 16**: **86.7 img/s** | Peak VRAM: **1,180.6 MB** (Selected standard)
  - Batch size 20: 98.1 img/s | Peak VRAM: 1,429.1 MB
  - Batch size 32: 88.0 img/s | Peak VRAM: 2,191.6 MB
- **Decision**: Lock batch size = 16 for all local and FL training. At 1.18 GB peak allocation, this utilizes < 30% of the 4 GB GPU, leaving ample headroom (> 2.8 GB) for Grad-CAM activation maps and thermal stability without OOM risk.
- **Thermal Calibration**:
  - Idle temperature: 47 °C
  - Loaded temperature: 54 °C (under continuous 100% compute load)
  - Clamped guards: `soft_c = 70.0 °C`, `hard_c = 75.0 °C`, `resume_c = 55.0 °C` (idle + 8 °C).
  - Telemetry source: Direct `nvidia-smi` query fallback active under WSL2.
- **Recomputed Run Matrix**:
  - Total pooled training cohort: ~4,000 images. At 86.7 img/s, 1 pooled epoch ≈ 46 seconds.
  - Local epoch times: VinDr (~3,800 train images) ≈ 44s; CMMD (~2,800 train images) ≈ 32s; CDD (~800 train images) ≈ 9.2s; INbreast (~320 train images) ≈ 3.7s.
  - FL simulation (20–30 global rounds, 1 local epoch): ~15–25 minutes per run. 3 seeds require ~1.5 hours total compute time, well within the 18-day sprint budget.

---

### D-003: CDD-CESM Low-Energy DM Acquisition
- **Date**: 2026-11-02 (Day 0)
- **Status**: Accepted
- **Decision**: In CDD-CESM, strictly use the **low-energy digital mammography (`DM`)** images (1,004 images, 1,000 annotated). Exclude all recombined subtracted contrast-enhanced spectral images (`CESM`).
- **Rationale**: Recombined images contain iodine contrast wash-in information that is physically absent in standard screening/diagnostic mammograms. Mixing modalities would introduce an artificial diagnostic shortcut.

---

### D-004: Thermal Telemetry Architecture in WSL2
- **Date**: 2026-11-02 (Day 0)
- **Status**: Accepted
- **Decision**: Implement dual-mode telemetry in `src/utils/thermal.py`: check for NVML bindings (`pynvml`), and seamlessly fall back to formatted `nvidia-smi` sub-process queries (`temperature.gpu,power.draw,utilization.gpu,clocks.sm`).
- **Rationale**: WSL2 does not always expose full NVML library hooks to userland Python packages, but standard `nvidia-smi` provides accurate hardware telemetry without requiring elevated kernel privileges.

---

### D-005: Phase P1 Data Manifest, Label Resolution, and Format Audit
- **Date**: 2026-11-02 (Day 0–1, Phase P1)
- **Status**: Accepted & Signed Off
- **Caveat Resolutions (AGENTS.md Section 1.2)**:
  1. **CMMD Contralateral Images**: Excluded 1,259 unlabeled images of the opposite breast. Retained 3,534 verified lesion-breast images across 1,673 patients.
  2. **CDD-CESM Occult Normals**: Excluded 34 low-energy DM images tagged "normal" where biopsy pathology confirmed occult lesions visible only on subtracted contrast views, plus 4 unannotated images. Retained 966 low-energy DM images across 326 patients.
  3. **INbreast FB View**: Excluded non-standard caudocranial view `50994300`. Retained 409 images across 108 patients.
  4. **VinDr-Mammo (mini)**: Filtered out 3,375 0-byte stub files (Kaggle upload quota limit artifact), preserving 1,497 complete, valid 16-bit DICOMs across 375 patients using the official 80/20 train/test split.
- **Image Format Audit & Domain Shortcut Notice (Section 2.1)**:
  - CDD-CESM: 8-bit lossy JPEG (`source_format: jpg`, `bit_depth: 8`).
  - CMMD: 8-bit DICOM (`source_format: dicom`, `bit_depth: 8`).
  - VinDr-Mammo & INbreast: 12–16 bit lossless DICOM (`source_format: dicom`, `bit_depth: 16`).
  - *Domain Shortcut Alert*: Formats and bit-depths strongly correlate with hospital sites. The site-ID probe achieved 95.30% accuracy (vs. 54.59% majority baseline), demonstrating pronounced visual domain shift across sites.
- **Rule 9 Geometry & Annotation Clamping**:
  - Radiologist annotations extending slightly outside raw image bounds (e.g., VinDr ground-truth $x = -26.8$, CDD $x = 1213$ on width 1159) are clamped to the $[0.0, 512.0]$ canvas to ensure all 9,948 annotations lie strictly within image boundaries.
- **Cache Checksum**:
  - Deterministic SHA-256: `884be5cb17374d7d02d8dcf63225e8681fce1afdad91ec8c9c660bec16944054`.

---

### D-006: Phase P2 Pre-Registered Mini-Sweep (BatchNorm & Learning Rate Selection)
- **Date**: 2026-11-02 (Day 1, Phase P2)
- **Status**: Completed & Locked
- **Sweep Cohort**: CMMD (biopsy family, classes {1, 2}, Seed 42, evaluated strictly on validation set; test split unpolluted).
- **Sweep Results (2×2 Pre-Registered Grid)**:
  1. `bn_mode: frozen`, `backbone_lr: 1e-4`: Best Epoch 24 | Val Macro-AUC = 0.7779 | Val Macro-F1 = 0.6899
  2. `bn_mode: frozen`, `backbone_lr: 3e-4`: Best Epoch 6  | Val Macro-AUC = 0.6475 | Val Macro-F1 = 0.0000
  3. `bn_mode: trainable`, `backbone_lr: 1e-4`: Best Epoch 9 | Val Macro-AUC = 0.8122 | Val Macro-F1 = 0.7262
  4. `bn_mode: trainable`, `backbone_lr: 3e-4`: Best Epoch 22 | Val Macro-AUC = **0.8123** | Val Macro-F1 = **0.7511** (WINNER)
- **Decision**:
  - Lock `bn_mode = "trainable"` and `backbone_lr = 3e-4` (head LR = 3e-3 via 10× multiplier).
  - Trainable BatchNorm in `layer3` and `layer4` delivers superior feature adaptation over frozen ImageNet affine statistics (+0.165 AUC at 3e-4 LR), comfortably exceeding the Gate 2 validation target ($\ge 0.75$ binary AUC).
  - Preserving trainable BN also directly satisfies the architectural requirement for FedBN in Phase P3.
  - This configuration is locked for all local, centralized, and FL runs.

---

### D-007: Early Stopping Patience Adjustment
- **Date**: 2026-11-02 (Day 1, Phase P2)
- **Status**: Accepted
- **Decision**: Update validation-based early stopping patience from 8 to 12 epochs across all local and centralized baseline training pipelines (`src/train/local.py`, `scripts/run_local_baselines.py`, `scripts/run_centralized.py`).
- **Rationale**: Mammography feature representations on small-sample digital cohorts (e.g. INbreast with 326 train images, CDD with 763 train images) exhibit intermittent validation AUC plateaus before fine-tuning transitions. Expanding patience to 12 prevents premature stopping while preserving the 40-epoch cap and atomic best-model restoration.

---

### D-008: Federated Learning Rate & Hygiene Specification
- **Date**: 2026-11-03 (Day 2, Phase P3 Kickoff)
- **Status**: Accepted & Locked
- **Decision**: Lock constant backbone learning rate `backbone_lr = 1e-4` (head LR `1e-3` via 10× multiplier) with zero schedule restarts for all Federated Learning simulation experiments (FedAvg, FedProx, FedBN, and Experiment E6).
- **Rationale**: While centralized continuous training stabilized at 3e-4 with cosine decay, multi-client federated training with distributed client models exhibits severe gradient shock and client-drift instability at 3e-4 across rounds. As observed in prior empirical attempts and specified in `AGENTS.md` Section 2.3, a constant LR of 1e-4 eliminates LR spikes and flat prediction collapse across federated communication rounds.





