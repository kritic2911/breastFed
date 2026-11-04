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

---

### D-009: Promotion of Per-Family Heads Architecture and Multi-Scanner Aggregation
- **Date**: 2026-11-03 (Day 2, Phase P3)
- **Status**: Accepted & Promoted from Stretch to Core P3 Variant
- **Motivation**: Confusion matrix analysis on 3 seeds of FedAvg (Sample-Weighted) revealed severe decision rule collapse on `birads` sites: VinDr and INbreast predicted ~82% of all test breasts as class 0 (normal), suppressing class 1 recall to 0.00–0.13. Analysis showed that while INbreast retained strong ranking signal (Class 1 OvR AUC $0.7998 \pm 0.0279$), VinDr Class 1 OvR AUC dropped to $0.5815 \pm 0.0384$. This semantic collision arises because CMMD/CDD define class 1 as biopsy-confirmed focal benign lesions, while VinDr defines class 1 as unbiopsied BI-RADS 2–3 (often diffuse or negative-appearing findings). Under a single shared head, gradient interference forces class 1 to conflict with class 0.
- **Architectural Specification (`MammographyPerFamilyResNet18`)**:
  - Shared feature backbone (ResNet-18 conv1 through layer4, adaptive avg pool).
  - Dedicated classification heads: `heads.birads` ($512 \to 3$) and `heads.biopsy` ($512 \to 3$).
- **Aggregation Mechanics (`aggregate_per_family_heads`)**:
  - Backbone weights ($\theta_{\text{backbone}}$) and BatchNorm statistics are averaged across **all K clients** to maximize feature transfer and domain generalization.
  - `heads.birads` parameters are averaged **only** across `birads` clients (`vindr`, `inbreast`).
  - `heads.biopsy` parameters are averaged **only** across `biopsy` clients (`cmmd`, `cdd`).
- **Decision Rule & Metric Reporting**:
  - Argmax is restricted to the site's `available_classes` (e.g. CMMD masks class 0).
  - Report per-class One-vs-Rest AUCs (class 0, 1, 2) alongside macro-AUC to evaluate true ranking capability independent of threshold shifts.

---

### D-010: Pre-Registered Design & Success Criteria for Sharpened Localization, Cluster Routing, and Calcification Expert Fusion
- **Date**: 2026-11-04 (Day 3–4, Phase P3b)
- **Status**: Pre-Registered & Locked Prior to Implementation
- **Context & Empirical Motivation**:
  - Phase P3b verification of Mass vs. Calcification test cohorts revealed a structural resolution bottleneck:
    * ResNet-18 operating on 512×512 inputs downsamples native digital mammograms ($3000 \times 4000$) by $6\times - 8\times$, yielding coarse $16 \times 16$ `layer4` feature maps (stride $\approx 32\,\text{px}$).
    * Focal masses ($N=524$) exhibit large continuous contrast gradients, achieving a **$61.02\%$** peak-in-lesion hit rate and $0.458$ TV distance.
    * Microcalcifications ($N=236$, typical size $<0.5\,\text{mm}$) undergo spatial gradient attenuation, yielding a lower peak hit rate (**$48.18\%$**) and more diffuse attribution (TV distance $0.552$), despite strong global textural malignancy ranking ($0.8199$ Class-2 AUC).
- **Three-Pillar Architectural Specification**:
  1. **Pillar 1: Localization Sharpening**:
     - *Mechanisms*:
       * Multiplicative map: $\text{Map}_{\text{final}} = \text{CAM} \times (\epsilon + \text{Detector Density})$. Coarse Grad-CAM specifies the macro-region, while the high-frequency speck detector snaps activation to actual microcalcifications.
       * CAM-guided crops: Peak coordinate $(y^*, x^*)$ guides a native-resolution bounding crop for second-stage feature extraction.
       * Agreement score: Energy-in-box overlap between Grad-CAM heat mass and detector bounding boxes, reported as an objective visual consistency metric.
     - *Reporting Integrity*: Pure Grad-CAM is strictly preserved for faithfulness occlusion and Adebayo parameter-randomization sanity checks. Any detector product map is explicitly labelled as **"Combined Localization"**.
     - *Evaluation Invariant*: Compare CAM-only vs. Detector-only vs. Product map on the 345 annotated test cases using Peak Hit Rate, 3×3 TV distance, and size error.
  2. **Pillar 2: Classical Detector & Cluster Router**:
     - *Detector*: Classical morphological white top-hat / difference-of-Gaussians (DoG) filter plus adaptive thresholding to detect bright high-frequency specks on high-resolution tissue.
     - *Cluster Aggregation*: Real calcifications present as clusters rather than isolated specks. Specks are merged via spatial clustering (e.g. DBSCAN or connected components within radius $R$).
     - *Router Decision Rule*: If cluster count $C \ge C_{\text{thresh}}$ and the fraction of small boxes exceeds threshold, route the breast to the specialized calcification expert.
     - *Hyperparameter Hygiene*: Thresholds ($C_{\text{thresh}}$, box size cutoff) are tuned strictly on training/validation splits, never hand-picked.
     - *Router Metric*: Evaluated directly as a binary classifier for "has calcification" on validation, reporting Precision, Recall, and F1 across all 4 sites.
     - *Privacy & Test Invariants*: Zero ground-truth boxes fed at test time; the router depends strictly on detector output. Detector boxes, patches, and coordinates remain local to client storage.
  3. **Pillar 3: Calcification Expert & Decision Fusion**:
     - *Training Patches*: Extracted from INbreast point annotations and VinDr bounding boxes with image-level labels (weakly-supervised Multiple Instance Learning with max or attention pooling).
     - *Fusion*: Logistic regression or linear calibration combining global ResNet-18 logits and calcification expert patch logits, fitted strictly on validation split.
- **Pre-Registered Success Criteria (Write Down First)**:
  1. **Criterion 1 (Localization Enhancement)**:
     $$\Delta \text{Hit Rate} = \text{Hit Rate}_{\text{Product}} - \text{Hit Rate}_{\text{CAM-Only}} \ge +5.0\%$$
     on the calcification test cohort ($N=137$ annotated), without increasing mean 3×3 TV distance by $>0.05$.
  2. **Criterion 2 (Detector-Based Router Discrimination)**:
     $$\text{Recall}_{\text{calc}} \ge 70.0\%, \quad \text{Precision}_{\text{calc}} \ge 50.0\% \quad (\text{F1} \ge 0.584)$$
     measured against clinical finding ground truth on the validation split.
  3. **Criterion 3 (Calcification Expert Decision Lift)**:
     $$\Delta \text{AUC}_{\text{calc}} = \text{AUC}_{\text{Global+Expert}} - \text{AUC}_{\text{Global}} \ge +0.030$$
     evaluated on the calcification test cohort ($N=236$), with a 1,000-resample paired patient bootstrap 95% confidence interval strictly excluding zero ($\text{CI}_{\text{lower}} > 0.000$).
- **Null Result & Fallback Protocol (AGENTS.md Rule 5)**:
  - If $\Delta \text{AUC}_{\text{calc}} < +0.030$ or the bootstrap CI crosses zero, the hypothesis that high-resolution patch routing overcomes 512 px pooling is rejected.
  - The detector is retained solely as an explainability and reporting tool (visual overlay and agreement metric).
  - The outcome is documented and published honestly as a **neutral/negative finding** with full statistical reporting (no p-hacking or silent tuning).
- **Experimental Execution & Empirical Outcomes**:
  1. *Criterion 1 (Localization Sharpening: Product Map vs CAM-Only)*:
     - Evaluated on $N=106$ annotated calcification test cases (`paper/tables/localization_sharpening_comparison.json`):
       * Pure Grad-CAM: Hit Rate = **$46.23\%$** | 3×3 TV Distance = **$0.5136$**
       * Classical Detector Only: Hit Rate = **$29.25\%$** | 3×3 TV Distance = **$0.6231$**
       * Product Map ($\text{CAM} \times \text{Detector}$): Hit Rate = **$44.34\%$** | 3×3 TV Distance = **$0.5139$** ($\Delta \text{Hit Rate} = -1.89\%$, $\Delta \text{TV} = +0.0003$)
     - **Status**: **FAILED / NULL RESULT** (Target $\Delta \text{Hit Rate} \ge +5.0\%$ not met). High-frequency top-hat filters detect parenchymal variations in dense tissue, pulling spatial peak attention away from subtle lesions. Pure Grad-CAM is preserved.
     - **Energy-in-Box Context**: While $71.72\%$ of Grad-CAM energy fell inside detected cluster bounding boxes, **the detected boxes covered $89.85\%$ of the breast tissue area on average**. Because the boxes occupy $\sim 90\%$ of the breast, containing $71.7\%$ of heat energy is lower than uniform random expectation and does not indicate selective localization.
  2. *Criterion 2 (Detector-Based Router Discrimination)*:
     - Evaluated on held-out test splits across sites (`paper/tables/p3b_rigorous_metric_audit.json`):
       * INbreast held-out test (prevalence $75.61\%$): Router ROC-AUC = **$0.6427$**, Specificity at $\ge 90\%$ recall = **$25.0\%$**, and at $C \ge 2$, Specificity = **$5.0\%$** ($81$ of $82$ images flagged; precision = $76.54\%$, matching prevalence).
       * VinDr held-out test (prevalence $14.47\%$): Router ROC-AUC = **$0.4981$** (chance), Specificity at $\ge 90\%$ recall = **$0.0\%$**, and at $C \ge 2$, Specificity = **$20.38\%$** ($236$ of $304$ images flagged; precision = $12.29\%$).
     - **Status**: **INCONCLUSIVE / POOR DISCRIMINATION**. The high-frequency filter lacks specificity in dense fibroglandular tissue, flagging nearly every image to achieve high recall. It is not viable as an automated routing gate.
  3. *Criterion 3 (Calcification Expert Decision Fusion)*:
     - Evaluated on held-out calcification test cohort ($N=236$ images across all 4 sites, `paper/tables/calc_expert_fusion_results.json`):
       * Global ResNet-18 Malignancy AUC: **$0.8073$**
       * Global + Routed Expert Fused AUC: **$0.8157$**
       * Measured $\Delta \text{AUC}_{\text{calc}}$: **$+0.0084$** (1,000-resample paired patient bootstrap 95% CI: $[-0.0032, +0.0211]$)
     - **Status**: **FAILED / NULL RESULT** (Target $\Delta \text{AUC} \ge +0.030$ not met; 95% bootstrap CI crosses zero). **No benefit observed in this pilot**.
- **Final Decision & Architectural Resolution**:
  - In strict compliance with Decision D-010 Fallback Protocol and AGENTS.md Rule 5 (Honest Reporting) and Rule 7 (Simplest Defensible Thing First):
    1. Weakly-supervised high-resolution patch MIL models provide no statistically significant lift over global ResNet-18 pooling ($\Delta \text{AUC} = +0.0084$, 95% CI: $[-0.0032, +0.0211]$).
    2. We do **not** federate the patch expert model, preventing communication and architectural bloat.
    3. The high-resolution microcalcification detector and DBSCAN clustering module are **retained exclusively for client-side explainability and reporting**: generating cluster counts, spatial descriptors, and visual overlays in structured reports without transmitting raw pixel ROIs.
    4. All null results are formally documented as honest, negative findings in the BTP deliverable.






