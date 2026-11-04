# Cross-Silo Federated Learning for Explainable Digital Mammography

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![CI](https://github.com/kc/breastFed/actions/workflows/ci.yml/badge.svg)](https://github.com/kc/breastFed/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end BTP (Bachelor of Technology Project) implementing and benchmarking **cross-silo federated learning (FL)** across heterogeneous clinical digital mammography cohorts. The framework delivers **explainable breast-level diagnostic findings** (calibrated probabilities and spatial localization validated against radiologist annotations) and rigorous **systems telemetry** (latency, jitter, communication bytes, and thermal guards).

---

## 1. Executive Summary & Core Contributions

Breast cancer screening across multi-institution hospital networks faces severe privacy and domain heterogeneity hurdles. Centralizing raw patient mammograms across hospitals is prohibited by health data governance regulations (HIPAA, GDPR). However, cross-silo FL across digital hospitals must navigate acute scanner, resolution, bit-depth, and clinical label-family disparities (`birads` vs. `biopsy`).

### Key Contributions
1. **Empirical Cross-Silo FL vs. Centralized vs. Local Baselines**: Rigorous $n=3$ seed benchmarking across 4 digital hospital sites comparing local isolated models, pooled/site-balanced centralized models, and federated algorithms (FedAvg sample-weighted & uniform, FedProx, FedBN).
2. **Handling Clinical Label-Family Heterogeneity**: Unified 3-class breast-level mapping reconciling screening BI-RADS sites (`birads` 1/2-3/4-5) with diagnostic surgical sites (`biopsy` normal/benign/malignant) without label leakage or artificial imputation.
3. **Hardware Telemetry & Systems Safety Rails**: Custom reactive thermal guard with automatic checkpointing and resumption (calibrated for laptop/workstation 4 GB GPUs under WSL2), paired with network emulation (`tc netem`) and secure aggregation overhead accounting.
4. **Explainable Diagnostic Reporting (Phase P3b)**: Spatial localization via Grad-CAM mapped to standard radiological mammographic quadrants (3×3 breast grid) validated quantitatively against ground-truth radiologist lesion annotations.

---

## 2. Hospital Sites & Dataset Landscape

Heterogeneity claims are strictly restricted to modern digital full-field scanners (film-digitized legacy collections such as CBIS-DDSM and BCDR are excluded per Decision **D-001** to avoid confounding scanner variation with digitization artifacts).

All patient cohorts undergo salted SHA-256 de-identification and patient-grouped stratified 80/20 train/test partitioning.

| Simulated Hospital Site | Label Family | Images (Audited) | Patients | Scanner Manufacturer(s) | Available Classes | Lesion Annotations |
|---|---|---|---|---|---|---|
| **VinDr-Mammo (mini)** | `birads` | 1,497 | 375 | Siemens Mammomat Inspiration, Planmed Nuance, Giotto | {0, 1, 2} | Finding bounding boxes |
| **CMMD** | `biopsy` | 3,534 | 1,673 | GE Senographe | {1, 2} | None (Breast-level) |
| **INbreast** | `birads` | 409 | 108 | Siemens Mammomat Novation | {0, 1, 2} | OsiriX contours & points |
| **CDD-CESM (Low-Energy)** | `biopsy` | 966 | 326 | GE Senographe | {0, 1, 2} | Lesion contour masks |
| **Total Cohort** | — | **6,406** | **2,481** | **4 Distinct Digital Scanners** | — | **9,948 Ground-Truth ROIs** |

### Label Mapping (3-Class Breast-Level)
- **Class 0 (Normal)**: VinDr BI-RADS 1, INbreast BI-RADS 1, CDD normal screening. *(CMMD contains no class 0; masked during training and evaluation)*.
- **Class 1 (Benign)**: VinDr BI-RADS 2–3, INbreast BI-RADS 2–3, CDD biopsied benign, CMMD biopsied benign.
- **Class 2 (Suspicious / Malignant)**: VinDr BI-RADS 4–5 (suspicious), INbreast BI-RADS 4–6, CDD confirmed malignant, CMMD confirmed malignant.

---

## 3. Architecture & Federated Training Hygiene

```
   Raw DICOM / Image (12/16-bit or 8-bit)
                  │
                  ▼
   ┌──────────────────────────────────────────────┐
   │ Dynamic Tissue Segmentation & P1 Preprocess  │
   │ - Pectoral/air background removal            │
   │ - Chest wall flush left, nipple right        │
   │ - Percentile tissue normalization [0.0, 1.0] │
   │ - Aspect-preserving pad & resize to 512×512  │
   └──────────────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────┐
   │ ResNet-18 Backbone (Torchvision ImageNet)    │
   │ - conv1, layer1, layer2: Frozen              │
   │ - layer3, layer4: Fine-tuned (LR = 1e-4 / 3e-4)│
   │ - Head: Linear(512, 3) (Head LR = 10x)       │
   │ - BatchNorm: Trainable (Decision D-006)      │
   └──────────────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────┐
   │ Dual-View Logit Mean Aggregation             │
   │ Breast Logits = 0.5 * (Logits_CC + Logits_MLO)│
   └──────────────────────────────────────────────┘
```

### Federated Learning Protocol (Phase P3)
To prevent federated gradient shock and representation collapse (Decision **D-008**):
- **Optimizer**: AdamW with constant learning rate (`1e-4` backbone, `1e-3` head) without intra-round cosine restarts.
- **Aggregation Strategies**:
  - **Sample-Weighted FedAvg**: Global model updated proportionally to client dataset size $w_k = \frac{n_k}{N}$.
  - **Uniform FedAvg**: Equal weighting $w_k = \frac{1}{K}$ to prevent CMMD (~12× larger than INbreast) from monopolizing optimization trajectories.
  - **FedProx**: Proximal regularization term ($\mu \in \{0.01, 0.1\}$) penalizing local client drift.
  - **FedBN**: Keeps local batch normalization affine parameters on-site to handle scanner domain shift.

---

## 4. Hardware Safety Rails & Run Discipline

All training and evaluation pipelines adhere strictly to hardware safety invariants:

- **Reactive Thermal Guard (`src/utils/thermal.py`)**:
  Continuous GPU temperature monitoring (NVML with fallback to formatted `nvidia-smi` under WSL2).
  - Soft pause at $70.0^\circ\text{C}$ until cooling to $55.0^\circ\text{C}$.
  - Hard pause & immediate checkpoint at $75.0^\circ\text{C}$.
  - Resumable exit (`code 75`) if thermal pause exceeds 10 minutes.
- **Atomic Checkpointing (`src/utils/ckpt.py`)**:
  All model checkpoints are saved atomically (`.tmp` write, `os.fsync`, rename). Checkpoint bundles contain model weights, optimizer state, scheduler, AMP GradScaler, RNG states across Python/NumPy/PyTorch/CUDA, resolved config hashes, and Git commit SHA.
- **Run Directory Schema (`results/runs/<run_id>/`)**:
  Every execution produces an immutable ledger row in `results/ledger.csv`, structured run directory with `config.yaml`, `env.json`, `metrics.jsonl`, `events.jsonl`, and per-breast test predictions in `preds/`.

---

## 5. Empirical Results & Milestone Gates

### Gate 1 (Phase P1 — Data & Pipeline Verification): PASSED
- Verified 6,406 cached arrays (Deterministic cache SHA-256: `884be5cb17374d7d02d8dcf63225e8681fce1afdad91ec8c9c660bec16944054`).
- Zero patient overlap between train and test splits; zero view separation.
- Site-ID classifier probe scored **95.30% accuracy** (vs. 54.59% majority baseline), demonstrating substantial scanner domain shift across hospitals.

### Gate 2 (Phase P2 — Baselines): PASSED
All pre-registered baseline gates were exceeded across $n=3$ seeds (`[42, 53, 64]`):
1. **CMMD binary test AUC $\ge 0.75$**: **0.8040** (Local), **0.8030** (Centralized).
2. **No site near chance ($< 0.55$)**: All sites achieved $\ge 0.6790$ Macro-AUC.
3. **Small-site centralized gains**: INbreast improved by **+9.81% AUC** (0.7348 $\to$ 0.8329); CDD improved by **+10.03% AUC** (0.7681 $\to$ 0.8684).

| Method | Site | Label Family | Test Macro-AUC (Mean $\pm$ Std) | Class-2 (Malignant) AUC | Sens @ 90% Spec | Macro F1 |
|---|---|---|---|---|---|---|
| **Local Baseline** | CDD-CESM | `biopsy` | $0.7681 \pm 0.0186$ | $0.8679 \pm 0.0253$ | $0.5185 \pm 0.0425$ | $0.5239 \pm 0.1181$ |
| **Local Baseline** | CMMD | `biopsy` | $0.8040 \pm 0.0164$ | $0.8040 \pm 0.0164$ | $0.5981 \pm 0.0346$ | $0.7018 \pm 0.0231$ |
| **Local Baseline** | INbreast | `birads` | $0.7348 \pm 0.0631$ | $0.8630 \pm 0.0196$ | $0.4815 \pm 0.2313$ | $0.4670 \pm 0.0637$ |
| **Local Baseline** | VinDr-Mammo | `birads` | $0.6790 \pm 0.0336$ | $0.7516 \pm 0.0496$ | $0.3571 \pm 0.0859$ | $0.3969 \pm 0.0268$ |
| **Centralized (Pooled)** | All Sites | Pooled | $0.8462 \pm 0.0028$ | $0.8689 \pm 0.0063$ | $0.6202 \pm 0.0122$ | $0.6197 \pm 0.0416$ |
| **Centralized (Balanced)**| All Sites | Pooled | $0.8467 \pm 0.0023$ | $0.8715 \pm 0.0048$ | $0.6121 \pm 0.0150$ | $0.6154 \pm 0.0389$ |
| **FedAvg (Sample-Weighted)**| All Sites | Federated | $0.8407 \pm 0.0029$ | $0.8661 \pm 0.0031$ | $0.6697 \pm 0.0185$ | $0.5012 \pm 0.0210$ |

---

## 6. Repository Layout

```
.
├── configs/                  # Experiment configuration files (YAML)
├── data/                     # Raw datasets & cached 512x512 numpy arrays (gitignored)
│   ├── manifest.csv          # Master dataset manifest with all transforms & metadata
│   ├── annotations.csv       # Standardized 512x512 lesion annotations
│   └── cache512/             # Preprocessed float32 image cache
├── docs/                     # Engineering documentation
│   ├── DECISIONS.md          # Immutable architectural & clinical decision records (D-001 - D-008)
│   └── PROGRESS.md           # Phase checkpoints, milestones, and telemetry logs
├── paper/                    # Paper tables and figures
│   ├── figures/              # Vector PDFs and QC image grids
│   └── tables/               # JSON and LaTeX tables compiled directly from ledger
├── results/
│   ├── ledger.csv            # Central run tracking ledger
│   └── runs/                 # Per-run outputs (checkpoints, metrics, event logs)
├── scripts/                  # Workflow scripts
│   ├── build_manifest_and_cache.py  # P1 preprocessing pipeline
│   ├── run_local_baselines.py       # P2 local training across sites & seeds
│   ├── run_centralized.py           # P2 centralized pooling & balancing
│   ├── run_fl_simulation.py         # P3 single-process FL simulation
│   └── run_resumable.sh             # Automatic resumption wrapper for code 75
├── src/                      # Source modules
│   ├── data/                 # Dataset loaders, manifest parsing, geometric transforms
│   ├── eval/                 # Evaluation metrics, bootstrapping, and floor baselines
│   ├── explain/              # Grad-CAM, quadrant mapping, and template report generator
│   ├── fl/                   # FL client, server, and aggregation logic (FedAvg, FedProx, FedBN)
│   ├── models/               # ResNet-18 backbone and custom linear heads
│   ├── systems/              # Network emulation & telemetry collectors
│   ├── train/                # Training loops with AMP, grad accumulation, and thermal guards
│   └── utils/                # Thermal guard, atomic checkpointing, provenance logging, seeding
├── tests/                    # Pytest test suite (unit & regression tests)
├── pyproject.toml            # Project configuration & ruff settings
└── requirements.txt          # Clean, pinned runtime dependencies
```

---

## 7. Setup & Quickstart

### Environment Setup
Create and activate a Python 3.12 virtual environment:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Running the Test Suite
Verify that all unit tests, atomic checkpointing, thermal guard mocks, and FL aggregation tests pass:
```bash
pytest tests/ -v
```

### Code Quality & Linting
Run Ruff check and formatting:
```bash
ruff check .
ruff format --check .
```

### Reproducing Baseline & FL Runs
1. **Preprocess & Cache Data**:
   ```bash
   python scripts/build_manifest_and_cache.py
   ```
2. **Run Local Baselines (Phase P2)**:
   ```bash
   python scripts/run_local_baselines.py --seed 42
   ```
3. **Run Centralized Baselines**:
   ```bash
   python scripts/run_centralized.py --mode pooled --seed 42
   ```
4. **Run Federated Simulation (Phase P3)**:
   ```bash
   python scripts/run_fl_simulation.py --strategy fedavg --mode sample_weighted --rounds 20 --seed 42
   ```

---

## 8. Governance & Attribution
- **Simulated Environment**: All experiments simulate hospital sites using publicly accessible, de-identified datasets. This project makes **no claims of clinical deployment or direct clinical validity**.
- **Data Governance**: Patient identifiers are cryptographically hashed using salted SHA-256 before entering any log, filename, checkpoint, or prediction file. Pixel arrays and DICOM headers are strictly processed locally and never transmitted to third-party hosted services.

