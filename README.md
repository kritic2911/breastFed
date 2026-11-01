# FL-Mammo: Federated Learning Mammography Classification

A reproducible benchmark evaluating **Federated Learning (FedAvg)** against **Local-Only** and **Centralized** baselines for binary breast cancer classification (malignant vs. benign) across four clinical sites:
- **CMMD**: 3,702 labeled mammograms (pathology/biopsy-confirmed, 70% positive)
- **CBIS-DDSM**: 770 labeled mammograms (pathology-confirmed, 40% positive)
- **BCDR**: 485 labeled mammograms (biopsy/pathology-confirmed, 15% positive)
- **INbreast**: 343 labeled mammograms (BI-RADS radiologist assessment, 29% positive; *labeled honestly as assessment, never clinical diagnosis*)

---

## 1. System Requirements & Architecture

- **Model**: ResNet-18 initialized with ImageNet weights (`IMAGENET1K_V1`), frozen BatchNorm running statistics and affine parameters to prevent cross-scanner feature distortion.
- **Hardware Profile**: NVIDIA GeForce RTX 2050 (4 GB VRAM) under WSL2 (Ubuntu 24.04, Python 3.12, PyTorch with CUDA). Single-client sequential GPU execution.
- **Experimental Evaluation**: Evaluated across three independent seeds (`42, 53, 64`) with 1,000-iteration patient-level bootstrap 95% confidence intervals.
- **Leakage Prevention**: Patient-grouped stratified train/val/test splits (`labels_split.csv`) with zero patient overlap across splits.

```
fl-mammo/
├── configs/               # Canonical configuration (experiment.yaml)
├── src/                   # Core modular library
│   ├── dataset.py         # PyTorch dataset & data loaders with RAM caching
│   ├── model.py           # ResNet18 model builder with frozen BN
│   ├── train.py           # Local training loop with class-weighted loss
│   ├── federated.py       # FedAvg simulation engine with round resumption
│   ├── evaluate.py        # AUC, sensitivity@spec90, patient bootstrap CIs
│   ├── provenance.py      # Config hashing, run_id, and structured JSONL logging
│   ├── storage.py         # Artifact bundle persistence
│   ├── synthetic.py       # Multi-site synthetic dataset generator
│   └── thermal.py         # GPU thermal management
├── scripts/               # CLI runners & utilities
│   ├── run_local.py       # Local-only baseline trainer
│   ├── run_centralized.py # Centralized pooled / site-balanced trainer
│   ├── run_federated.py   # Federated FedAvg runner
│   ├── run_cross_site.py  # Leave-One-Site-Out (LOSO) domain gap runner
│   ├── run_smoke.py       # Synthetic CPU smoke test runner
│   └── make_synthetic_data.py # Synthetic dataset CLI generator
├── tests/                 # Pytest invariant test suite
├── Makefile               # Standard developer interface (test, smoke, run, lint)
├── experiment_log.md      # Append-only experimental markdown ledger
└── pyproject.toml         # Tooling and ruff linter configuration
```

---

## 2. Quickstart & Verification

Activate the virtual environment and run the test suite and smoke verification:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run mathematical & dataset invariant test suite (16 tests)
make test

# 3. Run CPU synthetic smoke run (2 federated rounds in < 30 seconds)
make smoke

# 4. Run linter
make lint
```

Every run strictly produces all required Rule 3 artifacts:
- `resolved_config.yaml`: Canonical resolved YAML config.
- `logs.jsonl`: Line-delimited JSON events (`ts, run_id, event, round, site, split, metric, value`).
- `metrics.json`: Complete metrics dictionary with `run_id`, `config_hash`, `git_sha`, `seed`, and `environment`.
- `preds.csv`: Consolidated test predictions per sample.
- `best.pt`: PyTorch model weights at best validation checkpoint.
- `training_history.csv`: Per-epoch or per-round loss, AUC, VRAM, and timing.

---

## 3. Running Phase 1 Benchmarks (Seeds 42, 53, 64)

### 3.1 Local-Only Baselines
Train independent models for each site and evaluate both on self and cross-site:
```bash
# Single site, specific seed
python scripts/run_local.py --site CMMD --seed 42

# All 4 sites across evaluation seeds (42, 53, 64)
python scripts/run_local.py --all-seeds
# or specify explicitly:
python scripts/run_local.py --seeds 42 53 64
```

### 3.2 Centralized Baselines (Pooled Ceiling)
Train on pooled data from all 4 sites and evaluate on each site's test set:
```bash
# Evaluate on all sites across all 3 seeds
python scripts/run_centralized.py --all-sites --seeds 42 53 64

# Site-balanced centralized baseline
python scripts/run_centralized.py --all-sites --seeds 42 53 64 --site-balanced
```

### 3.3 Federated Learning (FedAvg)
Run multi-site collaborative FedAvg with frozen BatchNorm:
```bash
# Single seed
python scripts/run_federated.py --seed 42

# All 3 seeds sequentially (seeds 42, 53, 64)
python scripts/run_federated.py --all-seeds
```

---

## 4. Privacy & Governance Rules
1. **Zero Raw Patient Data in Git or CI**: All patient mammograms remain strictly local in gitignored `data/`. All automated testing and CI runs execute solely on synthetic data.
2. **Honest Label Semantics**: INbreast classifications are radiologist BI-RADS assessments, whereas CMMD and CBIS are pathology-confirmed. BI-RADS derived models are never termed clinical diagnoses.
3. **No Tuning on Test Sets**: Decision thresholds ($FPR \le 0.10$ for 90% target specificity) and early stopping checkpoints are derived strictly from validation sets. Test sets are touched exactly once.
