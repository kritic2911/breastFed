# Training Pipeline Usage Guide — Federated Mammography (FL-Mammo)

This guide documents the execution workflow, hardware requirements, acceleration optimizations (in-memory caching & batch processing), and output persistence for the cross-site mammography benchmark.

---

## 1. System & Architecture Overview

The experimental pipeline benchmarks **three training paradigms** across **4 clinical sites** using identical pretrained ResNet-18 architectures, data splits, and evaluation criteria:

| Method | Description | Role in Benchmark | Runner Script |
|---|---|---|---|
| **Local-Only** | Train 1 model per site on its own data | **Performance Floor** (isolated hospital) | `scripts/run_local.py` |
| **Centralized** | Pool all sites into a single dataset | **Performance Ceiling** (ideal privacy-free setting) | `scripts/run_centralized.py` |
| **Federated (FedAvg)** | Collaborative training via iterative weight averaging | **The Claim** (approaching ceiling without raw data exchange) | `scripts/run_federated.py` |
| **Leave-One-Site-Out (LOSO)** | Train on 3 sites, test on held-out 4th site | **Domain Gap Measurement** | `scripts/run_cross_site.py` |

### Key Experimental Controls
- **Model**: ResNet-18 initialized with ImageNet weights (`IMAGENET1K_V1`).
- **Frozen BatchNorm**: All BatchNorm running stats and affine parameters remain frozen at ImageNet defaults to eliminate cross-scanner batch norm contamination.
- **Decision Threshold**: Selected strictly on the validation set at target 90% specificity ($FPR \le 0.10$). Test sets are scored once at final checkpoint.
- **Patient-Level Bootstrap**: 1,000 bootstrap iterations resampled by `patient_id` to report 95% Confidence Intervals $[2.5\%, 97.5\%]$.
- **Data Splits**: Patient-grouped stratified splits generated with zero leakage in `data/<site>/labels_split.csv`.

---

## 2. Environment Activation & Hardware Verification

The project runs under Python 3.12 with PyTorch and CUDA support.

```bash
# 1. Activate project virtual environment
source /home/kc/venv/bin/activate

# 2. Verify GPU and dependencies
python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 3. High-Throughput Optimizations

### In-Memory Image Caching
- Config option: `cache_images: true` in `configs/experiment.yaml`.
- Images are pre-resized to target input resolution ($224 \times 224$) and cached in system RAM upon first read.
- Total memory footprint across all ~5,300 images is $< 700\text{ MB}$, well within standard system RAM limits ($7.6\text{ GB}$).
- **Speedup**: Epochs run directly from RAM, reducing epoch duration by $5\times - 10\times$ by eliminating repeated disk I/O and PNG decompression.

### GPU Batch Processing & Data Transfer
- `pin_memory=True`: Enables page-locked memory allocations for asynchronous host-to-GPU memory copies.
- `persistent_workers=True`: DataLoader background worker processes persist across epochs, eliminating process respawn latency.
- `prefetch_factor=2`: Workers prefetch batches ahead of GPU execution.
- `non_blocking=True`: Asynchronous CUDA memory streaming during training loops and inference evaluation.

---

## 4. Running the Pipelines

All scripts accept `--config`, `--device`, and flexible `--seed` / `--seeds` / `--all-seeds` flags.

### 4.1 Phase 1: Local-Only Baseline
Trains independent models per site and evaluates each model on its own test set and across all other sites.

```bash
# Single site, single seed
python scripts/run_local.py --site CMMD --seed 42

# All 4 sites, single seed
python scripts/run_local.py --seed 42

# Batch processing across all evaluation seeds (42, 43, 44)
python scripts/run_local.py --all-seeds
# or specify explicitly:
python scripts/run_local.py --seeds 42 43 44
```

### 4.2 Phase 2: Centralized Baseline (Pooled Ceiling)
Pools training data from all sites into a single dataset. Tests against a **particular dataset only** (rather than evaluating all datasets together).

```bash
# Test on a particular dataset (e.g., INbreast) using standard sample-weighted pooling
python scripts/run_centralized.py --site INbreast --seed 42

# Test on CMMD with site-balanced sampling (uniform representation across sites)
python scripts/run_centralized.py --site CMMD --seed 42 --site-balanced

# Batch test against all datasets individually in sequence (trains ONCE, evaluates each)
python scripts/run_centralized.py --all-sites --seed 42

# Batch process multiple seeds for a particular dataset
python scripts/run_centralized.py --site INbreast --seeds 42 43 44

# Force retraining the pooled model from scratch (trains once, then evaluates)
python scripts/run_centralized.py --all-sites --seed 42 --force-train

# Force re-evaluation only (re-tests using already-trained model without retraining)
python scripts/run_centralized.py --all-sites --seed 42 --force-eval
```

### 4.3 Phase 3: Federated Learning (FedAvg)
Runs simulated FedAvg rounds with local updates, server-side aggregation, and cosine round LR decay. Reuses trained model checkpoints on subsequent calls.

```bash
# Run FedAvg (default: 35 rounds, 2 local epochs/round = 70 epochs per client max, patience: 10)
python scripts/run_federated.py --seed 42

# Batch process all seeds
python scripts/run_federated.py --all-seeds

# Force retrain federated learning from round 1
python scripts/run_federated.py --seed 42 --force-train

# Force re-evaluation only (re-evaluates test sets with saved best.pt without retraining)
python scripts/run_federated.py --seed 42 --force-eval
```

### 4.4 Phase 4: Leave-One-Site-Out (Cross-Site Generalization Gap)
Trains on 3 sites and evaluates zero-shot transfer onto the held-out 4th site for both centralized and federated methods. Automatically reuses trained 3-site models to avoid redundant training.

```bash
# Single held-out site (e.g., test generalization to BCDR)
python scripts/run_cross_site.py --held-out BCDR --seed 42

# Run full cross-site gap analysis holding out each of the 4 sites in turn
python scripts/run_cross_site.py --seed 42

# Force retraining 3-site models from scratch
python scripts/run_cross_site.py --held-out BCDR --seed 42 --force-train

# Force re-evaluation only (recalculates gap & metrics using trained weights in seconds)
python scripts/run_cross_site.py --held-out BCDR --seed 42 --force-eval
```

---

## 5. Stored Artifacts & Analysis Outputs

Every execution systematically stores all weights, logs, and predictions under `results/`:

```
results/
├── all_runs_summary.csv                   # Master CSV tracking all runs and evaluations
├── local/
│   └── <site>/
│       └── seed_<seed>/
│           ├── best.pt                    # Best model checkpoint weights
│           ├── metrics.json               # Full evaluation metrics and configuration
│           ├── training_history.csv       # Per-epoch loss, val AUC, sens, VRAM, timing
│           ├── predictions_<eval_site>.csv# Per-sample test predictions
├── centralized_<variant>/
│   └── all_pooled/seed_<seed>/
│       ├── best.pt
│       ├── metrics.json
│       ├── training_history.csv
│       └── predictions_<site>.csv
└── federated_<weighting>/
    └── collaborative_4sites/seed_<seed>/
        ├── best.pt
        ├── metrics.json
        ├── training_history.csv
        └── predictions_<site>.csv
```

### Artifact Details
1. **`best.pt`**: PyTorch checkpoint saved at the epoch/round of maximum validation AUC.
2. **`training_history.csv`**: Contains `epoch`, `train_loss`, `val_auc`, `val_sensitivity`, `best_val_auc`, `time_sec`, `vram_mb`, and `improved`.
3. **`predictions_<site>.csv`**: Contains `patient_id`, `labels` (ground truth), `probs` (predicted probabilities), and `preds` (binary decision based on validation threshold). Enables ROC plotting and threshold calibration without re-running.
4. **`all_runs_summary.csv`**: Central tabular tracker automatically updated upon run completion for instant statistical analysis.
5. **`experiment_log.md`**: Clean human-readable Markdown tables appended in real time.

---

## 6. Configuration Reference (`configs/experiment.yaml`)

| Parameter | Default | Purpose |
|---|---|---|
| `img_size` | `224` | Input dimensions ($224 \times 224$) |
| `cache_images` | `true` | Pre-resize and hold images in RAM |
| `batch_size` | `32` | Minibatch size for training and inference |
| `max_epochs` | `30` | Maximum training epochs for local & centralized |
| `patience` | `5` | Early stopping patience (monitors validation AUC) |
| `lr` | `1.0e-4` | Adam learning rate |
| `weight_decay` | `1.0e-4` | L2 weight decay regularization |
| `freeze_bn` | `true` | Freeze BatchNorm layers across all models |
| `fed.rounds` | `30` | Communication rounds in FedAvg |
| `fed.local_epochs` | `2` | Local epochs completed per client before averaging |
| `fed.weighting` | `sample` | Client weight proportional to site sample count |
| `eval.seeds` | `[42, 43, 44]` | Default seeds for multi-run robustness |
| `eval.bootstrap_patients` | `1000` | Number of bootstrap resamples for 95% CIs |

---

## 7. Crash Resumption & Tracking Images Trained On

If a training session is interrupted midway (e.g., server reboot, power loss, or manual stop), the pipeline supports multi-level resumption:

### 7.1 Inspecting Images Already Trained On
- Every run automatically writes `train_images_manifest.csv` into its run directory (e.g. `results/local/<site>/seed_<seed>/train_images_manifest.csv`).
- This file lists the exact image filenames, `proc_path`, `patient_id`, and ground truth labels assigned to the training split for that run.
- Combined with `training_history.csv` (which records the last fully trained epoch), you can inspect precisely which images and batches were processed prior to interruption.

### 7.2 Epoch-Level & Round-Level Resumption
- **Local & Centralized**: At the conclusion of every epoch, `last_checkpoint.pt` is saved containing the model weights, optimizer state, best validation AUC, and history. When re-running the script, it detects `last_checkpoint.pt` and resumes from epoch $N+1$ without starting over.
- **Federated FedAvg**: At the conclusion of each communication round, `last_round.pt` is saved. Re-running `python scripts/run_federated.py` automatically picks up from round $R+1$ with client/server states preserved.

### 7.3 Pipeline-Level Resumption (Skipping Finished Sites)
- If a batch script (like `run_local.py` or `run_cross_site.py`) was training 4 sites and was interrupted at site 3, re-running the script will **automatically skip already completed sites** (where `metrics.json` exists) and continue immediately with the unfinished ones.
- To force retraining of an already completed run from scratch, pass the `--force` flag:
  ```bash
  python scripts/run_local.py --force --seed 42
  ```

---

## 8. GPU Thermal Monitoring & Dynamic Cooldown

To prevent thermal throttling on workstation/laptop GPUs (such as the RTX 2050), the pipeline includes dedicated thermal protection ([src/thermal.py](src/thermal.py)):

### How It Works:
1. **Periodic Interval Cooldown**: After every **10 epochs** (or rounds), training pauses automatically for **10 seconds** (`interval_cooldown_sec: 10`) to allow the GPU fan to vent accumulated heat.
2. **Dynamic Temperature Guard**: If the GPU temperature reaches or exceeds `temp_threshold` (**72°C**) at any point, an emergency 15-second cooldown is triggered.
3. **Per-Epoch Pause (Optional)**: If you also want a pause after *every* epoch, set `epoch_cooldown_sec`. By default it is `0` (disabled) so training runs at full speed between interval checkpoints.

### Configuration (`configs/experiment.yaml`):
```yaml
thermal:
  enabled: true
  temp_threshold: 72             # Emergency pause if GPU >= 72°C
  hot_cooldown_sec: 15           # Pause duration when hot (15s)
  cooldown_interval_epochs: 10   # Cooldown after every 10 epochs
  interval_cooldown_sec: 10      # Pause duration (10s)
  epoch_cooldown_sec: 0          # Optional pause after every epoch (0 = disabled)
```
