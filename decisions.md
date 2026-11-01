# FL-Mammo: Architectural & Methodological Decision Log

This document records the foundational technical and methodological decisions for the FL-Mammo system, explaining the context, alternatives considered, and verification procedures.

---

### Decision 1: Constant Learning Rate ($1.0 \times 10^{-4}$) vs Warmup + Cosine Annealing
- **What**: Enforce a constant learning rate of $1.0 \times 10^{-4}$ (`constant_lr: true`) across local, centralized, and federated pipelines, disabling linear warmup and cosine decay.
- **Why**: In previous iterations, local and centralized models ran with linear warmup and cosine decay peaking at $3.0 \times 10^{-4}$, while federated training ran at $1.0 \times 10^{-4}$ with round-level decay. This introduced an uncontrolled confounding variable where differences in performance were driven by optimizer scheduling rather than the training paradigm itself. Furthermore, LR spikes caused model collapse (flat probability outputs) on smaller sites.
- **How to verify**: Optimizer learning rate is logged on every epoch/round in `logs.jsonl` (`metric: "lr"`, `value: 0.0001`). Inspection of logs confirms zero LR drift.

---

### Decision 2: BatchNorm Frozen and Excluded from Federated Averaging
- **What**: Freeze all BatchNorm layers in ResNet-18 at ImageNet pretrained running statistics ($\mu, \sigma^2$) and affine parameters ($\gamma, \beta$), set `eval()` mode on all BN layers, and explicitly filter out any BN keys (`*.bn*`, `*running_mean*`, `*running_var*`, `*num_batches_tracked*`) during federated parameter averaging.
- **Why**: Medical imaging cross-site heterogeneity violates the IID assumption. Different mammography scanners (e.g., GE digital vs Hologic film) have vastly different intensity distributions. Averaging running statistics across scanners or allowing local batches to corrupt ImageNet normalization produces severe feature distortion and client drift (as documented by Li et al., 2021 [FedBN] and Hsieh et al., 2020).
- **How to verify**: Unit test `tests/test_federated.py::test_bn_keys_excluded_from_averaging` and `src/model.py::verify_bn_state` verify that BN weights/biases have `requires_grad=False`, all BN modules report `training=False`, and zero BN keys exist in the averaged state dictionary.

---

### Decision 3: Standard 224 px Input Resolution with Dedicated CBIS 384 px Test
- **What**: Use $224 \times 224$ as the default image resolution for all comparative benchmarks, and evaluate $384 \times 384$ in a targeted resolution study on CBIS-DDSM (`scripts/run_resolution_test.py`).
- **Why**: Training full $512 \times 512$ mammograms requires substantial VRAM and computational time (~4x increase per epoch), which exceeds our 20-day presentation timeline on a single 4 GB RTX 2050 GPU. CBIS-DDSM contains predominantly microcalcifications which are most sensitive to downsampling. If 384 px does not produce a statistically significant AUC gain outside the 95% CI on CBIS, retaining 224 px preserves efficiency without sacrificing validated performance.
- **Empirical Resolution Test Findings (Seeds 42, 53, 64)**:
  - 224 px: Mean AUC $0.4188 \pm 0.0198$, 95% CI $[0.3696, 0.4680]$
  - 384 px: Mean AUC $0.4527 \pm 0.0354$, 95% CI $[0.3649, 0.5405]$
  - Paired Difference: $+0.0339$, 95% CI $[-0.0480, +0.1158]$ (excludes zero: False)
  - Execution Time: 384 px was $3.5\times$ slower (~30-35s/epoch vs 8-10s/epoch at 224 px).
  - Decision: Under Rule 5, because the 95% CI of the paired difference includes zero, we cannot claim a statistically significant improvement. Standard 224 px is retained for efficiency across the benchmark matrix and this negative result is reported in the paper.

---

### Decision 4: Evaluation Seed Triplet (`[42, 53, 64]`)
- **What**: Standardize all experiments on three fixed random seeds: `42, 53, 64`.
- **Why**: Single-seed results showed up to $\pm 0.10$ AUC variance due to data ordering, weight initialization, and small validation batch stochasticity. A minimum of three seeds is required to report meaningful means and Student-$t$ 95% confidence intervals.
- **How to verify**: In `configs/experiment.yaml`, `eval.seeds: [42, 53, 64]`. The aggregation script `scripts/aggregate_results.py` asserts all three seeds are present before computing final summaries.

---

### Decision 5: Fixed Epoch Counts (20 Epochs) for Small Sites (BCDR, INbreast)
- **What**: In local training (`scripts/run_local.py`), train BCDR and INbreast for a fixed 20 epochs (`small_site_epochs: 20`) without early stopping.
- **Why**: BCDR has only ~193 patients (15% positive) and INbreast has 99 patients (29% positive). Their validation sets contain fewer than 45 images. Evaluating noisy validation batches during early training frequently triggered premature early stopping at epochs 10–11, before the classifier could converge.
- **How to verify**: Epoch count logged in `training_history.csv` confirms exactly 20 epochs executed for BCDR and INbreast local runs.

---

### Decision 6: Validation-Derived Decision Thresholds at 90% Target Specificity
- **What**: Select the binary classification decision threshold strictly on the validation set where False Positive Rate $FPR \le 0.10$ (i.e. $\text{Specificity} \ge 90\%$). Apply this frozen threshold to evaluate test set sensitivity.
- **Why**: Tuning decision thresholds directly on test sets violates ML evaluation hygiene and artificially inflates reported sensitivity. Setting a target of 90% specificity reflects clinical screening triage requirements, where unnecessary recalls must be minimized.
- **How to verify**: `metrics.json` records `val_decision_threshold` and confirms `sens_at_90_spec` on the test set is calculated using that exact threshold.

