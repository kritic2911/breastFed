# Experiment Log — Federated Learning Mammography

## 2026-10-20 | Session 1: Discovery & Planning

### Data inventory complete
| Site | Labeled images | Patients | Positive rate | Cached images |
|------|:-:|:-:|:-:|:-:|
| CMMD | 3,702 | 1,757 | 70% | 5,132 |
| CBIS | 770 | 461 | 40% | 881 |
| BCDR | 485 | 193 | 15% | 485 |
| INbreast | 343 | 99 | 29% | 410 |

### Key findings & Completed Setup
- **PyTorch & CUDA**: PyTorch 2.14.1+cu130, CUDA active on NVIDIA GeForce RTX 2050.
- **Data Splits**: Patient-grouped stratified splits generated with zero leakage across all 4 sites in `labels_split.csv`.
- **Pipeline Optimizations**: Enabled in-memory caching of pre-resized $224 \times 224$ images, persistent workers, prefetching, and non-blocking CUDA transfers.
- **Documentation**: Comprehensive usage guide created at [USAGE_GUIDE.md](USAGE_GUIDE.md).

---

## 2026-10-20 | Session 2: Phase 2 Local-Only Baseline (Seed 42)

### Run: Local Baseline | Trained on CMMD (Seed 42)
- **Val Decision Threshold**: 0.7924 (derived at 90% target specificity on CMMD validation set)
- **Early Stopping**: Epoch 26 / Best Val AUC: 0.7545

| Eval Site | Role | Test Image AUC [95% CI] | Sens @ 90% Spec | Patient AUC | Note |
|---|---|:---:|:---:|:---:|---|
| **CMMD** | **Own site** | **0.7212 [0.6741, 0.7657]** | 0.4663 | 0.7384 | Target baseline floor for CMMD |
| **CBIS** | Cross-site | 0.5254 [0.4248, 0.6168] | 0.3226 | 0.5361 | Near random guessing on CBIS |
| **BCDR** | Cross-site | 0.5953 [0.3727, 0.8085] | 0.2500 | 0.6750 | Wide CI due to small positive sample size |
| **INbreast** | Cross-site | 0.8133 [0.6865, 0.9238] | 0.6500 | 0.8596 | Strong zero-shot transfer |

### Run: Local Baseline | Trained on CBIS (Seed 42)
- **Val Decision Threshold**: 0.4412
- **Early Stopping**: Epoch 11 / Best Val AUC: 0.5113

| Eval Site | Role | Test Image AUC [95% CI] | Sens @ 90% Spec | Patient AUC | Note |
|---|---|:---:|:---:|:---:|---|
| **CBIS** | **Own site** | **0.3973 [0.2860, 0.5089]** | 0.0806 | 0.3849 | Small local sample size, severe underfitting |
| **CMMD** | Cross-site | 0.4300 [0.3859, 0.4793] | 0.3218 | 0.4197 | Fails to generalize to CMMD |
| **BCDR** | Cross-site | 0.4855 [0.2974, 0.6974] | 0.1250 | 0.5167 | Random-level transfer |
| **INbreast** | Cross-site | 0.3990 [0.2227, 0.5750] | 0.1500 | 0.4079 | Inverse/random performance |

### Run: Local Baseline | Trained on BCDR (Seed 42)
- **Val Decision Threshold**: 0.1226
- **Early Stopping**: Epoch 22 / Best Val AUC: 0.7389

| Eval Site | Role | Test Image AUC [95% CI] | Sens @ 90% Spec | Patient AUC | Note |
|---|---|:---:|:---:|:---:|---|
| **BCDR** | **Own site** | **0.5274 [0.3231, 0.7387]** | 0.0000 | 0.5125 | Highly imbalanced (15% positive), wide CI |
| **CMMD** | Cross-site | 0.5467 [0.4989, 0.5931] | 0.1118 | 0.5462 | Weak cross-scanner transfer |
| **CBIS** | Cross-site | 0.6278 [0.5369, 0.7171] | 0.2097 | 0.6481 | Moderate generalization |
| **INbreast** | Cross-site | 0.5388 [0.3729, 0.7044] | 0.1500 | 0.5895 | Near random generalization |

### Run: Local Baseline | Trained on INbreast (Seed 42)
- **Val Decision Threshold**: 0.2558
- **Early Stopping**: Epoch 10 / Best Val AUC: 0.6731

| Eval Site | Role | Test Image AUC [95% CI] | Sens @ 90% Spec | Patient AUC | Note |
|---|---|:---:|:---:|:---:|---|
| **INbreast** | **Own site** | **0.6561 [0.5141, 0.8091]** | 0.3500 | 0.6542 | Modest local sample size (219 train) |
| **CMMD** | Cross-site | 0.5232 [0.4824, 0.5638] | 0.1811 | 0.5284 | Limited transfer to large cohort |
| **CBIS** | Cross-site | 0.5378 [0.4425, 0.6393] | 0.1452 | 0.5428 | Marginal cross-site transfer |
| **BCDR** | Cross-site | 0.5373 [0.3967, 0.6750] | 0.1250 | 0.5625 | Poor transfer to film mammography |

---

### Local-Only Baseline: Full Cross-Site Performance Matrix (Test AUC)

Diagonal elements (**bold**) represent self-site evaluation (the "floor"). Off-diagonal elements show cross-site transfer.

| Trained On ↓ \ Evaluated On → | CMMD Test | CBIS Test | BCDR Test | INbreast Test | Average Cross-Site |
|---|:---:|:---:|:---:|:---:|:---:|
| **Local CMMD** | **0.7212** | 0.5254 | 0.5953 | 0.8133 | 0.6447 |
| **Local CBIS** | 0.4300 | **0.3973** | 0.5015 | 0.7276 | 0.5530 |
| **Local BCDR** | 0.5467 | 0.6278 | **0.5274** | 0.5388 | 0.5711 |
| **Local INbreast** | 0.5232 | 0.5378 | 0.5373 | **0.6561** | 0.5328 |

---



### Run: FEDERATED_SAMPLE | Train: `collaborative_4sites` | Seed: `42` (2026-10-20 22:25)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CBIS** | Cross-site | 0.4674 [0.3594, 0.5793] | 0.1935 | 0.4794 |
| **BCDR** | Cross-site | 0.5267 [0.3737, 0.6896] | 0.0000 | 0.6000 |
| **INbreast** | Cross-site | 0.6551 [0.5062, 0.7930] | 0.0000 | 0.5300 |

---

### Run: CENTRALIZED_SAMPLE_WEIGHTED | Train: `all_pooled` | Seed: `42` (2026-10-21 20:57)

- **Best Epoch**: 19 (Val AUC: 0.8144)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.7359 [0.6882, 0.7749] | 0.5992 | 0.7651 |
| **CBIS** | Cross-site | 0.6086 [0.5081, 0.7046] | 0.2097 | 0.5588 |
| **BCDR** | Cross-site | 0.7416 [0.5507, 0.9059] | 0.2500 | 0.8458 |
| **INbreast** | Cross-site | 0.9102 [0.7714, 1.0000] | 0.7000 | 0.9000 |

---

### Run: FEDERATED_SAMPLE | Train: `collaborative_4sites` | Seed: `42` (2026-10-22 01:02)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CBIS** | Cross-site | 0.4674 [0.3594, 0.5793] | 0.1935 | 0.4794 |
| **BCDR** | Cross-site | 0.5267 [0.3737, 0.6896] | 0.0000 | 0.6000 |
| **INbreast** | Cross-site | 0.6551 [0.5062, 0.7930] | 0.0000 | 0.5300 |

---

### Run: FEDERATED_SAMPLE | Train: `collaborative_4sites` | Seed: `42` (2026-10-22 01:25)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CBIS** | Cross-site | 0.4674 [0.3594, 0.5793] | 0.1935 | 0.4794 |
| **BCDR** | Cross-site | 0.5267 [0.3737, 0.6896] | 0.0000 | 0.6000 |
| **INbreast** | Cross-site | 0.6551 [0.5062, 0.7930] | 0.0000 | 0.5300 |

---

### Run: LOCAL | Train: `CMMD` | Seed: `42` (2026-10-22 10:40)

- **Best Epoch**: 30 (Val AUC: 0.7491)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Own site | 0.7411 [0.6921, 0.7815] | 0.4143 | 0.7450 |
| **CBIS** | Cross-site | 0.5249 [0.4177, 0.6296] | 0.2258 | 0.5098 |
| **BCDR** | Cross-site | 0.7035 [0.5135, 0.8900] | 0.1875 | 0.7333 |
| **INbreast** | Cross-site | 0.7439 [0.5160, 0.9146] | 0.5500 | 0.8000 |

---

### Run: LOCAL | Train: `CBIS` | Seed: `42` (2026-10-22 10:40)

- **Best Epoch**: 1 (Val AUC: 0.5591)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.4300 [0.3859, 0.4793] | 0.3218 | 0.4197 |
| **CBIS** | Own site | 0.3973 [0.2860, 0.5089] | 0.0806 | 0.3849 |
| **BCDR** | Cross-site | 0.5015 [0.3133, 0.6896] | 0.3125 | 0.6083 |
| **INbreast** | Cross-site | 0.7276 [0.6171, 0.8452] | 0.1000 | 0.6100 |

---

### Run: LOCAL | Train: `BCDR` | Seed: `42` (2026-10-22 10:41)

- **Best Epoch**: 3 (Val AUC: 0.5699)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.5048 [0.4489, 0.5591] | 0.0096 | 0.4564 |
| **CBIS** | Cross-site | 0.5703 [0.4699, 0.6679] | 0.2258 | 0.5212 |
| **BCDR** | Own site | 0.5793 [0.3416, 0.7822] | 0.0625 | 0.5375 |
| **INbreast** | Cross-site | 0.4500 [0.2660, 0.6194] | 0.2000 | 0.5500 |

---

### Run: LOCAL | Train: `INbreast` | Seed: `42` (2026-10-22 10:42)

- **Best Epoch**: 9 (Val AUC: 0.5881)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.6012 [0.5485, 0.6487] | 0.0906 | 0.5841 |
| **CBIS** | Cross-site | 0.6265 [0.5368, 0.7163] | 0.2258 | 0.5955 |
| **BCDR** | Cross-site | 0.5838 [0.3939, 0.7756] | 0.1875 | 0.5750 |
| **INbreast** | Own site | 0.6255 [0.4336, 0.7900] | 0.1000 | 0.6800 |

---

### Run: CENTRALIZED | Train: `sample_weighted` | Seed: `42` (2026-10-22 11:15)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **INbreast** | Cross-site | 0.9347 [0.8329, 0.9951] | 0.6000 | 0.9500 |

---

### Run: CENTRALIZED | Train: `sample_weighted` | Seed: `42` (2026-10-22 11:19)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.7301 [0.6778, 0.7714] | 0.5318 | 0.7364 |

---

### Run: CENTRALIZED | Train: `sample_weighted` | Seed: `42` (2026-10-22 11:19)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CBIS** | Cross-site | 0.6041 [0.4929, 0.7179] | 0.1290 | 0.6022 |

---

### Run: CENTRALIZED | Train: `sample_weighted` | Seed: `42` (2026-10-22 11:20)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **BCDR** | Cross-site | 0.7812 [0.6348, 0.9026] | 0.1250 | 0.8292 |

---

### Run: FEDERATED_SAMPLE | Train: `collaborative_4sites` | Seed: `42` (2026-10-22 11:50)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.7034 [0.6561, 0.7440] | 0.4335 | 0.7407 |
| **CBIS** | Cross-site | 0.5480 [0.4409, 0.6555] | 0.1774 | 0.5815 |
| **BCDR** | Cross-site | 0.6204 [0.4414, 0.7867] | 0.1875 | 0.5750 |
| **INbreast** | Cross-site | 0.8459 [0.6679, 0.9729] | 0.5500 | 0.8700 |

---

### Run: FEDERATED_SAMPLE | Train: `train_CBIS_BCDR_INbreast` | Seed: `42` (2026-10-22 12:07)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CBIS** | Cross-site | 0.4674 [0.3594, 0.5793] | 0.1935 | 0.4794 |
| **BCDR** | Cross-site | 0.5267 [0.3737, 0.6896] | 0.0000 | 0.6000 |
| **INbreast** | Cross-site | 0.6551 [0.5062, 0.7930] | 0.0000 | 0.5300 |

---

### Run: FEDERATED_SAMPLE | Train: `train_CMMD_BCDR_INbreast` | Seed: `42` (2026-10-22 12:55)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.7096 [0.6651, 0.7472] | 0.4855 | 0.7261 |
| **BCDR** | Cross-site | 0.6761 [0.4367, 0.8720] | 0.3125 | 0.6958 |
| **INbreast** | Cross-site | 0.8480 [0.6357, 0.9834] | 0.6000 | 0.8600 |

---

### Run: FEDERATED_SAMPLE | Train: `train_CMMD_CBIS_INbreast` | Seed: `42` (2026-10-22 15:36)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.7051 [0.6582, 0.7428] | 0.4143 | 0.7226 |
| **CBIS** | Cross-site | 0.4841 [0.3814, 0.5785] | 0.2258 | 0.5284 |
| **INbreast** | Cross-site | 0.8449 [0.6635, 0.9755] | 0.6000 | 0.8300 |

---

### Run: FEDERATED_SAMPLE | Train: `train_CMMD_CBIS_BCDR` | Seed: `42` (2026-10-22 17:30)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.7389 [0.6914, 0.7795] | 0.4489 | 0.7474 |
| **CBIS** | Cross-site | 0.5575 [0.4483, 0.6652] | 0.2419 | 0.5573 |
| **BCDR** | Cross-site | 0.6418 [0.4116, 0.8443] | 0.3750 | 0.6792 |

---

### Run: LOCAL | Train: `CMMD` | Seed: `42` (2026-10-23 00:14)

- **Best Epoch**: 44 (Val AUC: 0.7283)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Own site | 0.7110 [0.6621, 0.7541] | 0.4008 | 0.6960 |
| **CBIS** | Cross-site | 0.5312 [0.4257, 0.6292] | 0.1774 | 0.5227 |
| **BCDR** | Cross-site | 0.5556 [0.3812, 0.7653] | 0.3125 | 0.5708 |
| **INbreast** | Cross-site | 0.8194 [0.6335, 0.9607] | 0.7000 | 0.8600 |

---

### Run: LOCAL | Train: `CBIS` | Seed: `42` (2026-10-23 00:16)

- **Best Epoch**: 4 (Val AUC: 0.6341)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.5201 [0.4714, 0.5672] | 0.5222 | 0.5126 |
| **CBIS** | Own site | 0.5115 [0.4244, 0.6083] | 0.2903 | 0.5960 |
| **BCDR** | Cross-site | 0.5793 [0.4212, 0.7176] | 0.3125 | 0.5875 |
| **INbreast** | Cross-site | 0.6408 [0.4836, 0.7800] | 0.3000 | 0.5000 |

---

### Run: LOCAL | Train: `BCDR` | Seed: `42` (2026-10-23 00:17)

- **Best Epoch**: 1 (Val AUC: 0.7354)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.5547 [0.5065, 0.6011] | 0.2929 | 0.5533 |
| **CBIS** | Cross-site | 0.5385 [0.4427, 0.6366] | 0.3871 | 0.5630 |
| **BCDR** | Own site | 0.6143 [0.4739, 0.7597] | 0.0625 | 0.5500 |
| **INbreast** | Cross-site | 0.6490 [0.5144, 0.7878] | 0.3000 | 0.8400 |

---

### Run: LOCAL | Train: `INbreast` | Seed: `42` (2026-10-23 00:20)

- **Best Epoch**: 12 (Val AUC: 0.6747)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.5875 [0.5375, 0.6361] | 0.9364 | 0.5673 |
| **CBIS** | Cross-site | 0.3740 [0.2703, 0.4843] | 0.4516 | 0.4365 |
| **BCDR** | Cross-site | 0.4520 [0.2619, 0.6318] | 0.3750 | 0.4667 |
| **INbreast** | Own site | 0.6112 [0.3976, 0.8426] | 0.3500 | 0.6800 |

---

### Run: CENTRALIZED | Train: `sample_weighted` | Seed: `42` (2026-10-23 12:05)

- **Best Epoch**: 45 (Val AUC: 0.8018)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.7297 [0.6837, 0.7703] | 0.5761 | 0.7360 |

---

### Run: CENTRALIZED | Train: `sample_weighted` | Seed: `42` (2026-10-23 12:13)

- **Best Epoch**: 45 (Val AUC: 0.8018)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.7297 [0.6837, 0.7703] | 0.5761 | 0.7360 |

---

### Run: CENTRALIZED | Train: `sample_weighted` | Seed: `42` (2026-10-23 12:13)

- **Best Epoch**: 45 (Val AUC: 0.8018)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CBIS** | Cross-site | 0.5806 [0.4863, 0.6768] | 0.1774 | 0.6068 |

---

### Run: CENTRALIZED | Train: `sample_weighted` | Seed: `42` (2026-10-23 12:13)

- **Best Epoch**: 45 (Val AUC: 0.8018)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **BCDR** | Cross-site | 0.7081 [0.5547, 0.8560] | 0.0625 | 0.7375 |

---

### Run: CENTRALIZED | Train: `sample_weighted` | Seed: `42` (2026-10-23 12:14)

- **Best Epoch**: 45 (Val AUC: 0.8018)

| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **INbreast** | Cross-site | 0.9388 [0.8301, 0.9987] | 0.6000 | 0.9600 |

---

### Run: FEDERATED_SAMPLE | Train: `train_CBIS_BCDR_INbreast` | Seed: `42` (2026-10-23 13:03)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CBIS** | Cross-site | 0.4652 [0.3555, 0.5797] | 0.1935 | 0.4773 |
| **BCDR** | Cross-site | 0.5290 [0.3801, 0.6872] | 0.0000 | 0.6042 |
| **INbreast** | Cross-site | 0.6531 [0.5078, 0.7894] | 0.0000 | 0.5300 |

---

### Run: FEDERATED_SAMPLE | Train: `train_CMMD_CBIS_BCDR` | Seed: `42` (2026-10-23 14:49)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.7392 [0.6924, 0.7803] | 0.4547 | 0.7463 |
| **CBIS** | Cross-site | 0.5572 [0.4482, 0.6670] | 0.2258 | 0.5573 |
| **BCDR** | Cross-site | 0.6410 [0.4115, 0.8467] | 0.3750 | 0.6750 |

---

### Run: FEDERATED_SAMPLE | Train: `collaborative_4sites` | Seed: `42` (2026-10-23 16:11)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.6078 [0.5536, 0.6607] | 0.3796 | 0.5922 |
| **CBIS** | Cross-site | 0.3771 [0.2728, 0.4910] | 0.0000 | 0.3772 |
| **BCDR** | Cross-site | 0.5838 [0.4255, 0.7636] | 0.0000 | 0.6000 |
| **INbreast** | Cross-site | 0.6184 [0.4204, 0.8199] | 0.0500 | 0.5900 |

---

### Run: FEDERATED_SAMPLE | Train: `train_CBIS_BCDR_INbreast` | Seed: `42` (2026-10-23 16:33)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CBIS** | Cross-site | 0.4434 [0.3566, 0.5332] | 0.0645 | 0.4523 |
| **BCDR** | Cross-site | 0.2889 [0.1279, 0.5130] | 0.0625 | 0.2375 |
| **INbreast** | Cross-site | 0.6286 [0.4777, 0.7906] | 0.1500 | 0.5600 |

---

### Run: FEDERATED_SAMPLE | Train: `train_CMMD_BCDR_INbreast` | Seed: `42` (2026-10-23 17:03)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.5416 [0.4862, 0.5938] | 0.2659 | 0.5330 |
| **BCDR** | Cross-site | 0.4752 [0.2590, 0.6963] | 0.0000 | 0.4250 |
| **INbreast** | Cross-site | 0.6673 [0.4846, 0.8649] | 0.0000 | 0.6700 |

---

### Run: FEDERATED_SAMPLE | Train: `train_CMMD_CBIS_INbreast` | Seed: `42` (2026-10-23 17:48)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.5357 [0.4811, 0.5870] | 0.2524 | 0.5273 |
| **CBIS** | Cross-site | 0.3825 [0.2796, 0.4969] | 0.0000 | 0.4420 |
| **INbreast** | Cross-site | 0.6276 [0.4366, 0.8111] | 0.0000 | 0.5600 |

---

### Run: FEDERATED_SAMPLE | Train: `train_CMMD_CBIS_BCDR` | Seed: `42` (2026-10-23 18:17)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.6163 [0.5590, 0.6679] | 0.2601 | 0.5819 |
| **CBIS** | Cross-site | 0.4665 [0.3683, 0.5680] | 0.0000 | 0.4293 |
| **BCDR** | Cross-site | 0.5819 [0.3855, 0.7908] | 0.1250 | 0.5792 |

---

### Run: ABLATION_HARMONIZATION_PROC512 | Train: `collaborative_4sites` | Seed: `43` (2026-10-27 13:10)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.5208 [0.4688, 0.5721] | 0.1040 | 0.5238 |
| **CBIS** | Cross-site | 0.5290 [0.4330, 0.6291] | 0.2419 | 0.5516 |
| **BCDR** | Cross-site | 0.5991 [0.4002, 0.7790] | 0.0625 | 0.5792 |
| **INbreast** | Cross-site | 0.4735 [0.3182, 0.6291] | 0.0000 | 0.2800 |

---

### Run: ABLATION_HARMONIZATION_PROC512 | Train: `collaborative_4sites` | Seed: `43` (2026-10-27 22:30)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.5388 [0.4830, 0.5946] | 0.2794 | 0.5164 |
| **CBIS** | Cross-site | 0.3875 [0.2839, 0.5070] | 0.0484 | 0.4438 |
| **BCDR** | Cross-site | 0.4596 [0.2832, 0.6662] | 0.0000 | 0.3833 |
| **INbreast** | Cross-site | 0.5760 [0.3965, 0.7725] | 0.0500 | 0.5050 |

---

### Run: FEDERATED_SAMPLE | Train: `collaborative_4sites` | Seed: `0` (2026-10-29 10:25)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 0.7163 [0.6694, 0.7642] | 0.4682 | 0.7134 |
| **CBIS** | Cross-site | 0.5697 [0.4423, 0.6786] | 0.1935 | 0.5841 |
| **BCDR** | Cross-site | 0.6738 [0.4780, 0.9039] | 0.3125 | 0.6958 |
| **INbreast** | Cross-site | 0.8561 [0.6259, 0.9922] | 0.7500 | 0.8800 |

---

### Run: SMOKE_FEDAVG | Train: `collaborative_4sites` | Seed: `0` (2026-11-01 01:00)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 1.0000 [1.0000, 1.0000] | 0.0000 | 1.0000 |
| **CBIS** | Cross-site | 0.2500 [0.2500, 0.2500] | 0.0000 | 0.0000 |
| **BCDR** | Cross-site | 0.5000 [0.5000, 0.5000] | 0.5000 | 1.0000 |
| **INbreast** | Cross-site | 0.5000 [0.5000, 0.5000] | 0.0000 | 0.0000 |

---

### Run: SMOKE_FEDAVG | Train: `collaborative_4sites` | Seed: `0` (2026-11-01 01:01)


| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |
|---|---|:---:|:---:|:---:|
| **CMMD** | Cross-site | 1.0000 [1.0000, 1.0000] | 0.0000 | 1.0000 |
| **CBIS** | Cross-site | 0.2500 [0.2500, 0.2500] | 0.0000 | 0.0000 |
| **BCDR** | Cross-site | 0.5000 [0.5000, 0.5000] | 0.5000 | 1.0000 |
| **INbreast** | Cross-site | 0.5000 [0.5000, 0.5000] | 0.0000 | 0.0000 |

---
