# FL-Mammo: Implementation Plan and Agent Prompt

Oct 31, 2026 · @kriti

**Deadline: presentable by Friday 20 November 2026 (20 days).** The deliverable is the federated breast-cancer detection system you promised: four simulated hospitals, local vs centralized vs federated results with confidence intervals, domain-shift handling, a containerised federated run that survives failures, and explainable outputs. Sections 1 to 8 are the 20-day plan. Monitoring, full CI/CD, cloud, privacy attacks and consensus move to Part B, to do after the deadline.

## 1. Agent brief (paste this first)

This plan takes FL-Mammo from a shaky baseline to a federated breast-cancer detection system you can present on 20 November and defend in an interview. Each phase has what the agent builds, what you study, what to look for, and a gate that must pass before the next phase starts.

**Role.** You are a senior ML-systems engineer pair-programming with a student. Build in small, tested steps, and explain every non-trivial change in three lines: what, why, how to verify.

**Project.** Federated mammography classification (malignant vs benign) with four datasets acting as hospitals: CMMD, CBIS-DDSM, INbreast, BCDR. Model: ResNet-18, ImageNet init, frozen BN. Hardware: RTX 2050 with 4 GB VRAM under WSL2, so only one GPU client at a time.

**Rules of engagement**

1. Work one phase at a time. Do not start the next until the gate passes and results are in the ledger (section 8). GPU phases run overnight while you build the next one.
2. Never tune on the test set. Thresholds and epochs come from validation; each final config touches test once.
3. Every run writes: resolved config, config hash, git SHA, seed, environment, `logs.jsonl`, `metrics.json`, `preds.csv`.
4. No patient images or labels in git, Docker images or CI. CI uses synthetic data only. Check each dataset's licence before putting it anywhere beyond your laptop.
5. Three seeds minimum (42, 53, 64). Claim an improvement only when the 95% CI of the paired difference excludes zero. Report negative results.
6. If a gate fails twice, stop and report: what failed, the evidence, two hypotheses, the cheapest test for each.
7. Label honestly: INbreast labels come from BI-RADS (radiologist assessment), CMMD and CBIS from pathology. Never call a BI-RADS-derived output a diagnosis.
8. Scope is frozen. Anything not in the Must list in section 7 goes to Part B, however interesting.
9. Maintain a usage_guide.md file explaining how to run the system, how to interpret the results, and what the limitations of the system are. Update it whenever you learn something new or important.
10. Maintain experiment_log.md with all the experiments, their results, and lessons learned for regular review.

**Gate report format** (the agent posts this at every checkpoint)

- Done: what was built or run.
- Evidence: numbers, screenshots, test output.
- Failed or surprising: anything off.
- Decision: proceed, redo, or ask.

**Definition of done for the project**

- Reproducible on synthetic data with one command (`make smoke`, `docker compose up`).
- Local, centralized and federated compared on all four sites, 3 seeds, with CIs.
- Domain-shift measurement plus ablations A1 to A3 (A4 if time allows).
- Containerised federated run with TLS, round timeouts with partial aggregation, and checkpoint resume; fault table filled from observed runs.
- Calibrated probability, risk band, Grad-CAM with sanity checks, and per-image and per-run reports.
- Minimal CI (lint, tests, synthetic smoke test) green on every push.
- A ledger that regenerates every table and figure; a short report and a 12-minute deck with a demo.

## 2. Where you are now: what to change and rerun

Your federated pipeline is now sound (aggregation test, BN exclusion, constant LR, one-client run), but every number in the old results CSVs predates those fixes. Treat the old numbers as discarded and rerun everything in Phase 1.

| Issue | Evidence | Fix | Phase |
| --- | --- | --- | --- |
| Federated regression between runs | All 16 federated rows dropped (mean about 0.15 AUC) from Oct 22 to Oct 23 | LR spikes and BN bookkeeping fixed; rerun all federated | 1 |
| Local and centralized used the old LR schedule | Peak LR about 3e-4 vs config 1e-4 | Rerun at constant 1e-4 so comparisons are fair | 1 |
| One seed only | All rows seed 42; same-config reruns differed by 0.10 on average | Seeds 42, 53, 64 for every config | 1 |
| Tiny val and test sets | CI half-width: CMMD ±0.05, CBIS ±0.10, INbreast ±0.16, BCDR ±0.18 | Fixed epoch counts for small sites; report CIs; no claims inside overlap | 1 |
| Pooled val AUC is inflated | Sites differ in malignant prevalence, so the model can score "which site" | Log val AUC per site | 1 |
| Per-client loss 1.0 to 1.3 on BCDR, INbreast, CBIS | Worse than a constant prediction (about 0.69) | Compare mean predicted probability to site prevalence; per-site calibration | 1, 3 |
| BN flags unverified | BN params now excluded from averaging | Print `requires_grad` and train/eval mode for all BN layers | 1 |
| Resolution may hide calcifications | 224 px from 512 px; CBIS is mostly calcifications | Local CBIS at 224, 384, 512 with 3 seeds before paying for it everywhere | 1 |
| Result-file bookkeeping | Duplicate centralized CMMD row; `is_own_site` false for pooled rows | Dedupe on (run\_id, method, eval\_site); define the flag per method | 0 |
| Realistic ceilings | CBIS full-image classifiers typically land around 0.6 to 0.7 | Set expectations; a CBIS AUC near 0.6 is not a bug | 1 |

## 3. Phases 0 and 1: foundations and baseline repair

### Phase 0: repo, config and logging foundations (about 1 day)

**Agent does**

1. Git repo with layout `src/ configs/ scripts/ tests/ docker/ paper/ results/` (`results/` and all data gitignored).
2. One YAML config. Each run gets `run_id = <timestamp>-<config hash>-<git sha>` and saves its resolved config.
3. Structured logging: one JSON object per line in `logs.jsonl` (`ts, run_id, event, round, site, split, metric, value`), plus `metrics.json` and `preds.csv`.
4. `pytest` suite: aggregation identity (you have it), weights sum to 1, no patient appears in two splits, labels are in {0, 1}, BN keys are excluded from averaging.
5. `ruff` and a `Makefile` with `test`, `smoke`, `run`.
6. A synthetic dataset generator (noise images with a planted signal, four fake sites with different intensity statistics) for tests, smoke runs and the container demo.

**You study:** pytest fixtures and parametrisation; run provenance (config hash, git SHA, seed).

**Look for:** every number you will later put in the paper can be traced to a run directory.

**Gate:** `make test` is green; `make smoke` runs 2 federated rounds on synthetic data on CPU in under 2 minutes; a run directory holds all artifacts from rule 3.

### Phase 1: baseline repair and full rerun (about 6 days, mostly GPU time)

**Agent does**

1. Add per-site val AUC, mean predicted probability, probability std, and per-client loss to every round log. Print BN `requires_grad` and train/eval state once per run.
2. Resolution test: local CBIS at 224 and 384 px (512 px is too slow for a 20-day plan), seeds 42, 53, 64. Keep the larger size only if the AUC gain is outside the CI; otherwise stay at 224 and say so in the paper.
3. Rerun the full matrix at constant LR 1e-4: local (4 sites, with cross-site eval), centralized (pooled and site-balanced), federated sample-weighted. Seeds 42, 53, 64.
4. Small sites (BCDR, INbreast): fixed epoch count or merged train+val with cross-validation; no early stopping on a validation set of about 50 images.
5. Aggregation script: mean and 95% CI over seeds, plus patient-level bootstrap CIs per run, saved to the ledger.
6. Calibration: reliability curve and ECE per site; per-site temperature scaling fitted on val.

**Cost:** a 35-round federated run took about 130 s per round at 224 px, so about 76 minutes per seed, about 4 hours for three seeds. Start it overnight.

**You study:** bootstrap confidence intervals; why pooled AUC over sites is misleading; calibration (Guo et al. 2017).

**Look for:** per-site probability std staying above about 0.05 (below 0.01 means collapse); the federated model beating local training on the small sites; train-val gaps.

**Gate**

- A table of 3 methods × 4 sites, mean ± CI over three seeds (42, 53, 64).
- CMMD AUC at least about 0.70 in every method.
- Rerunning the same seed reproduces test AUC within 0.03.
- No site with a CI entirely below 0.5. If one appears, repeat the flip test and preview check before continuing.

## 4. Phase 2: domain-shift study and FL method ablations (about 1 week)

The goal is to turn "federated is worse on some sites" into a measured study of why, and which methods fix it. Every ablation changes one thing against the Phase 1 baseline.

**Agent does**

1. **Measure the shift before fixing it.** Extract frozen ResNet-18 embeddings for every image. (a) Train a logistic-regression probe to predict the site; accuracy near 100% means a strong shift. Repeat on `raw512` and `proc512`: the drop is your harmonization effect. (b) Per-site intensity histograms, raw vs processed. (c) MMD or CORAL distance between site embeddings. (d) t-SNE or UMAP coloured by site, then by label.
2. **Ablation ladder** (3 seeds each, same splits, per-site results plus leave-one-site-out):

| ID | Change vs baseline | Question it answers | Main reference |
| --- | --- | --- | --- |
| A1 | `raw512` instead of `proc512` | Does harmonization help convergence and cross-site AUC? | Tzortzis et al. 2025; your pitch |
| A2 | Uniform instead of sample-weighted aggregation | Does CMMD (71% of data) drown the small sites? | McMahan 2017; Quintana 2026 |
| A3 | FedProx, mu in {0.01, 0.1} | Does limiting client drift help? | Li 2020 |
| A4 (optional) | FedBN: trainable BN, BN params and stats kept local | Does handling film vs digital statistics locally help? | Li 2021 |

Moved to Part B: GroupNorm, server optimizers, Mammo-CLIP initialisation, SCAFFOLD.

3. **Shortcut checks.** Site-balanced sampling vs pooled; look for burned-in text and markers in Grad-CAM (Phase 3); compare the site-probe accuracy of embeddings before and after training.
4. **Statistics.** Paired patient-level bootstrap of every ablation against baseline per site. A change counts only when the CI excludes zero.

**You study**

- McMahan 2017 (FedAvg), Li 2020 (FedProx), Li 2021 (FedBN), Hsieh 2020 (why BN fails under non-IID data).
- Kairouz et al. 2021, the heterogeneity chapter.
- Roth et al. 2020: what changed when breast-density FL moved from simulation to real hospitals.
- Zech 2018 and DeGrave 2021: how models learn site and scanner shortcuts instead of pathology.

**Look for:** which site benefits most from federation (usually the smallest); whether harmonization shrinks the site-probe accuracy; whether any method helps CBIS beyond its CI.

**Gate**

- A domain-shift table (site probe accuracy, MMD) and one embedding plot.
- The ablation table with paired CIs; a one-paragraph conclusion per row: supported, not supported, or inconclusive.
- A1 to A3 completed (A4 if time allows). Inconclusive results are valid results if the CIs are honest.

## 5. Phase 3: explainability, calibration and BI-RADS-style reports (about 2 days)

Decision: keep benign/malignant as the trained target. Add a calibrated probability, a risk band, a saliency map and a templated report per image. Do not train a 5-class BI-RADS head across all sites: only INbreast has BI-RADS labels, and BI-RADS 4/5 means "suspicious", not confirmed malignant.

**Agent does**

1. **Calibration.** Per-site temperature scaling fitted on validation only. Report ECE before and after.
2. **Risk bands.** Low, Intermediate, High, with cut-offs chosen on validation (for example Low where sensitivity is at least 95%, High where specificity is at least 90%). Name it "model-suggested category, BI-RADS-style". Validate on INbreast's test set: Spearman correlation between probability and true BI-RADS, and a band × BI-RADS cross-tab. If the ordering is not monotone, say so.
3. **Grad-CAM** on the last ResNet block, overlaid on `proc512`. Sanity check: randomise the weights and confirm the map changes. Where lesion masks exist (CBIS, INbreast, BCDR) report the fraction of CAM mass inside the mask.
4. **Report generator.** A Jinja2 template fills: site, image id, view, laterality, calibrated probability, band, CAM overlay, and a fixed caveat line (research use, not a diagnosis, label provenance of the site). Reports are generated locally on each client; only aggregate metrics leave the site. Plus a per-run report: per-site AUC with CI, sensitivity at 90% specificity, ECE.
5. **Failure gallery.** Top false positives and negatives with CAMs; check for burned-in text and corner markers.

Moved to Part B: MC-dropout uncertainty, the INbreast ordinal BI-RADS head, and the model card.

**You study:** Grad-CAM (Selvaraju 2017); sanity checks for saliency maps (Adebayo 2018); calibration (Guo 2017).

**Look for:** CAMs that light up the breast edge, markers or text instead of tissue; high confidence on low-quality images; a band table that is not monotone.

**Gate**

- One per-run report and ten example per-image reports.
- The randomised-model CAM visibly differs from the trained-model CAM.
- ECE drops after calibration on at least three of four sites.
- The INbreast BI-RADS cross-tab, with an honest sentence on what it shows.

## 6. Phases 4 and 5: Docker and the federated system (about 8 days, alongside the GPU runs)

Decision: use Flower for the client-server transport. Your own gRPC layer is about a week of work you do not have, so it moves to Part B (B0). Keep your own trainer, aggregation function and BN handling; Flower only moves weights and metrics between containers. Check the docs for the version you install, because Flower's API has changed between releases.

The container demo runs on synthetic data or a tiny real subset. Your real results come from the simulator. Say so in the report and the talk, and never present synthetic-data runs as clinical results.

### Phase 4: containerise (1 to 2 days)

**Agent does**

1. One multi-stage `Dockerfile` (slim runtime, non-root user, pinned dependencies, a `.dockerignore` that excludes all data and results) with two entrypoints: `server` and `client`.
2. Data is mounted read-only at `/data/<site>`, never copied into an image.
3. `docker-compose.yml`: one server and four clients, each client with its own volume and `SITE_ID`; healthchecks and `restart: on-failure`.
4. GPU: NVIDIA Container Toolkit under WSL2. With 4 GB VRAM give the GPU to one client; the others run on CPU with synthetic data.

**You study:** multi-stage builds, layer caching, compose networks and volumes.

**Gate:** `docker compose up` runs 2 rounds on synthetic data.

### Phase 5: federated system (4 to 5 days)

**Agent does**

1. Wrap your trainer as a Flower client: load the global weights, train one local epoch, return weights, `n_samples` and metrics.
2. Server strategy: subclass FedAvg and call your own aggregation (sample-weighted or uniform, BN keys excluded). Set the minimum number of clients and a round timeout, and tolerate failures, so a missing client means partial aggregation with renormalised weights.
3. Atomic checkpoint after every round (write a temp file, `fsync`, rename) holding round, weights and RNG state. On startup, resume from the last complete checkpoint.
4. TLS: a small CA made with `openssl`; a server certificate, and client authentication if your Flower version supports it. Certificates are mounted as secrets, never baked into images.
5. Fault tests, driven by a short script or pytest against compose:

| Scenario | How to inject | Expected behaviour |
| --- | --- | --- |
| Client dies mid-round | `docker kill` | Round completes with the others after the timeout; the client rejoins |
| Server dies mid-round | `docker kill`, then restart | Resumes from the last checkpoint; no round skipped or repeated |
| Slow client | `sleep` in the client, or `tc netem` | Partial aggregation after the timeout |

6. Equivalence test: 3 rounds on synthetic data match the simulator's global weights within 1e-4.

**You study:** Flower docs (PyTorch quickstart, strategies, TLS), TLS basics, timeouts and partial failure.

**Look for:** a straggler stalling every round; a restarted client with a stale model; a server restart that replays a round.

**Gate**

- Fault table filled in with what you observed, not what you expected.
- A killed client and a killed server are both survived.
- The equivalence test passes.

## 7. 20-day schedule, cut lines and resume bullets

Day 1 is Sunday 1 November and day 20 is Friday 20 November. GPU phases run overnight while you build the container system. Freeze all experiments on 14 November so the last six days go to the report, slides and rehearsal.

&#91;embedded content: 20-day schedule by phase, 1 to 20 Nov 2026\]

Container work runs beside the GPU phases. The dashed line marks presentation day; the last two days are buffer.

**Cut lines**

- **Must:** Phases 42, 53, 64 (A1 to A3), 4 and 5 (TLS, timeouts, checkpoint resume), the section 8 ledger and report, and a minimal CI (half a day: GitHub Actions running `ruff`, `pytest` and `make smoke` on synthetic data).
- **Should:** Phase 3 explainability (calibration, Grad-CAM, reports); ablation A4.
- **Moved to Part B:** monitoring, full CI/CD, cloud, secure aggregation, differential privacy, attacks, consensus, your own gRPC layer.
- **If you fall behind, cut in this order:** A4, the mask-overlap analysis, the failure gallery, the slow-client fault test. Never cut the three-seed comparison.

**Daily rhythm:** start an overnight GPU run every evening. Each morning check the last run, update the ledger, and write three lines in the lab notebook. Hold a gate review in the section 1 format at the end of days 7, 13 and 14.

**Resume bullets** (fill the brackets from your ledger; remove any bullet you did not finish)

- Built a federated learning system for breast-cancer classification across four mammography datasets simulating hospitals; compared local, centralized and federated training over 3 seeds with patient-level bootstrap CIs.
- Quantified film-vs-digital domain shift and ablated harmonization, aggregation weighting and FedProx; \[method\] changed \[site\] AUC by \[X\] (95% CI \[a, b\]).
- Containerised the federated training as server and client services (Docker Compose, Flower) with TLS, round timeouts with partial aggregation, and atomic checkpoint resume; fault-tested for client and server failure.
- Added calibrated risk bands, Grad-CAM with sanity checks, and per-image reports generated locally at each site.
- Set up GitHub Actions CI (lint, tests, synthetic-data smoke test) and a ledger that regenerates every table and figure in the paper.

**Interview map**

| Topic | Where you used it | Be ready for |
| --- | --- | --- |
| Docker | Phase 4 | Multi-stage builds, layer caching, non-root users, volumes vs bind mounts, compose networking |
| CI | Minimal CI | Pipeline stages, caching, why CI uses synthetic data |
| Distributed systems | Phase 5 | Timeouts and partial aggregation, atomic checkpoints, idempotency, at-least-once delivery |
| Security | Phase 5 | TLS and private CAs, secret handling, why raw data never leaves a site |
| ML evaluation | Phases 1 to 3 | Patient-level splits, leakage, bootstrap CIs, calibration, non-IID data |
| Reproducibility | Phase 0 and section 8 | Seeds, config hashes, run provenance, why every number comes from the ledger |

Observability, cloud and IaC, consensus, secure aggregation and differential privacy join this table when you finish Part B.

## 8. Ledger, report and presentation (runs through every phase)

Start this in Phase 0 and keep it running. The rule: no number in the paper is typed by hand; every table and figure is regenerated from the ledger.

**Three layers**

1. **Run artifacts** (one directory per run): resolved `config.json`, `git_sha`, `env.txt` (pip freeze, CUDA, GPU, OS), `seed`, `logs.jsonl`, `metrics.json`, `preds.csv` (pseudonymous ids, kept out of public repos), `report/`.
2. **Experiment ledger** (append-only CSV or MLflow): `run_id, phase, hypothesis_id, method, variant, site, seed, metric, value, ci_low, ci_high, git_sha, split_hash, status, notes`. Never overwrite a row; a rerun is a new row.
3. **Paper pipeline:** `make paper` runs `make_tables.py` (LaTeX tables with mean ± CI from the ledger) and `make_figures.py` (PDF figures), then builds `paper/main.tex` (or Typst). A claims-to-evidence table maps each sentence that states a result to the ledger query that supports it.

**Habits that make the paper easy**

- **Pre-register hypotheses** in one line before running: "H2: uniform weighting raises BCDR and INbreast AUC by at least 0.03." Mark each supported, not supported, or inconclusive afterward.
- **Decision log** (`decisions.md`, one short entry each): why constant LR, why BN excluded from averaging, why 224 px.
- **Lab notebook** (`lab_notebook.md`, dated): include the failures. The debugging story (a flat-probability collapse traced to LR spikes and BN bookkeeping) is the best interview material you have.
- **Reproducibility checklist** (Pineau et al., the ML Reproducibility Checklist) filled in at the end.

**Figures to plan for**

1. System architecture (a simple diagram).
2. Domain-shift probe and embedding plot.
3. Main result grid: 3 methods x 4 sites with CIs.
4. Ablation bars with CIs.
5. Convergence curves per site.
6. Fault tests: the table, and a timeline of one round with a killed client.
7. Grad-CAM gallery and the BI-RADS cross-tab.

**Paper outline**

Abstract; Introduction with a contribution list; Related work (FL for mammography, domain shift); Data and sites (a table with label provenance and licences, and the BI-RADS-derived caveat); Methods; System design; Experiments and results; Explainability; Threat model (what the system does and does not protect; privacy attacks are future work); Limitations (small test sets, heterogeneous labels, simulated hospitals, not a clinical tool); Ethics; Reproducibility; Appendix. Aim for 8 to 10 pages. Workshops on distributed, collaborative and federated learning in medical imaging (the DeCaF/FAIMI series at MICCAI) are natural targets; check each year's call for dates and format.

**Presentation (12 minutes including a 2-minute demo)**

| Slot | Content |
| --- | --- |
| 1 min | Problem and why federated |
| 1 min | Sites, labels, licences |
| 2 min | Baseline findings, including the honest failure and its fix |
| 2 min | Domain shift and ablations |
| 2 min | System: containers, TLS, timeouts, checkpoint resume |
| 2 min | Live demo on synthetic data: kill a client, the round completes; restart the server, it resumes |
| 1 min | Explainability: risk band, Grad-CAM, report |
| 1 min | Limitations and what comes next (Part B) |

Record the demo as a backup video, and rehearse it with the network off.

**You study:** Pineau's reproducibility checklist; "Ten simple rules for reproducible computational research"; MLflow tracking; LaTeX or Typst; Keshav's "How to read a paper" (use it on each paper in section 9).

**Gate (end of project):** `make paper` on a clean clone regenerates every table and figure from the ledger, and the claims-to-evidence table has no blanks.

## Part B: after day 20 (learning track)

None of this is needed for the deadline. Do it once the core system is presented; each item adds a resume line and an interview topic. It assumes the containerised system from section 6 exists.

### B0. Replace Flower with your own gRPC layer (about 1 week)

- Protocol: `Register`, `GetRoundTask` (server streams weights in 4 MB chunks), `UploadUpdate` (client streams weights plus `n_samples` and metrics), `Heartbeat`. A ResNet-18 is about 47 MB in fp32.
- Idempotent uploads keyed by `(round_id, client_id)`; client retries with exponential backoff and jitter.
- mTLS with a private CA, one certificate per client, server allowlist on the certificate CN.
- Same fault table as section 6, plus duplicate-upload and bad-certificate cases.
- Equivalence test: 10 distributed rounds on synthetic data match the simulator within 1e-4.

## B1. Monitoring with Prometheus and Grafana (about 3 to 4 days)

Keep three things separate, because interviewers ask: **monitoring** (live operational time series: Prometheus and Grafana), **logs** (events: Loki), and **experiment tracking** (one-off ML runs and their parameters: MLflow). Each answers a different question.

**Agent does**

1. **Instrument the server and clients** with `prometheus_client`, exposing `/metrics` on port 9100. Follow Prometheus naming conventions (base units, `_total` for counters). Label by `site` only; never by image or patient id.

| Metric | Type | Why |
| --- | --- | --- |
| `fl_round_current` | gauge | Progress |
| `fl_round_duration_seconds` | histogram | p50 and p95 round time |
| `fl_clients_connected`, `fl_clients_participating` | gauge | Dropouts |
| `fl_client_update_latency_seconds{site}` | histogram | Stragglers |
| `fl_update_bytes_total{site,direction}` | counter | Communication cost |
| `fl_global_val_auc{site}`, `fl_client_train_loss{site}` | gauge | Learning curves |
| `fl_prob_std{site}` | gauge | Model-collapse detector (below 0.01 is a red flag) |
| `fl_round_timeouts_total`, `fl_aggregation_errors_total` | counter | Failures |
| `fl_checkpoint_age_seconds` | gauge | Is checkpointing alive? |
| `fl_build_info{version,git_sha}` | gauge (always 1) | Which release is running |

2. **System metrics:** `node_exporter` (host), `cAdvisor` (containers), `nvidia_gpu_exporter` (reads `nvidia-smi`, works on consumer GPUs; DCGM is aimed at data-centre cards).
3. **Prometheus config:** 5 s scrape interval for demos, 15 s otherwise; recording rules for p95 round time and per-site AUC.
4. **Grafana as code:** datasources and dashboards provisioned from JSON in `monitoring/`, nothing clicked by hand. Four dashboards: (a) FL training: per-site val AUC, loss, probability std, participation; (b) system: round duration, update latency, bytes, timeouts; (c) infrastructure: CPU, memory, GPU, containers; (d) chaos: annotations when a fault test starts and ends.
5. **Alerts** (Prometheus rules + Alertmanager, or Grafana alerting, to a Discord/Slack/email webhook): no round progress for 3× the median round time; participants below `min_clients`; `fl_prob_std` under 0.01 for 3 rounds; checkpoint age above a limit; a scrape target down.
6. **Logs:** ship `logs.jsonl` with Promtail or Grafana Alloy to Loki; learn LogQL; link from a dashboard panel to the matching logs.
7. **Stretch:** OpenTelemetry tracing with one trace per round and one span per client, viewed in Jaeger or Tempo.
8. `docker-compose.monitoring.yml` brings the whole stack up with no manual setup.

**You study**

- Prometheus data model and metric types; PromQL `rate()`, `histogram_quantile()`, `sum by`.
- The four golden signals (Google SRE book, "Monitoring Distributed Systems"); RED and USE methods.
- Grafana provisioning and alerting.
- Cardinality: why unbounded labels kill Prometheus.

**Look for:** dashboards that answer a question ("is training healthy?", "is the system healthy?") rather than showing every metric; alerts that fire on symptoms rather than causes.

**Gate**

- Screenshot-ready dashboards, committed as JSON.
- In a chaos demo, killing a client raises an alert within 1 minute and the dashboard shows the dip and the recovery.
- A fresh clone plus `docker compose up` shows live data in Grafana.

## B2. Full CI/CD with GitHub Actions (about 3 days)

CI never trains on real data. It checks that the code is correct, the images build and are free of known vulnerabilities, and the federated system survives a fault on synthetic data. CD ships a tagged release to a cloud VM.

**Agent does: CI (every push and pull request)**

1. **Lint and format:** `ruff check`, `ruff format --check`; optional `mypy` on `src/`.
2. **Unit tests:** `pytest` with coverage on Python 3.10 and 3.11 (matrix), CPU-only PyTorch wheels, pip cache.
3. **Smoke FL:** `make smoke`, 2 rounds, 3 clients, synthetic data, CPU, under 5 minutes.
4. **Container checks:** `hadolint` on Dockerfiles, `docker buildx` build with layer cache, Trivy scan failing on fixable HIGH or CRITICAL.
5. **Integration test:** compose up on synthetic data, kill a client, assert the round completes (the Phase 5 fault suite, trimmed).
6. **Guards:** `gitleaks` for secrets; a job that fails if any `.dcm`, image or label file is committed; all `configs/*.yaml` validated against the schema.
7. **Infrastructure:** `terraform fmt -check`, `validate`, `tflint`; `plan` on pull requests.

**Agent does: CD**

1. On a tag `vX.Y.Z`: build and push `fl-server` and `fl-client` to GitHub Container Registry, tagged with the semver and the git SHA. Optional: SBOM with `syft`, signing with `cosign`.
2. `deploy.yml` with two GitHub Environments: `staging` deploys automatically; `demo` needs manual approval.
3. Authenticate to the cloud with **OIDC** (short-lived role), not stored access keys. Then SSH or SSM into the coordinator VM and run `docker compose pull && docker compose up -d`.
4. Post-deploy check: health endpoint, `fl_build_info` shows the new version in Grafana, one synthetic round completes.
5. Rollback is redeploying the previous tag; write it as a one-line workflow input.
6. Nightly scheduled run of the full integration suite; Dependabot or Renovate for dependency updates.
7. A `paper` job that rebuilds tables and figures from the committed ledger (small aggregate CSVs only) and attaches the PDF as an artifact.

**Branching:** trunk-based; short pull requests; required status checks on `main`.

**You study**

- GitHub Actions: workflows, matrix builds, caching, environments, reusable workflows, OIDC to AWS or GCP.
- The four DORA metrics (deployment frequency, lead time, change failure rate, time to restore) and be ready to quote them for your own pipeline.
- Semantic versioning; the difference between continuous delivery and continuous deployment.
- Trivy and why you pin base images by digest.

**Look for:** CI that is slow (cache it) or flaky (fix or delete the test); secrets in logs; a deploy with no way back.

**Gate**

- A pull request with a deliberately failing test is blocked from merging.
- Pushing a tag publishes images and updates the cloud VM; Grafana shows the new version.
- A passing badge in the README.

## B3. Cloud deployment with Terraform (about 4 days)

The aim is a heterogeneous deployment that looks like real hospitals: a coordinator in the cloud, some clients on cloud VMs, and your laptop GPU as another hospital that dials out to the coordinator. Real accuracy runs stay on your laptop; the cloud proves the distributed behaviour.

**Design**

- **VM1, coordinator and monitoring** (about 2 vCPU, 4 GB): `fl-server`, Prometheus, Grafana, Loki, Alertmanager.
- **VM2 and VM3, cloud hospitals** (CPU only): `fl-client` on public-licence data (CMMD, CBIS) at reduced settings (small subset, 224 px or smaller, one local epoch), or on synthetic data. These runs demonstrate rounds, failures and monitoring, not accuracy.
- **Laptop, hospital 3:** GPU client with INbreast or BCDR, connecting outbound over mTLS (optionally through WireGuard or Tailscale). Hospitals do not open inbound ports, so clients always dial the server.
- **Data licences:** keep INbreast and BCDR (request-based access) on the laptop. Read the terms of any dataset before uploading it anywhere, and never put images in a public bucket.

**Agent does**

1. Terraform modules: `network` (VPC, subnet, security group allowing only the gRPC port from known client IPs and SSH from your IP, or use SSM instead of SSH), `compute` (VMs with cloud-init that installs Docker and pulls the compose file), `iam` (instance role limited to pulling images and reading its own secrets), `budget` (billing alarm).
2. Remote or local state, `terraform fmt`/`validate` in CI, outputs for IPs.
3. Secrets (CA key, client certificates) in the cloud secret store or GitHub environment secrets, never in images or the repo. Use IMDSv2 on AWS.
4. A `make cloud-up` and a `make cloud-down` (`terraform destroy`). Tag every resource; add a nightly auto-stop for the VMs.
5. Run a 10-round mixed run (2 cloud clients + laptop). Then stop a cloud VM mid-round and measure recovery.
6. Measure communication: each round moves about 47 MB per client in each direction, so 4 clients over 35 rounds is about 13 GB. Test fp16 weight transfer or sending only trainable layers to halve it, and record the effect on AUC and round time.

**Cost control:** free tiers and credits change, so check the provider's current terms before you start, set a budget alarm first, and destroy the stack after every session.

**You study:** VPCs, subnets and security groups; IAM roles and least privilege; Terraform state and modules (HashiCorp Learn tutorials); cloud-init; OIDC federation from GitHub to the cloud; egress costs.

**Look for:** an open security group; a role with more permissions than needed; a forgotten VM still billing; clients that cannot reach the server because of NAT or firewall rules.

**Gate**

- `terraform apply` produces the stack; a mixed cloud and laptop run completes 10 rounds.
- Grafana shows all participants; stopping a VM raises the alert.
- `terraform destroy` leaves nothing behind, checked in the billing console.

## B4. Privacy and security evaluation (about 1 week)

The narrative is red team first, then mitigation: show what a plain FedAvg update leaks, then what each defence costs in accuracy. The breast-specific FL papers I found do not run privacy attacks, so applying them to mammography is your contribution; cite the general attack papers for the methods.

**Agent does**

1. **Threat model (one page).** Honest-but-curious server (reads every client update), network eavesdropper (stopped by mTLS), and optionally one malicious client. State what is out of scope: collusion between clients and the server, side channels, and the fact that with only 4 clients the guarantees are weak.
2. **Secure aggregation.** Pairwise masking after Bonawitz et al. 2017: X25519 key agreement, masks derived from shared seeds, fixed-point integers so masks cancel exactly. Test: the server's sum equals the plain sum within quantization error, and a single masked update is statistically indistinguishable from noise. Handle dropouts by aborting the round, with Shamir secret-sharing recovery as a stretch. Report the overhead in time and bytes. Be honest that with 4 clients, three colluding parties reveal the fourth.
3. **Differential privacy.** Clip each client update and add Gaussian noise (DP-FedAvg, McMahan et al. 2018), and separately example-level DP-SGD with Opacus. Opacus rejects BatchNorm, so use GroupNorm there (which also serves ablation A5). Sweep noise multiplier σ in {0, 0.1, 0.5, 1.0}; plot AUC against σ and against ε. With 4 clients, client-level ε is not meaningful, so report example-level ε and say why.
4. **Gradient inversion attack** on a plain FedAvg update: Deep Leakage from Gradients (Zhu 2019) and Inverting Gradients (Geiping 2020) on small batches (1 to 4 images, 64×64 or 128×128). Show the reconstruction, score it with SSIM, then show it degrade with larger batches, more local epochs, DP noise, and under secure aggregation (the server cannot isolate one client).
5. **Membership inference.** Loss-threshold attack (Yeom 2018, Shokri 2017). Report attack AUC and true-positive rate at low false-positive rate (Carlini 2022) for local, federated, and DP models. Test and training patients are separated by patient, as in your splits.
6. **Poisoning (optional, cheap).** One client flips labels; compare FedAvg with coordinate-wise median or trimmed mean (Yin 2018; Krum, Blanchard 2017).
7. Add the secure-aggregation equivalence test to CI.

**You study:** the Bonawitz protocol and Shamir secret sharing; differential privacy basics (ε, δ, sensitivity, composition); Opacus tutorials; how a gradient reveals its input; why membership inference works when a model overfits.

**Look for:** reconstructions that look good on one image but fail on batches; a membership attack AUC near 0.5 that nonetheless has a high TPR at low FPR; a DP noise level that destroys AUC on small sites.

**Gate**

- An attack × defence table (attack success: SSIM or attack AUC) and a utility-vs-privacy curve with CIs.
- Secure aggregation passes the equivalence test.
- The threat model states plainly what is and is not protected.

## B5. Consensus and a replicated coordinator (3 to 4 days, or a weekend if toy Raft)

The coordinator is a single point of failure. Pick one route; route A is the one I recommend for the time you have.

| Route | What you build | Effort | What you can claim |
| --- | --- | --- | --- |
| A (recommended) | Three coordinator replicas with leader election and replicated round state on a 3-node etcd cluster | 3 to 4 days | "Highly available coordinator using Raft-based etcd; failover tested" |
| B (learning) | A toy Raft (leader election plus log replication) in Python, tested with partitions, separate from the FL code | 1 to 2 weekends | "Implemented Raft from the paper" |
| C | Document the single point of failure and the design you would use | 0.5 day | A design discussion only |

**Agent does (route A)**

1. Add a 3-node etcd cluster to compose. Store the round number, checkpoint pointer (path plus SHA-256 of the model file), and participant list under `/fl/state/`.
2. Run three coordinator replicas. Use etcd's lease-based election so exactly one is active. Use the etcd revision as a **fencing token**: a stale leader's writes are rejected.
3. Clients try the known coordinator endpoints and retry with backoff, so they find the new leader.
4. Chaos tests: kill the leader mid-round (a standby takes over and resumes from the last committed round); kill one etcd member (service continues, quorum 2 of 3); kill two (writes stop, which shows the consistency-over-availability choice).
5. Metrics: `fl_leader_changes_total`, `fl_failover_seconds`; Grafana annotation at each failover.
6. Write a one-page design note (ADR) covering why etcd, what the fencing token prevents, and the CAP trade-off you observed.

**You study:** the Raft paper (Ongaro and Ousterhout 2014) with the raft.github.io visualisation; the MIT 6.5840 (6.824) lecture on Raft; *DDIA* chapter 9; etcd docs on leases and elections; Kleppmann's note on fencing tokens.

**Look for:** split-brain (two leaders acting at once); a checkpoint file that exists but is not the one etcd points to; time-to-recover when a leader dies.

**Gate**

- A recorded failover run with measured recovery time.
- The run finishes with the same final weights as a no-fault run, within tolerance.
- The ADR is written.

## 9. Papers and what to take from each

For the 20-day plan, read these eight first: Roth 2020, the CSBJ 2025 mammography case study, Quintana 2026, McMahan 2017, Li 2020 (FedProx), Li 2021 (FedBN), Selvaraju 2017 (Grad-CAM) and Guo 2017 (calibration). The rest is for Part B.

The four mammography FL papers in the first table were confirmed by search on 31 Oct 2026. The rest are cited from memory with arXiv IDs; open each link and check title, authors and venue before you cite it. Read each with Keshav's three-pass method and write three lines per paper: what it claims, what you would reuse, what it does not show.

### Federated learning for mammography and breast imaging

| Paper | Take from it |
| --- | --- |
| [Roth et al., Federated Learning for Breast Density Classification: A Real-World Implementation](https://arxiv.org/pdf/2009.01871) | Multi-hospital FL in practice; how the federated model compares with local training; deployment lessons for your system section |
| [Jiménez-Sánchez et al., Memory-aware curriculum federated learning for breast cancer classification](https://arxiv.org/pdf/2107.02504) | Handling heterogeneous sites with a curriculum; an extra ablation if time allows |
| [Towards generalizable Federated Learning in medical imaging: a real-world case study on mammography data (CSBJ 2025)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11984999/) | Preprocessing and harmonization effects on cross-site generalisation: your A1 ablation and the framing of your domain-shift study |
| [Evaluating Federated Learning approaches for mammography under breast density heterogeneity (arXiv 2605.09137, 2026)](https://www.alphaxiv.org/abs/2605.09137) | Comparing aggregation strategies under heterogeneity: your A2 to A4 design and how to report it |

### FL algorithms and non-IID data

| Paper | Take from it |
| --- | --- |
| McMahan et al. 2017, Communication-Efficient Learning of Deep Networks from Decentralized Data ([1602.05629](https://arxiv.org/abs/1602.05629)) | FedAvg; the weighting rule you ablate |
| Li et al. 2020, Federated Optimization in Heterogeneous Networks, FedProx ([1812.06127](https://arxiv.org/abs/1812.06127)) | Proximal term and partial work from stragglers |
| Li et al. 2021, FedBN: Federated Learning on Non-IID Features via Local Batch Normalization ([2102.07623](https://arxiv.org/abs/2102.07623)) | Feature shift is your problem (film vs digital); keep BN local |
| Hsieh et al. 2020, The Non-IID Data Quagmire of Decentralized Machine Learning ([1910.00189](https://arxiv.org/abs/1910.00189)) | Why BatchNorm breaks under skew; GroupNorm (A5) |
| Karimireddy et al. 2020, SCAFFOLD ([1910.06378](https://arxiv.org/abs/1910.06378)) | Client drift and control variates (optional A9) |
| Reddi et al. 2021, Adaptive Federated Optimization ([2003.00295](https://arxiv.org/abs/2003.00295)) | FedAdam and server optimizers (A6) |
| Kairouz et al. 2021, Advances and Open Problems in Federated Learning ([1912.04977](https://arxiv.org/abs/1912.04977)) | Survey; heterogeneity, privacy and systems chapters for related work |
| Rieke et al. 2020, The future of digital health with federated learning (npj Digital Medicine) | Medical FL framing and challenges |
| Sheller et al. 2020, Federated learning in medicine (Scientific Reports) | Federated vs institutional baselines; protocol ideas |

### Domain shift, shortcuts and pretraining

| Paper | Take from it |
| --- | --- |
| Ghosh et al. 2024, Mammo-CLIP ([2405.12255](https://arxiv.org/abs/2405.12255)) | Domain-specific initialisation (A8); preprocessing sensitivity; licence is non-commercial, share-alike |
| Zech et al. 2018, Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs (PLOS Medicine) | Models learn the hospital, not the disease: motivation for your site probe |
| DeGrave et al. 2021, AI for radiographic COVID-19 detection selects shortcuts over signal (Nature Machine Intelligence) | Shortcut detection with saliency; ties to your Grad-CAM checks |
| Guan and Liu 2022, Domain adaptation for medical image analysis: a survey (IEEE TBME) | Menu of alignment methods beyond the ones you implement |

### Privacy and security

| Paper | Take from it |
| --- | --- |
| Bonawitz et al. 2017, Practical Secure Aggregation for Privacy-Preserving Machine Learning ([1611.04482](https://arxiv.org/abs/1611.04482)) | Pairwise masking and dropout recovery |
| Abadi et al. 2016, Deep Learning with Differential Privacy ([1607.00133](https://arxiv.org/abs/1607.00133)) | DP-SGD and the moments accountant |
| McMahan et al. 2018, Learning Differentially Private Recurrent Language Models ([1710.06963](https://arxiv.org/abs/1710.06963)) | Clip-and-noise at the client-update level (DP-FedAvg) |
| Yousefpour et al. 2021, Opacus ([2109.12298](https://arxiv.org/abs/2109.12298)) | Practical DP training; BN incompatibility |
| Zhu et al. 2019, Deep Leakage from Gradients ([1906.08935](https://arxiv.org/abs/1906.08935)) | The basic gradient-inversion attack |
| Geiping et al. 2020, Inverting Gradients ([2003.14053](https://arxiv.org/abs/2003.14053)) | A stronger attack and how batch size changes it |
| Shokri et al. 2017, Membership Inference Attacks Against Machine Learning Models ([1610.05820](https://arxiv.org/abs/1610.05820)) | The threat model and shadow-model attack |
| Carlini et al. 2022, Membership Inference Attacks From First Principles ([2112.03570](https://arxiv.org/abs/2112.03570)) | Report TPR at low FPR, not just AUC |
| Blanchard et al. 2017, Krum ([1703.02757](https://arxiv.org/abs/1703.02757)); Yin et al. 2018, trimmed mean and median ([1803.01498](https://arxiv.org/abs/1803.01498)) | Robust aggregation against a poisoning client |

### Explainability, calibration, uncertainty

| Paper | Take from it |
| --- | --- |
| Selvaraju et al. 2017, Grad-CAM ([1610.02391](https://arxiv.org/abs/1610.02391)) | The saliency method |
| Adebayo et al. 2018, Sanity Checks for Saliency Maps ([1810.03292](https://arxiv.org/abs/1810.03292)) | Randomisation tests your maps must pass |
| Guo et al. 2017, On Calibration of Modern Neural Networks ([1706.04599](https://arxiv.org/abs/1706.04599)) | ECE and temperature scaling |
| Gal and Ghahramani 2016, Dropout as a Bayesian Approximation ([1506.02142](https://arxiv.org/abs/1506.02142)) | MC-dropout uncertainty |
| Cao et al. 2020, Rank consistent ordinal regression (CORAL) ([1901.07884](https://arxiv.org/abs/1901.07884)) | Optional INbreast BI-RADS head |
| Mitchell et al. 2019, Model Cards for Model Reporting ([1810.03993](https://arxiv.org/abs/1810.03993)) | The model card in Phase 3 |

### Systems and consensus

| Paper or book | Take from it |
| --- | --- |
| Bonawitz et al. 2019, Towards Federated Learning at Scale: System Design ([1902.01046](https://arxiv.org/abs/1902.01046)) | Round structure, dropout handling, monitoring: the design you are miniaturising |
| Beutel et al. 2020, Flower ([2007.14390](https://arxiv.org/abs/2007.14390)) | The standard open framework to cross-check against |
| Ongaro and Ousterhout 2014, In Search of an Understandable Consensus Algorithm (Raft) | Leader election and log replication for Phase 10 |
| Kleppmann, *Designing Data-Intensive Applications*, chapters 8 and 9 | Faults, timeouts, consistency, consensus |
| Beyer et al., *Site Reliability Engineering* (Google, free online) | Monitoring and cascading failures chapters |
