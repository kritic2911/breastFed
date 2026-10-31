"""Centralized result storage, artifact persistence, and logging."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from src.provenance import (
    StructuredLogger,
    generate_run_id,
    get_environment_info,
    save_resolved_config,
)


def save_run_artifacts(
    method: str,
    train_site: str,
    seed: int,
    threshold: float,
    best_state: dict,
    history: list,
    own_results: dict,
    cross_results: dict,
    results_root: str = "results",
    extra_meta: dict = None,
    output_dir: Path = None,
    cfg: dict = None,
    run_id: str = None,
    config_hash: str = None,
    git_sha: str = None,
    logger: StructuredLogger = None,
) -> Path:
    """Save all artifacts for a run:
    - resolved_config.yaml: resolved config dict
    - metrics.json: full metrics dictionary including provenance & environment
    - logs.jsonl: structured event logs
    - preds.csv: consolidated per-image predictions across test sets
    - best.pt: model checkpoint weights
    - training_history.csv: per-epoch loss, val AUC, sensitivity, VRAM, timing
    - results/all_runs_summary.csv: appends standardized metrics for aggregate analysis
    - experiment_log.md: appends markdown summary
    """
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)

    # 1. Resolve provenance metadata
    if cfg is not None and (run_id is None or config_hash is None or git_sha is None):
        gen_id, gen_hash, gen_sha = generate_run_id(cfg)
        run_id = run_id or gen_id
        config_hash = config_hash or gen_hash
        git_sha = git_sha or gen_sha
    else:
        run_id = run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-default-unknown"
        config_hash = config_hash or "unknown"
        git_sha = git_sha or "unknown"

    env_info = get_environment_info()

    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = root / method / train_site / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Save resolved config
    if cfg is not None:
        save_resolved_config(cfg, out_dir)

    # 3. Model checkpoint
    if best_state is not None:
        torch.save(best_state, out_dir / "best.pt")

    # 4. Per-epoch training history CSV
    if history:
        df_hist = pd.DataFrame(history)
        df_hist.to_csv(out_dir / "training_history.csv", index=False)

    # 5. Save test predictions and consolidated preds.csv
    consolidated_preds = []
    for eval_site, res in cross_results.items():
        if "predictions" in res and res["predictions"]:
            preds_dict = res["predictions"]
            df_preds = pd.DataFrame(preds_dict)
            df_preds.to_csv(out_dir / f"predictions_{eval_site}.csv", index=False)

            pids = preds_dict.get("patient_ids", [])
            lbls = preds_dict.get("labels", [])
            probs = preds_dict.get("probs", [])
            binary_preds = preds_dict.get("preds", [])
            image_ids = preds_dict.get("image_ids", [f"{pids[i]}_{i}" for i in range(len(pids))])

            for i in range(len(pids)):
                consolidated_preds.append({
                    "run_id": run_id,
                    "site": eval_site,
                    "split": "test",
                    "patient_id": pids[i],
                    "image_id": image_ids[i] if i < len(image_ids) else f"{pids[i]}_{i}",
                    "label": lbls[i] if i < len(lbls) else None,
                    "prob": probs[i] if i < len(probs) else None,
                    "pred": binary_preds[i] if i < len(binary_preds) else None,
                    "threshold": float(threshold),
                })

        # Save standalone site metric file
        site_metrics_payload = {
            "eval_site": eval_site,
            "threshold": float(threshold),
            "metrics": {k: v for k, v in res.items() if k != "predictions"}
        }
        with open(out_dir / f"metrics_{eval_site}.json", "w") as sf:
            json.dump(site_metrics_payload, sf, indent=2, default=str)

    # Save consolidated preds.csv
    if consolidated_preds:
        df_cons = pd.DataFrame(consolidated_preds)
        df_cons.to_csv(out_dir / "preds.csv", index=False)
    else:
        # Create empty placeholder if no predictions
        pd.DataFrame(columns=[
            "run_id", "site", "split", "patient_id", "image_id", "label", "prob", "pred", "threshold"
        ]).to_csv(out_dir / "preds.csv", index=False)

    # 6. Save metrics.json (merging cross_site_test if already exists)
    merged_cross = dict(cross_results)
    if (out_dir / "metrics.json").exists():
        try:
            with open(out_dir / "metrics.json") as f:
                prev_data = json.load(f)
            if "cross_site_test" in prev_data and isinstance(prev_data["cross_site_test"], dict):
                prev_cross = prev_data["cross_site_test"]
                prev_cross.update(merged_cross)
                merged_cross = prev_cross
        except Exception:
            pass

    results_payload = {
        "run_id": run_id,
        "config_hash": config_hash,
        "git_sha": git_sha,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "environment": env_info,
        "method": method,
        "trained_on": train_site,
        "threshold": float(threshold),
        "own_site_test": own_results,
        "cross_site_test": merged_cross,
        "history": history,
    }
    if extra_meta:
        results_payload.update(extra_meta)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results_payload, f, indent=2, default=str)

    # 7. Structured logger completion record
    if logger is None:
        logger = StructuredLogger(out_dir / "logs.jsonl", run_id=run_id)

    logger.log(event="run_completed", metric="threshold", value=float(threshold))
    for eval_site, res in cross_results.items():
        auc_val = res.get("image_level", {}).get("auc")
        if auc_val is not None:
            logger.log(event="test_evaluation", site=eval_site, split="test", metric="auc", value=float(auc_val))

    # 8. Append to central all_runs_summary.csv
    summary_csv_path = root / "all_runs_summary.csv"
    summary_rows = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    best_epoch = None
    best_val_auc = None
    if history:
        best_rec = max(history, key=lambda x: x.get("val_auc", -1))
        best_epoch = best_rec.get("epoch")
        best_val_auc = best_rec.get("val_auc")

    for eval_site, res in cross_results.items():
        img_m = res.get("image_level", {})
        pat_m = res.get("patient_level", {})
        ci = res.get("bootstrap_auc_ci", {})
        summary_rows.append({
            "timestamp": ts,
            "run_id": run_id,
            "method": method,
            "train_site": train_site,
            "eval_site": eval_site,
            "is_own_site": (train_site == eval_site),
            "seed": seed,
            "auc": img_m.get("auc"),
            "auc_ci_lower": ci.get("lower"),
            "auc_ci_upper": ci.get("upper"),
            "sensitivity_at_spec90": img_m.get("sensitivity"),
            "specificity": img_m.get("specificity"),
            "patient_auc": pat_m.get("auc"),
            "patient_sensitivity": pat_m.get("sensitivity"),
            "threshold": threshold,
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
        })

    df_new = pd.DataFrame(summary_rows)
    if summary_csv_path.exists():
        df_new.to_csv(summary_csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(summary_csv_path, mode="w", header=True, index=False)

    # 9. Append to experiment_log.md
    log_file = Path("experiment_log.md")
    if log_file.exists():
        append_to_experiment_log(log_file, method, train_site, seed, cross_results, best_epoch, best_val_auc)

    print(f"  [Artifacts Saved] Checkpoint, logs, predictions -> {out_dir}")
    print(f"  [Summary Updated] Appended {len(summary_rows)} rows -> {summary_csv_path}")

    return out_dir


def append_to_experiment_log(log_file: Path, method: str, train_site: str, seed: int,
                             cross_results: dict, best_epoch: int, best_val_auc: float):
    """Format a clean markdown table entry in experiment_log.md."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"\n### Run: {method.upper()} | Train: `{train_site}` | Seed: `{seed}` ({ts})\n",
        f"- **Best Epoch**: {best_epoch} (Val AUC: {best_val_auc:.4f})" if best_epoch else "",
        "\n| Eval Site | Role | Test AUC (95% CI) | Sens @ 90% Spec | Patient AUC |",
        "|---|---|:---:|:---:|:---:|",
    ]
    for eval_site, res in cross_results.items():
        role = "Own site" if eval_site == train_site else "Cross-site"
        img_m = res.get("image_level", {})
        pat_m = res.get("patient_level", {})
        ci = res.get("bootstrap_auc_ci", {})
        auc_str = f"{img_m.get('auc', 0):.4f} [{ci.get('lower', 0):.4f}, {ci.get('upper', 0):.4f}]"
        sens_str = f"{img_m.get('sensitivity', 0):.4f}"
        pat_auc_str = f"{pat_m.get('auc', 0):.4f}" if 'auc' in pat_m else "N/A"
        lines.append(f"| **{eval_site}** | {role} | {auc_str} | {sens_str} | {pat_auc_str} |")
    lines.append("\n---\n")

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write("\n".join(filter(None, lines)))
