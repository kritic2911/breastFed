"""FedAvg — simulated in a single process.

Each round:
  1. Server sends global weights to every client.
  2. Each client trains for `local_epochs` on its own data.
  3. Clients upload their updated weights.
  4. Server averages: w_global = sum(n_i / n_total * w_i)   [sample-weighted]

BN layers are frozen (cfg["freeze_bn"]=True), so their running stats stay at
ImageNet values and are never averaged. This sidesteps the cross-scanner BN
contamination problem.
"""

import copy
import json
import math
import time
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from src.model import build_model, model_size_bytes, get_bn_keys, get_trainable_params
from src.train import train_local_epochs, set_seed
from src.dataset import build_site_loader
from src.evaluate import predict, compute_metrics, evaluate_on_site
from src.storage import save_run_artifacts
from src.thermal import gpu_thermal_cooldown


# ────────────────────────────────────────────────────────────────
#  FedAvg core
# ────────────────────────────────────────────────────────────────

def fedavg_aggregate(global_state, client_states, client_sizes,
                     weighting="sample", exclude_keys=None):
    """Aggregate client state dicts into a new global state dict.

    Parameters
    ----------
    global_state : dict (used as template for keys)
    client_states : list of state dicts
    client_sizes : list of ints (number of training samples per client)
    weighting : "sample" or "uniform"
    exclude_keys : set of keys to skip (used by FedBN)
    """
    exclude_keys = exclude_keys or set()
    n_clients = len(client_states)
    total = sum(client_sizes)

    if weighting == "sample":
        weights = [n / total for n in client_sizes]
    else:  # uniform
        weights = [1.0 / n_clients] * n_clients

    new_state = {}
    for key in global_state.keys():
        if key in exclude_keys:
            # Keep global (or don't update — for FedBN this key stays local)
            new_state[key] = global_state[key].clone()
            continue
        new_state[key] = sum(
            w * client_states[i][key].float()
            for i, w in enumerate(weights)
        ).to(global_state[key].dtype)

    return new_state


# ────────────────────────────────────────────────────────────────
#  Full federated run
# ────────────────────────────────────────────────────────────────

def run_federated(cfg, seed, device="cpu", output_dir=None, resume=True,
                  force_train=False, force_eval=False, method_name=None):
    """Run full FedAvg experiment with automatic round-level resumption.

    Returns
    -------
    results : dict with per-round history and final test metrics
    """
    set_seed(seed)
    sites = cfg["sites"]
    data_root = cfg["data_root"]
    n_rounds = cfg["fed"]["rounds"]
    weighting = cfg["fed"]["weighting"]
    effective_method = method_name or f"federated_{weighting}"

    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        best_pt_path = out / "best.pt"
        metrics_json_path = out / "metrics.json"

        if not force_train and not force_eval and metrics_json_path.exists():
            print(f"  [Skipping] Federated already completed at {output_dir}. Use --force-eval to re-evaluate or --force to retrain.")
            try:
                with open(metrics_json_path) as f:
                    return json.load(f)
            except Exception:
                pass

    # ── Build per-site loaders ──
    site_train_loaders = {}
    site_val_loaders = {}
    site_test_loaders = {}
    site_sizes = {}

    for site in sites:
        _, train_loader = build_site_loader(data_root, site, "train", cfg, shuffle=True)
        _, val_loader = build_site_loader(data_root, site, "val", cfg)
        _, test_loader = build_site_loader(data_root, site, "test", cfg)
        site_train_loaders[site] = train_loader
        site_val_loaders[site] = val_loader
        site_test_loaders[site] = test_loader
        site_sizes[site] = len(train_loader.dataset)

    client_sizes = [site_sizes[s] for s in sites]
    print(f"Sites: {sites}")
    print(f"Training sizes: {site_sizes}")
    print(f"Weighting: {weighting}")

    # Check if we are explicitly re-evaluating an already trained checkpoint without training
    if out and force_eval and (out / "best.pt").exists():
        print(f"\n{'='*65}")
        print(f"FEDERATED BASELINE | Seed: {seed}")
        print(f"  [Reusing Trained Model] Loading federated checkpoint: {out / 'best.pt'}")
        print(f"{'='*65}")
        best_global_state = torch.load(out / "best.pt", map_location=device, weights_only=True)
        is_fedbn = bool(cfg.get("fed", {}).get("fedbn", False))
        best_client_bn_states = {}
        if is_fedbn and (out / "best_client_bn.pt").exists():
            best_client_bn_states = torch.load(out / "best_client_bn.pt", map_location=device, weights_only=False)
        history = []
        if (out / "training_history.csv").exists():
            try:
                history = pd.read_csv(out / "training_history.csv").to_dict(orient="records")
            except Exception:
                pass
        best_round = history[-1].get("best_round", n_rounds) if history else n_rounds
        best_avg_val_auc = history[-1].get("best_avg_val_auc", -1.0) if history else -1.0
        total_bytes_sent = 0
        total_train_steps = 0
    else:
        # Clean old checkpoints if force retraining
        if out and force_train:
            for fname in ["last_round.pt", "best.pt", "best_checkpoint.pt", "best_client_bn.pt"]:
                p = out / fname
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
            resume = False

        # ── Initialize global model ──
        is_fedbn = bool(cfg.get("fed", {}).get("fedbn", False))
        fedprox_mu = float(cfg.get("fed", {}).get("mu", 0.0))
        if is_fedbn:
            print("  [FedBN Enabled] BatchNorm weights and running stats remain strictly local per site.")
        if fedprox_mu > 0.0:
            print(f"  [FedProx Enabled] Proximal penalty mu = {fedprox_mu}")

        global_model = build_model(cfg, device=device)
        global_state = copy.deepcopy(global_model.state_dict())
        total_model_bytes = model_size_bytes(global_model)
        freeze_bn = bool(cfg.get("freeze_bn", True))
        exclude_keys = set()
        if is_fedbn or freeze_bn:
            exclude_keys = set(get_bn_keys(global_model))
        client_bn_states = {s: {k: v.clone() for k, v in global_state.items() if k in exclude_keys} for s in sites} if is_fedbn else {}
        best_client_bn_states = copy.deepcopy(client_bn_states)

        # ── Training loop ──
        history = []
        best_avg_val_auc = -1.0
        best_round = 0
        best_global_state = None
        total_bytes_sent = 0
        total_train_steps = 0
        start_round = 1
        rounds_no_improve = 0
        patience = cfg.get("fed", {}).get("patience", None)
        fed_lr = float(cfg.get("fed", {}).get("lr", 1.0e-4))
        use_constant_lr = bool(cfg.get("fed", {}).get("constant_lr", True))

        last_round_file = out / "last_round.pt" if out else None
        if resume and last_round_file and last_round_file.exists():
            try:
                ckpt = torch.load(last_round_file, map_location=device, weights_only=False)
                start_round = ckpt["round"] + 1
                global_state = ckpt["global_state"]
                best_round = ckpt["best_round"]
                best_avg_val_auc = ckpt["best_avg_val_auc"]
                best_global_state = ckpt["best_global_state"]
                history = ckpt["history"]
                total_bytes_sent = ckpt["total_bytes_sent"]
                total_train_steps = ckpt["total_train_steps"]
                rounds_no_improve = ckpt.get("rounds_no_improve", 0)
                if is_fedbn and "client_bn_states" in ckpt:
                    client_bn_states = ckpt["client_bn_states"]
                    best_client_bn_states = ckpt.get("best_client_bn_states", copy.deepcopy(client_bn_states))
                print(f"  [Resume] Resuming federated training from round {start_round}/{n_rounds} "
                      f"(loaded round {ckpt['round']} checkpoint, best avg val AUC: {best_avg_val_auc:.4f})")
            except Exception as err:
                print(f"  [Resume Warning] Could not resume from {last_round_file} ({err}), starting fresh.")

        for rnd in range(start_round, n_rounds + 1):
            t0 = time.time()
            client_states = []

            # Federated learning rate schedule (constant or cosine decay)
            if use_constant_lr:
                current_round_lr = fed_lr
            else:
                round_progress = (rnd - 1) / max(n_rounds - 1, 1)
                current_round_lr = fed_lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * round_progress)))

            # Snapshot global weights for FedProx penalty
            global_params = [p.detach().clone() for p in get_trainable_params(global_model)] if fedprox_mu > 0.0 else None

            client_losses = {}
            for site in sites:
                # Copy global weights to client (plus local BN state if FedBN)
                client_model = build_model(cfg, device=device)
                site_state = copy.deepcopy(global_state)
                if is_fedbn:
                    site_state.update(client_bn_states[site])
                client_model.load_state_dict(site_state)

                # Local training (supports FedAvg, FedProx, adaptive LR)
                client_model, avg_loss = train_local_epochs(
                    client_model, site_train_loaders[site], cfg, device=device,
                    lr=current_round_lr, global_params=global_params, mu=fedprox_mu
                )
                client_losses[site] = float(avg_loss)

                c_state = copy.deepcopy(client_model.state_dict())
                client_states.append(c_state)
                if is_fedbn:
                    client_bn_states[site] = {k: c_state[k].clone() for k in exclude_keys}

                steps = len(site_train_loaders[site]) * cfg["fed"]["local_epochs"]
                total_train_steps += steps

                # Communication: upload + download per client per round
                total_bytes_sent += total_model_bytes * 2

            # Aggregate
            global_state = fedavg_aggregate(
                global_state, client_states, client_sizes,
                weighting=weighting, exclude_keys=exclude_keys
            )

            # Evaluate on each site's val set (AUC, prob_std, prob_mean)
            round_val_aucs = {}
            round_val_stds = {}
            round_val_means = {}
            for site in sites:
                eval_model = build_model(cfg, device=device)
                eval_state = copy.deepcopy(global_state)
                if is_fedbn:
                    eval_state.update(client_bn_states[site])
                eval_model.load_state_dict(eval_state)
                eval_model.eval()

                probs, labels, _ = predict(eval_model, site_val_loaders[site], device)
                metrics = compute_metrics(labels, probs)
                round_val_aucs[site] = float(metrics["auc"])
                round_val_stds[site] = float(metrics["prob_std"])
                round_val_means[site] = float(metrics["prob_mean"])

            avg_val_auc = float(np.mean(list(round_val_aucs.values())))
            elapsed = time.time() - t0

            if avg_val_auc > best_avg_val_auc:
                best_avg_val_auc = avg_val_auc
                best_round = rnd
                best_global_state = copy.deepcopy(global_state)
                if is_fedbn:
                    best_client_bn_states = copy.deepcopy(client_bn_states)
                rounds_no_improve = 0
                if out:
                    torch.save(best_global_state, out / "best_checkpoint.pt")
                    torch.save(best_global_state, out / "best.pt")
                    if is_fedbn:
                        torch.save(best_client_bn_states, out / "best_client_bn.pt")
            else:
                rounds_no_improve += 1

            record = {
                "round": rnd,
                "lr": round(current_round_lr, 7),
                "client_losses": client_losses,
                "val_aucs": round_val_aucs,
                "val_prob_stds": round_val_stds,
                "val_prob_means": round_val_means,
                "avg_val_auc": float(avg_val_auc),
                "best_avg_val_auc": float(best_avg_val_auc),
                "time_sec": elapsed,
            }
            # Flatten per-site metrics for convenient CSV inspection
            for s in sites:
                record[f"{s}_loss"] = client_losses.get(s, np.nan)
                record[f"{s}_val_auc"] = round_val_aucs.get(s, np.nan)
                record[f"{s}_prob_std"] = round_val_stds.get(s, np.nan)

            history.append(record)
            if out:
                pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
                round_ckpt = {
                    "round": rnd,
                    "global_state": global_state,
                    "best_round": best_round,
                    "best_avg_val_auc": best_avg_val_auc,
                    "best_global_state": best_global_state,
                    "best_client_bn_states": best_client_bn_states if is_fedbn else None,
                    "client_bn_states": client_bn_states if is_fedbn else None,
                    "history": history,
                    "total_bytes_sent": total_bytes_sent,
                    "total_train_steps": total_train_steps,
                    "rounds_no_improve": rounds_no_improve,
                }
                torch.save(round_ckpt, out / "last_round.pt")

            print(f"  Round {rnd:2d}/{n_rounds:2d} | avg_val_auc {avg_val_auc:.4f} (best {best_avg_val_auc:.4f} r{best_round}) | lr {current_round_lr:.1e} | {elapsed:.1f}s")
            print(f"    Loss:      " + " | ".join(f"{s}={client_losses.get(s, 0.0):.4f}" for s in sites))
            print(f"    Val AUC:   " + " | ".join(f"{s}={round_val_aucs[s]:.4f}" for s in sites))
            print(f"    Prob Std:  " + " | ".join(f"{s}={round_val_stds[s]:.4f}" for s in sites), flush=True)

            if patience and rounds_no_improve >= patience:
                print(f"  [Early Stopping] Federated training stopped early at round {rnd} "
                      f"(no improvement for {patience} consecutive rounds)", flush=True)
                break

            if rnd < n_rounds:
                gpu_thermal_cooldown(cfg, verbose=True, epoch=rnd)

    # ── Final test evaluation (once, at best round) ──
    print(f"\nBest round: {best_round} (avg_val_auc={best_avg_val_auc:.4f})")
    print("Running final test evaluation...")

    # Get threshold from validation set (pooled across sites)
    all_val_probs, all_val_labels = [], []
    for site in sites:
        val_model = build_model(cfg, device=device)
        site_state = copy.deepcopy(best_global_state)
        if is_fedbn and site in best_client_bn_states:
            site_state.update(best_client_bn_states[site])
        val_model.load_state_dict(site_state)
        val_model.eval()
        probs, labels, _ = predict(val_model, site_val_loaders[site], device)
        all_val_probs.append(probs)
        all_val_labels.append(labels)
    val_probs = np.concatenate(all_val_probs)
    val_labels = np.concatenate(all_val_labels)
    val_metrics = compute_metrics(val_labels, val_probs)
    threshold = val_metrics["threshold"]
    print(f"Threshold from val: {threshold:.4f}")

    # Test on each site
    test_results = {}
    for site in sites:
        test_model = build_model(cfg, device=device)
        site_state = copy.deepcopy(best_global_state)
        if is_fedbn and site in best_client_bn_states:
            site_state.update(best_client_bn_states[site])
        test_model.load_state_dict(site_state)
        test_model.eval()
        site_results = evaluate_on_site(
            test_model, site_test_loaders[site], device,
            threshold=threshold, n_bootstrap=cfg["eval"]["bootstrap_patients"],
            seed=seed,
        )
        test_results[site] = site_results
        auc = site_results["image_level"]["auc"]
        ci = site_results["bootstrap_auc_ci"]
        print(f"  {site:10s} | AUC {auc:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}] | "
              f"sens {site_results['image_level']['sensitivity']:.4f}")

    # ── Save all artifacts ──
    sites_label = "collaborative_4sites" if len(sites) == 4 else f"train_{'_'.join(sites)}"
    save_run_artifacts(
        method=effective_method,
        train_site=sites_label,
        seed=seed,
        threshold=threshold,
        best_state=best_global_state,
        history=history,
        own_results={},
        cross_results=test_results,
        results_root=cfg.get("results_root", "results"),
        output_dir=out,
        extra_meta={
            "weighting": weighting,
            "best_round": best_round,
            "best_avg_val_auc": float(best_avg_val_auc),
            "total_rounds": n_rounds,
            "total_bytes_sent": total_bytes_sent,
            "total_train_steps": total_train_steps,
        }
    )

    results = {
        "method": effective_method,
        "weighting": weighting,
        "seed": seed,
        "best_round": best_round,
        "best_avg_val_auc": float(best_avg_val_auc),
        "total_rounds": n_rounds,
        "total_bytes_sent": total_bytes_sent,
        "total_train_steps": total_train_steps,
        "threshold": float(threshold),
        "per_site_test": test_results,
        "history": history,
        "best_state": best_global_state,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(best_global_state, out / "best.pt")
        with open(out / "metrics.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    return results
