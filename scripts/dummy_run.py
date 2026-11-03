#!/usr/bin/env python3
"""
Dummy training runner for Phase 0 verification.

Exercises:
  - Seeding
  - RunLogger (config.yaml, env.json, metrics.jsonl, events.jsonl, results/ledger.csv)
  - Atomic CheckpointManager (last.pt, best.pt, state restoration)
  - ThermalGuard (telemetry and reactive monitoring)
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Auto-delegate to local .venv if run with system python
_venv_dir = Path(__file__).resolve().parent.parent / ".venv"
_venv_py = _venv_dir / "bin" / "python3"
if _venv_py.exists() and sys.prefix != str(_venv_dir):
    os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.ckpt import CheckpointManager
from src.utils.provenance import RunLogger, compute_config_hash, get_git_info
from src.utils.seed import set_seed
from src.utils.thermal import ThermalGuard


def parse_args():
    parser = argparse.ArgumentParser(description="P0 Dummy Training Runner")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path or 'auto'")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run ID")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = {
        "experiment_id": "P0-DUMMY",
        "method": "dummy_trainer",
        "site": "synthetic",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": 1e-3,
    }

    # Setup RunLogger
    logger = RunLogger(cfg=cfg, run_id=args.run_id)
    print(f"[P0] Initialized RunLogger: {logger.run_id} at {logger.run_dir}")

    # Setup CheckpointManager
    cfg_hash = compute_config_hash(cfg)
    git_commit = get_git_info()["commit"]
    ckpt_mgr = CheckpointManager(
        checkpoint_dir=logger.ckpt_dir,
        config_hash=cfg_hash,
        git_commit=git_commit,
    )

    # Setup ThermalGuard
    thermal_guard = ThermalGuard(
        event_logger=lambda ev: logger.log_event(ev["event"], ev),
        check_every_iters=5,
    )

    # Simple 2-layer model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 64 * 64, 64),
        nn.ReLU(),
        nn.Linear(64, 3),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    # Dummy Dataset (64 samples, 3x64x64)
    x = torch.randn(64, 3, 64, 64)
    y = torch.randint(0, 3, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    # Handle resume
    if args.resume:
        ckpt_target = None if args.resume == "auto" else Path(args.resume)
        start_epoch, global_step, loaded_metric, extra = ckpt_mgr.load(
            ckpt_path=ckpt_target,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        if loaded_metric is not None:
            best_loss = loaded_metric
        logger.log_event("training_resumed", {"epoch": start_epoch, "step": global_step})
        print(f"[P0] Resumed training at epoch {start_epoch}, global_step {global_step}")

    wall_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for step, (bx, by) in enumerate(loader):
            global_step += 1
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    pred = model(bx)
                    loss = criterion(pred, by)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Thermal guard check
            def on_hard():
                ckpt_mgr.save(
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    metric_value=loss.item(),
                    is_best=False,
                )
                logger.log_event("checkpoint_saved", {"reason": "thermal_hard_threshold"})

            thermal_guard.step(on_hard=on_hard)

        avg_loss = total_loss / max(num_batches, 1)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        # Save checkpoint
        ckpt_mgr.save(
            epoch=epoch + 1,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            metric_value=avg_loss,
            is_best=is_best,
        )
        logger.log_event("checkpoint_saved", {"epoch": epoch + 1, "loss": avg_loss, "is_best": is_best})

        # Log metrics
        logger.log_metric({
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": avg_loss,
            "best_loss": best_loss,
            "paused_seconds": thermal_guard.total_paused_seconds,
        })
        print(f"[P0] Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f} (Best: {best_loss:.4f})")

    wall_time = time.time() - wall_start

    # Record in ledger
    logger.append_ledger(
        experiment_id=cfg["experiment_id"],
        method=cfg["method"],
        site=cfg["site"],
        seed=cfg["seed"],
        primary_metric="train_loss",
        metric_value=best_loss,
        wall_time_s=wall_time,
        paused_time_s=thermal_guard.total_paused_seconds,
        status="completed",
    )
    print(f"[P0] Completed run {logger.run_id} in {wall_time:.2f}s (Paused: {thermal_guard.total_paused_seconds:.2f}s)")


if __name__ == "__main__":
    main()
