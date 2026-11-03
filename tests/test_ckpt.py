"""Unit tests for atomic checkpointing, corruption recovery, and resume."""

import pytest
import torch
from torch import nn

from src.utils.ckpt import CheckpointManager


def test_checkpoint_save_and_load_integrity(tmp_path):
    """Verify that model, optimizer, step, and metric states are preserved identically."""
    ckpt_dir = tmp_path / "ckpt"
    mgr = CheckpointManager(checkpoint_dir=ckpt_dir, config_hash="test1234", git_commit="abc1234")

    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Do one synthetic step
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    orig_weight = model.weight.clone()

    mgr.save(
        epoch=3,
        global_step=45,
        model=model,
        optimizer=optimizer,
        metric_value=0.42,
        is_best=True,
    )

    assert (ckpt_dir / "last.pt").exists()
    assert (ckpt_dir / "best.pt").exists()

    # Create new fresh model and restore
    new_model = nn.Linear(4, 2)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)

    epoch, step, metric, extra = mgr.load(
        ckpt_path=ckpt_dir / "last.pt",
        model=new_model,
        optimizer=new_optimizer,
    )

    assert epoch == 3
    assert step == 45
    assert metric == pytest.approx(0.42)
    assert torch.equal(new_model.weight, orig_weight)


def test_checkpoint_corruption_fallback(tmp_path):
    """Verify that corrupt last.pt automatically falls back to last_backup.pt."""
    ckpt_dir = tmp_path / "ckpt"
    mgr = CheckpointManager(checkpoint_dir=ckpt_dir, config_hash="test1234")

    model = nn.Linear(2, 2)

    # First save (creates last.pt)
    mgr.save(epoch=1, global_step=10, model=model, metric_value=0.5)
    # Second save (backs up last.pt to last_backup.pt, then updates last.pt)
    mgr.save(epoch=2, global_step=20, model=model, metric_value=0.3)

    assert (ckpt_dir / "last.pt").exists()
    assert (ckpt_dir / "last_backup.pt").exists()

    # Corrupt last.pt
    with open(ckpt_dir / "last.pt", "wb") as f:
        f.write(b"CORRUPTED_BYTES_NOT_A_VALID_TORCH_FILE")

    # Attempt load -> should fall back to last_backup.pt (epoch 1)
    new_model = nn.Linear(2, 2)
    epoch, step, metric, _ = mgr.load(ckpt_path=ckpt_dir / "last.pt", model=new_model)
    assert epoch == 1
    assert step == 10
    assert metric == pytest.approx(0.5)


def test_kill_and_resume_continuation(tmp_path):
    """Verify simulate kill -9 mid-epoch and resume from last checkpoint."""
    ckpt_dir = tmp_path / "ckpt"
    mgr = CheckpointManager(checkpoint_dir=ckpt_dir, config_hash="hash_run")

    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Phase 1: Train for 2 epochs, save at each epoch
    for ep in range(1, 3):
        mgr.save(epoch=ep, global_step=ep * 10, model=model, optimizer=optimizer, metric_value=1.0 / ep)

    # Simulate crash and resume
    resumed_model = nn.Linear(3, 1)
    resumed_opt = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
    epoch, step, metric, _ = mgr.load(model=resumed_model, optimizer=resumed_opt)

    assert epoch == 2
    assert step == 20
    assert metric == pytest.approx(0.5)

    # Continue training from epoch 2 to epoch 4
    for ep in range(epoch, 4):
        mgr.save(epoch=ep + 1, global_step=step + (ep - epoch + 1) * 10, model=resumed_model, optimizer=resumed_opt)

    final_epoch, final_step, _, _ = mgr.load(model=resumed_model)
    assert final_epoch == 4
    assert final_step == 40
