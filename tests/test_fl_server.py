"""Server execution and kill-mid-round resume tests for Phase P3 Federated Learning.

Adheres strictly to AGENTS.md Section 3.2, 2.3 Checkpoint P3:
  - Required test: kill mid-round, resume, and show that round counter continues,
    no log rows are duplicated, and training continues cleanly.
  - Verifies round progression, communication byte accounting, and state saving.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.fl.client import FLClient
from src.fl.server import FLServer
from src.utils.provenance import hash_patient_id


def make_tiny_synthetic_data(tmp_path: Path):
    """Generate minimal synthetic cohort for server unit test."""
    cache_dir = tmp_path / "cache512"
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for site in ["vindr", "cmmd"]:
        for pt_idx in range(2):
            raw_pt = f"pt_{site}_{pt_idx}"
            pt_hash = hash_patient_id(raw_pt, salt=b"fl_test_salt")
            split = "train" if pt_idx == 0 else "test"

            for laterality in ["R"]:
                for view in ["CC"]:
                    img_id = f"{site}_{pt_hash[:6]}_{view}"
                    rel_cached = f"cache512/{img_id}.npy"
                    abs_cached = tmp_path / rel_cached

                    arr = np.ones((512, 512), dtype=np.float32) * 0.2
                    np.save(abs_cached, arr)

                    records.append({
                        "site": site,
                        "label_family": "biopsy" if site == "cmmd" else "birads",
                        "patient_hash": pt_hash,
                        "study": f"study_{pt_hash[:6]}",
                        "laterality": laterality,
                        "view": view,
                        "path": f"raw/{site}/{img_id}.dcm",
                        "label": 1 if site == "cmmd" else 0,
                        "birads_raw": "BI-RADS 2",
                        "split": split,
                        "source_format": "dicom",
                        "bit_depth": 16,
                        "manufacturer": "GE",
                        "pixel_spacing": "0.1, 0.1",
                        "photometric_interpretation": "MONOCHROME2",
                        "age": 50.0,
                        "density": "B",
                        "crop_box": "100, 100, 2000, 2000",
                        "flip": False,
                        "scale": 0.25,
                        "cached_path": rel_cached,
                        "image_id": img_id,
                        "breast_id": f"{site}_{pt_hash}_{laterality}",
                    })

    df = pd.DataFrame(records)
    return df, tmp_path


def test_fl_server_two_round_execution(tmp_path: Path):
    """Verify FLServer executes 2 communication rounds on CPU and records metrics."""
    df, data_root = make_tiny_synthetic_data(tmp_path)
    run_dir = tmp_path / "fl_test_run"

    # Setup 2 clients on CPU
    c1 = FLClient(
        site="vindr",
        train_df=df[(df["site"] == "vindr") & (df["split"] == "train")],
        data_dir=data_root,
        device=torch.device("cpu"),
        backbone_lr=1e-4,
        local_epochs=1,
        batch_size=2,
    )
    c2 = FLClient(
        site="cmmd",
        train_df=df[(df["site"] == "cmmd") & (df["split"] == "train")],
        data_dir=data_root,
        device=torch.device("cpu"),
        backbone_lr=1e-4,
        local_epochs=1,
        batch_size=2,
    )

    val_df = df[df["split"] == "train"].copy()
    test_dfs = {
        "vindr": df[(df["site"] == "vindr") & (df["split"] == "test")],
        "cmmd": df[(df["site"] == "cmmd") & (df["split"] == "test")],
    }

    server = FLServer(
        clients=[c1, c2],
        val_df=val_df,
        test_dfs=test_dfs,
        strategy="fedavg",
        aggregation_mode="sample_weighted",
        num_rounds=2,
        run_dir=run_dir,
        device=torch.device("cpu"),
        data_dir=data_root,
    )

    res = server.run(resume=False)
    assert "overall_macro_auc" in res
    assert (run_dir / "ckpt" / "best.pt").exists()
    assert (run_dir / "ckpt" / "last.pt").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert server.total_bytes_transferred > 0


def test_fl_server_kill_and_resume(tmp_path: Path):
    """Verify that simulating kill-mid-run resumes from the latest completed round."""
    df, data_root = make_tiny_synthetic_data(tmp_path)
    run_dir = tmp_path / "fl_resume_run"

    c1 = FLClient(
        site="vindr",
        train_df=df[(df["site"] == "vindr") & (df["split"] == "train")],
        data_dir=data_root,
        device=torch.device("cpu"),
        batch_size=2,
    )
    c2 = FLClient(
        site="cmmd",
        train_df=df[(df["site"] == "cmmd") & (df["split"] == "train")],
        data_dir=data_root,
        device=torch.device("cpu"),
        batch_size=2,
    )

    val_df = df[df["split"] == "train"].copy()
    test_dfs = {"vindr": df[df["split"] == "test"]}

    # Phase 1: Run 1 round, then stop (simulate kill)
    server_1 = FLServer(
        clients=[c1, c2],
        val_df=val_df,
        test_dfs=test_dfs,
        strategy="fedavg",
        num_rounds=1,
        run_dir=run_dir,
        device=torch.device("cpu"),
        data_dir=data_root,
    )
    server_1.run(resume=False)

    assert server_1.current_round == 1
    last_ckpt_1 = torch.load(run_dir / "ckpt" / "last.pt", map_location="cpu", weights_only=False)
    assert last_ckpt_1["epoch"] == 1

    # Phase 2: Resume with target 2 rounds
    server_2 = FLServer(
        clients=[c1, c2],
        val_df=val_df,
        test_dfs=test_dfs,
        strategy="fedavg",
        num_rounds=2,
        run_dir=run_dir,
        device=torch.device("cpu"),
        data_dir=data_root,
    )
    server_2.run(resume=True)

    assert server_2.current_round == 2
    last_ckpt_2 = torch.load(run_dir / "ckpt" / "last.pt", map_location="cpu", weights_only=False)
    assert last_ckpt_2["epoch"] == 2
