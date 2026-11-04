"""Reproducible seeding utilities."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed across Python, NumPy, and PyTorch (CPU and CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True


def get_rng_states() -> dict:
    """Capture current RNG states across all generators."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict) -> None:
    """Restore RNG states across all generators."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch_state = states["torch"]
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.cpu().to(torch.uint8)
        torch.set_rng_state(torch_state)
    if "cuda" in states and torch.cuda.is_available():
        cuda_states = states["cuda"]
        if isinstance(cuda_states, (list, tuple)):
            cuda_states = [s.cpu().to(torch.uint8) if isinstance(s, torch.Tensor) else s for s in cuda_states]
        elif isinstance(cuda_states, torch.Tensor):
            cuda_states = cuda_states.cpu().to(torch.uint8)
        try:
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception as e:
            logger.debug("Failed restoring CUDA RNG state: %s", e)
