"""GPU thermal management and cooldown utility.

Monitors GPU die temperature using nvidia-smi and applies cooldown pauses:
1. Dynamic thermal guard: Pauses for 15s if GPU exceeds a configured temperature threshold (default: 72°C).
2. Inter-epoch pause: Pauses for 10-15s at the end of every epoch/round to allow heat dissipation.
"""

import subprocess
import time

import torch


def get_gpu_temperature(device_index: int = 0) -> int | None:
    """Query current GPU temperature in Celsius via nvidia-smi."""
    if not torch.cuda.is_available():
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={device_index}", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=2.0
        ).decode().strip()
        return int(out)
    except Exception:
        return None


def gpu_thermal_cooldown(cfg: dict = None, verbose: bool = True, epoch: int = None):
    """Perform thermal management pause.

    Parameters
    ----------
    cfg : dict
        Experiment configuration dict containing optional 'thermal' section:
          thermal:
            enabled: bool (default True)
            temp_threshold: int (default 72°C)
            hot_cooldown_sec: int (default 15s)
            cooldown_interval_epochs: int (default 10)  # Pause every N epochs
            interval_cooldown_sec: int (default 10)     # Pause duration for interval
            epoch_cooldown_sec: int (default 0)         # Per-epoch pause (0 = disabled)
    verbose : bool
        Whether to print status to stdout.
    epoch : int, optional
        Current completed epoch or round number.
    """
    if not torch.cuda.is_available():
        return

    thermal_cfg = (cfg or {}).get("thermal", {})
    if not thermal_cfg.get("enabled", True):
        return

    temp_threshold = thermal_cfg.get("temp_threshold", 72)
    hot_cooldown = thermal_cfg.get("hot_cooldown_sec", 15)
    interval_epochs = thermal_cfg.get("cooldown_interval_epochs", 10)
    interval_cooldown = thermal_cfg.get("interval_cooldown_sec", 10)
    epoch_cooldown = thermal_cfg.get("epoch_cooldown_sec", 0)

    current_temp = get_gpu_temperature()
    temp_str = f"{current_temp}°C" if current_temp is not None else "N/A"

    # 1. Hot temperature threshold triggered (emergency cooldown)
    if current_temp is not None and current_temp >= temp_threshold:
        if verbose:
            print(f"  [Thermal Guard] GPU temp is {temp_str} (>= {temp_threshold}°C threshold). "
                  f"Pausing {hot_cooldown}s for thermal relief...", flush=True)
        time.sleep(hot_cooldown)
        after_temp = get_gpu_temperature()
        if verbose:
            after_str = f"{after_temp}°C" if after_temp is not None else "N/A"
            print(f"  [Thermal Guard] Cooldown complete. GPU temp now: {after_str}", flush=True)

    # 2. Periodic interval cooldown (every N epochs, e.g. after 10 epochs for 10s)
    elif epoch is not None and interval_epochs > 0 and (epoch % interval_epochs == 0):
        if verbose:
            print(f"  [Cooldown] Completed {epoch} epochs. Pausing {interval_cooldown}s (GPU: {temp_str})...", flush=True)
        time.sleep(interval_cooldown)
        after_temp = get_gpu_temperature()
        if verbose and after_temp is not None:
            print(f"  [Cooldown] Cooldown complete. GPU temp now: {after_temp}°C", flush=True)

    # 3. Optional per-epoch cooldown
    elif epoch_cooldown > 0:
        if verbose:
            print(f"  [Cooldown] Pausing {epoch_cooldown}s between epochs (GPU: {temp_str})...", flush=True)
        time.sleep(epoch_cooldown)
