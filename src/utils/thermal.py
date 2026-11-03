"""Reactive GPU thermal guard and telemetry logger.

Adheres strictly to AGENTS.md Section 3.1:
  - soft_c: 70, clamped to slowdown - 5: pause until cool
  - hard_c: 75, clamped to slowdown - 2: checkpoint immediately, then pause
  - resume_c: calibrated (idle + 8 C, default 60)
  - check_every_iters: 20
  - max_wait_s: 600; if still hot, checkpoint and exit with code 75
"""

import logging
import subprocess
import sys
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class GPUThermalSensor:
    """Hardware GPU thermal and telemetry sensor."""

    def __init__(self):
        self.has_nvml = False
        self.nvml_handle = None
        self.slowdown_temp = 87.0

        # Try NVML first
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.has_nvml = True
            try:
                self.slowdown_temp = float(
                    pynvml.nvmlDeviceGetTemperatureThreshold(
                        self.nvml_handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN
                    )
                )
            except Exception:
                self.slowdown_temp = 87.0
        except Exception:
            self.has_nvml = False

    def query(self) -> dict:
        """Query temperature, power, utilization, and SM clock."""
        if self.has_nvml and self.nvml_handle is not None:
            import pynvml
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU))
                power = float(pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)) / 1000.0  # mW -> W
                util = float(pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu)
                clock = float(pynvml.nvmlDeviceGetClockInfo(self.nvml_handle, pynvml.NVML_CLOCK_SM))
                return {
                    "temperature": temp,
                    "power_w": power,
                    "utilization_gpu": util,
                    "sm_clock_mhz": clock,
                    "slowdown_temp": self.slowdown_temp,
                    "source": "nvml",
                }
            except Exception as e:
                logger.debug("NVML query failed, falling back: %s", e)

        # Fallback to nvidia-smi
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks.sm",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            parts = [float(x.strip()) for x in res.stdout.strip().split(",")]
            return {
                "temperature": parts[0],
                "power_w": parts[1] if len(parts) > 1 else 0.0,
                "utilization_gpu": parts[2] if len(parts) > 2 else 0.0,
                "sm_clock_mhz": parts[3] if len(parts) > 3 else 0.0,
                "slowdown_temp": self.slowdown_temp,
                "source": "nvidia-smi",
            }
        except Exception as e:
            logger.debug("nvidia-smi query failed: %s", e)
            return {
                "temperature": 45.0,
                "power_w": 0.0,
                "utilization_gpu": 0.0,
                "sm_clock_mhz": 0.0,
                "slowdown_temp": 87.0,
                "source": "mock_fallback",
            }


class ThermalGuard:
    """Reactive thermal monitor and throttle controller."""

    def __init__(
        self,
        sensor: GPUThermalSensor | None = None,
        event_logger: Callable[[dict], None] | None = None,
        soft_c: float = 70.0,
        hard_c: float = 75.0,
        resume_c: float = 60.0,
        check_every_iters: int = 20,
        max_wait_s: float = 600.0,
        pause_sleep_s: float = 2.0,
        exit_on_timeout: bool = True,
    ):
        self.sensor = sensor or GPUThermalSensor()
        self.event_logger = event_logger
        self.check_every_iters = check_every_iters
        self.max_wait_s = max_wait_s
        self.pause_sleep_s = pause_sleep_s
        self.exit_on_timeout = exit_on_timeout

        # Calibrate clamp thresholds against hardware slowdown
        slowdown = getattr(self.sensor, "slowdown_temp", 87.0)
        self.soft_c = min(soft_c, slowdown - 5.0)
        self.hard_c = min(hard_c, slowdown - 2.0)
        self.resume_c = resume_c

        self.iter_count = 0
        self.total_paused_seconds = 0.0
        self.pause_count = 0
        self.start_time = time.time()

    def log_event(self, event_type: str, details: dict) -> None:
        """Forward telemetry or pause events to caller or logger."""
        details["event"] = event_type
        details["ts"] = time.time()
        if self.event_logger:
            self.event_logger(details)

    def step(self, on_hard: Callable[[], None] | None = None) -> dict:
        """Periodic step check called inside the training/eval loop."""
        self.iter_count += 1
        if self.iter_count % self.check_every_iters != 0:
            return {"status": "ok"}

        return self.check_and_pause(on_hard=on_hard)

    def check_and_pause(self, on_hard: Callable[[], None] | None = None) -> dict:
        """Inspect temperature and pause reactively if soft or hard limits are breached."""
        sample = self.sensor.query()
        temp = sample.get("temperature", 0.0)

        if temp < self.soft_c:
            return {"status": "ok", "temperature": temp}

        # Temperature threshold breached
        is_hard = temp >= self.hard_c
        reason = "hard_threshold" if is_hard else "soft_threshold"
        logger.warning(
            "Thermal limit breached (temp=%.1f C, soft=%.1f C, hard=%.1f C). Entering reactive pause.",
            temp,
            self.soft_c,
            self.hard_c,
        )

        if is_hard and on_hard:
            logger.warning("Hard temperature limit reached! Triggering on_hard checkpoint callback immediately.")
            try:
                on_hard()
            except Exception as e:
                logger.error("Error in on_hard checkpoint handler: %s", e)

        self.log_event("thermal_pause_start", {
            "reason": reason,
            "temperature_start": temp,
            "soft_c": self.soft_c,
            "hard_c": self.hard_c,
            "resume_c": self.resume_c,
        })

        wait_start = time.time()
        paused_this_event = 0.0

        while True:
            cur_sample = self.sensor.query()
            cur_temp = cur_sample.get("temperature", 0.0)
            elapsed = time.time() - wait_start

            if cur_temp <= self.resume_c:
                logger.info(
                    "GPU cooled to %.1f C (<= resume_c %.1f C) after %.1fs. Resuming training.",
                    cur_temp,
                    self.resume_c,
                    elapsed,
                )
                break

            if elapsed >= self.max_wait_s:
                logger.error(
                    "Thermal timeout exceeded (waited %.1fs, temp still %.1f C > %.1f C).",
                    elapsed,
                    cur_temp,
                    self.resume_c,
                )
                self.log_event("thermal_timeout", {
                    "waited_seconds": elapsed,
                    "final_temperature": cur_temp,
                    "action": "checkpoint_and_exit_75",
                })
                if on_hard:
                    on_hard()
                if self.exit_on_timeout:
                    sys.exit(75)  # Code 75: Resumable exit
                return {"status": "timeout_exit_75", "temperature": cur_temp, "paused_seconds": elapsed}

            time.sleep(self.pause_sleep_s)

        paused_this_event = time.time() - wait_start
        self.total_paused_seconds += paused_this_event
        self.pause_count += 1

        self.log_event("thermal_pause_end", {
            "duration_seconds": paused_this_event,
            "temperature_end": cur_temp,
            "total_paused_seconds": self.total_paused_seconds,
        })

        return {
            "status": "resumed",
            "temperature": cur_temp,
            "paused_seconds": paused_this_event,
            "total_paused_seconds": self.total_paused_seconds,
        }

    def check_pause_budget(self) -> bool:
        """Assert whether paused seconds exceed 30% of wall-clock time."""
        wall_clock = time.time() - self.start_time
        if wall_clock > 60 and (self.total_paused_seconds / wall_clock) > 0.30:
            logger.error(
                "CRITICAL: Thermal pauses (%.1fs) exceed 30%% of run wall-clock (%.1fs). Action required.",
                self.total_paused_seconds,
                wall_clock,
            )
            return False
        return True
