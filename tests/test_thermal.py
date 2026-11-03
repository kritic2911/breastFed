"""Unit tests for ThermalGuard reactive monitoring and safety rails."""

from src.utils.thermal import ThermalGuard


class MockSensor:
    """Mock sensor returning pre-configured temperature trajectories."""

    def __init__(self, temps, slowdown_temp=87.0):
        self.temps = list(temps)
        self.slowdown_temp = slowdown_temp
        self.idx = 0

    def query(self):
        val = self.temps[min(self.idx, len(self.temps) - 1)]
        self.idx += 1
        return {
            "temperature": val,
            "power_w": 25.0,
            "utilization_gpu": 50.0,
            "sm_clock_mhz": 1200.0,
            "slowdown_temp": self.slowdown_temp,
            "source": "mock",
        }


def test_thermal_soft_pause_and_cooling():
    """Verify that a soft thermal breach pauses and resumes once cooled."""
    # Starts at 72 C (soft breach >= 70), then drops to 65 C, then 58 C (<= resume_c 60 C)
    sensor = MockSensor([72.0, 65.0, 58.0])
    events = []
    guard = ThermalGuard(
        sensor=sensor,
        event_logger=events.append,
        soft_c=70.0,
        hard_c=75.0,
        resume_c=60.0,
        check_every_iters=1,
        pause_sleep_s=0.01,
    )

    res = guard.step()
    assert res["status"] == "resumed"
    assert res["temperature"] <= 60.0
    assert len(events) >= 2  # pause_start and pause_end


def test_thermal_hard_limit_calls_on_hard():
    """Verify that exceeding hard limit calls on_hard callback immediately."""
    sensor = MockSensor([76.0, 55.0])  # >= 75 C hard breach
    hard_called = False

    def on_hard_callback():
        nonlocal hard_called
        hard_called = True

    guard = ThermalGuard(
        sensor=sensor,
        soft_c=70.0,
        hard_c=75.0,
        resume_c=60.0,
        check_every_iters=1,
        pause_sleep_s=0.01,
    )

    guard.step(on_hard=on_hard_callback)
    assert hard_called is True


def test_thermal_timeout_exits_code_75():
    """Verify that persistent overheating past max_wait_s triggers exit 75."""
    # Constantly at 80 C
    sensor = MockSensor([80.0] * 10)
    hard_called = False

    def on_hard_callback():
        nonlocal hard_called
        hard_called = True

    guard = ThermalGuard(
        sensor=sensor,
        soft_c=70.0,
        hard_c=75.0,
        resume_c=60.0,
        check_every_iters=1,
        max_wait_s=0.05,
        pause_sleep_s=0.01,
        exit_on_timeout=False,  # Don't kill test runner process, return status
    )

    res = guard.step(on_hard=on_hard_callback)
    assert res["status"] == "timeout_exit_75"
    assert hard_called is True
