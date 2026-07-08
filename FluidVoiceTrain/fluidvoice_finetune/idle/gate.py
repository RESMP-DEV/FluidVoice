"""Power / idle / thermal gate for on-device training.

Pure-python (no pyobjc): reads ``pmset -g batt``, ``pmset -g``, and
``ioreg -c IOHIDSystem`` via subprocess. All functions degrade gracefully
(return permissive defaults) on non-macOS or parse failure, so the gate never
*blocks* training due to a parsing glitch — the caller decides whether to
require strict gating.
"""
from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Literal


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.stdout if r.returncode == 0 else ""
    except (subprocess.SubprocessError, OSError):
        return ""


def is_on_ac_power() -> bool:
    """True if drawing from AC power (not battery). Permissive default True."""
    if not _is_macos():
        return True
    out = _run(["pmset", "-g", "batt"])
    # Lines like: "Now drawing from 'AC Power'" or "'Battery Power'".
    if "AC Power" in out:
        return True
    if "Battery Power" in out:
        return False
    return True  # unknown → permissive


def is_low_power_mode() -> bool:
    """True if macOS Low Power Mode is on. Permissive default False."""
    if not _is_macos():
        return False
    out = _run(["pmset", "-g"])
    # "lowpowermode" appears in the active assertions / settings.
    return bool(re.search(r"lowpowermode\s+\d", out)) and "lowpowermode              1" in out


def idle_seconds() -> float:
    """Seconds since last HID input (keyboard/mouse/trackpad). Permissive 0.0."""
    if not _is_macos():
        return 0.0
    out = _run(["ioreg", "-c", "IOHIDSystem", "-d", "4"])
    # "HIDIdleTime" = nanoseconds since last input.
    m = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', out)
    if not m:
        return 0.0
    try:
        return int(m.group(1)) / 1_000_000_000
    except (ValueError, ZeroDivisionError):
        return 0.0


def thermal_state() -> Literal["nominal", "fair", "serious", "critical", "unknown"]:
    """Coarse thermal state. There's no clean CLI; approximate via `pmset -g therm`.
    Permissive default 'nominal'."""
    if not _is_macos():
        return "nominal"
    out = _run(["pmset", "-g", "therm"])
    if "CPU_Scheduler_Limit" in out:
        # Non-zero scheduler limit / thermal pressure suggests >= fair.
        # This is a rough proxy; NSProcessInfo.thermalState is the real signal.
        for line in out.splitlines():
            if "CPU_Scheduler_Limit" in line:
                try:
                    val = int(line.split("=")[-1].strip())
                    return "fair" if val > 0 else "nominal"
                except ValueError:
                    pass
    return "nominal"


@dataclass
class GateResult:
    """Outcome of the full gate check. ``ok`` = safe to train now."""
    ok: bool
    on_ac: bool
    low_power: bool
    idle_s: float
    thermal: str
    reasons: list[str]

    def __bool__(self) -> bool:
        return self.ok


def gate(
    *,
    require_ac: bool = True,
    min_idle_seconds: float = 600.0,  # 10 min
    max_thermal: str = "fair",  # nominal, fair, serious, critical
    allow_low_power: bool = False,
) -> GateResult:
    """Check all conditions for polite idle training.

    Defaults: must be on AC, idle ≥10 min, thermal at-or-below 'fair', and Low
    Power Mode off. Returns a :class:`GateResult` with the individual signals
    and a list of human-readable reasons the gate failed (empty if ok).
    """
    on_ac = is_on_ac_power()
    low_power = is_low_power_mode()
    idle_s = idle_seconds()
    therm = thermal_state()

    thermal_rank = {"nominal": 0, "fair": 1, "serious": 2, "critical": 3, "unknown": 0}
    max_rank = thermal_rank.get(max_thermal, 1)
    therm_ok = thermal_rank.get(therm, 0) <= max_rank

    reasons: list[str] = []
    if require_ac and not on_ac:
        reasons.append("not on AC power")
    if not allow_low_power and low_power:
        reasons.append("Low Power Mode is on")
    if idle_s < min_idle_seconds:
        reasons.append(f"idle {idle_s:.0f}s < {min_idle_seconds:.0f}s threshold")
    if not therm_ok:
        reasons.append(f"thermal state {therm!r} exceeds {max_thermal!r}")

    return GateResult(
        ok=not reasons,
        on_ac=on_ac, low_power=low_power, idle_s=idle_s, thermal=therm,
        reasons=reasons,
    )
