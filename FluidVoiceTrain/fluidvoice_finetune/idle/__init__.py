"""On-device idle-time training (macOS Apple Silicon).

Public API:
    - :func:`gate` — check whether it's safe/polite to train now.
    - :func:`run_idle_train` — gated, incremental, checkpointed Gemma train run.
    - :func:`install` / :func:`uninstall` — launchd + pmset scheduling.

Avoids a pyobjc dependency: power/idle/thermal state is read by shelling out to
``pmset`` and ``ioreg``. This keeps the install story simple for non-developers.
"""
from __future__ import annotations

from .gate import gate, GateResult, is_on_ac_power, is_low_power_mode, idle_seconds
from .runner import run_idle_train
from .scheduler import install, uninstall, status

__all__ = [
    "gate", "GateResult", "is_on_ac_power", "is_low_power_mode", "idle_seconds",
    "run_idle_train", "install", "uninstall", "status",
]
