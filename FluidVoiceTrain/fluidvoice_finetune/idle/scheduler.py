"""launchd + pmset scheduling for the idle trainer.

``install()`` writes a LaunchAgent that fires the gated trainer at a nightly
calendar time (default 01:00), wrapped in ``caffeinate -is -w <pid>`` so the
machine stays awake for the duration. A ``pmset repeat wakeorpoweron`` schedule
wakes the machine if it's asleep at that time.

    fluidvoice-finetune idle-install --time 01:00 --venv /path/to/venv
    fluidvoice-finetune idle-uninstall
    fluidvoice-finetune idle-status
"""
from __future__ import annotations

import plistlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LABEL = "com.fluidvoice.idle-train"
LAUNCH_AGENT_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


@dataclass
class ScheduleConfig:
    hour: int = 1
    minute: int = 0
    venv_dir: Path | None = None  # venv containing the fluidvoice-finetune entrypoint
    corpus_dir: Path = Path.home() / "Library" / "Application Support" / "FluidVoice" / "FluidIntelligenceTraining" / "corpus"
    checkpoint_dir: Path = Path.home() / "Library" / "Application Support" / "FluidVoice" / "FluidIntelligenceTraining" / "checkpoints"
    extra_args: list[str] | None = None


def _entrypoint(venv_dir: Path | None) -> str:
    """Resolve the fluidvoice-finetune binary, preferring a venv."""
    if venv_dir is not None:
        candidate = venv_dir / "bin" / "fluidvoice-finetune"
        if candidate.exists():
            return str(candidate)
    # Fall back to whatever is on PATH (or sys.executable's bin dir).
    import shutil
    found = shutil.which("fluidvoice-finetune")
    return found or str(Path(sys.executable).parent / "fluidvoice-finetune")


def build_plist(cfg: ScheduleConfig) -> dict:
    """Build the launchd plist dict (without writing it)."""
    entrypoint = _entrypoint(cfg.venv_dir)
    train_cmd = [
        entrypoint, "idle-train",
        "--corpus", str(cfg.corpus_dir),
        "--checkpoint-dir", str(cfg.checkpoint_dir),
    ]
    if cfg.extra_args:
        train_cmd.extend(cfg.extra_args)
    # Wrap in caffeinate -is -w <pid>: prevent idle+system sleep while training.
    # We use a wrapper script approach: bash -c 'caffeinate -is -w $!'
    wrapped = ["bash", "-c", f"{' '.join(train_cmd)} & caffeinate -is -w $!"]
    return {
        "Label": LABEL,
        "ProgramArguments": wrapped,
        "StartCalendarInterval": {"Hour": cfg.hour, "Minute": cfg.minute},
        "RunAtLoad": False,  # don't fire on load; only at the scheduled time (set True for catch-up)
        "StandardOutPath": str(cfg.checkpoint_dir / "idle-train.stdout.log"),
        "StandardErrorPath": str(cfg.checkpoint_dir / "idle-train.stderr.log"),
        "EnvironmentVariables": {"PATH": "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin"},
    }


def _pmset_schedule(hour: int, minute: int) -> str:
    """The day-pattern + time string for `pmset repeat wakeorpoweron`.
    MTWRFSU = Mon Tue Wed Thu Fri Sat Sun (pmset's day codes)."""
    return f"MTWRFSU {hour:02d}:{minute:02d}:00"


def install(cfg: ScheduleConfig, *, dry_run: bool = False) -> dict:
    """Install the LaunchAgent + pmset wake schedule.

    Returns a dict describing what was (or would be) written/done.
    """
    plist = build_plist(cfg)
    pmset_when = _pmset_schedule(cfg.hour, cfg.minute)
    actions = {
        "plist_path": str(LAUNCH_AGENT_PATH),
        "pmset_command": f"pmset repeat wakeorpoweron {pmset_when}",
        "load_command": f"launchctl load {LAUNCH_AGENT_PATH}",
        "plist": plist,
    }
    if dry_run:
        actions["dry_run"] = True
        return actions

    LAUNCH_AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LAUNCH_AGENT_PATH.open("wb") as fh:
        plistlib.dump(plist, fh)
    # Schedule the wake.
    subprocess.run(["pmset", "repeat", "wakeorpoweron", pmset_when], check=False)
    # Load the agent.
    subprocess.run(["launchctl", "load", str(LAUNCH_AGENT_PATH)], check=False)
    return actions


def uninstall(*, dry_run: bool = False) -> dict:
    """Remove the LaunchAgent and clear the pmset wake schedule."""
    actions = {
        "plist_path": str(LAUNCH_AGENT_PATH),
        "unload_command": f"launchctl unload {LAUNCH_AGENT_PATH}",
        "pmset_clear": "pmset repeat cancel",
    }
    if dry_run:
        actions["dry_run"] = True
        return actions

    subprocess.run(["launchctl", "unload", str(LAUNCH_AGENT_PATH)], check=False)
    if LAUNCH_AGENT_PATH.exists():
        LAUNCH_AGENT_PATH.unlink()
    # Clear the repeating wake schedule.
    subprocess.run(["pmset", "repeat", "cancel"], check=False)
    return actions


def status() -> dict:
    """Report install state + last/next run hints."""
    installed = LAUNCH_AGENT_PATH.exists()
    out = {"installed": installed, "plist_path": str(LAUNCH_AGENT_PATH)}
    if installed:
        try:
            plist = plistlib.loads(LAUNCH_AGENT_PATH.read_bytes())
            out["schedule_hour"] = plist.get("StartCalendarInterval", {}).get("Hour")
            out["schedule_minute"] = plist.get("StartCalendarInterval", {}).get("Minute")
        except Exception:
            out["plist_parse_error"] = True
    # pmset scheduled wakes
    try:
        r = subprocess.run(["pmset", "-g", "sched"], capture_output=True, text=True, timeout=5)
        out["pmset_sched"] = r.stdout.strip() if r.returncode == 0 else ""
    except subprocess.SubprocessError:
        out["pmset_sched"] = ""
    return out
