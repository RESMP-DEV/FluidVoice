"""Tests for the idle-training package (gate parsing, plist generation,
watermark increment). These run WITHOUT mlx-lm/llama.cpp installed by using
``--dry-run-prepare`` and direct function calls."""
from __future__ import annotations

import json
import plistlib
import sys
from pathlib import Path

import pytest

# Import the package first so the submodule lands in sys.modules, then grab the
# module object (NOT the re-exported function of the same name) for monkeypatching.
import fluidvoice_finetune.idle  # noqa: F401
from fluidvoice_finetune.idle import scheduler
from fluidvoice_finetune.idle.gate import GateResult, gate

gate_mod = sys.modules["fluidvoice_finetune.idle.gate"]


# --------------------------------------------------------------------------- #
# gate.py — power/idle parsing
# --------------------------------------------------------------------------- #
def test_is_on_ac_power_parses_ac(monkeypatch):
    monkeypatch.setattr(gate_mod, "_run", lambda cmd: "Now drawing from 'AC Power'\n")
    assert gate_mod.is_on_ac_power() is True


def test_is_on_ac_power_parses_battery(monkeypatch):
    monkeypatch.setattr(gate_mod, "_run", lambda cmd: "Now drawing from 'Battery Power'\n")
    assert gate_mod.is_on_ac_power() is False


def test_is_on_ac_power_unknown_is_permissive(monkeypatch):
    monkeypatch.setattr(gate_mod, "_run", lambda cmd: "garbage\n")
    assert gate_mod.is_on_ac_power() is True  # permissive default


def test_idle_seconds_parses_nanoseconds(monkeypatch):
    # 60 seconds = 60_000_000_000 ns
    monkeypatch.setattr(gate_mod, "_run",
                        lambda cmd: '"HIDIdleTime" = 60000000000\n')
    assert gate_mod.idle_seconds() == pytest.approx(60.0)


def test_idle_seconds_missing_returns_zero(monkeypatch):
    monkeypatch.setattr(gate_mod, "_run", lambda cmd: "")
    assert gate_mod.idle_seconds() == 0.0


def test_gate_passes_when_all_conditions_met(monkeypatch):
    monkeypatch.setattr(gate_mod, "is_on_ac_power", lambda: True)
    monkeypatch.setattr(gate_mod, "is_low_power_mode", lambda: False)
    monkeypatch.setattr(gate_mod, "idle_seconds", lambda: 900.0)
    monkeypatch.setattr(gate_mod, "thermal_state", lambda: "nominal")
    g = gate(min_idle_seconds=600.0)
    assert g.ok is True
    assert g.reasons == []


def test_gate_fails_on_battery(monkeypatch):
    monkeypatch.setattr(gate_mod, "is_on_ac_power", lambda: False)
    monkeypatch.setattr(gate_mod, "is_low_power_mode", lambda: False)
    monkeypatch.setattr(gate_mod, "idle_seconds", lambda: 900.0)
    monkeypatch.setattr(gate_mod, "thermal_state", lambda: "nominal")
    g = gate()
    assert g.ok is False
    assert "not on AC power" in g.reasons


def test_gate_fails_when_not_idle_enough(monkeypatch):
    monkeypatch.setattr(gate_mod, "is_on_ac_power", lambda: True)
    monkeypatch.setattr(gate_mod, "is_low_power_mode", lambda: False)
    monkeypatch.setattr(gate_mod, "idle_seconds", lambda: 30.0)  # too recent
    monkeypatch.setattr(gate_mod, "thermal_state", lambda: "nominal")
    g = gate(min_idle_seconds=600.0)
    assert g.ok is False
    assert any("idle" in r for r in g.reasons)


def test_gate_fails_on_low_power_mode(monkeypatch):
    monkeypatch.setattr(gate_mod, "is_on_ac_power", lambda: True)
    monkeypatch.setattr(gate_mod, "is_low_power_mode", lambda: True)
    monkeypatch.setattr(gate_mod, "idle_seconds", lambda: 900.0)
    monkeypatch.setattr(gate_mod, "thermal_state", lambda: "nominal")
    g = gate()
    assert g.ok is False
    assert "Low Power Mode" in g.reasons[0]


def test_gate_result_is_truthy_when_ok():
    assert bool(GateResult(ok=True, on_ac=True, low_power=False, idle_s=900.0,
                           thermal="nominal", reasons=[])) is True
    assert bool(GateResult(ok=False, on_ac=True, low_power=False, idle_s=900.0,
                           thermal="nominal", reasons=["x"])) is False


# --------------------------------------------------------------------------- #
# scheduler.py — plist generation
# --------------------------------------------------------------------------- #
def test_build_plist_has_required_keys(tmp_path: Path):
    cfg = scheduler.ScheduleConfig(hour=1, minute=30, venv_dir=None,
                                   corpus_dir=tmp_path / "corpus",
                                   checkpoint_dir=tmp_path / "ckpt")
    plist = scheduler.build_plist(cfg)
    assert plist["Label"] == scheduler.LABEL
    assert plist["StartCalendarInterval"] == {"Hour": 1, "Minute": 30}
    assert plist["ProgramArguments"][0] == "bash"
    # The wrapped command must invoke the idle-train subcommand with the corpus.
    joined = " ".join(plist["ProgramArguments"])
    assert "idle-train" in joined
    assert "caffeinate -is -w $!" in joined
    assert str(tmp_path / "corpus") in joined


def test_build_plist_uses_venv_entrypoint_when_provided(tmp_path: Path):
    fake_venv = tmp_path / "venv"
    (fake_venv / "bin").mkdir(parents=True)
    entrypoint = fake_venv / "bin" / "fluidvoice-finetune"
    entrypoint.write_text("#!/bin/sh\n")
    cfg = scheduler.ScheduleConfig(venv_dir=fake_venv, corpus_dir=tmp_path,
                                   checkpoint_dir=tmp_path)
    plist = scheduler.build_plist(cfg)
    assert str(entrypoint) in " ".join(plist["ProgramArguments"])


def test_pmset_schedule_format():
    assert scheduler._pmset_schedule(1, 0) == "MTWRFSU 01:00:00"
    assert scheduler._pmset_schedule(14, 30) == "MTWRFSU 14:30:00"


def test_install_dry_run_does_not_write(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(scheduler, "LAUNCH_AGENT_PATH", tmp_path / "agent.plist")
    cfg = scheduler.ScheduleConfig(corpus_dir=tmp_path, checkpoint_dir=tmp_path)
    actions = scheduler.install(cfg, dry_run=True)
    assert actions["dry_run"] is True
    assert not (tmp_path / "agent.plist").exists()
    # plist content is included for inspection.
    assert "plist" in actions


def test_install_writes_valid_plist(tmp_path: Path, monkeypatch):
    agent_path = tmp_path / "agent.plist"
    monkeypatch.setattr(scheduler, "LAUNCH_AGENT_PATH", agent_path)
    # Stub out the subprocess calls (don't actually touch launchd/pmset).
    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **k: None)
    cfg = scheduler.ScheduleConfig(corpus_dir=tmp_path, checkpoint_dir=tmp_path)
    scheduler.install(cfg)
    assert agent_path.exists()
    # Must be a valid plist.
    plist = plistlib.loads(agent_path.read_bytes())
    assert plist["Label"] == scheduler.LABEL


def test_uninstall_removes_plist(tmp_path: Path, monkeypatch):
    agent_path = tmp_path / "agent.plist"
    agent_path.write_bytes(plistlib.dumps({"Label": scheduler.LABEL}))
    monkeypatch.setattr(scheduler, "LAUNCH_AGENT_PATH", agent_path)
    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **k: None)
    scheduler.uninstall()
    assert not agent_path.exists()


def test_status_reports_installed_state(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(scheduler, "LAUNCH_AGENT_PATH", tmp_path / "agent.plist")
    s = scheduler.status()
    assert s["installed"] is False
    # Now write a plist and re-check.
    agent_path = tmp_path / "agent.plist"
    agent_path.write_bytes(plistlib.dumps({
        "Label": scheduler.LABEL,
        "StartCalendarInterval": {"Hour": 2, "Minute": 15},
    }))
    s = scheduler.status()
    assert s["installed"] is True
    assert s["schedule_hour"] == 2


# --------------------------------------------------------------------------- #
# runner.py — watermark increment + dry-run prepare
# --------------------------------------------------------------------------- #
@pytest.fixture
def aqua_corpus(tmp_path: Path) -> Path:
    """Tiny corpus with 2 genuine-correction entries + audio."""
    src = tmp_path / "corpus"
    (src / "audio").mkdir(parents=True)
    entries = [
        {"audio": "audio/a1.wav", "raw": "hi", "corrected": "Hi.",
         "duration": 1.0, "timestamp": "2026-07-07T00:00:00Z", "has_correction": True},
        {"audio": "audio/a2.wav", "raw": "bye", "corrected": "Goodbye!",
         "duration": 1.0, "timestamp": "2026-07-08T00:00:00Z", "has_correction": True},
    ]
    with (src / "manifest.jsonl").open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    for e in entries:
        (src / e["audio"]).write_bytes(b"x")
    return src


def test_idle_train_gate_fails_returns_early(aqua_corpus: Path, tmp_path: Path, monkeypatch):
    """When the gate fails, no training happens and new_records=0."""
    from fluidvoice_finetune.idle.runner import run_idle_train
    monkeypatch.setattr(gate_mod, "is_on_ac_power", lambda: False)  # fails gate
    result = run_idle_train(aqua_corpus, tmp_path / "ckpt", gate_check=True)
    assert result.trained is False
    assert result.gate_ok is False
    assert result.new_records == 0


def test_idle_train_dry_run_prepare_counts_new_records(aqua_corpus: Path, tmp_path: Path, monkeypatch):
    """--dry-run-prepare runs the gate + prepare but skips training."""
    from fluidvoice_finetune.idle.runner import run_idle_train
    # Force gate to pass.
    monkeypatch.setattr(gate_mod, "is_on_ac_power", lambda: True)
    monkeypatch.setattr(gate_mod, "is_low_power_mode", lambda: False)
    monkeypatch.setattr(gate_mod, "idle_seconds", lambda: 900.0)
    monkeypatch.setattr(gate_mod, "thermal_state", lambda: "nominal")

    result = run_idle_train(aqua_corpus, tmp_path / "ckpt", gate_check=True,
                            dry_run_prepare=True)
    assert result.gate_ok is True
    assert result.new_records == 2  # both are genuine corrections
    assert result.trained is False
    assert "Dry run" in result.detail


def test_watermark_persists_and_filters(aqua_corpus: Path, tmp_path: Path, monkeypatch):
    """Second dry-run with a watermark should only count newer records."""
    from fluidvoice_finetune.idle.runner import run_idle_train
    from fluidvoice_finetune.gemma.mlx_train import write_watermark
    monkeypatch.setattr(gate_mod, "is_on_ac_power", lambda: True)
    monkeypatch.setattr(gate_mod, "is_low_power_mode", lambda: False)
    monkeypatch.setattr(gate_mod, "idle_seconds", lambda: 900.0)
    monkeypatch.setattr(gate_mod, "thermal_state", lambda: "nominal")

    ckpt = tmp_path / "ckpt"
    # Pretend we already trained everything up to 2026-07-07T23:59:59Z.
    write_watermark(ckpt, "2026-07-07T23:59:59Z")

    result = run_idle_train(aqua_corpus, ckpt, gate_check=True, dry_run_prepare=True)
    # Only the 2026-07-08 entry is newer than the watermark.
    assert result.new_records == 1


def test_idle_train_no_new_records(aqua_corpus: Path, tmp_path: Path, monkeypatch):
    """When watermark covers everything, no training happens."""
    from fluidvoice_finetune.idle.runner import run_idle_train
    from fluidvoice_finetune.gemma.mlx_train import write_watermark
    monkeypatch.setattr(gate_mod, "is_on_ac_power", lambda: True)
    monkeypatch.setattr(gate_mod, "is_low_power_mode", lambda: False)
    monkeypatch.setattr(gate_mod, "idle_seconds", lambda: 900.0)
    monkeypatch.setattr(gate_mod, "thermal_state", lambda: "nominal")

    ckpt = tmp_path / "ckpt"
    write_watermark(ckpt, "2099-01-01T00:00:00Z")  # future → nothing newer

    result = run_idle_train(aqua_corpus, ckpt, gate_check=True, dry_run_prepare=True)
    assert result.trained is False
    assert result.new_records == 0
    assert "No new training records" in result.detail
