"""Tests for the CLI dispatch and config validation."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from fluidvoice_finetune.cli import cli
from fluidvoice_finetune.config import (
    GemmaSize, Model, RunConfig, Target,
)


def test_help_works():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.output
    assert "prepare" in result.output
    assert "publish" in result.output


def test_train_help_lists_options():
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--target" in result.output
    assert "--size" in result.output
    assert "--upload" in result.output


def test_prepare_command_runs_on_messages(messages_jsonl: Path, tmp_path: Path):
    """`prepare` should not require any ML deps — pure data shaping."""
    runner = CliRunner()
    out = tmp_path / "prepared"
    result = runner.invoke(cli, [
        "prepare", "--dataset", str(messages_jsonl),
        "--out", str(out), "--model", "gemma", "--val-split", "0.5",
    ])
    assert result.exit_code == 0, result.output
    assert (out / "train.jsonl").exists()
    assert (out / "val.jsonl").exists()


def test_run_config_parakeet_rejects_mlx():
    """Parakeet + MLX target must be rejected at validation time."""
    cfg = RunConfig(
        model=Model.PARAKEET,
        dataset=Path("/dev/null"),
        extra={"target": Target.MLX.value},
    )
    with pytest.raises(ValueError, match="mlx is not supported"):
        cfg.validate()


def test_run_config_val_split_bounds():
    cfg = RunConfig(model=Model.GEMMA, dataset=Path("/dev/null"), val_split=-0.1)
    with pytest.raises(ValueError, match="val_split"):
        cfg.validate()
    cfg2 = RunConfig(model=Model.GEMMA, dataset=Path("/dev/null"), val_split=1.0)
    with pytest.raises(ValueError, match="val_split"):
        cfg2.validate()


def test_gemma_size_artifact_filename_matches_swift_convention():
    """The GGUF filename MUST match FluidIntelligenceModelVariant.artifactFilename
    in the Swift app: fluid-1-<size>-q4_k_m.gguf."""
    assert GemmaSize.E2B.artifact_filename == "fluid-1-e2b-q4_k_m.gguf"
    assert GemmaSize.E4B.artifact_filename == "fluid-1-e4b-q4_k_m.gguf"


def test_gemma_size_base_checkpoint():
    assert GemmaSize.E2B.base_checkpoint == "google/gemma-3n-e2b"
    assert GemmaSize.E4B.base_checkpoint == "google/gemma-3n-e4b"
