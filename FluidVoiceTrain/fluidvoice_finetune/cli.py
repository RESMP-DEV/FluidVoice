"""CLI entrypoint: fluidvoice-finetune.

Mirrors the tracesmith (agent-trace-share) click-based style.

Examples
--------
# Gemma E4B via MLX on this Mac, upload to FluidIntelligence as the new artifact
fluidvoice-finetune train --model gemma --size e4b --target mlx \
    --dataset ./data/messages.jsonl --upload altic-dev/FluidIntelligence \
    --path-in-repo models/

# Parakeet via NeMo on a remote CUDA box
fluidvoice-finetune train --model parakeet --dataset ./data/manifest.jsonl \
    --upload FluidInference/parakeet-tdt-0.6b-v2-finetuned-coreml
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from .config import DatasetFormat, GemmaSize, Model, RunConfig, Target


@click.group()
def cli() -> None:
    """FluidVoice unified fine-tuning pipeline (Gemma + Parakeet)."""


def _build_run_config(
    model: str,
    dataset: Path,
    target: str,
    size: str,
    dataset_format: str,
    val_split: float,
    seed: int,
    upload: str | None,
    path_in_repo: str | None,
    lora_rank: int,
    learning_rate: float,
    num_epochs: int,
    private_repo: bool,
    skip_upload: bool,
) -> RunConfig:
    cfg = RunConfig(
        model=Model(model),
        dataset=dataset,
        dataset_format=DatasetFormat(dataset_format),
        val_split=val_split,
        seed=seed,
        extra={"target": target} if model == "gemma" else {},
    )
    cfg.gemma.size = GemmaSize(size)
    cfg.gemma.target = Target(target)
    cfg.gemma.lora_rank = lora_rank
    cfg.gemma.learning_rate = learning_rate
    cfg.gemma.num_epochs = num_epochs
    cfg.parakeet.learning_rate = learning_rate
    cfg.parakeet.num_epochs = num_epochs
    cfg.publish.repo_id = upload
    cfg.publish.path_in_repo = path_in_repo
    cfg.publish.private = private_repo
    cfg.publish.skip_upload = skip_upload
    cfg.validate()
    return cfg


@cli.command("train")
@click.option("--model", type=click.Choice([m.value for m in Model]), required=True)
@click.option("--dataset", type=click.Path(exists=True, path_type=Path), required=True,
              help="Path to the prepared dataset (JSONL) or a HF dataset id.")
@click.option("--dataset-format", type=click.Choice([f.value for f in DatasetFormat]),
              default=DatasetFormat.AUTO.value)
@click.option("--target", type=click.Choice([t.value for t in Target]), default=Target.MLX.value,
              help="Compute target. Gemma supports mlx|cuda; parakeet is cuda-only.")
@click.option("--size", type=click.Choice([s.value for s in GemmaSize]), default=GemmaSize.E4B.value,
              help="Gemma base checkpoint size (ignored for parakeet).")
@click.option("--val-split", default=0.05, type=float, show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--lora-rank", default=16, type=int, show_default=True, help="LoRA rank (Gemma only).")
@click.option("--learning-rate", default=None, type=float,
              help="Override learning rate. Default: 2e-4 (gemma) / 1e-4 (parakeet).")
@click.option("--num-epochs", default=None, type=int, help="Override epoch count.")
@click.option("--upload", default=None, help="HF repo id to publish to, e.g. altic-dev/FluidIntelligence")
@click.option("--path-in-repo", default=None, help="Subpath within the HF repo, e.g. models/")
@click.option("--private-repo", is_flag=True, help="Create the HF repo as private.")
@click.option("--skip-upload", is_flag=True, help="Train + export locally; do not push to HF.")
def train_cmd(model: str, dataset: Path, dataset_format: str, target: str, size: str,
              val_split: float, seed: int, lora_rank: int, learning_rate: float | None,
              num_epochs: int | None, upload: str | None, path_in_repo: str | None,
              private_repo: bool, skip_upload: bool) -> None:
    """Fine-tune the selected model and (optionally) publish to HF Hub."""
    cfg = _build_run_config(
        model=model, dataset=dataset, target=target, size=size,
        dataset_format=dataset_format, val_split=val_split, seed=seed,
        upload=upload, path_in_repo=path_in_repo, lora_rank=lora_rank,
        learning_rate=learning_rate if learning_rate is not None else (2e-4 if model == "gemma" else 1e-4),
        num_epochs=num_epochs if num_epochs is not None else (3 if model == "gemma" else 5),
        private_repo=private_repo, skip_upload=skip_upload,
    )
    click.echo(f"Run: model={cfg.model.value} dataset={cfg.dataset}")
    if cfg.model is Model.GEMMA:
        from .gemma import run_gemma
        run_gemma(cfg)
    else:
        from .parakeet import run_parakeet
        run_parakeet(cfg)


@cli.command("prepare")
@click.option("--dataset", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--model", type=click.Choice([m.value for m in Model]), required=True,
              help="Target model whose format to emit.")
@click.option("--dataset-format", type=click.Choice([f.value for f in DatasetFormat]),
              default=DatasetFormat.AUTO.value)
@click.option("--val-split", default=0.05, type=float, show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
def prepare_cmd(dataset: Path, out: Path, model: str, dataset_format: str,
                val_split: float, seed: int) -> None:
    """Convert a raw dataset into the backend's expected format (no training)."""
    from .data.prepare import prepare_dataset
    target_fmt = DatasetFormat.MESSAGES if model == "gemma" else DatasetFormat.NEMO_MANIFEST
    counts = prepare_dataset(dataset, out, target_fmt=target_fmt, source_fmt=DatasetFormat(dataset_format),
                             val_split=val_split, seed=seed)
    click.echo(f"Wrote {counts} records to {out}")


@cli.command("prepare-aqua")
@click.option("--src", type=click.Path(exists=True, path_type=Path), required=True,
              help="Corpus dir containing manifest.jsonl + audio/.")
@click.option("--out", type=click.Path(path_type=Path), required=True,
              help="Output dir for train.jsonl / val.jsonl.")
@click.option("--target", type=click.Choice([DatasetFormat.MESSAGES.value,
                                             DatasetFormat.NEMO_MANIFEST.value]),
              required=True, help="messages (Gemma) or nemo-manifest (Parakeet).")
@click.option("--val-split", default=0.1, type=float, show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--watermark", default=None,
              help="ISO timestamp; only entries newer than this are emitted (incremental).")
@click.option("--keep-all/--filter-corrections", "require_genuine_correction", default=None,
              help="Gemma defaults to filtering non-corrections; Parakeet keeps all. Override here.")
@click.option("--require-audio/--skip-missing-audio", default=None,
              help="Parakeet requires audio; Gemma ignores it. Override here.")
def prepare_aqua_cmd(src: Path, out: Path, target: str, val_split: float, seed: int,
                     watermark: str | None, require_genuine_correction: bool | None,
                     require_audio: bool | None) -> None:
    """Prepare an Aqua Voice / FluidVoice corpus for training."""
    from .data.prepare import prepare_aqua_corpus
    summary = prepare_aqua_corpus(
        src, out, target_fmt=DatasetFormat(target), val_split=val_split, seed=seed,
        watermark=watermark,
        require_genuine_correction=require_genuine_correction,
        require_audio=require_audio,
    )
    click.echo(json.dumps(summary, indent=2))
    click.echo(f"→ {out}")


@cli.command("publish")
@click.option("--artifact", type=click.Path(exists=True, path_type=Path), required=True,
              help="Local file/dir to upload (GGUF file, .mlmodelc bundle, or .nemo checkpoint).")
@click.option("--repo-id", required=True, help="HF repo id, e.g. altic-dev/FluidIntelligence")
@click.option("--path-in-repo", default=None, help="Subpath within the repo.")
@click.option("--model", type=click.Choice([m.value for m in Model]), required=True,
              help="Which kind of artifact (drives the generated model card).")
@click.option("--size", type=click.Choice([s.value for s in GemmaSize]), default=GemmaSize.E4B.value)
@click.option("--private-repo", is_flag=True)
def publish_cmd(artifact: Path, repo_id: str, path_in_repo: str | None, model: str,
                size: str, private_repo: bool) -> None:
    """Publish an already-produced artifact to HF Hub with a model card."""
    from .publish.huggingface import publish_artifact
    publish_artifact(artifact, repo_id=repo_id, path_in_repo=path_in_repo,
                     model=Model(model), size=GemmaSize(size), private=private_repo)


# --------------------------------------------------------------------------- #
# On-device idle training (macOS Apple Silicon)
# --------------------------------------------------------------------------- #
@cli.command("idle-train")
@click.option("--corpus", type=click.Path(exists=True, path_type=Path), required=True,
              help="Corpus dir with manifest.jsonl + audio/ (Aqua/FluidVoice shape).")
@click.option("--checkpoint-dir", type=click.Path(path_type=Path), required=True,
              help="Where adapters + watermark live across runs.")
@click.option("--size", type=click.Choice([s.value for s in GemmaSize]), default=GemmaSize.E4B.value,
              show_default=True)
@click.option("--iters", type=int, default=None, help="Override MLX iteration count.")
@click.option("--learning-rate", type=float, default=None)
@click.option("--lora-rank", type=int, default=16, show_default=True)
@click.option("--publish", "publish_repo", default=None, help="HF repo to publish the GGUF to.")
@click.option("--no-gate", is_flag=True, help="Skip the power/idle/thermal gate.")
@click.option("--min-idle-seconds", default=600.0, type=float, show_default=True)
@click.option("--dry-run-prepare", is_flag=True,
              help="Prepare the corpus but skip training (test the pipeline).")
def idle_train_cmd(corpus: Path, checkpoint_dir: Path, size: str, iters: int | None,
                   learning_rate: float | None, lora_rank: int, publish_repo: str | None,
                   no_gate: bool, min_idle_seconds: float, dry_run_prepare: bool) -> None:
    """Run one gated, incremental Gemma training pass (the launchd job entrypoint)."""
    from .idle.runner import run_idle_train
    result = run_idle_train(
        corpus, checkpoint_dir, size=GemmaSize(size), iters=iters,
        learning_rate=learning_rate, lora_rank=lora_rank, publish_repo=publish_repo,
        gate_check=not no_gate, min_idle_seconds=min_idle_seconds,
        dry_run_prepare=dry_run_prepare,
    )
    if not result.gate_ok:
        click.echo(f"Gate failed: {'; '.join(result.gate_reasons)}", err=True)
        raise SystemExit(1)
    click.echo(result.detail)
    if result.gguf_path:
        click.echo(f"GGUF: {result.gguf_path}")


@cli.command("idle-install")
@click.option("--time", "time_str", default="01:00", show_default=True,
              help="HH:MM for the nightly run.")
@click.option("--venv", type=click.Path(exists=True, path_type=Path), default=None,
              help="venv containing the fluidvoice-finetune entrypoint.")
@click.option("--corpus", type=click.Path(path_type=Path), default=None)
@click.option("--checkpoint-dir", type=click.Path(path_type=Path), default=None)
@click.option("--dry-run", is_flag=True, help="Print what would be done; write nothing.")
def idle_install_cmd(time_str: str, venv: Path | None, corpus: Path | None,
                     checkpoint_dir: Path | None, dry_run: bool) -> None:
    """Install the launchd LaunchAgent + pmset wake schedule for nightly training."""
    from .idle.scheduler import ScheduleConfig, install
    try:
        hour, minute = (int(x) for x in time_str.split(":"))
    except ValueError:
        raise click.BadParameter("time must be HH:MM, e.g. 01:00")
    cfg = ScheduleConfig(hour=hour, minute=minute, venv_dir=venv,
                         corpus_dir=corpus or ScheduleConfig().corpus_dir,
                         checkpoint_dir=checkpoint_dir or ScheduleConfig().checkpoint_dir)
    actions = install(cfg, dry_run=dry_run)
    click.echo(json.dumps(actions, indent=2, default=str))


@cli.command("idle-uninstall")
@click.option("--dry-run", is_flag=True)
def idle_uninstall_cmd(dry_run: bool) -> None:
    """Remove the launchd LaunchAgent and clear the pmset wake schedule."""
    from .idle.scheduler import uninstall
    actions = uninstall(dry_run=dry_run)
    click.echo(json.dumps(actions, indent=2, default=str))


@cli.command("idle-status")
def idle_status_cmd() -> None:
    """Show install state, scheduled time, and last/next run hints."""
    from .idle.scheduler import status
    click.echo(json.dumps(status(), indent=2, default=str))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
