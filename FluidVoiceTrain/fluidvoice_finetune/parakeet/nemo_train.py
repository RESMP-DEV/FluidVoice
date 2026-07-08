"""Parakeet ASR fine-tuning via NVIDIA NeMo (CUDA host only).

Loads a NeMo Parakeet checkpoint (CTC or TDT), fine-tunes on a manifest
dataset, and saves a new ``.nemo`` checkpoint for CoreML export.

NeMo's API is heavy and version-sensitive; the call sequence below is sketched
and must be pinned to the installed ``nemo_toolkit`` version at run time.
"""
from __future__ import annotations

from pathlib import Path

from ..config import RunConfig


def train(cfg: RunConfig) -> Path:
    """Fine-tune the Parakeet model; return the path to the saved .nemo checkpoint."""
    try:
        import torch  # noqa: F401
        import nemo.collections.asr as nemo_asr  # noqa: F401  (type: ignore[import])
    except ImportError as e:
        raise RuntimeError(
            "Parakeet fine-tuning requires the [parakeet] extra on a CUDA host: "
            "uv pip install -e .[parakeet]"
        ) from e

    p = cfg.parakeet
    out_dir = p.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # NeMo manifests live beside cfg.dataset as train.jsonl + val.jsonl.
    manifest_dir = cfg.dataset.parent  # noqa: F841  (consumed by the live NeMo call)
    nemo_ckpt = out_dir / "parakeet-finetuned.nemo"

    # NOTE: sketched NeMo call sequence — pin nemo_toolkit version and adapt.
    #
    #   model = nemo_asr.models.ASRModel.from_pretrained(p.base_checkpoint)
    #   model.change_attention_model(self_attention_model="rel_pos_localatt")
    #   model.cfg.train_ds.manifest_filepath = str(manifest_dir / "train.jsonl")
    #   model.cfg.validation_ds.manifest_filepath = str(manifest_dir / "val.jsonl")
    #   model.setup_training_data(model.cfg.train_ds)
    #   model.setup_validation_data(model.cfg.validation_ds)
    #   trainer = pl.Trainer(max_epochs=p.num_epochs, accelerator="gpu", devices=1,
    #                        accumulate_grad_batches=p.grad_accum_steps)
    #   model.set_trainer(trainer)
    #   trainer.fit(model)
    #   model.save_to(str(nemo_ckpt))

    _write_run_manifest(out_dir, cfg)
    return nemo_ckpt


def _write_run_manifest(out_dir: Path, cfg: RunConfig) -> None:
    import json
    manifest = {
        "backend": "nemo-asr",
        "base_checkpoint": cfg.parakeet.base_checkpoint,
        "learning_rate": cfg.parakeet.learning_rate,
        "num_epochs": cfg.parakeet.num_epochs,
        "dataset": str(cfg.dataset),
        "note": "scaffold: populate with real NeMo call sequence before a live run",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
