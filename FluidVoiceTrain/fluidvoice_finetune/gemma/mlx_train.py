"""Gemma LoRA/SFT on Apple Silicon via mlx-lm.

Real (not sketched) implementation against the current mlx-lm API:

1. Generate a ``lora_config.yaml`` from :class:`RunConfig`.
2. Run ``mlx_lm.lora --config <yaml>`` via subprocess → adapter in ``adapters/``.
3. Run ``mlx_lm.fuse --model <base> --adapter-path adapters --save-path merged``
   → merged HuggingFace-format model directory.

``mlx_lm.fuse --export-gguf`` is **hard-blocked for non-llama architectures**
(it raises ``ValueError`` for Gemma), so this module stops at the merged HF
folder. :mod:`export_gguf` then runs llama.cpp's ``convert_hf_to_gguf.py`` +
``llama-quantize`` to produce the final ``fluid-1-<size>-q4_k_m.gguf``.

Incremental training: a ``trained_watermark.txt`` (max timestamp consumed) is
read by the caller and passed back via ``resume_adapter_file`` in the YAML so
MLX resumes from the last checkpoint.

Heavy imports (``mlx_lm``) are deferred to call time so the module imports
cleanly without the ``[gemma-mlx]`` extra.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from ..config import RunConfig


def _write_lora_config(cfg: RunConfig, data_dir: Path, adapter_path: Path,
                       resume_adapter: Path | None) -> Path:
    """Generate the mlx_lm.lora YAML config. Returns its path.

    Gemma 4 specifics (vs 3n):
    - Base checkpoint is ``google/gemma-4-E2B`` / ``E4B`` (or the MLX 4-bit
      quant at ``mlx-community/gemma-4-<size>-it-4bit`` for QLoRA).
    - LoRA target keys target attention + MLP projections: ``q_proj``, ``k_proj``,
      ``v_proj``, ``o_proj`` (note Gemma 4 has asymmetric global/sliding heads,
      so k/v projections matter, not just q/v), plus ``gate_proj``/``up_proj``/
      ``down_proj`` for the double-wide MLP. Do NOT use 3n's ``self_attn.*``
      paths or target the (nonexistent) ``altup_*`` / ``laurel.*`` modules.
    - ``num_layers``: Gemma 4 E4B has 35 blocks; default to applying LoRA to
      all of them (mlx-lm converts the last ``num_layers`` blocks).
    """
    g = cfg.gemma
    size_token = "e2b" if g.size.value == "e2b" else "e4b"
    config: dict[str, Any] = {
        "model": g.base_checkpoint if not g.quantize_base
        else f"mlx-community/gemma-4-{size_token}-it-4bit",
        "train": True,
        "fine_tune_type": "lora",
        "data": str(data_dir),
        "seed": cfg.seed,
        "num_layers": -1,  # all transformer blocks; Gemma 4 E4B has 35
        "batch_size": g.per_device_batch_size * g.grad_accum_steps,
        "iters": max(1, int(g.num_epochs * 50)),  # rough; caller can override
        "val_batches": -1,
        "learning_rate": g.learning_rate,
        "steps_per_report": 10,
        "steps_per_eval": 100,
        "grad_accumulation_steps": g.grad_accum_steps,
        "adapter_path": str(adapter_path),
        "save_every": 50,
        "max_seq_length": g.max_seq_length,
        "mask_prompt": True,  # loss on completion (corrected text) only
        "lora_parameters": {
            # Gemma 4 module paths, RELATIVE to each transformer block (mlx-lm
            # matches these via `k in keys` over `block.named_modules()`). Verified
            # against mlx-community/gemma-4-e2b-it-4bit: each block has
            # self_attn.{q_proj,o_proj} on all layers; self_attn.{k_proj,v_proj}
            # only on the first (35 - num_kv_shared_layers) blocks (shared KV);
            # mlp.{gate_proj,up_proj,down_proj} on all. Targeting q/o + the three
            # MLP projections gives a balanced adapter; k/v are included for the
            # blocks that have them.
            "keys": [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            ],
            "rank": g.lora_rank,
            "scale": float(g.lora_alpha),
            "dropout": g.lora_dropout,
        },
    }
    if resume_adapter and resume_adapter.exists():
        config["resume_adapter_file"] = str(resume_adapter)

    config_path = g.output_dir / "lora_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def _run(cmd: list[str]) -> None:
    """Run a subprocess, streaming output, raising on failure."""
    print(f"[mlx_train] $ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def train(cfg: RunConfig, data_dir: Path | None = None,
          resume_adapter: Path | None = None) -> Path:
    """Run LoRA training via mlx-lm, fuse, return the merged HF model directory.

    Args:
        data_dir: Directory containing train.jsonl (+ valid.jsonl). Defaults to
            ``cfg.dataset.parent`` (the prepare step writes splits beside it).
        resume_adapter: Path to a previously-saved adapter to resume from
            (incremental training). None = train from scratch.

    Returns the merged HF-format model directory (consumable by export_gguf).
    """
    try:
        # Apply the transformers-5 register() shim BEFORE importing mlx_lm, which
        # otherwise crashes on AutoTokenizer.register("NewlineTokenizer", ...).
        from .. import _mlx_compat  # noqa: F401  (side-effect: patches register)
        _mlx_compat.apply_shim()
        import mlx_lm  # noqa: F401  (verify availability; the real call is via CLI)
    except ImportError as e:
        raise RuntimeError(
            "MLX target requires the [gemma-mlx] extra: "
            "uv pip install -e '.[gemma-mlx]'  (or the [idle] extra)"
        ) from e

    g = cfg.gemma
    if data_dir is None:
        data_dir = cfg.dataset.parent
    adapter_path = g.output_dir / "adapters"
    merged_dir = g.output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate + run the LoRA training config.
    #    Invoke via the _mlx_cli wrapper so the transformers-5 register() shim
    #    is applied before mlx_lm imports (the bare `mlx_lm.lora` console script
    #    would otherwise crash on AutoTokenizer.register).
    config_path = _write_lora_config(cfg, data_dir, adapter_path, resume_adapter)
    _run([sys.executable, "-m", "fluidvoice_finetune._mlx_cli", "lora",
          "--config", str(config_path)])

    # 2. Fuse adapter into base → merged HF folder.
    #    NOTE: do NOT pass --export-gguf here; mlx_lm.fuse hard-blocks it for
    #    Gemma (ValueError for non-llama/mixtral/mistral). export_gguf.py runs
    #    llama.cpp's convert_hf_to_gguf.py on the merged folder instead.
    base_model = _read_base_model_from_config(config_path)
    # Pre-fetch the full model snapshot (with network) before fuse: mlx_lm.fuse
    # calls snapshot_download(local_files_only=True), which crashes if a prior
    # training run only pulled the model files (not README/.gitattributes).
    _ensure_snapshot(base_model)
    _run([sys.executable, "-m", "fluidvoice_finetune._mlx_cli", "fuse",
          "--model", base_model,
          "--adapter-path", str(adapter_path),
          "--save-path", str(merged_dir),
          "--dequantize",  # produce clean fp/bf16 HF folder for GGUF conversion
    ])

    return merged_dir


def _read_base_model_from_config(config_path: Path) -> str:
    """Read the `model:` field back out of the generated YAML."""
    data = yaml.safe_load(config_path.read_text())
    return str(data["model"])


def _ensure_snapshot(repo_id: str) -> None:
    """Pre-fetch a HF model snapshot so mlx_lm.fuse's local_files_only call succeeds.

    mlx_lm.fuse resolves the model path via snapshot_download(local_files_only=True),
    which raises IncompleteSnapshotError if a prior training run only pulled the
    weight files (e.g. README.md / .gitattributes missing). Fetch the full
    snapshot here first, with network access.
    """
    if not repo_id or "/" not in repo_id or Path(repo_id).exists():
        return  # local path, not a HF repo id
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id)
    except Exception as e:
        # Non-fatal: if the snapshot is already complete, this is a no-op; if
        # it genuinely can't be fetched, mlx_lm.fuse will raise the real error.
        print(f"[mlx_train] snapshot pre-fetch warning for {repo_id}: {e}", file=sys.stderr)


def write_watermark(checkpoint_dir: Path, watermark: str) -> None:
    """Persist the max timestamp consumed, for incremental training."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "trained_watermark.txt").write_text(watermark)


def read_watermark(checkpoint_dir: Path) -> str | None:
    f = checkpoint_dir / "trained_watermark.txt"
    return f.read_text().strip() if f.exists() else None
