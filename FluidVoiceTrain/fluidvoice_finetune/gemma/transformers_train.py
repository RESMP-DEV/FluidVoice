"""Gemma LoRA/SFT on a CUDA host via transformers + PEFT + TRL.

Remote alternative to the MLX path. Uses QLoRA (4-bit base + LoRA) when
``RunConfig.gemma.quantize_base`` is set and bitsandbytes is available; falls
back to fp16 base + LoRA otherwise (relevant for older GPUs like the Tesla P100,
which lacks bf16/fp8 and has limited bnb support).

Returns a *merged* HuggingFace-format model directory consumable by
:mod:`export_gguf`.
"""
from __future__ import annotations

from pathlib import Path

from ..config import RunConfig


def train(cfg: RunConfig) -> Path:
    """Run QLoRA/LoRA via TRL's SFTTrainer, merge, return the merged dir."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
        import trl  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "CUDA target requires the [gemma-cuda] extra: "
            "uv pip install -e .[gemma-cuda]"
        ) from e

    g = cfg.gemma
    merged_dir = g.output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: sketched call sequence — pin versions and adapt to the installed
    # transformers/peft/trl when running for real. The shape is:
    #
    #   from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    #   from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    #   from trl import SFTTrainer, SFTConfig
    #   from datasets import load_dataset
    #
    #   bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    #                            bnb_4bit_compute_dtype=torch.float16) if g.quantize_base else None
    #   model = AutoModelForCausalLM.from_pretrained(g.base_checkpoint,
    #       quantization_config=bnb, torch_dtype=torch.float16)
    #   if bnb: model = prepare_model_for_kbit_training(model)
    #   tok = AutoTokenizer.from_pretrained(g.base_checkpoint)
    #   peft_cfg = LoraConfig(r=g.lora_rank, lora_alpha=g.lora_alpha,
    #                        lora_dropout=g.lora_dropout, bias="none", task_type="CAUSAL_LM")
    #   ds = load_dataset("json", data_files={"train": str(data_dir/"train.jsonl"),
    #                                          "validation": str(data_dir/"val.jsonl")})
    #   trainer = SFTTrainer(model=model, args=SFTConfig(...), train_dataset=ds["train"],
    #                        eval_dataset=ds["validation"], peft_config=peft_cfg,
    #                        max_seq_length=g.max_seq_length, tokenizer=tok)
    #   trainer.train()
    #   merged = trainer.model.merge_and_unload()
    #   merged.save_pretrained(merged_dir); tok.save_pretrained(merged_dir)

    _write_run_manifest(merged_dir, cfg, backend="transformers-qlora")
    return merged_dir


def _write_run_manifest(merged_dir: Path, cfg: RunConfig, backend: str) -> None:
    import json
    manifest = {
        "backend": backend,
        "base_checkpoint": cfg.gemma.base_checkpoint,
        "size": cfg.gemma.size.value,
        "lora_rank": cfg.gemma.lora_rank,
        "learning_rate": cfg.gemma.learning_rate,
        "num_epochs": cfg.gemma.num_epochs,
        "quantize_base": cfg.gemma.quantize_base,
        "dataset": str(cfg.dataset),
        "note": "scaffold: populate with real TRL/PEFT call sequence before a live run",
    }
    (merged_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
