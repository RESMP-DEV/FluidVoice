"""Gemma 4 LoRA/SFT on a CUDA host via transformers + PEFT + TRL.

The CUDA path (vs the MLX path). Targets multi-GPU boxes like Strix
(4× RTX 3090 Ti). Uses QLoRA (4-bit NF4 base + LoRA) when
``RunConfig.gemma.quantize_base`` is set so the model + optimizer state fit in
24GB VRAM; full fp16/bf16 base + LoRA otherwise (needs more VRAM).

Returns a *merged* HuggingFace-format model directory consumable by
:mod:`export_gguf` (which runs llama.cpp's convert_hf_to_gguf.py on it).

Gemma 4 LoRA targets (same as the MLX path): q/k/v/o projections + gate/up/down
MLP projections. PEFT does substring matching on module names, so the bare
projection names work without the ``self_attn.``/``mlp.`` prefix.
"""
from __future__ import annotations

from pathlib import Path

from ..config import RunConfig


def train(cfg: RunConfig, data_dir: Path | None = None) -> Path:
    """Run QLoRA/LoRA via TRL's SFTTrainer, merge, return the merged dir.

    Args:
        data_dir: Directory containing train.jsonl + valid.jsonl. Defaults to
            ``cfg.dataset.parent`` (the prepare step writes splits beside it).

    Returns the merged HF-format model directory (consumable by export_gguf).
    """
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise RuntimeError(
            "CUDA target requires the [gemma-cuda] extra: "
            "uv pip install -e .[gemma-cuda]"
        ) from e

    g = cfg.gemma
    if data_dir is None:
        data_dir = cfg.dataset.parent
    merged_dir = g.output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    base = g.base_checkpoint  # google/gemma-4-E2B or ...E4B
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # --- 4-bit quantization config (QLoRA) ---
    bnb_config = None
    if g.quantize_base:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    # --- Load base model + tokenizer ---
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",  # Gemma 4 works reliably with eager; SDPA can OOM on the sliding window
    )
    if g.quantize_base:
        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- LoRA config (Gemma 4 targets) ---
    # Gemma 4 is multimodal: the language model projections are plain nn.Linear
    # (PEFT-compatible), but the vision/audio towers wrap their projections in
    # Gemma4ClippableLinear, which PEFT rejects ("not supported"). The vision
    # tower paths look like `model.vision_tower.encoder.layers.N.self_attn.q_proj`,
    # so bare `self_attn.q_proj` matches BOTH the LM (good) and the vision tower
    # (rejected). Scope targets to the language model only with a regex anchored
    # on `language_model`. (Verified via probing the quantized model: all 232
    # Gemma4ClippableLinear modules live under model.vision_tower / model.audio.)
    peft_config = LoraConfig(
        r=g.lora_rank,
        lora_alpha=g.lora_alpha,
        lora_dropout=g.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # PEFT with task_type=CAUSAL_LM operates on the .language_model subtree,
        # where paths are 'layers.N.self_attn.q_proj' (no 'language_model.' prefix).
        # Match the projection names with a negative lookahead to exclude the
        # vision/audio towers (whose Gemma4ClippableLinear wrappers PEFT rejects).
        # 'q_proj.linear' etc. are the ClippableLinear inner linears — exclude by
        # requiring the name to END on the projection (.q_proj$ not .q_proj.linear$).
        target_modules=r"^(?!.*(vision_tower|audio|\.linear)).*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$",
    )

    # --- Dataset (messages format → chat template applied by SFTTrainer) ---
    data_files = {"train": str(data_dir / "train.jsonl")}
    if (data_dir / "valid.jsonl").exists():
        data_files["validation"] = str(data_dir / "valid.jsonl")
    dataset = load_dataset("json", data_files=data_files)

    # --- Training config ---
    sft_config = SFTConfig(
        output_dir=str(g.output_dir / "trl-out"),
        num_train_epochs=g.num_epochs,
        per_device_train_batch_size=g.per_device_batch_size,
        gradient_accumulation_steps=g.grad_accum_steps,
        learning_rate=g.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=20,
        eval_strategy="steps" if "validation" in dataset else "no",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        max_length=g.max_seq_length,
        packing=False,  # keep examples separate for clean loss attribution
        dataset_kwargs={"skip_prepare_dataset": False},
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    # --- Merge LoRA into base → save HF folder for GGUF export ---
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    _write_run_manifest(merged_dir, cfg, backend="transformers-qlora" if g.quantize_base else "transformers-lora")
    return merged_dir


def _write_run_manifest(merged_dir: Path, cfg: RunConfig, backend: str) -> None:
    import json
    manifest = {
        "backend": backend,
        "base_checkpoint": cfg.gemma.base_checkpoint,
        "size": cfg.gemma.size.value,
        "lora_rank": cfg.gemma.lora_rank,
        "lora_alpha": cfg.gemma.lora_alpha,
        "learning_rate": cfg.gemma.learning_rate,
        "num_epochs": cfg.gemma.num_epochs,
        "quantize_base": cfg.gemma.quantize_base,
        "dataset": str(cfg.dataset),
        "note": "CUDA training via TRL SFTTrainer + PEFT LoRA",
    }
    (merged_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
