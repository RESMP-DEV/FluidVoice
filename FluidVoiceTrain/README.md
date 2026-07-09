# FluidVoiceTrain

**FluidVoice's on-device training component.** This is part of the FluidVoice
repository (not a standalone package) and is bundled into the FluidVoice.app at
release build time. End users never install Python or touch this directory —
FluidVoice auto-discovers the bundled copy at
`FluidVoice.app/Contents/Resources/FluidVoiceTrain/` and runs it transparently.

It produces the on-device artifacts FluidVoice loads at runtime:

- **Gemma** — the "Fluid Intelligence" enhancement LLM (a modified **Gemma 4**
  derivative — confirmed from the GGUF metadata, `general.architecture = "gemma4"`
  — shipped as `fluid-1-*-q4_k_m.gguf` at
  [`altic-dev/FluidIntelligence`](https://huggingface.co/altic-dev/FluidIntelligence)).
  Supports two sizes behind the FluidVoice **E2B / E4B** toggle:
  `google/gemma-4-E2B` and `google/gemma-4-E4B`.
- **Parakeet** — on-device ASR (NVIDIA NeMo CTC/TDT checkpoints re-exported to
  CoreML `.mlmodelc` bundles for the `FluidInference/*-coreml` HF org).

The CLI dispatches by `--model {gemma,parakeet}`; Gemma further branches on
`--target {mlx,cuda}` (local Apple Silicon vs. remote GPU). Parakeet is
CUDA-only.

## For FluidVoice maintainers / contributors

### How it's bundled (release)

The release build (`scripts/build_with_training.sh` at the repo root, layered on
top of the private-FI build) does three things:
1. Runs `FluidVoiceTrain/scripts/build_bundle.sh` →
   `FluidVoiceTrain/dist/FluidVoiceTrain/` (a relocatable dir containing
   `bin/fluidvoice-finetune`, `bin/llama-quantize`, `bin/convert_hf_to_gguf.py`,
   and all Python deps — mlx-lm, transformers, torch).
2. Runs the private-FI build to produce the base `FluidVoice.app`.
3. Copies `dist/FluidVoiceTrain/` → `FluidVoice.app/Contents/Resources/FluidVoiceTrain/`
   and re-signs.

The Swift app discovers it via `FluidIntelligenceTrainer.bundledTrainerDirectory()`:
`Contents/Resources/FluidVoiceTrain/` first (release), then
`<repo>/FluidVoiceTrain/dist/FluidVoiceTrain/` (dev builds, resolved via the
source tree). No user-facing path setting exists.

### Local development

```bash
cd FluidVoiceTrain
uv venv .venv --python 3.11
uv pip install -e ".[dev]"          # core + tests
uv pip install -e ".[idle,gguf]"    # + mlx-lm + torch for the real chain
pytest                               # 56 tests
```

Build a local bundle (so an unsigned dev FluidVoice build can find it):
```bash
./scripts/build_bundle.sh            # → dist/FluidVoiceTrain/
```

The built bundle and venv are gitignored — only the source is tracked.

## Status

**Scaffold.** The data formatters, CLI dispatch, HF publish layer, and model-card
generator are implemented and tested. The training call sequences (MLX / TRL /
NeMo) and the GGUF / CoreML export steps are sketched as documented seams —
they are pinned to specific library versions and filled in when a dataset is
available and a live run is scheduled. The `lora → fuse → convert → quantize`
chain itself has been verified end-to-end on Apple Silicon (see "Verified
end-to-end chain" below); the sketched seams are the higher-level runners that
compose it.
deps gated behind optional extras).

## Install (contributors)

See "Local development" above for the standard dev setup. Optional extras keep
the Mac install light — the CUDA/NeMo deps aren't pulled on Apple Silicon:

```bash
uv pip install -e ".[dev]"        # core + tests
uv pip install -e ".[gemma-mlx]"  # + Gemma on Apple Silicon
uv pip install -e ".[gemma-cuda]" # + Gemma on a CUDA host
uv pip install -e ".[parakeet]"   # + Parakeet (NeMo + CoreML), CUDA host
```

## Usage

### Prepare a dataset (no training, no ML deps)

```bash
fluidvoice-finetune prepare \
    --dataset ./data/pairs.csv \
    --out ./prepared/gemma \
    --model gemma \
    --val-split 0.05
# → ./prepared/gemma/{train,val}.jsonl  (ShareGPT messages)
```

```bash
fluidvoice-finetune prepare \
    --dataset ./data/asr_manifest.jsonl \
    --out ./prepared/parakeet \
    --model parakeet \
    --val-split 0.05
# → ./prepared/parakeet/{train,val}.jsonl  (NeMo manifest)
```

### Train + export + publish (one command)

```bash
# Gemma E4B via MLX on this Mac → GGUF → upload to FluidIntelligence
fluidvoice-finetune train \
    --model gemma --size e4b --target mlx \
    --dataset ./prepared/gemma/train.jsonl \
    --upload altic-dev/FluidIntelligence \
    --path-in-repo models/

# Parakeet via NeMo on a remote CUDA box → CoreML → upload
fluidvoice-finetune train \
    --model parakeet \
    --dataset ./prepared/parakeet/train.jsonl \
    --upload FluidInference/parakeet-tdt-0.6b-v2-finetuned-coreml
```

### Publish an already-produced artifact

```bash
fluidvoice-finetune publish \
    --artifact ./output/gemma/fluid-1-e4b-q4_k_m.gguf \
    --repo-id altic-dev/FluidIntelligence \
    --path-in-repo models/ \
    --model gemma --size e4b
```

## Compute matrix

| Model | Target | Host | Framework | Notes |
|---|---|---|---|---|
| Gemma | `mlx` | Apple Silicon (this Mac) | `mlx-lm` LoRA | Primary local path; produces the E4B GGUF. |
| Gemma | `cuda` | NVIDIA GPU | transformers + PEFT + TRL (QLoRA) | Remote alternative. **P100 caveat:** Tesla P100 (Pascal) lacks bf16/fp8 and has limited `bitsandbytes` support — set `--learning-rate` conservatively and prefer fp16 base + LoRA over 4-bit QLoRA on that box. An Ampere+ GPU is recommended for 4-bit. |
| Parakeet | `cuda` | NVIDIA GPU (Ampere+ recommended) | NVIDIA NeMo | CUDA-only; `--target mlx` is rejected. Exports to CoreML via NeMo → ONNX → `coremltools`. |

The FluidVoice Swift app loads both artifact types with compute units
`.cpuAndNeuralEngine` on Apple Silicon.

## GGUF export requirements

`gemma/export_gguf.py` shells out to `llama.cpp`:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_NATIVE=OFF -DLLAMA_CURL=OFF
cmake --build build --target llama-quantize
export LLAMA_CPP_DIR="$(pwd)"
```

The HF→GGUF converter (`convert_hf_to_gguf.py`) also needs `torch`:

```bash
uv pip install -e ".[gguf]"   # pulls torch (CPU-only is fine on Mac)
```

The output filename is **pinned by the Swift toggle**:
`fluid-1-<size>-q4_k_m.gguf` (see
`FluidIntelligenceModelVariant.artifactFilename` in
`FluidVoice/Sources/Fluid/Services/PrivateAIProvider.swift`). The two must stay
in sync.

### Verified end-to-end chain

The full `lora → fuse → convert → quantize` chain has been smoke-tested on
Apple Silicon (mlx-lm 0.31.3, transformers 5.x, llama.cpp, torch):
- `mlx_lm.lora` (2 iters on a 1B test model) → adapter saved ✓
- `mlx_lm.fuse --dequantize` → merged HF folder (config.json + safetensors + tokenizer) ✓
- `convert_hf_to_gguf.py --outtype bf16` (147 tensors, 2.3 GB f16) ✓
- `llama-quantize Q4_K_M` → 770 MB final GGUF ✓

> **Note on mlx-lm × transformers 5:** mlx-lm 0.31.x calls
> `AutoTokenizer.register("NewlineTokenizer", ...)` with a string, which
> transformers 5 rejects. `fluidvoice_finetune/_mlx_compat.py` patches this at
> import time, and `mlx_lm.lora`/`fuse` are invoked via the
> `python -m fluidvoice_finetune._mlx_cli` wrapper so the patch applies in
> subprocesses too. Remove the shim once mlx-lm ships a fix.
>
> **Note on snapshot completeness:** `mlx_lm.fuse` calls
> `snapshot_download(local_files_only=True)`, which crashes if a prior training
> run only pulled weight files. `mlx_train.train()` pre-fetches the full
> snapshot (with network) before fusing to avoid this.

## Tests

```bash
pytest
```

Tests cover the data formatters (round-trips + validation), the CLI dispatch
(help / prepare / config-compatibility matrix), and the model-card generator —
without importing any ML dependencies.

## License

Apache-2.0. Trained artifacts are derivatives: Gemma artifacts carry the Gemma
Terms of Use; Parakeet artifacts carry the NVIDIA NeMo license. Both are noted
automatically on the generated model cards.
