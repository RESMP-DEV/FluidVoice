"""Convert a merged HuggingFace Gemma 4 model to the Fluid Intelligence GGUF.

Pipeline (the merge happens in :mod:`mlx_train` via ``mlx_lm.fuse``):

  1. ``convert_hf_to_gguf.py <merged> --outfile <size>-f16.gguf --outtype bf16``
     (llama.cpp's converter; the ``Gemma4Model`` class in ``conversion/gemma.py``
     handles ``gemma4`` natively — no ALTUP, uses proportional RoPE).
  2. ``llama-quantize <f16>.gguf <final>.gguf Q4_K_M``.
  3. Rename to the Fluid Intelligence filename convention.

Both ``convert_hf_to_gguf.py`` and ``llama-quantize`` are invoked as
subprocesses. ``convert_hf_to_gguf.py`` lives in a llama.cpp checkout whose
path is configured via ``LLAMA_CPP_DIR`` (default: ``./llama.cpp``);
``llama-quantize`` must be on PATH (built from that same checkout).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..config import RunConfig


def _llama_cpp_dir() -> Path:
    return Path(os.environ.get("LLAMA_CPP_DIR", "./llama.cpp"))


def _require_convert_script() -> Path:
    script = _llama_cpp_dir() / "convert_hf_to_gguf.py"
    if not script.exists():
        raise RuntimeError(
            f"convert_hf_to_gguf.py not found at {script}. Set LLAMA_CPP_DIR to "
            f"a llama.cpp checkout (git clone https://github.com/ggml-org/llama.cpp)."
        )
    return script


def _require_quantize_bin() -> str:
    import shutil
    bin_name = "llama-quantize"
    # Prefer a binary in LLAMA_CPP_DIR/build before falling back to PATH.
    local = _llama_cpp_dir() / "build" / "bin" / bin_name
    if local.exists():
        return str(local)
    found = shutil.which(bin_name)
    if not found:
        raise RuntimeError(
            f"{bin_name!r} not found. Build llama.cpp "
            f"(cd $LLAMA_CPP_DIR && cmake -B build && cmake --build build)."
        )
    return found


def export_gguf(merged_dir: Path, cfg: RunConfig) -> Path:
    """Convert ``merged_dir`` (HF folder) → ``fluid-1-<size>-q4_k_m.gguf``.

    Returns the path to the final GGUF file under ``cfg.gemma.output_dir``.
    """
    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged model dir not found: {merged_dir}")

    g = cfg.gemma
    out_dir = g.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    final_path = out_dir / g.size.artifact_filename  # fluid-1-<size>-q4_k_m.gguf
    f16_gguf = out_dir / f"{g.size.value}-f16.gguf"

    convert_script = _require_convert_script()
    quantize_bin = _require_quantize_bin()

    # 1. HF → unquantized GGUF (bf16). llama.cpp's Gemma4Model handles gemma4.
    subprocess.run(
        [sys_exec(), str(convert_script), str(merged_dir),
         "--outfile", str(f16_gguf), "--outtype", "bf16"],
        check=True,
    )

    # 2. Quantize to Q4_K_M. Quant type is case-insensitive.
    subprocess.run(
        [quantize_bin, str(f16_gguf), str(final_path), g.quant.upper()],
        check=True,
    )

    # 3. Clean up the intermediate f16 if the quantized file landed.
    if final_path.exists() and f16_gguf.exists():
        f16_gguf.unlink()

    return final_path


def sys_exec() -> str:
    """The python interpreter to use for convert_hf_to_gguf.py."""
    import sys
    return sys.executable
