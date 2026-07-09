"""Tests for the model-card / manifest generator."""
from __future__ import annotations

from pathlib import Path

import pytest

from fluidvoice_finetune.config import GemmaSize, Model
from fluidvoice_finetune.publish.manifest import (
    file_size_bytes, render_card, render_gemma_card, render_parakeet_card,
    sha256_file,
)


def test_sha256_file_matches_known(tmp_path: Path):
    import hashlib
    p = tmp_path / "f.bin"
    payload = b"fluid intelligence"
    p.write_bytes(payload)
    assert sha256_file(p) == hashlib.sha256(payload).hexdigest()
    assert file_size_bytes(p) == len(payload)


def test_render_gemma_card_has_required_fields(tmp_path: Path):
    gguf = tmp_path / "fluid-1-e4b-q4_k_m.gguf"
    gguf.write_bytes(b"\x00" * 100)
    card = render_gemma_card(gguf, GemmaSize.E4B, "altic-dev/FluidIntelligence")
    assert "fluid-1-e4b-q4_k_m.gguf" in card
    assert "google/gemma-4-E4B" in card
    assert "Q4_K_M" in card
    assert "modified Gemma model derivative" in card
    assert "Gemma Terms of Use apply" in card
    # SHA + size present
    assert "SHA-256" in card and "Size:" in card


def test_render_parakeet_card_has_required_fields(tmp_path: Path):
    bundle = tmp_path / "parakeet.mlmodelc"
    bundle.mkdir()
    card = render_parakeet_card(bundle, "FluidInference/parakeet-finetuned-coreml",
                                base_checkpoint="nvidia/parakeet-tdt-0.6b-v2")
    assert "CoreML" in card
    assert "nvidia/parakeet-tdt-0.6b-v2" in card
    assert "NeMo" in card


def test_render_card_dispatches_by_model(tmp_path: Path):
    gguf = tmp_path / "fluid-1-e2b-q4_k_m.gguf"
    gguf.write_bytes(b"x")
    card_g = render_card(gguf, Model.GEMMA, "altic-dev/FluidIntelligence", size=GemmaSize.E2B)
    assert "Gemma" in card_g

    bundle = tmp_path / "p.mlmodelc"
    bundle.mkdir()
    card_p = render_card(bundle, Model.PARAKEET, "FluidInference/x",
                         base_checkpoint="nvidia/parakeet-tdt-0.6b-v2")
    assert "Parakeet" in card_p


def test_render_gemma_card_requires_size(tmp_path: Path):
    gguf = tmp_path / "x.gguf"
    gguf.write_bytes(b"x")
    with pytest.raises(ValueError, match="size is required"):
        render_card(gguf, Model.GEMMA, "altic-dev/FluidIntelligence")
