"""Compatibility shim for the mlx-lm × transformers 5.x register() break.

transformers 5.x made ``AutoTokenizer.register(config_class, ...)`` require an
actual config *class* (it reads ``config_class.__module__``), but mlx-lm
``tokenizer_utils.py`` calls it with a *string* (``"NewlineTokenizer"``). This
crashes mlx-lm import on otherwise-compatible transformers 5.x, which is what
the ``[idle]`` extra pulls in.

The fix here is a one-line wrap of ``AutoTokenizer.register`` that tolerates a
string ``config_class`` (no-op instead of crashing). Importing this module
*before* ``mlx_lm`` makes the latter importable.

This is a known upstream mismatch; the shim is intentionally minimal and only
active at import time. Safe to remove once mlx-lm ships a fix.
"""
from __future__ import annotations


def apply_shim() -> None:
    try:
        from transformers.models.auto.tokenization_auto import AutoTokenizer
    except ImportError:
        return  # transformers not installed; nothing to shim

    if getattr(AutoTokenizer.register, "_fluidvoice_shimmed", False):
        return

    original = AutoTokenizer.register

    @staticmethod
    def _safe_register(config_class, *args, **kwargs):  # type: ignore[no-untyped-def]
        # transformers 5 reads config_class.__module__; skip if it's a string
        # (mlx-lm's case) rather than crashing. The NewlineTokenizer registration
        # is non-essential for our LoRA/fuse flow.
        if isinstance(config_class, str):
            return
        return original(config_class, *args, **kwargs)

    _safe_register._fluidvoice_shimmed = True  # type: ignore[attr-defined]
    AutoTokenizer.register = _safe_register  # type: ignore[assignment]


# Apply on import.
apply_shim()
