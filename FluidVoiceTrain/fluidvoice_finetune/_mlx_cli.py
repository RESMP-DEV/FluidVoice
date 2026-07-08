"""Thin wrapper to invoke ``mlx_lm.lora`` / ``mlx_lm.fuse`` through the
transformers-5 register() shim.

Why this exists: the bare ``mlx_lm.lora`` console-script subprocess crashes on
import under transformers 5.x (see :mod:`fluidvoice_finetune._mlx_compat`).
Running the same entrypoint via this wrapper lets us apply the shim first.

Usage (what mlx_train.py shells out to):

    python -m fluidvoice_finetune._mlx_cli lora --config <yaml>
    python -m fluidvoice_finetune._mlx_cli fuse --model <base> --adapter-path ...

Any subcommand other than ``lora``/``fuse`` is passed through to ``mlx_lm``.
"""
from __future__ import annotations

import sys


def main() -> None:
    # Apply the shim before touching mlx_lm.
    from . import _mlx_compat
    _mlx_compat.apply_shim()

    if len(sys.argv) < 2:
        print("usage: python -m fluidvoice_finetune._mlx_cli {lora,fuse,...} [args]", file=sys.stderr)
        raise SystemExit(2)

    sub = sys.argv[1]
    rest = sys.argv[2:]

    if sub in ("lora", "fuse"):
        # Import the corresponding mlx_lm module and call its main() with the
        # remaining args. We splice sys.argv so argparse sees only `rest`.
        sys.argv = [f"mlx_lm.{sub}"] + rest
        import importlib
        mod = importlib.import_module(f"mlx_lm.{sub}")
        main_fn = getattr(mod, "main", None)
        if main_fn is None:
            print(f"mlx_lm.{sub} has no main(); falling back to CLI entry", file=sys.stderr)
            from mlx_lm import version  # noqa: F401
            raise SystemExit(1)
        main_fn()
    else:
        # Generic passthrough: treat `sub` as a mlx_lm subcommand module.
        sys.argv = [f"mlx_lm.{sub}"] + rest
        import importlib
        try:
            mod = importlib.import_module(f"mlx_lm.{sub}")
        except ImportError as e:
            print(f"unknown mlx_lm subcommand {sub!r}: {e}", file=sys.stderr)
            raise SystemExit(2)
        if hasattr(mod, "main"):
            mod.main()
        else:
            print(f"mlx_lm.{sub} has no main()", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
