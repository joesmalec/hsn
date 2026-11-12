"""Argument handling without importing the optional terminal dependency."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from hsn.chat.session import Session, Settings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hsn chat", description="Chat with a local HSN checkpoint"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir")
    source.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-id", type=int, default=0)
    parser.add_argument("--max-bytes", type=int, default=512)
    parser.add_argument("--format", choices=["plain-v1", "raw"], default="plain-v1")
    parser.add_argument("--no-filter-fillers", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings(
        args.temperature,
        args.top_p,
        args.seed,
        args.context_id,
        args.max_bytes,
        args.format,
        not args.no_filter_fillers,
    )
    try:
        settings.validate()
    except ValueError as exc:
        parser.error(str(exc))
    try:
        from hsn.chat.app import ChatApp
    except ModuleNotFoundError as exc:
        if exc.name != "textual":
            raise
        parser.error("Install the chat extra: uv sync --extra chat --inexact")
    app = ChatApp(
        lambda: Session.load(
            run_dir=args.run_dir,
            checkpoint=args.checkpoint,
            config=args.config,
            device=args.device,
            settings=settings,
        )
    )
    app.run()
    return 0
