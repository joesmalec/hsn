"""Byte-level inference and conversation serialization, independent of the UI."""

from __future__ import annotations

import codecs
import json
import math
import re
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from hsn.config import config_hash, load_config
from hsn.model.hsn import HSN
from hsn.train.checkpoint import latest_checkpoint, load_checkpoint


@dataclass
class Settings:
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int = 0
    context_id: int = 0
    max_bytes: int = 512
    format: str = "plain-v1"
    filter_fillers: bool = True

    def validate(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and >= 0")
        if not 0 < self.top_p <= 1 or self.max_bytes < 1:
            raise ValueError("top-p must be in (0, 1]; max-bytes must be positive")
        if self.format not in ("plain-v1", "raw"):
            raise ValueError("format must be plain-v1 or raw")


def clean_input(text: str, enabled: bool = True) -> str:
    """Remove only unambiguous hesitation words, preserving quoted text verbatim."""
    if not enabled:
        return text
    parts = re.split(r'("[^"\n]*"|\x27[^\x27\n]*\x27|`[^`]*`)', text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"(?i)(?<![\w-])(?:um|uh|erm)(?![\w-])[, ]*", "", parts[i])
    return "".join(parts).strip()


def safe_text(text: str) -> str:
    return "".join(c if c in "\n\t" or c.isprintable() else f"\\x{ord(c):02x}" for c in text)


def serialize(text: str, fmt: str, first: bool) -> bytes:
    if fmt == "raw":
        return text.encode("utf-8")
    return (("" if first else "\n") + f"User: {text}\nAssistant: ").encode("utf-8")


class Session:
    def __init__(self, model: HSN, settings: Settings, checkpoint: str = "") -> None:
        settings.validate()
        if not 0 <= settings.context_id < model.cfg.n_context:
            raise ValueError("context-id is outside the model's context vocabulary")
        self.model = model.eval()
        self.settings = settings
        self.checkpoint = checkpoint
        self.cancel = threading.Event()
        self.messages: list[dict[str, Any]] = []
        self.reset()

    @classmethod
    def load(
        cls,
        *,
        run_dir: str | None,
        checkpoint: str | None,
        config: str | None,
        device: str,
        settings: Settings,
    ) -> Session:
        path = Path(checkpoint) if checkpoint else latest_checkpoint(run_dir or ".")
        if path is None:
            raise ValueError("No checkpoint found; supply --run-dir or --checkpoint")
        cfg = load_config(config or str(Path(run_dir or path.parent) / "config.resolved.yaml"))
        if cfg.model_name != "hsn" or cfg.model.vocab_size != 256:
            raise ValueError("Chat requires an HSN model with a 256-byte vocabulary")
        model = HSN.from_config(cfg)
        payload = load_checkpoint(path, model)
        extra = payload.get("extra", {})
        if extra.get("model_name", "hsn") != "hsn":
            raise ValueError("Checkpoint model identity is not hsn")
        if extra.get("config_hash") not in (None, config_hash(cfg)):
            raise ValueError("Checkpoint configuration hash differs from the supplied config")
        model.set_train_step(payload["step"])
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable; use --device cpu")
        return cls(model.to(device), settings, str(path.resolve()))

    def reset(self) -> None:
        self.history = bytearray()
        self.state = self.model.init_state(1)
        self.previous: torch.Tensor | None = None
        self.logits: torch.Tensor | None = None
        self.rng = torch.Generator(device="cpu").manual_seed(self.settings.seed)
        self.messages.clear()
        self.failed = False
        self.cancel.clear()
        self.generated_bytes = 0
        self.saved_bytes = 0
        self.finish_reason = "ready"

    def feed(self, value: int) -> None:
        token = torch.tensor([value], device=self.model.device)
        ctx = torch.tensor([self.settings.context_id], device=self.model.device)
        result = self.model.step(token, ctx, self.state, prev_tokens=self.previous)
        self.state, self.logits, self.previous = result.state, result.logits[0], token

    def sample(self) -> int:
        assert self.logits is not None
        logits = self.logits.float().cpu()
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced non-finite logits; reset the conversation")
        if self.settings.temperature == 0:
            return int(logits.argmax().item())
        probs = (logits / self.settings.temperature).softmax(-1)
        probs, indices = probs.sort(descending=True)
        probs[probs.cumsum(0) - probs >= self.settings.top_p] = 0
        return int(indices[torch.multinomial(probs, 1, generator=self.rng)].item())

    def generate(self, original: str) -> Iterator[str]:
        if self.failed:
            raise ValueError("Reset the conversation after an inference error")
        text = clean_input(original, self.settings.filter_fillers)
        if not text.strip():
            raise ValueError("Message is empty after filtering; disable the filter or enter text")
        prompt = serialize(text, self.settings.format, not self.messages)
        self.saved_bytes = len(original.encode()) - len(text.encode())
        self.generated_bytes = 0
        self.messages.append({"role": "user", "text": original, "model_text": text})
        reply: dict[str, Any] = {"role": "assistant", "text": "", "finish_reason": "error"}
        self.messages.append(reply)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = b""
        stop = b"\nUser:" if self.settings.format == "plain-v1" else b""
        self.finish_reason = "limit"
        try:
            with torch.inference_mode():
                for value in prompt:
                    if self.cancel.is_set():
                        # A partially consumed prompt cannot support another turn safely.
                        self.failed = True
                        break
                    self.feed(value)
                    self.history.append(value)
                for _ in range(self.settings.max_bytes):
                    if self.cancel.is_set():
                        self.finish_reason = "cancelled"
                        break
                    value = self.sample()
                    self.feed(value)
                    self.generated_bytes += 1
                    pending += bytes([value])
                    self.history.append(value)
                    if stop and pending == stop:
                        # Rebuild on the next turn from the visible conversation, excluding stop.
                        self.finish_reason = "stop"
                        pending = b""
                        del self.history[-len(stop) :]
                        break
                    while pending and not (stop and stop.startswith(pending)):
                        chunk = decoder.decode(pending[:1])
                        pending = pending[1:]
                        reply["text"] += chunk
                        if chunk:
                            yield chunk
                chunk = decoder.decode(pending, final=True)
                reply["text"] += chunk
                if chunk:
                    yield chunk
                if self.finish_reason == "stop":
                    # Stop bytes have already entered the recurrent state. Replay exact history.
                    self.state = self.model.init_state(1)
                    self.previous = None
                    for byte in self.history:
                        if self.cancel.is_set():
                            self.failed = True
                            self.finish_reason = "cancelled"
                            break
                        self.feed(byte)
            reply["finish_reason"] = self.finish_reason
        except Exception:
            self.failed = True
            self.finish_reason = "error"
            raise

    def export(self, path: Path) -> None:
        with path.open("x", encoding="utf-8") as file:
            json.dump(
                {
                    "version": 1,
                    "checkpoint": self.checkpoint,
                    "settings": asdict(self.settings),
                    "messages": self.messages,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")
