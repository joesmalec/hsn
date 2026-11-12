"""CPU inference contract and terminal integration checks."""

import asyncio
import json
from pathlib import Path

import pytest
import torch
from hsn.chat.session import Session, Settings, clean_input, safe_text, serialize
from hsn.config import config_hash, dump_resolved, load_config
from hsn.model.hsn import HSN
from hsn.train.checkpoint import capture_rng_state, save_checkpoint


@pytest.fixture
def session():
    cfg = load_config("configs/tiny_gradcheck.yaml", overrides=["model.vocab_size=256"])
    torch.manual_seed(0)
    return Session(HSN.from_config(cfg), Settings(max_bytes=4, format="raw"))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Um, explain this", "explain this"),
        ("I uh do not like that", "I do not like that"),
        ("umbrella drum uh-oh", "umbrella drum uh-oh"),
        ('say "um" and `uh`', 'say "um" and `uh`'),
        ("erm", ""),
    ],
)
def test_filter(source, expected):
    assert clean_input(source) == expected
    assert clean_input(source, False) == source


def test_real_generation_reset_and_weights(session):
    before = {
        k: v.clone() for k, v in session.model.state_dict().items() if isinstance(v, torch.Tensor)
    }
    result = "".join(session.generate("hello"))
    assert session.generated_bytes == 4
    assert session.state.pos.item() == 9
    session.reset()
    assert "".join(session.generate("hello")) == result
    for k, v in before.items():
        torch.testing.assert_close(session.model.state_dict()[k], v, rtol=0, atol=0)


def test_forward_parity(session):
    tokens = torch.tensor([[65, 66, 67]])
    with torch.inference_mode():
        logits, state, _ = session.model(
            tokens,
            torch.zeros_like(tokens),
            session.model.init_state(1),
            torch.zeros_like(tokens, dtype=torch.bool),
        )
        for b in tokens[0]:
            session.feed(int(b))
    torch.testing.assert_close(session.logits, logits[0, -1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(session.state.h, state.h, atol=1e-5, rtol=1e-5)


def test_unicode_stop_and_exact_replay(session, monkeypatch):
    session.settings.format = "plain-v1"
    session.settings.max_bytes = 30
    values = iter("é".encode() + b"\xff\nUser:")
    monkeypatch.setattr(session, "sample", lambda: next(values))
    assert "".join(session.generate("Hi")) == "é�"
    assert session.finish_reason == "stop"
    assert session.history == serialize("Hi", "plain-v1", True) + "é".encode() + b"\xff"
    assert session.state.pos.item() == len(session.history)


def test_cancel_and_reset(session):
    it = session.generate("hello")
    next(it)
    session.cancel.set()
    list(it)
    assert session.finish_reason == "cancelled"
    assert not session.failed
    session.reset()
    session.cancel.set()
    assert list(session.generate("hello")) == []
    assert session.failed
    session.reset()
    assert not session.failed


def test_error_and_export(session, monkeypatch, tmp_path):
    def fail():
        raise RuntimeError("bad logits")

    monkeypatch.setattr(session, "sample", fail)
    with pytest.raises(RuntimeError):
        list(session.generate("um, hello"))
    assert session.failed
    with pytest.raises(ValueError, match="Reset"):
        list(session.generate("hello"))
    path = tmp_path / "chat.json"
    session.export(path)
    exported = json.loads(path.read_text())
    assert exported["messages"][0]["model_text"] == "hello"
    assert exported["messages"][0]["text"] == "um, hello"
    with pytest.raises(FileExistsError):
        session.export(path)
    assert safe_text("\x1b[31m[bold]\x00") == "\\x1b[31m[bold]\\x00"


def test_load_checkpoint(tmp_path):
    cfg = load_config("configs/tiny_gradcheck.yaml", overrides=["model.vocab_size=256"])
    model = HSN.from_config(cfg)
    dump_resolved(cfg, tmp_path / "config.resolved.yaml")
    for step in [1, 3]:
        save_checkpoint(
            tmp_path / f"step_{step:08d}.pt",
            model,
            None,
            None,
            step,
            capture_rng_state(),
            None,
            extra={"model_name": "hsn", "config_hash": config_hash(cfg)},
        )
    loaded = Session.load(
        run_dir=str(tmp_path),
        checkpoint=None,
        config=None,
        device="cpu",
        settings=Settings(max_bytes=1),
    )
    assert loaded.checkpoint.endswith("step_00000003.pt")
    assert loaded.state.pos.item() == 0
    assert len(list(loaded.generate("hello"))) <= 1
    cfg = cfg.model_copy(update={"seed": cfg.seed + 1})
    dump_resolved(cfg, tmp_path / "config.resolved.yaml")
    with pytest.raises(ValueError, match="hash"):
        Session.load(
            run_dir=str(tmp_path), checkpoint=None, config=None, device="cpu", settings=Settings()
        )


def test_terminal(session, tmp_path, monkeypatch):
    pytest.importorskip("textual")
    from hsn.chat.app import ChatApp
    from textual.widgets import TextArea

    monkeypatch.chdir(tmp_path)

    async def exercise():
        app = ChatApp(lambda: session)
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            app.query_one("#composer", TextArea).text = "um, hello"
            await pilot.click("#send")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(session.messages) == 2
            assert session.saved_bytes == 4
            app.action_export()
            assert list(Path(".").glob("hsn-chat-*.json"))
            await pilot.resize_terminal(45, 20)
            app.action_filter()
            assert not session.settings.filter_fillers
            app.action_reset_chat()
            assert not session.messages

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "settings",
    [
        Settings(temperature=float("nan")),
        Settings(top_p=0),
        Settings(max_bytes=0),
        Settings(context_id=999),
    ],
)
def test_invalid_settings(session, settings):
    with pytest.raises(ValueError):
        Session(session.model, settings)


def test_empty_filtered_input(session):
    with pytest.raises(ValueError, match="empty"):
        list(session.generate("um, uh"))
    assert not session.messages


def test_bad_checkpoints(tmp_path):
    options = dict(
        run_dir=str(tmp_path), checkpoint=None, config=None, device="cpu", settings=Settings()
    )
    with pytest.raises(ValueError, match="No checkpoint"):
        Session.load(**options)
    cfg = load_config("configs/tiny_gradcheck.yaml", overrides=["model.vocab_size=256"])
    dump_resolved(cfg, tmp_path / "config.resolved.yaml")
    torch.save({"format": -1}, tmp_path / "step_00000001.pt")
    with pytest.raises(ValueError, match="format"):
        Session.load(**options)
    cfg = cfg.model_copy(update={"model_name": "gru"})
    dump_resolved(cfg, tmp_path / "config.resolved.yaml")
    with pytest.raises(ValueError, match="HSN model"):
        Session.load(**options)


def test_terminal_load_error_and_retry(session):
    from hsn.chat.app import ChatApp

    attempts = []

    def load():
        attempts.append(True)
        if len(attempts) == 1:
            raise ValueError("missing checkpoint")
        return session

    async def exercise():
        app = ChatApp(load)
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            assert app.session is None
            await pilot.press("ctrl+n")
            await app.workers.wait_for_complete()
            assert app.session is session
            assert not app.busy

    asyncio.run(exercise())


def test_terminal_stop_and_overlap(session, monkeypatch):
    import threading

    from hsn.chat.app import ChatApp
    from textual.widgets import TextArea

    entered = threading.Event()
    release = threading.Event()

    def sample():
        entered.set()
        release.wait(timeout=5)
        return 65

    monkeypatch.setattr(session, "sample", sample)

    async def exercise():
        app = ChatApp(lambda: session)
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            app.query_one("#composer", TextArea).text = "hello"
            await pilot.press("ctrl+enter")
            await asyncio.to_thread(entered.wait, 5)
            assert app.busy
            app.action_send()
            assert len(session.messages) == 2
            await pilot.press("escape")
            release.set()
            await app.workers.wait_for_complete()
            assert session.finish_reason == "cancelled"
            assert not app.busy
            await pilot.press("ctrl+n")
            assert not session.messages

    try:
        asyncio.run(exercise())
    finally:
        release.set()


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_smoke(session, device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    session.model.to(device)
    session.reset()
    list(session.generate("hello"))
    assert session.generated_bytes == 4
    assert session.state.h.device.type == device
