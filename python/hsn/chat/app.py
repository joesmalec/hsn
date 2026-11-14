"""Optional full-screen terminal frontend."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button, Footer, Header, Static, TextArea

from hsn.chat.session import Session, safe_text


class ChatApp(App[None]):
    TITLE = "HSN · local chat"
    STATUS_REFRESH = 0.2  # seconds between status-line refreshes
    CSS = """
    #transcript { height: 1fr; padding: 1 2; }
    #composer { height: 6; margin: 0 2; }
    #status { height: auto; padding: 0 2; }
    Button { margin: 0 2; }
    """
    BINDINGS = [
        Binding("ctrl+enter", "send", "Send", priority=True),
        Binding("escape", "stop", "Stop", priority=True),
        Binding("ctrl+n", "reset_chat", "New chat", priority=True),
        Binding("ctrl+e", "export", "Export", priority=True),
        Binding("ctrl+f", "filter", "Filler filter", priority=True),
        Binding("ctrl+q", "leave", "Quit", priority=True),
    ]

    def __init__(self, loader: Callable[[], Session]) -> None:
        super().__init__()
        self.loader = loader
        self.session: Session | None = None
        self.busy = True
        self.started = time.monotonic()
        self.reply: Static | None = None
        self.reply_text = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="transcript"):
            yield Static("Load a language-trained checkpoint for conversational replies.")
        yield Static("Loading checkpoint…", id="status", markup=False)
        yield TextArea(id="composer", disabled=True)
        yield Button("Send", id="send", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        # The status line owns its refresh timer. Textual stops a widget's timers before the
        # widget leaves the DOM, so no tick can query a status line that is gone. A timer on the
        # App outlives the screen's widgets when App._shutdown prunes the screen first and stops
        # the App's timers only afterwards, as App.run_test does on exit.
        self.query_one("#status", Static).set_interval(self.STATUS_REFRESH, self.refresh_status)
        self.load_model()

    @work(thread=True, exit_on_error=False)
    def load_model(self) -> None:
        try:
            session = self.loader()
            self.call_from_thread(self.loaded, session)
        except Exception as exc:
            self.call_from_thread(self.show_error, str(exc))

    def loaded(self, session: Session) -> None:
        self.session = session
        self.set_busy(False)
        self.query_one("#composer", TextArea).focus()

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.query_one("#composer", TextArea).disabled = busy
        self.query_one("#send", Button).disabled = busy
        self.refresh_status()

    def show_error(self, message: str) -> None:
        self.query_one("#transcript", VerticalScroll).mount(
            Static("Error: " + safe_text(message), markup=False)
        )
        if self.session is None:
            self.query_one("#status", Static).update(
                "Load failed · Ctrl+N to retry · Ctrl+Q to quit"
            )

    def refresh_status(self) -> None:
        if self.session is None:
            return
        s = self.session
        state = "generating" if self.busy else ("reset required" if s.failed else s.finish_reason)
        self.query_one("#status", Static).update(
            f"{safe_text(s.checkpoint)} · {s.model.device} · {state} · "
            f"{s.generated_bytes} bytes · {time.monotonic() - self.started:.1f}s · "
            f"filter {'on' if s.settings.filter_fillers else 'off'} · {s.saved_bytes} bytes saved"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send":
            self.action_send()

    def action_send(self) -> None:
        if self.busy or self.session is None:
            return
        composer = self.query_one("#composer", TextArea)
        text = composer.text
        if not text.strip():
            return
        if self.session.failed:
            self.show_error("Press Ctrl+N to reset before continuing")
            return
        composer.clear()
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(Static("You: " + safe_text(text), markup=False))
        self.reply = Static("HSN: ", markup=False)
        transcript.mount(self.reply)
        self.reply_text = ""
        self.started = time.monotonic()
        self.session.cancel.clear()
        self.set_busy(True)
        self.respond(text)

    @work(thread=True, exit_on_error=False)
    def respond(self, text: str) -> None:
        assert self.session is not None
        try:
            for chunk in self.session.generate(text):
                self.call_from_thread(self.append_reply, chunk)
        except Exception as exc:
            self.call_from_thread(self.show_error, str(exc))
        finally:
            self.call_from_thread(self.set_busy, False)

    def append_reply(self, chunk: str) -> None:
        self.reply_text += chunk
        if self.reply is not None:
            self.reply.update("HSN: " + safe_text(self.reply_text))
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    def action_stop(self) -> None:
        if self.session is not None:
            self.session.cancel.set()

    def action_reset_chat(self) -> None:
        if self.session is None:
            if not self.workers:
                self.load_model()
            return
        if self.busy:
            self.action_stop()
            return
        self.session.reset()
        self.query_one("#transcript", VerticalScroll).remove_children()
        self.refresh_status()

    def action_filter(self) -> None:
        if self.session is not None and not self.busy:
            self.session.settings.filter_fillers = not self.session.settings.filter_fillers
            self.refresh_status()

    def action_export(self) -> None:
        if self.session is None or self.busy:
            return
        path = Path("hsn-chat-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
        try:
            self.session.export(path)
            self.notify(f"Saved {path.resolve()}")
        except OSError as exc:
            self.show_error(str(exc))

    def action_leave(self) -> None:
        self.action_stop()
        self.exit()
