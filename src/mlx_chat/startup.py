"""Model loading indicator."""

from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from .terminal import short_model
from .theme import ACCENT, BORDER, MUTED


class StartupStatus:
    """Display model loading progress."""

    def __init__(self, console, model, message):
        self.console = console
        self.model = model
        self.message = message
        self.live = Live(self.render(), console=console, transient=True, refresh_per_second=8)

    def render(self):
        message = self.message.plain if isinstance(self.message, Text) else str(self.message)
        content = Group(
            Text("◆ mlx-chat", style=f"bold {ACCENT}"),
            Text(short_model(self.model), style="bold"),
            Text("Local MLX inference · Apple Silicon", style="dim"),
            Text(""),
            Spinner("dots", text=Text(message, style=MUTED), style=ACCENT),
        )
        return Panel(
            content,
            box=box.ROUNDED,
            border_style=BORDER,
            padding=(1, 2),
            width=min(68, self.console.width),
        )

    def update(self, message):
        self.message = message
        self.live.update(self.render())

    def __enter__(self):
        self.live.__enter__()
        return self

    def __exit__(self, *args):
        return self.live.__exit__(*args)
