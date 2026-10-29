"""Terminal colors."""

CANVAS = "#0b0c10"
SURFACE = "#16181f"
POPUP = "#20232c"
ACCENT = "#a5b4fc"
MUTED = "#8b93a3"
BORDER = "#3a3f4b"
SELECTION = "#4b5578"
SUCCESS = "#86efac"
WARNING = "#fcd34d"
DANGER = "#fca5a5"
# The wordmark fades between these two.
BRAND_START = "#c084fc"
BRAND_END = "#22d3ee"


def gradient(text: str, start: str = BRAND_START, end: str = BRAND_END):
    """Per-character colors from start to end, as (character, hex color) pairs."""
    first = [int(start[index : index + 2], 16) for index in (1, 3, 5)]
    last = [int(end[index : index + 2], 16) for index in (1, 3, 5)]
    steps = max(1, len(text) - 1)
    for position, character in enumerate(text):
        mix = position / steps
        channels = (round(a + (b - a) * mix) for a, b in zip(first, last, strict=True))
        yield character, "#{:02x}{:02x}{:02x}".format(*channels)
