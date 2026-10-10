"""Audio stop-key handling."""

from __future__ import annotations

import select
import sys
import threading
from contextlib import closing, contextmanager

from prompt_toolkit.input import create_input
from prompt_toolkit.keys import Keys


def stop_requested(keyboard, key: Keys) -> bool:
    if keyboard is None or not select.select([keyboard.fileno()], [], [], 0)[0]:
        return False
    return any(press.key in {key, Keys.ControlC} for press in keyboard.read_keys())


@contextmanager
def stop_key_listener(key: Keys):
    """Listen in raw mode and restore terminal input before returning to the prompt."""
    stopped = threading.Event()
    finished = threading.Event()
    if not sys.stdin.isatty():
        yield stopped
        return
    with closing(create_input()) as keyboard, keyboard.raw_mode():

        def listen():
            while not finished.wait(0.02):
                if stop_requested(keyboard, key):
                    stopped.set()
                    return

        worker = threading.Thread(target=listen, daemon=True)
        worker.start()
        try:
            yield stopped
        finally:
            finished.set()
            worker.join()
