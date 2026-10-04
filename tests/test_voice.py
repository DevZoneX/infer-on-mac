from prompt_toolkit.input import create_pipe_input

from mlx_chat.voice import recording_stop_requested


def test_recording_stop_handles_f2_and_ctrl_c_without_blocking():
    with create_pipe_input() as keyboard:
        assert not recording_stop_requested(keyboard)
        keyboard.send_text("\x1bOP")
        assert not recording_stop_requested(keyboard)
        keyboard.send_text("x")
        assert not recording_stop_requested(keyboard)
        keyboard.send_text("\x1bOQ")
        assert recording_stop_requested(keyboard)
        keyboard.send_text("\x1b[12~")
        assert recording_stop_requested(keyboard)
        keyboard.send_text("\x03")
        assert recording_stop_requested(keyboard)
    assert not recording_stop_requested(None)
