#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "mlx-chat requires an Apple Silicon Mac (arm64)." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv sync --python 3.11 --extra voice
else
  PYTHON_BIN=${PYTHON_BIN:-python3}
  "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' || {
    echo "Python 3.11+ is required. Install uv (recommended) or set PYTHON_BIN." >&2
    exit 1
  }
  "$PYTHON_BIN" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e '.[voice]'
fi

BIN_DIR=${MLX_CHAT_BIN_DIR:-"$HOME/.local/bin"}
COMMAND_PATH="$BIN_DIR/mlx-chat"
COMMAND_TARGET="$PROJECT_DIR/.venv/bin/mlx-chat"
COMMAND_READY=1

mkdir -p "$BIN_DIR"
if [ -L "$COMMAND_PATH" ]; then
  CURRENT_TARGET=$(readlink "$COMMAND_PATH")
  if [ "$CURRENT_TARGET" != "$COMMAND_TARGET" ]; then
    echo "Not replacing existing symlink: $COMMAND_PATH -> $CURRENT_TARGET" >&2
    echo "Set MLX_CHAT_BIN_DIR to install the command somewhere else." >&2
    COMMAND_READY=0
  fi
elif [ -e "$COMMAND_PATH" ]; then
  echo "Not replacing existing file: $COMMAND_PATH" >&2
  echo "Set MLX_CHAT_BIN_DIR to install the command somewhere else." >&2
  COMMAND_READY=0
else
  ln -s "$COMMAND_TARGET" "$COMMAND_PATH"
fi

echo
if [ "$COMMAND_READY" -eq 1 ]; then
  echo "Installed command: $COMMAND_PATH"
  echo "Start chatting from any directory with:"
  echo "  mlx-chat"
  echo "  mlx-chat mlx-community/Qwen3-4B-4bit"
else
  echo "The project environment is ready, but the global command was not installed."
fi

if [ "$COMMAND_READY" -eq 1 ]; then
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
      echo
      echo "$BIN_DIR is not currently on PATH. Add this to your shell profile:"
      echo "  export PATH=\"$BIN_DIR:\$PATH\""
      ;;
  esac
fi
