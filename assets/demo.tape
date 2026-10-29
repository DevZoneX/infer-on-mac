# Re-record the README demo with VHS (https://github.com/charmbracelet/vhs):
#   brew install vhs && brew install --cask font-jetbrains-mono
#   mlx-chat serve medium --port 8081        # load the model first so the take skips loading
#   cd /path/to/a/project/with/a/failing/test # with pytest on PATH (e.g. a local .venv)
#   git checkout . && git clean -fd           # reset between takes
#   vhs /path/to/infer-on-mac/assets/demo.tape && mv demo.gif /path/to/infer-on-mac/assets/
# Recorded with Qwen3-30B-A3B-4bit on a project with todo.py, tests/test_todo.py,
# AGENTS.md ("Run tests with `python -m pytest -q`") and a failing empty-title test.
Output demo.gif
Set Shell zsh
Set FontFamily "JetBrains Mono"
Set FontSize 15
Set Width 1000
Set Height 760
Set Padding 18
Set WindowBar Colorful
Set Theme { "background": "#0b0c10", "foreground": "#d4d7de", "cursor": "#a5b4fc", "selection": "#4b5578", "black": "#16181f", "brightBlack": "#8b93a3", "red": "#fca5a5", "brightRed": "#fca5a5", "green": "#86efac", "brightGreen": "#86efac", "yellow": "#fcd34d", "brightYellow": "#fcd34d", "blue": "#a5b4fc", "brightBlue": "#a5b4fc", "magenta": "#c084fc", "brightMagenta": "#c084fc", "cyan": "#22d3ee", "brightCyan": "#22d3ee", "white": "#d4d7de", "brightWhite": "#ffffff" }
Set TypingSpeed 45ms

Hide
Type "export PATH=$PWD/.venv/bin:$PATH && clear" Enter
Type "mlx-chat chat medium --attach --port 8081 --ephemeral" Enter
Wait+Screen@1m /Qwen3-30B/
Sleep 3s
Show

Sleep 1s
Type "The empty-title test fails. Fix it and run the tests."
Sleep 500ms
Enter
# The model runs the tests first: always allow Run for the session.
Wait+Screen@3m /Allow Run\?/
Sleep 1.5s
Type "a"
Enter
# Then approve its edit once.
Wait+Screen@3m /Allow Edit\?/
Sleep 2.5s
Enter
Wait+Screen@3m /agent ·/
Sleep 4s
