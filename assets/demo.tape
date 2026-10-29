# Re-record the README demo with VHS (https://github.com/charmbracelet/vhs):
#   brew install vhs
#   cd /path/to/a/project/with/a/failing/test
#   vhs /path/to/infer-on-mac/assets/demo.tape
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
Type "mlx-chat medium" Enter
Sleep 8s
Show

Sleep 1s
Type "The empty-title test fails. Fix it and run the tests."
Sleep 500ms
Enter
# Wait for the edit approval card, then approve once.
Wait+Screen /Allow/
Sleep 1.5s
Enter
Sleep 1s
# Always allow run_command for the session.
Wait+Screen /Allow/
Sleep 1.2s
Type "a"
Enter
Wait+Screen@2m /reused from cache/
Sleep 3s
