# Real VHS capture of the compiled secant binary. Not hand-drawn.
# Reproduce from native/secant-rs (see docs/quickstart.md, "Watch It Run"):
#   cargo build --release
#   SECANT_DEMO_DIR=/path/to/demo vhs docs/assets/vhs/secant-analyze.tape
Output docs/assets/secant-analyze.gif
Output docs/assets/vhs/secant-analyze.txt

Set Shell "bash"
Set FontSize 18
Set Width 1240
Set Height 580
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms
Set Framerate 25
Set PlaybackSpeed 1.0

Hide
Type 'export PATH="$PWD/target/release:$PATH"; cd "${SECANT_DEMO_DIR:?set SECANT_DEMO_DIR}"; rm -f analysis.json; clear'
Enter
Sleep 500ms
Show

Type "# Rank layers by captured per-layer SNR (Borda across 4 subspaces)"
Sleep 200ms
Enter
Type "secant analyze --in layer_snr.json --out analysis.json"
Sleep 200ms
Enter
Wait@20s
Sleep 4s
