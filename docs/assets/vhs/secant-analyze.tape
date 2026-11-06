# Real VHS capture of the compiled secant binary. Not hand-drawn.
# Reproduce from native/secant-rs (see docs/quickstart.md, "Watch It Run"):
#   cargo build --release
#   SECANT_DEMO_DIR=/path/to/demo vhs docs/assets/vhs/secant-analyze.tape
Output docs/assets/secant-analyze.gif
Output docs/assets/vhs/secant-analyze.txt

Set Shell "bash"
Set FontSize 14
Set Width 1560
Set Height 700
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms
Set Framerate 20
Set PlaybackSpeed 1.0

Hide
Type 'export PATH="$PWD/target/release:$PATH"; cd "${SECANT_DEMO_DIR:?set SECANT_DEMO_DIR}"; rm -f analysis.json; clear'
Enter
Sleep 500ms
Show

Type "# Rank layers by captured per-layer SNR (Borda across 4 subspaces)"
Enter
Type "secant analyze --in layer_snr.json --out analysis.json"
Enter
Wait@20s
Sleep 5s
