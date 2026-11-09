# Real VHS capture of the compiled secant binary. Not hand-drawn.
# Reproduce from native/secant-rs (see docs/quickstart.md, "Watch It Run"):
#   cargo build --release
#   SECANT_DEMO_DIR=/path/to/demo vhs docs/assets/vhs/secant-compose.tape
Output docs/assets/secant-compose.gif
Output docs/assets/vhs/secant-compose.txt

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
Type 'export PATH="$PWD/target/release:$PATH"; cd "${SECANT_DEMO_DIR:?set SECANT_DEMO_DIR}"; rm -f plan.json; clear'
Enter
Sleep 500ms
Show

Type "# Orthogonalize B2/B3 directions in float64 and calibrate the plan"
Sleep 200ms
Enter
Type "secant compose --profile profile-smollm2-b2b3.json --vectors vectors --out plan.json"
Sleep 200ms
Enter
Wait@20s
Sleep 4s
