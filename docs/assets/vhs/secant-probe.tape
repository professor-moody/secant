# Real VHS capture of the compiled secant binary. Not hand-drawn.
# Reproduce from native/secant-rs (see docs/quickstart.md, "Watch It Run"):
#   cargo build --release
#   SECANT_DEMO_DIR=/path/to/demo vhs docs/assets/vhs/secant-probe.tape
Output docs/assets/secant-probe.gif
Output docs/assets/vhs/secant-probe.txt

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
Type 'export PATH="$PWD/target/release:$PATH"; cd "${SECANT_DEMO_DIR:?set SECANT_DEMO_DIR}"; rm -rf fixtures probe-receipt.json; clear'
Enter
Sleep 500ms
Show

Type "# Generate deterministic NaN/Inf boundary fixtures in both containers"
Sleep 200ms
Enter
Type "secant probe --class class3 --out-dir fixtures --receipt probe-receipt.json --attempt-id vhs-demo"
Sleep 200ms
Enter
Wait@20s
Type "shasum -a 256 fixtures/*"
Sleep 200ms
Enter
Wait@10s
Type "secant inspect --model fixtures/class3-nan-sort-arrays.gguf"
Sleep 200ms
Enter
Wait@10s
Sleep 4s
