# Real VHS capture of the compiled secant binary. Not hand-drawn.
# Reproduce from native/secant-rs (see docs/quickstart.md, "Watch It Run"):
#   cargo build --release
#   SECANT_DEMO_DIR=/path/to/demo vhs docs/assets/vhs/secant-patch.tape
Output docs/assets/secant-patch.gif
Output docs/assets/vhs/secant-patch.txt

Set Shell "bash"
Set FontSize 15
Set Width 1240
Set Height 620
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms
Set Framerate 20
Set PlaybackSpeed 1.0

Hide
Type 'export PATH="$PWD/target/release:$PATH"; cd "${SECANT_DEMO_DIR:?set SECANT_DEMO_DIR}"; rm -f patched.safetensors; clear'
Enter
Sleep 500ms
Show

Type "# Apply the plan to a copy of the 3.4 GB checkpoint, then hash the output"
Enter
Type "secant patch --base smollm2-1.7b.safetensors --plan plan.json --out patched.safetensors"
Enter
Wait@120s
Type "shasum -a 256 patched.safetensors"
Enter
Wait@60s
Sleep 5s
