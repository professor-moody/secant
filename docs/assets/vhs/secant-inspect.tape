# Real VHS capture of the compiled secant binary. Not hand-drawn.
# Reproduce from native/secant-rs (see docs/quickstart.md, "Watch It Run"):
#   cargo build --release
#   SECANT_DEMO_DIR=/path/to/demo vhs docs/assets/vhs/secant-inspect.tape
Output docs/assets/secant-inspect.gif
Output docs/assets/vhs/secant-inspect.txt

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
Type 'export PATH="$PWD/target/release:$PATH"; cd "${SECANT_DEMO_DIR:?set SECANT_DEMO_DIR}"; clear'
Enter
Sleep 500ms
Show

Type "# SmolLM2-1.7B-Instruct, pinned HF commit 31b70e2e, 3.4 GB BF16"
Sleep 200ms
Enter
Type "secant inspect --model smollm2-1.7b.safetensors"
Sleep 200ms
Enter
Wait@20s
Sleep 4s
