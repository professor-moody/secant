# Real VHS capture of the compiled secant binary. Not hand-drawn.
# Reproduce from native/secant-rs (see docs/quickstart.md, "Watch It Run"):
#   cargo build --release
#   SECANT_DEMO_DIR=/path/to/demo vhs docs/assets/vhs/secant-patch.tape
#
# Patching and hashing the 3.4 GB checkpoint takes ~27 s and ~8 s of disk I/O.
# The tape hides those disk I/O waits (Hide/Wait/Show) after the command is typed
# so the GIF has no blank screen pause.
Output docs/assets/secant-patch.gif
Output docs/assets/vhs/secant-patch.txt

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
Type 'export PATH="$PWD/target/release:$PATH"; cd "${SECANT_DEMO_DIR:?set SECANT_DEMO_DIR}"; rm -f patched.safetensors; clear'
Enter
Sleep 500ms
Show

Type "# Patch a copy of the 3.4 GB checkpoint, then hash the output"
Sleep 200ms
Enter
Type "secant patch --base smollm2-1.7b.safetensors --plan plan.json --out patched.safetensors"
Sleep 400ms
Enter
Hide
Wait@180s
Show
Sleep 1s
Type "shasum -a 256 patched.safetensors"
Sleep 400ms
Enter
Hide
Wait@60s
Show
Sleep 4s
