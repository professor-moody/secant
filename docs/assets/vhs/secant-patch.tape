# Real VHS capture of the compiled secant binary. Not hand-drawn.
# Reproduce from native/secant-rs (see docs/quickstart.md, "Watch It Run"):
#   cargo build --release
#   SECANT_DEMO_DIR=/path/to/demo vhs docs/assets/vhs/secant-patch.tape
#
# Patching and hashing the 3.4 GB checkpoint takes about 27 s and 8 s of CPU time.
# The tape hides those waits (Hide/Wait/Show) so the GIF shows no blank screen.
# Nothing is pre-printed: each command really runs, and bash `time` prints its real
# elapsed time on screen, so the cut is visible in the recording itself.
Output docs/assets/secant-patch.gif
Output docs/assets/vhs/secant-patch.txt

Set Shell "bash"
Set FontSize 15
Set Width 1240
Set Height 700
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set Framerate 30
Set PlaybackSpeed 1.0

Hide
Type 'export PATH="$PWD/target/release:$PATH" TIMEFORMAT="real %Rs (wait cut from this recording)"; cd "${SECANT_DEMO_DIR:?set SECANT_DEMO_DIR}"; rm -f patched.safetensors; clear'
Enter
Sleep 500ms
Show

Type "# Patch a copy of the 3.4 GB checkpoint, then hash the output"
Enter
Type "time secant patch --base smollm2-1.7b.safetensors --plan plan.json --out patched.safetensors"
Enter
Hide
Wait@180s
Show
Sleep 2s
Type "time shasum -a 256 patched.safetensors"
Enter
Hide
Wait@60s
Show
Sleep 6s
