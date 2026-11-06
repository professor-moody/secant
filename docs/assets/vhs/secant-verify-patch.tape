# Real VHS capture of the compiled secant binary. Not hand-drawn.
# Reproduce from native/secant-rs (see docs/quickstart.md, "Watch It Run"):
#   cargo build --release
#   SECANT_DEMO_DIR=/path/to/demo vhs docs/assets/vhs/secant-verify-patch.tape
#
# Reading both 3.4 GB files takes about 7 s. The tape hides that wait (Hide/Wait/Show)
# so the GIF shows no blank screen. The command really runs, and bash `time` prints
# its real elapsed time on screen, so the cut is visible in the recording itself.
Output docs/assets/secant-verify-patch.gif
Output docs/assets/vhs/secant-verify-patch.txt

Set Shell "bash"
Set FontSize 15
Set Width 1240
Set Height 560
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set Framerate 30
Set PlaybackSpeed 1.0

Hide
Type 'export PATH="$PWD/target/release:$PATH" TIMEFORMAT="real %Rs (wait cut from this recording)"; cd "${SECANT_DEMO_DIR:?set SECANT_DEMO_DIR}"; clear'
Enter
Sleep 500ms
Show

Type "# Prove every non-target tensor is byte-identical to the base"
Enter
Type 'time secant verify-patch --base smollm2-1.7b.safetensors --patched patched.safetensors \'
Enter
Type "  --expect-changed model.layers.14.self_attn.o_proj.weight,model.layers.15.mlp.down_proj.weight"
Enter
Hide
Wait@120s
Show
Sleep 6s
