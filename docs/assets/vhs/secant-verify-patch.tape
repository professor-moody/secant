# Real VHS capture of the compiled secant binary. Not hand-drawn.
# Reproduce from native/secant-rs (see docs/quickstart.md, "Watch It Run"):
#   cargo build --release
#   SECANT_DEMO_DIR=/path/to/demo vhs docs/assets/vhs/secant-verify-patch.tape
#
# Reading both 3.4 GB files takes ~7 s of disk I/O. The tape hides that disk I/O
# wait (Hide/Wait/Show) after the command is typed so the GIF has no blank pause.
Output docs/assets/secant-verify-patch.gif
Output docs/assets/vhs/secant-verify-patch.txt

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

Type "# Prove every non-target tensor is byte-identical to the base"
Sleep 200ms
Enter
Type 'secant verify-patch --base smollm2-1.7b.safetensors --patched patched.safetensors \'
Sleep 200ms
Enter
Type "  --expect-changed model.layers.14.self_attn.o_proj.weight,model.layers.15.mlp.down_proj.weight"
Sleep 400ms
Enter
Hide
Wait@120s
Show
Sleep 400ms
Type "shasum -a 256 smollm2-1.7b.safetensors patched.safetensors"
Sleep 400ms
Enter
Hide
Wait@120s
Show
Sleep 4s
