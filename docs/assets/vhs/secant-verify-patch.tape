# Real VHS capture of the compiled secant binary. Not hand-drawn.
# Reproduce from native/secant-rs (see docs/quickstart.md, "Watch It Run"):
#   cargo build --release
#   SECANT_DEMO_DIR=/path/to/demo vhs docs/assets/vhs/secant-verify-patch.tape
Output docs/assets/secant-verify-patch.gif
Output docs/assets/vhs/secant-verify-patch.txt

Set Shell "bash"
Set FontSize 15
Set Width 1240
Set Height 560
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms
Set Framerate 20
Set PlaybackSpeed 1.0

Hide
Type 'export PATH="$PWD/target/release:$PATH"; cd "${SECANT_DEMO_DIR:?set SECANT_DEMO_DIR}";  clear'
Enter
Sleep 500ms
Show

Type "# Prove every non-target tensor is byte-identical to the base"
Enter
Type 'secant verify-patch --base smollm2-1.7b.safetensors --patched patched.safetensors \'
Enter
Type "  --expect-changed model.layers.14.self_attn.o_proj.weight,model.layers.15.mlp.down_proj.weight"
Enter
Wait@120s
Sleep 5s
