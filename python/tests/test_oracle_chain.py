"""Torch -> Secant -> Go-oracle chain.

A ProjectThenHealLinear trained-style adapter is exported with the RAW B_tilde,
healed by BOTH `secant heal` and the Go oracle `crucible wsp heal`, and the
results are compared:

  * the two healed LoRA files must be byte-identical (sha256);
  * B_locked read back from the file must equal the module's own float64
    P(P B_tilde) rounded to float32 (the torch module and the Rust CLI compute
    the same lock independently).

SECANT_ORACLE_BIN is three-state: absent => skip with an explicit reason;
present-but-empty => error; value => the oracle binary. The Rust arithmetic
policy is chosen from the ORACLE's CPU architecture (read from its Mach-O/ELF
header): arm64 oracles fuse multiply-adds, so they are compared with
`--go-arm64-fma`.
"""

import hashlib
import os
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="SKIP-NO-TORCH: needs torch")

import secant
from secant.nn import ProjectThenHealLinear
from secant._safetensors import read_safetensors

from conftest import real_binary_or_skip


def _oracle_or_skip():
    v = os.environ.get("SECANT_ORACLE_BIN")
    if v is None:
        pytest.skip("SKIP-NO-ORACLE: SECANT_ORACLE_BIN is absent")
    if v == "":
        pytest.fail("SECANT_ORACLE_BIN is present but empty: refusing to treat it as unset")
    return v


def _oracle_arch(path):
    with open(path, "rb") as f:
        h = f.read(32)
    if h[:4] == b"\xcf\xfa\xed\xfe":
        return {0x0100000C: "aarch64", 0x01000007: "x86_64"}.get(int.from_bytes(h[4:8], "little"), "unknown")
    if h[:4] == b"\x7fELF":
        return {0xB7: "aarch64", 0x3E: "x86_64"}.get(int.from_bytes(h[18:20], "little"), "unknown")
    return "unknown"


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_heal_chain_matches_oracle_and_module(tmp_path, dtype):
    binary = real_binary_or_skip()
    oracle = _oracle_or_skip()
    arch = _oracle_arch(oracle)
    assert arch in ("aarch64", "x86_64"), f"cannot determine the oracle architecture ({arch})"
    fma = arch == "aarch64"

    torch.manual_seed(1234)
    d_out, d_in, r, k = 40, 24, 4, 3
    dirs = torch.randn(k, d_out, dtype=torch.float64)
    mod = ProjectThenHealLinear(d_in, d_out, r, dirs, alpha=8.0, dtype=dtype, b_init_std=0.5, module_name="base_model.model.model.layers.0.mlp.down_proj")
    # raw (unlocked) export: the CLI must perform the lock itself
    lora = str(tmp_path / "raw.safetensors")
    mod.export_lora(lora, locked=False)
    vec_dir = tmp_path / "vecs"
    vec_dir.mkdir()
    for i in range(k):
        np.save(vec_dir / f"d_{i}.npy", dirs[i].numpy().astype("<f8"))

    out_rs = str(tmp_path / "healed_rs.safetensors")
    out_go = str(tmp_path / "healed_go.safetensors")
    rs = secant.heal(lora, str(vec_dir), out_rs, binary=binary, go_arm64_fma=fma, json=True)
    assert rs.exit_code == 0, rs.stderr
    go = subprocess.run([oracle, "wsp", "heal", "--lora", lora, "--vectors", str(vec_dir), "--out", out_go, "--json"], capture_output=True, text=True)
    assert go.returncode == 0, go.stderr
    assert _sha(out_rs) == _sha(out_go), f"healed LoRA differs from the Go oracle (oracle arch {arch}, fma={fma})"

    tensors, _ = read_safetensors(out_rs)
    name = mod.module_name
    b_file = tensors[f"{name}.lora_B.weight"]
    want = mod.lora_tensors(locked=True)[f"{name}.lora_B.weight"]
    # same float64 projection in torch and Rust, rounded once to the export dtype
    scale = float(np.abs(want).max())
    assert np.allclose(b_file, want, rtol=0, atol=2.5e-7 * scale if dtype == torch.float32 else 1e-12 * scale)
    # lora_A is copied byte-for-byte
    assert np.array_equal(tensors[f"{name}.lora_A.weight"], mod.lora_tensors(locked=False)[f"{name}.lora_A.weight"])
