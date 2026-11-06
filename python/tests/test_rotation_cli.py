"""Independent PyTorch AdamW and real ProjectThenHealLinear parity oracle."""
import json
import subprocess

import numpy as np
import pytest

from conftest import real_binary_or_skip


def test_native_rotation_matches_torch_adamw_and_structural_layer(tmp_path):
    torch = pytest.importorskip("torch")
    from secant.nn import ProjectThenHealLinear

    binary = real_binary_or_skip()
    rows, cols, rank, steps, seed = 6, 8, 2, 8, 7
    w = np.array([(i * 17 % 29 - 14) / 7 for i in range(rows * cols)], dtype=np.float64).reshape(rows, cols)
    d = np.array([[1., 2., 3., 0., 0., 0.]], dtype=np.float64)
    np.save(tmp_path / "w.npy", w)
    np.save(tmp_path / "d.npy", d)
    args = [binary, "rotation-test", "--tensor-npy", str(tmp_path / "w.npy"), "--directions-npy", str(tmp_path / "d.npy"),
            "--rank", str(rank), "--steps", str(steps), "--lr", "0.01", "--wd", "0.01", "--seed", str(seed), "--json"]
    out = subprocess.run(args, capture_output=True, text=True, check=True)
    report = json.loads(out.stdout)
    assert report["schema"] == "secant.rotation-test.v1"
    assert report["inputs"]["unchanged"] and report["lock_passed"]
    assert subprocess.run(args, capture_output=True, text=True, check=True).stdout == out.stdout
    import secant
    wrapped = secant.rotation_test(tmp_path / "w.npy", tmp_path / "d.npy", rank=rank, steps=steps, seed=seed, binary=binary)
    assert wrapped.ok and wrapped.json == report

    # Pin the documented PRNG, independently of torch's changing RNG internals.
    mask, state, values = (1 << 64) - 1, seed, []
    for _ in range(rank * cols):
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z ^= z >> 31
        values.append(((z >> 11) / (1 << 52) - 1) / np.sqrt(cols))
    a0 = torch.tensor(values, dtype=torch.float64).reshape(rank, cols)
    target = torch.from_numpy(w / np.sqrt(np.mean(w * w)))
    basis = torch.from_numpy(d / np.linalg.norm(d))
    for arm_index, arm in enumerate(report["arms"]):
        if arm_index == 2:
            mod = ProjectThenHealLinear(cols, rows, rank, d, dtype=torch.float64, weight=torch.zeros(rows, cols, dtype=torch.float64))
            with torch.no_grad(): mod.lora_A.copy_(a0)
            a, b = mod.lora_A, mod.lora_B_tilde
        else:
            a = torch.nn.Parameter(a0.clone())
            b = torch.nn.Parameter(torch.zeros(rows, rank, dtype=torch.float64))
        opt = torch.optim.AdamW([a, b], lr=0.01, weight_decay=0.01, foreach=False)
        for step in range(steps + 1):
            delta = mod(torch.eye(cols, dtype=torch.float64)).T if arm_index == 2 else b @ a
            loss = ((delta - target) ** 2).mean()
            n = delta.norm().item()
            leak = (basis @ delta).norm().item() / n if n else 0.0
            expected = arm["trajectory"][step]
            assert loss.item() == pytest.approx(expected["loss"], abs=2e-12)
            assert leak == pytest.approx(expected["relative_leakage"], abs=2e-12)
            assert n == pytest.approx(expected["delta_fro"], abs=2e-12)
            if step == steps: break
            opt.zero_grad()
            loss.backward()
            if arm_index == 1:
                with torch.no_grad():
                    for _ in range(2): b.grad.sub_(basis.T @ (basis @ b.grad))
            opt.step()
