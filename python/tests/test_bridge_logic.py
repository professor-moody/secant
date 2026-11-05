"""Bridge logic tests against a fake executable: discovery, tri-state env,
argument construction, JSON handling. These say nothing about the real CLI
(see test_bridge_real.py)."""

import json
import os
import sys

import pytest

import secant
from secant import _bridge


@pytest.fixture(autouse=True)
def _isolate_discovery(monkeypatch, tmp_path):
    """No ambient SECANT_*, an empty PATH, a cwd without target/, and no source-tree workspace."""
    monkeypatch.delenv("SECANT_BIN", raising=False)
    monkeypatch.delenv("SECANT_LIB", raising=False)
    empty = tmp_path / "emptypath"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(_bridge, "_PKG_WORKSPACE", tmp_path / "no-workspace")
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


# ------------------------------------------------------------------ Part 1: discovery, three states


def test_env_absent_nothing_found_raises_not_found():
    with pytest.raises(secant.SecantNotFound, match="searched"):
        secant.find_binary()


def test_env_present_but_empty_raises_even_if_path_has_a_binary(monkeypatch, fake_secant):
    monkeypatch.setenv("PATH", str(fake_secant.parent))
    monkeypatch.setenv("SECANT_BIN", "")
    with pytest.raises(secant.SecantEnvError, match="empty"):
        secant.find_binary()
    with pytest.raises(secant.SecantEnvError):
        secant.run(["--version"])


def test_env_present_with_value_is_used_and_must_exist(monkeypatch, fake_secant, tmp_path):
    monkeypatch.setenv("SECANT_BIN", str(fake_secant))
    assert secant.find_binary() == fake_secant
    monkeypatch.setenv("SECANT_BIN", str(tmp_path / "nope"))
    with pytest.raises(secant.SecantEnvError, match="does not exist"):
        secant.find_binary()  # no silent fallback to PATH


def test_explicit_argument_beats_env_and_empty_explicit_raises(monkeypatch, fake_secant, tmp_path):
    monkeypatch.setenv("SECANT_BIN", str(tmp_path / "nope"))
    assert secant.find_binary(fake_secant) == fake_secant
    with pytest.raises(secant.SecantEnvError):
        secant.find_binary("")
    with pytest.raises(secant.SecantNotFound):
        secant.find_binary(tmp_path / "missing")


def test_env_absent_searches_path_then_cwd_target_release_before_debug(monkeypatch, fake_secant, tmp_path, _isolate_discovery):
    monkeypatch.setenv("PATH", str(fake_secant.parent))
    assert secant.find_binary() == fake_secant
    monkeypatch.setenv("PATH", str(tmp_path / "emptypath"))
    cwd = _isolate_discovery
    for sub in ("debug", "release"):
        d = cwd / "target" / sub
        d.mkdir(parents=True)
        (d / "secant").write_bytes(fake_secant.read_bytes())
        (d / "secant").chmod(0o755)
    assert secant.find_binary() == cwd / "target" / "release" / "secant"
    (cwd / "target" / "release" / "secant").unlink()
    assert secant.find_binary() == cwd / "target" / "debug" / "secant"


def test_library_discovery_same_three_states(monkeypatch, tmp_path, _isolate_discovery):
    with pytest.raises(secant.SecantNotFound):
        secant.find_library()
    monkeypatch.setenv("SECANT_LIB", "")
    with pytest.raises(secant.SecantEnvError):
        secant.find_library()
    fake = tmp_path / "libfake.so"
    fake.write_bytes(b"not a real library")
    monkeypatch.setenv("SECANT_LIB", str(fake))
    assert secant.find_library() == fake
    monkeypatch.delenv("SECANT_LIB")
    d = _isolate_discovery / "target" / "debug"
    d.mkdir(parents=True)
    name = "libsecant.dylib" if sys.platform == "darwin" else "libsecant.so"
    (d / name).write_bytes(b"x")
    assert secant.find_library() == d / name
    assert secant.find_library(fake) == fake


def test_loading_a_non_library_is_an_error_not_a_crash(tmp_path):
    bogus = tmp_path / "libbogus.so"
    bogus.write_bytes(b"garbage")
    with pytest.raises(secant.SecantError, match="could not load"):
        secant.load_library(bogus)


# ------------------------------------------------------------------ Part 1b: run() and wrappers against the fake


def test_run_returns_tuple_json_and_never_uses_a_shell(fake_secant):
    code, out, err = secant.run(["inspect", "--model", "a b; rm -rf x", "--json"], binary=fake_secant)
    assert code == 0 and err == "fake-stderr"
    res = secant.run(["inspect", "--model", "a b; rm -rf x", "--json"], binary=fake_secant)
    assert res.json == {"argv": ["inspect", "--model", "a b; rm -rf x", "--json"]}  # one argv element, unsplit
    assert res.json_error is None and res.ok and not res.gate_failed
    plain = secant.run(["x"], binary=fake_secant)
    assert plain.json is None and plain.json_error is None  # not asked for
    with pytest.raises(TypeError):
        secant.run("inspect --model x", binary=fake_secant)


def test_exit_codes_and_json_errors_are_reported_not_hidden(fake_secant, monkeypatch):
    monkeypatch.setenv("FAKE_SECANT_EXIT", "1")
    r = secant.run(["x", "--json"], binary=fake_secant)
    assert r.exit_code == 1 and r.gate_failed and not r.ok
    with pytest.raises(secant.SecantError):
        secant.run(["x"], binary=fake_secant, check=True)
    monkeypatch.setenv("FAKE_SECANT_STDOUT", "not json")
    bad = secant.run(["x", "--json"], binary=fake_secant)
    assert bad.json is None and "not valid JSON" in bad.json_error
    monkeypatch.setenv("FAKE_SECANT_STDOUT", "")
    assert secant.run(["x", "--json"], binary=fake_secant).json_error == "stdout is empty"


def test_helpers_raise_on_exit_2_but_return_gate_failures(fake_secant, monkeypatch, tmp_path):
    kw = dict(binary=fake_secant)
    monkeypatch.setenv("FAKE_SECANT_EXIT", "1")
    r = secant.verify_subspace("b", "h", "v", ["t1", "t2"], tol=1e-5, **kw)
    assert r.exit_code == 1  # a leak is an answer
    assert r.json["argv"] == ["verify-subspace", "--base", "b", "--healed", "h", "--vectors", "v", "--tensors", "t1,t2", "--tol", "1e-05", "--json"]
    monkeypatch.setenv("FAKE_SECANT_EXIT", "2")
    with pytest.raises(secant.SecantError, match="could not be determined"):
        secant.verify_subspace("b", "h", "v", "t1", **kw)
    assert secant.verify_subspace("b", "h", "v", "t1", strict=False, **kw).exit_code == 2
    monkeypatch.setenv("FAKE_SECANT_EXIT", "0")


def test_wrapper_argument_construction(fake_secant, tmp_path):
    kw = dict(binary=fake_secant)
    assert secant.inspect("m.gguf", **kw).json["argv"] == ["inspect", "--model", "m.gguf", "--json"]
    assert secant.inspect("m.gguf", json=False, **kw).json is None
    a = secant.read("m.st", tensor="blk.*", top_cols=8, top_k_singular=2, vocab_tensor="out.weight", detect_anomalies=True, vectors_out=tmp_path / "o", **kw).json["argv"]
    assert a == ["read", "--model", "m.st", "--tensor", "blk.*", "--top-cols", "8", "--top-k-singular", "2", "--vocab-tensor", "out.weight", "--detect-anomalies", "--vectors-out", str(tmp_path / "o"), "--json"]
    h = secant.heal("l.st", "vecs", "out.st", base="b.gguf", merge_out="m.gguf", scale=0.5, tol=1e-6, **kw).json["argv"]
    assert h == ["heal", "--lora", "l.st", "--vectors", "vecs", "--out", "out.st", "--base", "b.gguf", "--merge-out", "m.gguf", "--scale", "0.5", "--tol", "1e-06", "--json"]
    with pytest.raises(ValueError):
        secant.heal("l", "v", "o", base="b", **kw)
    c = secant.compose("p", "plan.json", vectors="v", target_shift=3.0, **kw).json["argv"]
    assert c == ["compose", "--profile", "p", "--vectors", "v", "--target-shift", "3.0", "--out", "plan.json", "--json"]
    assert secant.verify_patch("a", "b", expect_changed=["t1", "t2"], **kw).json["argv"] == ["verify-patch", "--base", "a", "--patched", "b", "--expect-changed", "t1,t2", "--json"]
    assert secant.patch("a", "p", "o", **kw).json["argv"] == ["patch", "--base", "a", "--plan", "p", "--out", "o", "--json"]
    r = secant.run(["--version"], go_arm64_fma=True, **kw)
    assert r.json is None and r.argv[1] == "--go-arm64-fma"


def test_timeout_raises(fake_secant, tmp_path):
    slow = tmp_path / "slowbin" / "secant"
    slow.parent.mkdir()
    slow.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    slow.chmod(0o755)
    with pytest.raises(secant.SecantError, match="timed out"):
        secant.run(["x"], binary=slow, timeout=0.5)
