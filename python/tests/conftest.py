"""Shared helpers. Tests that need the real binary or library skip with an
explicit reason (printed by `-ra`); they never pass vacuously."""

from __future__ import annotations

import os
import stat
import sys

import pytest

from secant import _bridge


def real_binary_or_skip():
    try:
        return _bridge.find_binary()
    except _bridge.SecantError as e:
        pytest.skip("SKIP-NO-BINARY: no secant binary (SECANT_BIN unset; PATH, ./target, workspace target searched)" if isinstance(e, _bridge.SecantNotFound) else f"SKIP-BAD-BINARY: {e}")


def real_library_or_skip():
    try:
        return _bridge.load_library()
    except _bridge.SecantError as e:
        pytest.skip("SKIP-NO-LIBRARY: no libsecant (SECANT_LIB unset; ./target, workspace target searched)" if isinstance(e, _bridge.SecantNotFound) else f"SKIP-BAD-LIBRARY: {e}")


@pytest.fixture
def fake_secant(tmp_path):
    """A stand-in `secant` executable (a Python script) that prints its argv as
    JSON on stdout and exits with $FAKE_SECANT_EXIT (default 0). It tests the
    bridge's own logic only; it says nothing about the real CLI."""
    if os.name != "posix":
        pytest.skip("SKIP-NOT-POSIX: fake executable needs a shebang")
    p = tmp_path / "fakebin" / "secant"
    p.parent.mkdir()
    p.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if os.environ.get('FAKE_SECANT_STDOUT') is not None:\n"
        "    sys.stdout.write(os.environ['FAKE_SECANT_STDOUT'])\n"
        "else:\n"
        "    json.dump({'argv': sys.argv[1:]}, sys.stdout)\n"
        "sys.stderr.write('fake-stderr')\n"
        "sys.exit(int(os.environ.get('FAKE_SECANT_EXIT', '0')))\n"
    )
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p
