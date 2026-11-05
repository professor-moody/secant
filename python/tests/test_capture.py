"""Contrastive-direction capture: analytic checks (numpy) and a hook test (torch)."""

import math
import os

import numpy as np
import pytest

from secant import capture
from secant.capture import capture_contrastive_directions, evaluate_cell, write_vectors

# ---- analytic setup: pos ~ N(+g/2 e1, s^2 I), neg ~ N(-g/2 e1, s^2 I)
D_DIM, N_CLS, G, S = 32, 20000, 2.0, 1.0


def _gauss_pair(seed=7, d=D_DIM, n=N_CLS, g=G, s=S):
    rng = np.random.default_rng(seed)
    delta = np.zeros(d)
    delta[0] = g
    pos = rng.standard_normal((n, d)) * s + delta / 2
    neg = rng.standard_normal((n, d)) * s - delta / 2
    return pos, neg


def test_known_gaussian_snr_and_cosine_within_stated_tolerances():
    pos, neg = _gauss_pair()
    r = capture_contrastive_directions(pos, neg, stream="toy", layer=3, seed=11)[("toy", 3)]
    assert r.status == "ok" and r.reason == ""
    nh = N_CLS // 2
    sigma2 = S**2 * (1 / nh + 1 / nh)  # per-coordinate variance of a half's mean difference
    # Analytic expectations for large n (derivations in the module docstring):
    snr_proj_expect = G**2 / (2 * S**2) * (G**2 / (G**2 + D_DIM * sigma2))  # = g^2/(2 s^2) up to cos^2(u, delta)
    snr_trace_expect = (G**2 + D_DIM * sigma2) / (2 * D_DIM * S**2)
    cos_expect = G**2 / (G**2 + D_DIM * sigma2)
    # Tolerances: 5% relative on the SNRs (>= 3 sampling sigma at this n), 0.01 absolute on the cosine.
    assert r.snr_proj == pytest.approx(snr_proj_expect, rel=0.05)
    assert r.snr_proj_a_to_b == pytest.approx(snr_proj_expect, rel=0.07)
    assert r.snr_proj_b_to_a == pytest.approx(snr_proj_expect, rel=0.07)
    assert r.snr_proj == pytest.approx(0.5 * (r.snr_proj_a_to_b + r.snr_proj_b_to_a), rel=1e-12)
    assert r.snr_trace == pytest.approx(snr_trace_expect, rel=0.05)
    assert abs(r.split_half_cosine - cos_expect) < 0.01
    # direction: unit norm to 1e-15, aligned with the true e1
    assert abs(np.linalg.norm(r.direction) - 1.0) < 1e-14
    assert r.direction[0] > 0.999
    assert abs(np.linalg.norm(r.direction_a) - 1.0) < 1e-14 and abs(np.linalg.norm(r.direction_b) - 1.0) < 1e-14
    assert (r.n_pos, r.n_neg, r.dim, r.seed) == (N_CLS, N_CLS, D_DIM, 11)


def test_snr_trace_is_direction_free_snr_proj_is_not():
    # Same delta norm spread over 8 axes: snr_trace is unchanged, direction recovered anyway.
    rng = np.random.default_rng(8)
    d, n = 32, 20000
    u = np.zeros(d)
    u[:8] = 1 / math.sqrt(8)
    pos = rng.standard_normal((n, d)) + G * u / 2
    neg = rng.standard_normal((n, d)) - G * u / 2
    r = evaluate_cell(pos, neg)
    assert float(r.direction @ u) > 0.999
    assert r.snr_trace == pytest.approx(((G**2) + d * 2 / (n // 2)) / (2 * d), rel=0.05)


def test_degenerate_identical_distributions_snr_near_zero_cosine_random():
    rng = np.random.default_rng(9)
    d, n = 256, 2000
    pos, neg = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    r = evaluate_cell(pos, neg, seed=3)
    assert r.status == "ok"  # a noise direction exists; the metrics are what say it is noise
    # expected values: snr_proj ~ chi2_1 * 0.001, snr_trace ~ 0.001 (see test analysis); bounds are >= 20x / 2x the means
    assert 0.0 <= r.snr_proj < 0.02
    assert r.snr_trace < 0.002
    assert abs(r.split_half_cosine) < 5 / math.sqrt(d)  # |cos| of independent noise ~ N(0, 1/d)
    # and the contrast with a real signal is large
    sp, sn = _gauss_pair(d=d, n=n, g=2.0)
    real = evaluate_cell(sp, sn, seed=3)
    assert real.snr_proj > 50 * r.snr_proj and real.split_half_cosine > 0.5


def test_exactly_identical_classes_are_undetermined_not_a_number():
    rng = np.random.default_rng(10)
    x = rng.standard_normal((100, 16))
    r = evaluate_cell(x, x.copy())
    assert r.status == "undetermined" and "zero" in r.reason
    assert np.all(np.isnan(r.direction))
    with pytest.raises(ValueError, match="undetermined"):
        write_vectors(os.devnull + "_unused", {("default", 0): r})


def test_constant_features_zero_variance_is_undetermined():
    pos = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1))
    neg = np.tile(np.array([-1.0, 0.0, 0.0]), (10, 1))
    r = evaluate_cell(pos, neg)
    assert r.status == "undetermined" and math.isnan(r.snr_proj) and math.isnan(r.snr_trace)


def test_held_out_scores_expose_overfitting_that_in_sample_scores_hide():
    rng = np.random.default_rng(12)
    d, n = 500, 12  # more dimensions than samples, NO real signal
    pos, neg = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    u = capture._diff_direction(pos, neg)
    in_sample = capture._snr_proj(u, pos, neg)
    r = evaluate_cell(pos, neg)
    assert in_sample > 5.0  # the fitted direction separates the training data perfectly
    assert r.snr_proj < 1.0  # but not data it has not seen
    assert abs(r.split_half_cosine) < 0.3


def test_deterministic_and_independent_of_cell_order():
    pos, neg = _gauss_pair(n=400)
    cells = {("a", 1): (pos, neg), ("b", 2): (neg, pos)}
    r1 = capture_contrastive_directions(activations=cells, seed=5)
    r2 = capture_contrastive_directions(activations=dict(reversed(list(cells.items()))), seed=5)
    for k in cells:
        assert r1[k].metrics() == r2[k].metrics()
        assert np.array_equal(r1[k].direction, r2[k].direction)
    r3 = capture_contrastive_directions(activations=cells, seed=6)
    assert r3[("a", 1)].split_half_cosine != r1[("a", 1)].split_half_cosine
    # swapping classes flips the direction and leaves the unsigned metrics alone
    assert np.allclose(r1[("a", 1)].direction, -evaluate_cell(neg, pos, stream="a", layer=1, seed=5).direction)


def test_input_validation():
    pos, neg = _gauss_pair(n=50, d=8)
    with pytest.raises(ValueError):
        capture_contrastive_directions(pos)  # neg missing
    with pytest.raises(ValueError):
        capture_contrastive_directions(pos, neg, activations={("a", 0): (pos, neg)})  # two modes
    with pytest.raises(ValueError):
        evaluate_cell(pos[:3], neg[:3])  # too few
    with pytest.raises(ValueError):
        evaluate_cell(pos, neg[:, :4])  # dim mismatch
    bad = pos.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate_cell(bad, neg)


def test_write_vectors_npy_v1_roundtrip_and_refusals(tmp_path):
    pos, neg = _gauss_pair(n=200, d=16)
    res = capture_contrastive_directions(activations={("resid", 12): (pos, neg), ("mlp-x", 3): (pos, neg)}, seed=1)
    out = tmp_path / "vecs"
    written = write_vectors(out, res)
    assert sorted(os.path.basename(p) for p in written.values()) == ["d_mlp-x_L3.npy", "d_resid_L12.npy"]
    for (stream, layer), path in written.items():
        raw = open(path, "rb").read()
        assert raw[:8] == b"\x93NUMPY\x01\x00"  # magic + format version 1.0
        hlen = int.from_bytes(raw[8:10], "little")
        assert (10 + hlen) % 64 == 0  # numpy's alignment
        header = raw[10 : 10 + hlen].decode("latin1")
        assert "'descr': '<f8'" in header and "'fortran_order': False" in header and "'shape': (16,)" in header
        back = np.load(path, allow_pickle=False)
        assert back.dtype == np.dtype("<f8") and back.shape == (16,)
        assert np.array_equal(back, res[(stream, layer)].direction)
    with pytest.raises(FileExistsError):
        write_vectors(out, res)
    write_vectors(out, res, overwrite=True)
    # a bad token anywhere writes nothing at all
    fresh = tmp_path / "fresh"
    bad = dict(res)
    bad[("../evil", 0)] = evaluate_cell(pos, neg, stream="../evil", layer=0)
    with pytest.raises(ValueError, match="token"):
        write_vectors(fresh, bad)
    assert not fresh.exists()


# ---- hook mode (needs torch)


def test_hook_mode_recovers_known_direction_and_cleans_up():
    torch = pytest.importorskip("torch", reason="SKIP-NO-TORCH: hook capture test needs torch")
    from torch import nn

    class Block(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.lin = nn.Linear(d_in, d_out, bias=False)

        def forward(self, x):
            return (self.lin(x),)  # tuple output, as many transformer blocks return

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(6, 8), Block(8, 8)])
            self.unused = nn.Linear(2, 2)

        def forward(self, x):
            for blk in self.layers:
                x = blk(x)[0]
            return x

    torch.manual_seed(0)
    model = Tiny().train()
    n, T = 96, 3
    g = torch.Generator().manual_seed(1)
    base = torch.randn(2 * n, T, 6, generator=g) * 0.3
    pos = [base[i].clone() for i in range(n)]
    neg = [base[n + i].clone() for i in range(n)]
    for t in pos:
        t[-1, 0] += 3.0  # signal only at the LAST position, input channel 0
    hp = [("h", 0, "layers.0"), ("h", 1, model.layers[1])]
    res = capture_contrastive_directions(model=model, pos_prompts=pos, neg_prompts=neg, hook_points=hp, batch_size=7, position=-1, seed=2)
    assert set(res) == {("h", 0), ("h", 1)}
    w0 = model.layers[0].lin.weight.detach().double().numpy()  # [8, 6]
    expect0 = w0[:, 0] / np.linalg.norm(w0[:, 0])
    r0 = res[("h", 0)]
    assert r0.status == "ok" and float(r0.direction @ expect0) > 0.99
    assert r0.snr_proj > 5.0 and r0.split_half_cosine > 0.9
    expect1 = model.layers[1].lin.weight.detach().double().numpy() @ w0[:, 0]
    expect1 /= np.linalg.norm(expect1)
    assert float(res[("h", 1)].direction @ expect1) > 0.99
    # position 0 carries no signal: held-out metrics must say so
    res0 = capture_contrastive_directions(model=model, pos_prompts=pos, neg_prompts=neg, hook_points=hp[:1], position=0, seed=2)
    assert res0[("h", 0)].snr_proj < 0.5 * r0.snr_proj and res0[("h", 0)].snr_proj < 0.5
    # cleanup: no hooks left behind, training mode restored
    assert all(len(m._forward_hooks) == 0 for m in model.modules())
    assert model.training is True
    # a module off the forward path is an error, not an empty result
    with pytest.raises(RuntimeError, match="never fired"):
        capture_contrastive_directions(model=model, pos_prompts=pos, neg_prompts=neg, hook_points=[("u", 0, "unused")])
    assert all(len(m._forward_hooks) == 0 for m in model.modules())
