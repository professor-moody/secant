//! Rank-k spectral energy and vocabulary coherence for `secant audit-trojan
//! --rank-k <K>` (K in {1, 2, 4, 8}).
//!
//! A rank-1 edit puts all of `Delta W` in one singular direction, but a
//! trojan inserted as a rank-k adapter (or a k-token backdoor) spreads its
//! energy over k directions, so `sigma_1^2 / ||M||_F^2` understates it. The
//! cumulative top-k energy
//!
//! ```text
//! E_k(M) = sum_{i=1..k} sigma_i^2 / ||M||_F^2
//! ```
//!
//! is the fraction of the matrix `M` (the tensor `W`, or `W - W_ref` under
//! `--reference`, or `B A` of a LoRA adapter) that the leading k directions
//! explain. `E_1` is the existing `sigma1_energy_fraction`, bit for bit.
//!
//! The reader's sigmas are Rayleigh quotients of mutually orthonormal
//! iterates (power iteration with deflation), so by the Ky Fan maximum
//! principle the computed `E_k` is never above the true `E_k` beyond
//! rounding: an unconverged spectrum under-states concentration, never
//! invents it. `converged` says whether every computed sigma met the
//! reader's tolerance.
//!
//! Vocabulary coherence generalises "top promoted tokens of `u_1`" to the
//! rank-k output subspace `Q = span(u_1..u_k)`. For a vocabulary
//! (unembedding) row `e_t`
//!
//! ```text
//! c_t = ||Q^T e_t||^2 / ||e_t||^2  in [0, 1]
//! ```
//!
//! is the fraction of the token's direction that lies inside the subspace; a
//! random token sits near `k / d_model` (`chance_level`). It does not depend
//! on the sign or the basis chosen inside the subspace (the singular vectors
//! beyond `u_1` are only defined up to sign), which a per-direction list does.
//! Per-direction signed projections `z_i = E u_i` are reported beside it. The
//! `u_i` are re-orthonormalised before projecting because `u_i = W v_i /
//! sigma_i` is only orthogonal for converged singular vectors.
//!
//! Honest limit: trained unembeddings are anisotropic, so coherence is a
//! lead to compare against sibling layers or the base checkpoint, never a
//! proof of a backdoor.

use crate::error::Result;
use crate::mp::{sibling_stats, Sibling};
use crate::read::{dot_nf, norm2_nf, VecEntry};
use serde::Serialize;

/// The accepted `--rank-k` values.
pub const RANK_K_CHOICES: [usize; 4] = [1, 2, 4, 8];

/// Tokens listed per tensor by subspace coherence.
const COHERENT_TOKENS: usize = 16;
/// Promoted tokens listed per singular direction.
const DIRECTION_TOKENS: usize = 8;
/// `E_k` at or above `1 - EXHAUSTED_TOL` means the computed spectrum is the whole matrix (the reader's
/// sigmas carry up to ~2e-10 relative error, so this sits above that and far below any real bulk).
pub const EXHAUSTED_TOL: f64 = 1e-8;

/// A vector whose residual after projecting out the earlier ones is below
/// this (they are unit length on entry) is linearly dependent, not a new axis.
const ORTHO_FLOOR: f64 = 1e-8;

/// `--rank-k` validation: 0 means off; otherwise one of 1, 2, 4, 8.
pub fn validate_rank_k(k: usize) -> Result<()> {
    if k != 0 && !RANK_K_CHOICES.contains(&k) {
        return Err(crate::error::Error::msg(format!("rank-k must be one of 1, 2, 4, 8 (got {})", k)));
    }
    Ok(())
}

pub fn rule_text() -> String {
    "E_k = sum_{i<=k} sigma_i^2 / ||M||_F^2 with M = W (single file), W - W_ref (--reference) or B A (LoRA adapter); sigma_i are Rayleigh quotients of orthonormal iterates, so a computed E_k never exceeds the true E_k beyond rounding (Ky Fan); c_t = ||Q^T e_t||^2 / ||e_t||^2 over the re-orthonormalised top-k left singular vectors Q; chance level is k/d_model. Leads to compare against sibling layers or the base checkpoint, not proof of tampering"
        .into()
}

/// One point of the cumulative energy curve.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EnergyPoint {
    pub k: usize,
    pub energy: f64,
}

/// One singular direction's strongest signed token projections.
#[derive(Debug, Clone, Serialize)]
pub struct DirectionTokens {
    /// 1-based singular rank.
    pub rank: usize,
    pub sigma: f64,
    pub promoted: Vec<VecEntry>,
}

/// Rank-k vocabulary coherence of one tensor (see the module docs).
#[derive(Debug, Clone, Default, Serialize)]
pub struct RankKVocab {
    /// `ok`, or `skipped: <reason>` / `undefined: <reason>`.
    pub status: String,
    #[serde(skip_serializing_if = "is_zero")]
    pub vocab_size: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub d_model: usize,
    /// Orthonormal directions actually used (<= the computed rank).
    #[serde(skip_serializing_if = "is_zero")]
    pub basis_rank: usize,
    /// Expected `c_t` of a random token: `basis_rank / d_model`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chance_level: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_coherence: Option<f64>,
    /// `top_coherence / chance_level`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_over_chance: Option<f64>,
    /// The most coherent tokens, `value` = `c_t` (descending, ties by index).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tokens: Vec<VecEntry>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub per_direction: Vec<DirectionTokens>,
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}

/// Rank-k block of one tensor.
#[derive(Debug, Clone, Serialize)]
pub struct RankK {
    /// `ok`, or `undefined: <reason>` (never a number computed from nothing).
    pub status: String,
    /// `w`, `delta_w` or `lora_ba`: what the energy is a fraction of.
    pub basis: String,
    pub requested_k: usize,
    /// Singular values actually available (a short spectrum is exhausted, not zero-padded).
    pub k_computed: usize,
    /// `||M||_F^2`.
    pub frobenius_sq: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sigmas: Vec<f64>,
    /// `E_k` at `k_computed` (== at `requested_k`: the rest is below the rank floor).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cumulative_energy: Option<f64>,
    /// `E_k` at each of 1, 2, 4, 8 not above `requested_k`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub prefix_energy: Vec<EnergyPoint>,
    /// Every computed sigma met the reader's convergence tolerance.
    pub converged: bool,
    /// The computed spectrum explains all of `||M||_F^2` (to [`EXHAUSTED_TOL`]): `E_k` is
    /// 1 by construction at `k_computed`, so it cannot tell tensors apart there.
    pub exhausted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab: Option<RankKVocab>,
}

/// `E_k` from the computed sigmas and the total energy. `energy_at(k)` sums
/// the first `min(k, len)` squares in order, so `energy_at(1)` is exactly
/// `sigma_1 * sigma_1 / frob_sq` (the `sigma1_energy_fraction` formula).
fn energy_at(sigmas: &[f64], frob_sq: f64, k: usize) -> f64 {
    let mut s = 0f64;
    for x in sigmas.iter().take(k) {
        let sq = x * x;
        s += sq;
    }
    s / frob_sq.max(f64::MIN_POSITIVE)
}

/// Build the energy half of a [`RankK`] block. `sigmas` is the computed
/// descending spectrum (at least `requested_k` long unless exhausted).
pub fn energy_report(basis: &str, requested_k: usize, sigmas: &[f64], converged: bool, frob_sq: f64) -> RankK {
    let mut r = RankK {
        status: "ok".into(),
        basis: basis.into(),
        requested_k,
        k_computed: sigmas.len().min(requested_k),
        frobenius_sq: frob_sq,
        sigmas: sigmas.iter().take(requested_k).cloned().collect(),
        cumulative_energy: None,
        prefix_energy: vec![],
        converged,
        exhausted: false,
        vocab: None,
    };
    if frob_sq.is_nan() || frob_sq.is_infinite() || frob_sq <= 0.0 || sigmas.is_empty() {
        r.status = "undefined: zero or non-finite energy, or no computed singular value".into();
        r.sigmas.clear();
        r.k_computed = 0;
        return r;
    }
    let total = energy_at(sigmas, frob_sq, requested_k);
    r.cumulative_energy = Some(total);
    r.exhausted = total >= 1.0 - EXHAUSTED_TOL;
    for k in RANK_K_CHOICES.iter().filter(|k| **k <= requested_k) {
        r.prefix_energy.push(EnergyPoint { k: *k, energy: energy_at(sigmas, frob_sq, *k) });
    }
    r
}

/// A degenerate rank-k block for a tensor with no usable spectrum.
pub fn undefined(basis: &str, requested_k: usize, why: &str) -> RankK {
    RankK {
        status: format!("undefined: {}", why),
        basis: basis.into(),
        requested_k,
        k_computed: 0,
        frobenius_sq: 0.0,
        sigmas: vec![],
        cumulative_energy: None,
        prefix_energy: vec![],
        converged: false,
        exhausted: false,
        vocab: None,
    }
}

/// Modified Gram-Schmidt with one re-orthogonalisation pass. Vectors that are
/// linearly dependent on the earlier ones (residual below [`ORTHO_FLOOR`]) are
/// dropped, so the result spans the same subspace with an orthonormal basis.
pub fn orthonormalize(vs: &[&[f64]]) -> Vec<Vec<f64>> {
    let mut q: Vec<Vec<f64>> = Vec::new();
    for v in vs {
        let n0 = norm2_nf(v);
        if n0 == 0.0 || !n0.is_finite() {
            continue;
        }
        let mut x: Vec<f64> = v.iter().map(|a| a / n0).collect();
        for _pass in 0..2 {
            for b in &q {
                let p = dot_nf(&x, b);
                for i in 0..x.len() {
                    let d = p * b[i];
                    x[i] -= d;
                }
            }
        }
        let n = norm2_nf(&x);
        if n < ORTHO_FLOOR {
            continue;
        }
        for a in x.iter_mut() {
            *a /= n;
        }
        q.push(x);
    }
    q
}

fn top_by(values: &[f64], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|a, b| values[*b].partial_cmp(&values[*a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
    idx.truncate(n.min(values.len()));
    idx
}

/// Rank-k coherence of a vocabulary against the left singular directions
/// `dirs` (`(sigma_i, u_i)` for i = 1..k, descending). `row(t, buf)` fills
/// `buf[..d_model]` with token `t`'s finite unembedding row.
pub fn vocab_coherence<F>(mut row: F, vocab: usize, d_model: usize, dirs: &[(f64, Vec<f64>)]) -> Result<RankKVocab>
where
    F: FnMut(usize, &mut [f64]) -> Result<()>,
{
    let mut out = RankKVocab { status: "ok".into(), vocab_size: vocab, d_model, ..Default::default() };
    let usable: Vec<&(f64, Vec<f64>)> = dirs.iter().filter(|(_, u)| u.len() == d_model).collect();
    if usable.len() != dirs.len() || usable.is_empty() {
        out.status = format!("skipped: u_i lives in a different dimension than the vocab tensor's model dim {}", d_model);
        return Ok(out);
    }
    let basis = orthonormalize(&usable.iter().map(|(_, u)| u.as_slice()).collect::<Vec<_>>());
    if basis.is_empty() || vocab == 0 {
        out.status = "undefined: no non-zero singular direction or an empty vocabulary".into();
        return Ok(out);
    }
    out.basis_rank = basis.len();
    let mut coh = vec![0f64; vocab];
    let mut z: Vec<Vec<f64>> = vec![vec![0f64; vocab]; usable.len()];
    let mut buf = vec![0f64; d_model];
    for t in 0..vocab {
        row(t, &mut buf)?;
        let en = dot_nf(&buf, &buf);
        for (i, (_, u)) in usable.iter().enumerate() {
            z[i][t] = dot_nf(&buf, u);
        }
        if en > 0.0 {
            let mut s = 0f64;
            for b in &basis {
                let p = dot_nf(&buf, b);
                let sq = p * p;
                s += sq;
            }
            coh[t] = (s / en).min(1.0);
        }
    }
    let chance = basis.len() as f64 / d_model as f64;
    out.chance_level = Some(chance);
    let top = top_by(&coh, COHERENT_TOKENS);
    out.top_coherence = top.first().map(|&i| coh[i]);
    out.top_over_chance = out.top_coherence.map(|c| c / chance);
    out.tokens = top.iter().map(|&i| VecEntry { index: i, value: coh[i] }).collect();
    for (i, (sigma, _)) in usable.iter().enumerate() {
        out.per_direction.push(DirectionTokens {
            rank: i + 1,
            sigma: *sigma,
            promoted: top_by(&z[i], DIRECTION_TOKENS).into_iter().map(|t| VecEntry { index: t, value: z[i][t] }).collect(),
        });
    }
    Ok(out)
}

/// One row of a report-level rank-k ranking.
#[derive(Debug, Clone, Serialize)]
pub struct RankKOutlier {
    pub tensor: String,
    pub requested_k: usize,
    pub k_computed: usize,
    /// `E_k` at `requested_k`.
    pub cumulative_energy: f64,
    /// `E_1`, for reading the curve without the per-tensor block.
    pub sigma1_energy: f64,
    /// The k this ranking orders and judges siblings by: `requested_k`, unless
    /// some computed spectrum is exhausted there (`E_k` = 1 for a rank-4
    /// adapter at k = 4, which says nothing), in which case the largest
    /// choice below the shortest exhausted spectrum. One k for the whole report.
    pub ranking_k: usize,
    /// `E_k` at `ranking_k`: the number the order and the sibling statistics use.
    pub ranking_energy: f64,
    /// `ranking_energy` is 1 (to [`EXHAUSTED_TOL`]): every direction is inside k,
    /// so this row cannot be told apart from another saturated one.
    pub saturated: bool,
    pub converged: bool,
    /// `E_{ranking_k}` against the same tensor in the other layers.
    pub sibling: Sibling,
    /// Most coherent tokens of the rank-k subspace (with a vocabulary).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub coherent_tokens: Vec<VecEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_over_chance: Option<f64>,
}

/// Number of coherent tokens copied into a ranking row.
const RANKING_TOKENS: usize = 8;

/// The k at which `r` still discriminates: `requested_k`, unless its spectrum
/// is exhausted, then the largest choice strictly below `k_computed` (1 when none).
fn informative_k(r: &RankK) -> usize {
    if r.exhausted {
        RANK_K_CHOICES.iter().rev().copied().find(|k| *k < r.k_computed).unwrap_or(1)
    } else {
        r.requested_k
    }
}

fn energy_for(r: &RankK, k: usize) -> Option<f64> {
    r.prefix_energy.iter().find(|p| p.k == k).map(|p| p.energy)
}

/// Rank tensors by the energy at one report-wide informative k (descending,
/// ties by name), each judged against its layer family (`mp::family_key`)
/// like the MP spike ratio. That k is `requested_k` unless some tensor's
/// computed spectrum is exhausted at it (see [`RankKOutlier::ranking_k`]).
pub fn ranking<'a>(items: impl Iterator<Item = (&'a str, &'a RankK)>) -> Vec<RankKOutlier> {
    let items: Vec<(&str, &RankK)> = items.filter(|(_, r)| r.cumulative_energy.is_some()).collect();
    let key_k = items.iter().map(|(_, r)| informative_k(r)).min().unwrap_or(1);
    let mut out: Vec<RankKOutlier> = items
        .iter()
        .filter_map(|(name, r)| {
            let e = r.cumulative_energy?;
            let ek = energy_for(r, key_k)?;
            Some(RankKOutlier {
                tensor: name.to_string(),
                requested_k: r.requested_k,
                k_computed: r.k_computed,
                cumulative_energy: e,
                sigma1_energy: r.prefix_energy.first().map(|p| p.energy).unwrap_or(e),
                ranking_k: key_k,
                ranking_energy: ek,
                saturated: ek >= 1.0 - EXHAUSTED_TOL,
                converged: r.converged,
                sibling: Sibling {
                    family: crate::mp::family_key(name),
                    family_size: 0,
                    family_median: None,
                    robust_z: None,
                    sibling_outlier: None,
                },
                coherent_tokens: match &r.vocab {
                    Some(v) if v.status == "ok" => v.tokens.iter().take(RANKING_TOKENS).cloned().collect(),
                    _ => vec![],
                },
                top_over_chance: r.vocab.as_ref().filter(|v| v.status == "ok").and_then(|v| v.top_over_chance),
            })
        })
        .collect();
    let mut families: std::collections::BTreeMap<String, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, o) in out.iter().enumerate() {
        families.entry(o.sibling.family.clone()).or_default().push(i);
    }
    for (fam, idx) in families {
        let vals: Vec<f64> = idx.iter().map(|&i| out[i].ranking_energy).collect();
        for (&i, s) in idx.iter().zip(sibling_stats(&fam, &vals)) {
            out[i].sibling = s;
        }
    }
    out.sort_by(|a, b| b.ranking_energy.partial_cmp(&a.ranking_energy).unwrap_or(std::cmp::Ordering::Equal).then(a.tensor.cmp(&b.tensor)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choices_are_validated() {
        for k in [0, 1, 2, 4, 8] {
            assert!(validate_rank_k(k).is_ok(), "{}", k);
        }
        for k in [3, 5, 6, 7, 9, 16, 64] {
            assert!(validate_rank_k(k).is_err(), "{}", k);
        }
    }

    #[test]
    fn energy_curve_prefixes_and_exhaustion() {
        // sigmas 4,3,0,0 over ||M||^2 = 25: E_1 = 16/25, E_2 = 1.
        let r = energy_report("w", 8, &[4.0, 3.0], true, 25.0);
        assert_eq!(r.status, "ok");
        assert_eq!(r.k_computed, 2);
        assert_eq!(r.cumulative_energy, Some(1.0));
        let pts: Vec<(usize, f64)> = r.prefix_energy.iter().map(|p| (p.k, p.energy)).collect();
        assert_eq!(pts, vec![(1, 16.0 / 25.0), (2, 1.0), (4, 1.0), (8, 1.0)]);
        // requested 2 lists only the points not above it
        let r = energy_report("w", 2, &[4.0, 3.0, 1.0], true, 26.0);
        assert_eq!(r.prefix_energy.iter().map(|p| p.k).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(r.sigmas, vec![4.0, 3.0]);
        // E_1 is exactly the sigma1_energy_fraction expression
        assert_eq!(r.prefix_energy[0].energy, 4.0f64 * 4.0 / 26.0f64.max(f64::MIN_POSITIVE));
    }

    #[test]
    fn zero_or_empty_spectrum_is_undefined_not_a_number() {
        for (s, f) in [(&[][..], 5.0), (&[1.0][..], 0.0), (&[1.0][..], f64::NAN), (&[1.0][..], f64::INFINITY)] {
            let r = energy_report("w", 4, s, true, f);
            assert!(r.status.starts_with("undefined"), "{:?}", r);
            assert!(r.cumulative_energy.is_none() && r.prefix_energy.is_empty() && r.k_computed == 0);
        }
        assert!(undefined("delta_w", 2, "identical").status.starts_with("undefined: identical"));
    }

    #[test]
    fn orthonormalize_drops_dependent_and_fixes_skew() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [1.0, 1e-3, 0.0, 0.0]; // nearly parallel to a, still independent
        let c = [2.0, 0.0, 0.0, 0.0]; // dependent on a
        let d = [0.0, 0.0, 0.0, 0.0]; // zero
        let q = orthonormalize(&[&a, &b, &c, &d]);
        assert_eq!(q.len(), 2);
        for (i, x) in q.iter().enumerate() {
            assert!((dot_nf(x, x) - 1.0).abs() < 1e-12);
            for y in &q[..i] {
                assert!(dot_nf(x, y).abs() < 1e-12);
            }
        }
    }

    fn rows(e: &[Vec<f64>]) -> impl FnMut(usize, &mut [f64]) -> Result<()> + '_ {
        move |t, buf: &mut [f64]| {
            buf.copy_from_slice(&e[t]);
            Ok(())
        }
    }

    #[test]
    fn coherence_finds_tokens_inside_the_subspace_and_ignores_sign() {
        // d_model 6, directions e0 and (-e1): a token on either axis (any sign) is fully coherent.
        let u1 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let u2 = vec![0.0, -1.0, 0.0, 0.0, 0.0, 0.0];
        let e = vec![
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0], // 0: orthogonal
            vec![0.3, 0.0, 0.0, 0.0, 0.0, 0.0], // 1: on u1 (small norm: coherence is scale-free)
            vec![0.0, 5.0, 0.0, 0.0, 0.0, 0.0], // 2: on -u2's axis
            vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0], // 3: half inside
            vec![0.0; 6],                       // 4: zero row
        ];
        let v = vocab_coherence(rows(&e), 5, 6, &[(3.0, u1), (2.0, u2)]).unwrap();
        assert_eq!(v.status, "ok");
        assert_eq!(v.basis_rank, 2);
        assert!((v.chance_level.unwrap() - 2.0 / 6.0).abs() < 1e-15);
        let toks: Vec<(usize, f64)> = v.tokens.iter().map(|t| (t.index, t.value)).collect();
        assert_eq!(toks[0].0, 1);
        assert_eq!(toks[1].0, 2);
        assert!((toks[0].1 - 1.0).abs() < 1e-12 && (toks[1].1 - 1.0).abs() < 1e-12);
        assert!((toks[2].1 - 0.5).abs() < 1e-12 && toks[2].0 == 3);
        // zero row: coherence 0, never NaN
        assert!(toks.iter().all(|(_, c)| c.is_finite()));
        assert_eq!(toks.last().unwrap().1, 0.0);
        assert!((v.top_over_chance.unwrap() - 3.0).abs() < 1e-9);
        // per-direction signed projections (z_t = e_t . u_i): token 3 has the largest e0 component
        // (1.0) then token 1 (0.3); u2 = -e1 suppresses token 2 (z = -5), it is never promoted
        assert_eq!(v.per_direction[0].promoted[0].index, 3);
        assert_eq!(v.per_direction[0].promoted[1].index, 1);
        assert_eq!(v.per_direction[1].rank, 2);
        assert!(v.per_direction[1].promoted[0].value >= 0.0);
        assert_ne!(v.per_direction[1].promoted[0].index, 2);
    }

    #[test]
    fn skewed_directions_are_orthonormalised_before_projecting() {
        // Two nearly parallel (skewed) directions span a plane; a token in the
        // plane is fully coherent, which a plain sum of squared projections
        // onto the raw skewed vectors would overstate (>1) or understate.
        let u1 = vec![1.0, 0.0, 0.0];
        let s = (0.5f64).sqrt();
        let u2 = vec![s, s, 0.0];
        let e = vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let v = vocab_coherence(rows(&e), 2, 3, &[(2.0, u1), (1.0, u2)]).unwrap();
        assert_eq!(v.basis_rank, 2);
        assert!((v.tokens[0].value - 1.0).abs() < 1e-12 && v.tokens[0].index == 0);
        assert_eq!(v.tokens[1].value, 0.0);
    }

    #[test]
    fn dimension_mismatch_is_skipped_not_guessed() {
        let e = vec![vec![1.0, 0.0]];
        let v = vocab_coherence(rows(&e), 1, 2, &[(1.0, vec![1.0, 0.0, 0.0])]).unwrap();
        assert!(v.status.starts_with("skipped"), "{}", v.status);
        assert!(v.tokens.is_empty() && v.chance_level.is_none());
    }

    #[test]
    fn exhausted_spectra_are_ranked_at_an_informative_k_not_at_one() {
        // Rank-4 adapters scanned at k = 4: E_4 is 1 for every module, so ordering by it is noise.
        // The informative k is the largest choice below the spectrum length (2); E_2 separates them.
        let sig = |s: [f64; 4]| {
            let f: f64 = s.iter().map(|x| x * x).sum();
            energy_report("lora_ba", 4, &s, true, f)
        };
        let rs = [
            sig([1.0, 0.9, 0.8, 0.7]),
            sig([1.1, 0.9, 0.8, 0.7]),
            sig([1.0, 0.95, 0.8, 0.7]),
            sig([1.0, 0.9, 0.85, 0.7]),
            sig([9.0, 0.3, 0.2, 0.1]),
        ];
        assert!(rs.iter().all(|r| r.exhausted && r.k_computed == 4));
        assert!(rs.iter().all(|r| (r.cumulative_energy.unwrap() - 1.0).abs() < 1e-12));
        let names = ["m.0.w", "m.1.w", "m.2.w", "m.3.w", "m.4.w"];
        let rk = ranking(names.iter().cloned().zip(rs.iter()));
        assert!(rk.iter().all(|r| r.ranking_k == 2 && r.requested_k == 4), "{:?}", rk.iter().map(|r| r.ranking_k).collect::<Vec<_>>());
        assert_eq!(rk[0].tensor, "m.4.w");
        assert_eq!(rk[0].sibling.sibling_outlier, Some(true));
        assert!(rk.iter().all(|r| !r.saturated));
        assert!(rk[0].ranking_energy > rk[1].ranking_energy + 0.3);
        // a rank-1 spectrum has nothing below k = 1: ranked at 1 and labelled saturated, not hidden
        let r1 = energy_report("lora_ba", 4, &[2.0], true, 4.0);
        let rk = ranking([("a.0.w", &r1), ("a.1.w", &r1), ("a.2.w", &r1), ("a.3.w", &r1)].into_iter());
        assert!(rk.iter().all(|r| r.ranking_k == 1 && r.saturated));
        // a non-exhausted tensor in the same report does not lower k below an exhausted one's informative k
        let wide = energy_report("w", 4, &[3.0, 2.0, 1.5, 1.0], true, 100.0);
        assert!(!wide.exhausted);
        let rk = ranking([("x", &wide), ("y", &rs[0])].into_iter());
        assert!(rk.iter().all(|r| r.ranking_k == 2));
    }

    #[test]
    fn ranking_orders_by_energy_and_judges_siblings() {
        let mk = |e: f64| energy_report("w", 2, &[e.sqrt(), 0.0f64.max(0.0)], true, 1.0);
        let names = ["blk.0.w", "blk.1.w", "blk.2.w", "blk.3.w", "blk.4.w", "other"];
        let rs: Vec<RankK> = [0.10, 0.12, 0.11, 0.13, 0.9, 0.5].iter().map(|e| mk(*e)).collect();
        let rk = ranking(names.iter().cloned().zip(rs.iter()));
        assert_eq!(rk[0].tensor, "blk.4.w");
        assert_eq!(rk[0].sibling.sibling_outlier, Some(true));
        assert_eq!(rk[0].sibling.family_size, 5);
        let other = rk.iter().find(|r| r.tensor == "other").unwrap();
        assert_eq!(other.sibling.family_size, 1);
        assert!(other.sibling.sibling_outlier.is_none());
        assert!(rk.windows(2).all(|w| w[0].cumulative_energy >= w[1].cumulative_energy));
    }
}
