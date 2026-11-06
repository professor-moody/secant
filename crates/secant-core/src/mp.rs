//! Marchenko-Pastur bulk edge for `secant read --marchenko-pastur` and
//! `secant audit-trojan` (Milestone S-2).
//!
//! For a `d_out x d_in` matrix with i.i.d. zero-mean entries of standard
//! deviation `s`, the singular values fill a bulk whose upper edge is
//! `s * (sqrt(d_out) + sqrt(d_in))`. Written as in the roadmap:
//!
//! ```text
//! sigma_bulk = s * sqrt(d_out)
//! sigma_MP   = sigma_bulk * (1 + sqrt(d_in / d_out))
//! ```
//!
//! A rank-1 edit `theta * u v^T` strong enough to matter (and many trojan
//! insertions) detaches a singular value from that bulk, so `sigma_1 /
//! sigma_MP` well above 1 is a spectral spike.
//!
//! `s` is estimated from the energy the computed spike does NOT explain:
//! `s^2 = (||W||_F^2 - sum of outlier sigma_i^2) / (d_out * d_in)`, iterated
//! until the outlier set is stable (removing an outlier lowers the edge, which
//! can expose the next one). Only the top-k singular values the reader
//! computed can be removed, so with more than k outliers the edge is an
//! over-estimate and the count is a lower bound (`spectrum_exhausted`).
//!
//! Honest limits: trained weights are not i.i.d., and ratios above 1 are common
//! in genuine checkpoints. The ratio is evidence for comparison (against
//! sibling layers, or against the base checkpoint before an edit), not a proof
//! of tampering. When the computed spectrum explains all of the energy (a
//! matrix of rank <= k, or a vector), no bulk remains and the result is
//! `undefined`, never a clean or flagged verdict.

use serde::Serialize;

/// Relative bulk energy below which no bulk can be told apart from the
/// computed spectrum's own error. The reader converges each sigma to 1e-13
/// relative with a 1e-10 residual, so each removed sigma^2 carries up to about
/// 2e-10 relative error and up to 64 are removed: anything under 1e-8 of the
/// total energy is estimation error (an exactly low-rank F32 matrix leaves
/// ~1e-16 of round-off), never a bulk to divide by.
pub const EMPTY_BULK: f64 = 1e-8;

#[derive(Debug, Clone, Serialize)]
pub struct SpectralBulk {
    /// `ok`, or `undefined: <reason>`.
    pub status: String,
    pub d_out: usize,
    pub d_in: usize,
    /// `d_in / d_out`.
    pub aspect_ratio: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sigma_entry: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sigma_bulk: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sigma_mp: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sigma1_over_mp: Option<f64>,
    /// Computed singular values above `sigma_mp`.
    pub outliers: usize,
    /// `sigma_i / sigma_mp` of each outlier, descending.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub outlier_ratios: Vec<f64>,
    /// Every computed singular value is an outlier: the true count may be larger.
    pub spectrum_exhausted: bool,
    /// Fraction of `||W||_F^2` left to the bulk.
    pub bulk_energy_fraction: f64,
    pub threshold: f64,
    /// `sigma1_over_mp > threshold` (false when undefined).
    pub flagged: bool,
    pub rule: String,
}

pub fn rule_text(threshold: f64) -> String {
    format!(
        "sigma_MP = sigma_bulk * (1 + sqrt(d_in/d_out)), sigma_bulk = sigma_entry * sqrt(d_out), sigma_entry^2 = (||W||_F^2 - sum of outlier sigma_i^2) / (d_out*d_in), outliers iterated to a fixed point among the computed singular values; flagged when sigma_1/sigma_MP > {}. i.i.d. null model: compare against sibling layers or the base checkpoint, not as proof of tampering",
        crate::numeric::go_fmt_g(threshold, None)
    )
}

/// Bulk edge of a `d_out x d_in` matrix with total energy `frob_sq` and
/// computed leading singular values `sigmas` (descending).
pub fn marchenko_pastur(frob_sq: f64, d_out: usize, d_in: usize, sigmas: &[f64], threshold: f64) -> SpectralBulk {
    let mut r = SpectralBulk {
        status: "ok".into(),
        d_out,
        d_in,
        aspect_ratio: if d_out > 0 { d_in as f64 / d_out as f64 } else { 0.0 },
        sigma_entry: None,
        sigma_bulk: None,
        sigma_mp: None,
        sigma1_over_mp: None,
        outliers: 0,
        outlier_ratios: vec![],
        spectrum_exhausted: false,
        bulk_energy_fraction: 0.0,
        threshold,
        flagged: false,
        rule: rule_text(threshold),
    };
    if d_out == 0 || d_in == 0 || sigmas.is_empty() || frob_sq.is_nan() || frob_sq <= 0.0 || frob_sq.is_infinite() {
        r.status = "undefined: empty matrix, zero energy or no computed singular value".into();
        return r;
    }
    let n = d_out as f64 * d_in as f64;
    let shape = 1.0 + (d_in as f64 / d_out as f64).sqrt();
    let edge_of = |energy: f64| (energy.max(0.0) / n).sqrt() * (d_out as f64).sqrt() * shape;
    let mut removed = 0usize;
    let mut energy = frob_sq;
    loop {
        let edge = edge_of(energy);
        let count = sigmas.iter().take_while(|&&s| s > edge).count();
        if count <= removed {
            break;
        }
        removed = count;
        energy = frob_sq - sigmas[..removed].iter().map(|s| s * s).sum::<f64>();
    }
    r.outliers = removed;
    r.spectrum_exhausted = removed == sigmas.len();
    r.bulk_energy_fraction = (energy / frob_sq).max(0.0);
    if removed >= d_out.min(d_in) || r.bulk_energy_fraction <= EMPTY_BULK {
        r.status =
            "undefined: the computed singular values explain all of the energy (to the spectrum's precision); no bulk remains to estimate"
                .into();
        r.outliers = removed;
        return r;
    }
    let entry = (energy.max(0.0) / n).sqrt();
    let bulk = entry * (d_out as f64).sqrt();
    let mp = bulk * shape;
    r.sigma_entry = Some(entry);
    r.sigma_bulk = Some(bulk);
    r.sigma_mp = Some(mp);
    r.sigma1_over_mp = Some(sigmas[0] / mp);
    r.outlier_ratios = sigmas[..removed].iter().map(|s| s / mp).collect();
    r.flagged = sigmas[0] / mp > threshold;
    r
}

/// Minimum family size for a sibling comparison.
pub const SIBLING_MIN: usize = 4;
/// Iglewicz-Hoaglin modified z-score cut-off.
pub const SIBLING_Z: f64 = 3.5;

/// The layer family of a tensor name: every dot-separated component that is
/// all digits becomes `*` (`blk.12.ffn_down.weight` -> `blk.*.ffn_down.weight`,
/// `model.layers.3.mlp.down_proj.weight` -> `model.layers.*.mlp.down_proj.weight`).
pub fn family_key(name: &str) -> String {
    name.split('.').map(|c| if !c.is_empty() && c.bytes().all(|b| b.is_ascii_digit()) { "*" } else { c }).collect::<Vec<_>>().join(".")
}

fn median(v: &[f64]) -> f64 {
    let mut c = v.to_vec();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = c.len();
    if n % 2 == 1 {
        c[n / 2]
    } else {
        (c[n / 2 - 1] + c[n / 2]) / 2.0
    }
}

/// Sibling comparison of one value against its family (itself included).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sibling {
    pub family: String,
    pub family_size: usize,
    /// Median ratio of the family; absent below `SIBLING_MIN` members.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family_median: Option<f64>,
    /// `0.6745 * (x - median) / MAD`; absent below `SIBLING_MIN` members or
    /// when the family's MAD is zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub robust_z: Option<f64>,
    /// `robust_z > 3.5`; with a zero MAD, any value above the median. `None`
    /// (serialized as null) when the family is too small to judge.
    pub sibling_outlier: Option<bool>,
}

/// Modified z-scores of `values` within one family (same order).
pub fn sibling_stats(family: &str, values: &[f64]) -> Vec<Sibling> {
    let n = values.len();
    if n < SIBLING_MIN {
        return values
            .iter()
            .map(|_| Sibling { family: family.into(), family_size: n, family_median: None, robust_z: None, sibling_outlier: None })
            .collect();
    }
    let med = median(values);
    let mad = median(&values.iter().map(|x| (x - med).abs()).collect::<Vec<_>>());
    values
        .iter()
        .map(|&x| {
            let (z, out) = if mad > 0.0 {
                let z = 0.6745 * (x - med) / mad;
                (Some(z), z > SIBLING_Z)
            } else {
                (None, x > med)
            };
            Sibling { family: family.into(), family_size: n, family_median: Some(med), robust_z: z, sibling_outlier: Some(out) }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::top_singular_triplets;

    #[test]
    fn families_and_sibling_outliers() {
        assert_eq!(family_key("blk.12.ffn_down.weight"), "blk.*.ffn_down.weight");
        assert_eq!(family_key("model.layers.3.mlp.down_proj.weight"), "model.layers.*.mlp.down_proj.weight");
        assert_eq!(family_key("output.weight"), "output.weight");
        assert_eq!(family_key("blk.1a.x"), "blk.1a.x");
        // genuine-like spread plus one planted layer
        let v = [2.4, 2.9, 3.1, 3.0, 2.7, 3.3, 9.5];
        let s = sibling_stats("f", &v);
        assert_eq!(s.iter().filter(|x| x.sibling_outlier == Some(true)).count(), 1);
        assert_eq!(s[6].sibling_outlier, Some(true));
        assert!(s[6].robust_z.unwrap() > 3.5);
        // too few siblings: undefined, never a verdict
        let s = sibling_stats("g", &[1.0, 50.0, 2.0]);
        assert!(s.iter().all(|x| x.sibling_outlier.is_none() && x.robust_z.is_none()));
        // zero MAD: only values above the shared median stand out
        let s = sibling_stats("h", &[2.0, 2.0, 2.0, 2.0, 5.0]);
        assert_eq!(
            s.iter().map(|x| x.sibling_outlier).collect::<Vec<_>>(),
            vec![Some(false), Some(false), Some(false), Some(false), Some(true)]
        );
    }

    fn uniform(seed: &mut u64) -> f64 {
        // splitmix64 -> [-1, 1)
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    fn noise(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..rows * cols).map(|_| uniform(&mut s)).collect()
    }

    fn spectrum(w: &[f64], rows: usize, cols: usize, k: usize) -> (f64, Vec<f64>) {
        let t = top_singular_triplets(w, rows, cols, k).unwrap();
        (w.iter().map(|x| x * x).sum(), t.iter().map(|x| x.sigma).collect())
    }

    #[test]
    fn iid_noise_sits_at_the_edge_and_is_not_flagged() {
        let (r, c) = (120, 240);
        let w = noise(r, c, 7);
        let (f, s) = spectrum(&w, r, c, 4);
        let b = marchenko_pastur(f, r, c, &s, 1.1);
        assert_eq!(b.status, "ok");
        let ratio = b.sigma1_over_mp.unwrap();
        assert!((0.9..1.05).contains(&ratio), "iid sigma_1/sigma_MP = {}", ratio);
        // the entry estimate recovers the uniform[-1,1) standard deviation 1/sqrt(3)
        assert!((b.sigma_entry.unwrap() - (1.0f64 / 3.0).sqrt()).abs() < 0.02, "{:?}", b.sigma_entry);
        assert!(!b.flagged);
        // the formula as written in the roadmap
        let mp = b.sigma_bulk.unwrap() * (1.0 + (c as f64 / r as f64).sqrt());
        assert_eq!(b.sigma_mp.unwrap(), mp);
    }

    #[test]
    fn a_planted_rank_one_edit_detaches_and_is_flagged() {
        let (r, c) = (120, 240);
        let mut w = noise(r, c, 11);
        let edge = (1.0f64 / 3.0).sqrt() * ((r as f64).sqrt() + (c as f64).sqrt());
        let theta = 4.0 * edge;
        let (u, v): (Vec<f64>, Vec<f64>) =
            ((0..r).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect(), (0..c).map(|j| if j < 8 { 1.0 } else { 0.0 }).collect());
        let (nu, nv) = (u.iter().map(|x| x * x).sum::<f64>().sqrt(), v.iter().map(|x| x * x).sum::<f64>().sqrt());
        for i in 0..r {
            for j in 0..c {
                w[i * c + j] += theta * u[i] / nu * v[j] / nv;
            }
        }
        let (f, s) = spectrum(&w, r, c, 4);
        let b = marchenko_pastur(f, r, c, &s, 1.1);
        assert_eq!(b.status, "ok");
        assert_eq!(b.outliers, 1, "{:?}", b.outlier_ratios);
        assert!(b.sigma1_over_mp.unwrap() > 3.5, "{:?}", b.sigma1_over_mp);
        assert!(b.flagged && !b.spectrum_exhausted);
        // removing the spike's energy keeps the bulk estimate unbiased
        assert!((b.sigma_entry.unwrap() - (1.0f64 / 3.0).sqrt()).abs() < 0.02, "{:?}", b.sigma_entry);
    }

    #[test]
    fn low_rank_or_degenerate_inputs_are_undefined_not_verdicts() {
        // exactly rank 1: the top singular value explains all the energy
        let (r, c) = (6, 10);
        let w: Vec<f64> = (0..r * c).map(|k| ((k / c) as f64 + 1.0) * ((k % c) as f64 - 4.5)).collect();
        let (f, s) = spectrum(&w, r, c, 4);
        let b = marchenko_pastur(f, r, c, &s, 1.1);
        assert!(b.status.starts_with("undefined"), "{}", b.status);
        assert!(!b.flagged && b.sigma1_over_mp.is_none());
        // rank 1 in F32 precision: round-off leaves ~1e-16 of the energy, which is not a bulk
        let w32: Vec<f64> = w.iter().map(|x| *x as f32 as f64 * (1.0 + 1e-9)).collect();
        let (f, s) = spectrum(&w32, r, c, 1);
        let b = marchenko_pastur(f, r, c, &s, 1.1);
        assert!(b.status.starts_with("undefined") && !b.flagged, "{:?}", b);
        assert!(b.bulk_energy_fraction < EMPTY_BULK);
        for b in
            [marchenko_pastur(0.0, 4, 4, &[1.0], 1.1), marchenko_pastur(1.0, 4, 4, &[], 1.1), marchenko_pastur(f64::NAN, 4, 4, &[1.0], 1.1)]
        {
            assert!(b.status.starts_with("undefined") && !b.flagged);
        }
    }
}
