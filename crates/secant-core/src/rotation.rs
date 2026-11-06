//! Deterministic float64 LoRA reconstruction witness. The input tensor divided
//! by its RMS is the target residual; the objective is mean((B A - target)^2).
//! A frozen base cancels out of that residual objective. This is an optimizer
//! geometry experiment, not model inference or a task-quality evaluation.
use crate::{Error, Result};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct RotationOptions {
    pub rank: usize,
    pub steps: usize,
    pub lr: f64,
    pub wd: f64,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Measurement {
    pub step: usize,
    pub loss: f64,
    pub delta_fro: f64,
    pub relative_leakage: f64,
    pub delta_is_zero: bool,
}

#[derive(Debug, Serialize)]
pub struct Arm {
    pub name: String,
    pub step_1: Measurement,
    pub step_t: Measurement,
    pub trajectory: Vec<Measurement>,
}

#[derive(Debug, Serialize)]
pub struct RotationReport {
    pub schema: &'static str,
    pub objective: &'static str,
    pub arithmetic: &'static str,
    pub shape: [usize; 2],
    pub directions: usize,
    pub options: RotationOptions,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub target_rms: f64,
    pub locked_loss_floor: f64,
    pub lock_tolerance: f64,
    pub lock_passed: bool,
    pub arms: Vec<Arm>,
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

fn project(x: &mut [f64], cols: usize, basis: &[Vec<f64>]) {
    // P(P X), matching ProjectThenHealLinear's float64 projection and backward.
    for _ in 0..2 {
        for d in basis {
            for c in 0..cols {
                let dot: f64 = d.iter().enumerate().map(|(i, v)| v * x[i * cols + c]).sum();
                for (i, v) in d.iter().enumerate() {
                    x[i * cols + c] -= v * dot;
                }
            }
        }
    }
}

fn product(b: &[f64], a: &[f64], rows: usize, cols: usize, rank: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for h in 0..rank {
            for j in 0..cols {
                out[i * cols + j] += b[i * rank + h] * a[h * cols + j];
            }
        }
    }
    out
}

fn norm(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |n, x| n.hypot(*x))
}

fn protected_norm(x: &[f64], cols: usize, basis: &[Vec<f64>]) -> f64 {
    let mut n: f64 = 0.0;
    for d in basis {
        for j in 0..cols {
            let v: f64 = d.iter().enumerate().map(|(i, d)| d * x[i * cols + j]).sum();
            n = n.hypot(v);
        }
    }
    n
}

fn adamw(p: &mut [f64], g: &[f64], m: &mut [f64], v: &mut [f64], step: usize, o: &RotationOptions) {
    let c1 = 1.0 - 0.9f64.powi(step as i32);
    let c2 = 1.0 - 0.999f64.powi(step as i32);
    for i in 0..p.len() {
        m[i] = 0.9 * m[i] + 0.1 * g[i];
        v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
        p[i] = p[i] * (1.0 - o.lr * o.wd) - o.lr * (m[i] / c1) / ((v[i] / c2).sqrt() + 1e-8);
    }
}

/// `basis` must have orthonormal rows. Reject invalid dimensions and non-finite
/// arithmetic before publishing a report. Zero updates are explicitly vacuous.
pub fn rotation_test(w: &[f64], rows: usize, cols: usize, basis: &[Vec<f64>], o: &RotationOptions) -> Result<RotationReport> {
    if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(w.len()) || w.iter().any(|x| !x.is_finite()) {
        return Err(Error::msg("rotation-test: tensor must be a nonempty finite [d_out, d_in] matrix"));
    }
    if o.rank == 0
        || o.rank > rows.min(cols)
        || o.steps == 0
        || o.steps > 100_000
        || !o.lr.is_finite()
        || o.lr <= 0.0
        || !o.wd.is_finite()
        || o.wd < 0.0
        || o.lr * o.wd >= 1.0
    {
        return Err(Error::msg(
            "rotation-test: require 1 <= rank <= min(shape), 1 <= steps <= 100000, finite lr > 0, wd >= 0 and lr*wd < 1",
        ));
    }
    if basis.is_empty() || basis.len() >= rows || basis.iter().any(|d| d.len() != rows || d.iter().any(|x| !x.is_finite())) {
        return Err(Error::msg("rotation-test: directions must span a nonempty proper output subspace"));
    }
    for (i, d) in basis.iter().enumerate() {
        for (j, e) in basis.iter().enumerate() {
            let dot: f64 = d.iter().zip(e).map(|(a, b)| a * b).sum();
            if (dot - if i == j { 1.0 } else { 0.0 }).abs() > 1e-10 {
                return Err(Error::msg("rotation-test: basis is not orthonormal"));
            }
        }
    }
    let rms = norm(w) / (w.len() as f64).sqrt();
    if !rms.is_finite() || rms == 0.0 {
        return Err(Error::msg("rotation-test: tensor RMS must be finite and nonzero"));
    }
    let target: Vec<f64> = w.iter().map(|x| x / rms).collect();
    let floor = protected_norm(&target, cols, basis).powi(2) / w.len() as f64;
    let mut rng = Rng(o.seed);
    let a0: Vec<f64> = (0..o.rank * cols).map(|_| rng.unit() / (cols as f64).sqrt()).collect();
    let mut arms = Vec::new();
    for (arm, name) in ["AdamW", "Gradient-Projected", "ProjectThenHealLinear"].iter().enumerate() {
        let mut a = a0.clone();
        let mut b = vec![0.0; rows * o.rank];
        let (mut ma, mut va) = (vec![0.0; a.len()], vec![0.0; a.len()]);
        let (mut mb, mut vb) = (vec![0.0; b.len()], vec![0.0; b.len()]);
        let mut trajectory = Vec::new();
        for step in 0..=o.steps {
            let mut used_b = b.clone();
            if arm == 2 {
                project(&mut used_b, o.rank, basis);
            }
            let dw = product(&used_b, &a, rows, cols, o.rank);
            let residual: Vec<f64> = dw.iter().zip(&target).map(|(x, y)| x - y).collect();
            let n = norm(&dw);
            let m = Measurement {
                step,
                loss: norm(&residual).powi(2) / w.len() as f64,
                delta_fro: n,
                relative_leakage: if n == 0.0 { 0.0 } else { protected_norm(&dw, cols, basis) / n },
                delta_is_zero: n == 0.0,
            };
            if !m.loss.is_finite() || !m.relative_leakage.is_finite() || !n.is_finite() {
                return Err(Error::msg(format!("rotation-test: non-finite arithmetic in {} at step {}; lower --lr", name, step)));
            }
            trajectory.push(m);
            if step == o.steps {
                break;
            }
            let (mut ga, mut gb) = (vec![0.0; a.len()], vec![0.0; b.len()]);
            for i in 0..rows {
                for h in 0..o.rank {
                    for j in 0..cols {
                        let g = 2.0 * residual[i * cols + j] / w.len() as f64;
                        gb[i * o.rank + h] += g * a[h * cols + j];
                        ga[h * cols + j] += g * used_b[i * o.rank + h];
                    }
                }
            }
            if arm != 0 {
                project(&mut gb, o.rank, basis);
            }
            adamw(&mut a, &ga, &mut ma, &mut va, step + 1, o);
            adamw(&mut b, &gb, &mut mb, &mut vb, step + 1, o);
        }
        arms.push(Arm { name: name.to_string(), step_1: trajectory[1].clone(), step_t: trajectory[o.steps].clone(), trajectory });
    }
    let lock_passed = arms[2].trajectory.iter().skip(1).all(|m| !m.delta_is_zero && m.relative_leakage <= 1e-12);
    Ok(RotationReport {
        schema: "secant.rotation-test.v1",
        objective: "mean((B A - W / rms(W))^2); scale=1; A seeded uniform; B=0",
        arithmetic: "float64; splitmix64; serial reductions; two-pass output projection",
        shape: [rows, cols],
        directions: basis.len(),
        options: o.clone(),
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        target_rms: rms,
        locked_loss_floor: floor,
        lock_tolerance: 1e-12,
        lock_passed,
        arms,
    })
}
