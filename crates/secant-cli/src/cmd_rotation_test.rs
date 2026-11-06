use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::{
    container::file_sha256,
    heal::load_protected_directions,
    rotation::{rotation_test, RotationOptions},
    subspace::load_numpy_file,
    Error,
};

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Run a deterministic float64 three-arm LoRA reconstruction experiment. All arms share seeded A and zero B, fitting W/rms(W). AdamW updates both factors; gradient projection projects grad_B before AdamW; ProjectThenHealLinear projects B in every forward and the gradient in backward. Report ||D dW||_F / ||dW||_F at step 1, step T and every intermediate step. This is a numerical witness, not an inference benchmark. A zero update is flagged as vacuous and fails the structural lock gate.",
    after_help = "Example:\n  secant rotation-test --tensor-npy w.npy --directions-npy d.npy --rank 4 --steps 60 --lr 0.01 --wd 0.01 --seed 0 --json"
)]
pub struct Args {
    /// Finite C-order F32/F64 matrix [d_out, d_in]
    #[arg(long)]
    tensor_npy: String,
    /// Protected output directions: vector or [k, d_out] .npy, orthonormalized on load
    #[arg(long)]
    directions_npy: String,
    /// LoRA rank, at most min(d_out, d_in)
    #[arg(long, default_value_t = 4)]
    rank: usize,
    /// Optimizer steps (1..100000)
    #[arg(long, default_value_t = 60)]
    steps: usize,
    /// Positive finite learning rate
    #[arg(long, default_value_t = 0.01)]
    lr: f64,
    /// Nonnegative decoupled weight decay, lr*wd < 1
    #[arg(long, default_value_t = 0.01)]
    wd: f64,
    /// SplitMix64 initialization seed
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Emit the versioned report with full trajectory
    #[arg(long)]
    json: bool,
    /// Optional pipeline identity included in JSON
    #[arg(long, default_value = "")]
    attempt_id: String,
}

pub fn run(a: &Args) -> i32 {
    let result = (|| -> Result<_, Error> {
        let pre_w = file_sha256(&a.tensor_npy)?.0;
        let pre_d = file_sha256(&a.directions_npy)?.0;
        let w = load_numpy_file(&a.tensor_npy)?;
        if w.shape.len() != 2 {
            return Err(Error::msg("--tensor-npy must have shape [d_out, d_in]"));
        }
        let d = load_protected_directions(&a.directions_npy)?;
        let report = rotation_test(
            &w.to_f64(),
            w.shape[0],
            w.shape[1],
            &d.basis,
            &RotationOptions { rank: a.rank, steps: a.steps, lr: a.lr, wd: a.wd, seed: a.seed },
        )?;
        let post_w = file_sha256(&a.tensor_npy)?.0;
        let post_d = file_sha256(&a.directions_npy)?.0;
        if pre_w != post_w || pre_d != post_d {
            return Err(Error::msg("rotation-test: input changed during execution"));
        }
        let mut v = serde_json::to_value(&report).map_err(|e| Error::msg(e.to_string()))?;
        v["attempt_id"] = a.attempt_id.clone().into();
        v["inputs"] = serde_json::json!({"tensor": {"path": a.tensor_npy, "pre_sha256": pre_w, "post_sha256": post_w}, "directions": {"path": a.directions_npy, "pre_sha256": pre_d, "post_sha256": post_d}, "unchanged": true});
        Ok((report, v))
    })();
    match result {
        Err(e) => fail(&e),
        Ok((r, v)) => {
            if a.json {
                if let Err(e) = emit_json(&v) {
                    return fail(&e);
                }
            } else {
                section("AdamW subspace rotation");
                field("Objective", r.objective);
                field("Shape / rank", format!("{:?} / {}", r.shape, r.options.rank));
                field("Locked loss floor", format!("{:.6e}", r.locked_loss_floor));
                field("Structural lock gate", badge(r.lock_passed));
                table(
                    &["ARM", "LEAKAGE STEP 1", "LEAKAGE STEP T", "FINAL LOSS", "DELTA FROBENIUS"],
                    &r.arms
                        .iter()
                        .map(|arm| {
                            vec![
                                arm.name.clone(),
                                format!("{:.6e}", arm.step_1.relative_leakage),
                                format!("{:.6e}", arm.step_t.relative_leakage),
                                format!("{:.6e}", arm.step_t.loss),
                                format!("{:.6e}", arm.step_t.delta_fro),
                            ]
                        })
                        .collect::<Vec<_>>(),
                );
            }
            if r.lock_passed {
                0
            } else {
                1
            }
        }
    }
}
