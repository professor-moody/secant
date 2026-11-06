use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::subspace::{compose, ComposeOptions};
use secant_core::{plan, subspace::ORTHOGONALITY_TOLERANCE};
use serde_json::json;

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Build a deterministic edit plan from protected directions and a built-in or JSON profile. Directions are orthogonalized in float64; calibration uses a target activation shift or relative Frobenius norm. The plan is written to --out.",
    after_help = "Example:\n  secant compose --profile security-auditor-b1b2b3 --vectors directions/ --target-shift 1 --out plan.json"
)]
pub struct Args {
    /// Built-in profile name or path to a secant.wsp.profile.v1 JSON file
    #[arg(long)]
    profile: Option<String>,
    /// Directory holding d_*/h_* vectors as .npy or .json
    #[arg(long, default_value = "")]
    vectors: String,
    /// secant.wsp.analyze.v1 report from `secant analyze`; its L_loop_star/L_ref_star/L_root_star assignments bind auto slots (Borda order) instead of sorted vector filenames
    #[arg(long)]
    analysis: Option<String>,
    /// Scale-invariant calibration: shift <h_ref, delta_W^T d_hat> by this amount
    #[arg(long, allow_negative_numbers = true)]
    target_shift: Option<f64>,
    /// Relative Frobenius calibration: ||delta_W||_F = gamma * ||W||_F (needs --base or a profile ref_fro_norm)
    #[arg(long, allow_negative_numbers = true)]
    gamma: Option<f64>,
    /// Base model used to measure ||W||_F for --gamma
    #[arg(long, default_value = "")]
    base: String,
    /// Output plan path (secant.wsp.plan.v1)
    #[arg(long)]
    out: Option<String>,
    #[arg(long)]
    json: bool,
}

pub fn run(a: &Args) -> i32 {
    let Some(profile) = &a.profile else {
        return usage(&format!("--profile is required (built-in name or a {} file)", plan::PROFILE_SCHEMA));
    };
    let Some(out) = &a.out else { return usage("--out is required (the plan path)") };
    let mut o = ComposeOptions {
        profile_name: profile.clone(),
        vectors_dir: a.vectors.clone(),
        target_shift: a.target_shift.unwrap_or(0.0),
        has_shift: a.target_shift.is_some(),
        gamma: a.gamma.unwrap_or(0.0),
        has_gamma: a.gamma.is_some(),
        base_model: a.base.clone(),
        analysis_path: a.analysis.clone().unwrap_or_default(),
        ..Default::default()
    };
    if profile.ends_with(".json") {
        o.profile_path = profile.clone();
        o.profile_name.clear();
    }
    let plan = match compose(o) {
        Ok(p) => p,
        Err(e) => return fail(&e),
    };
    if let Err(e) = plan.write(out) {
        return fail(&e);
    }
    if a.json {
        let v = json!({
            "schema": plan::COMPOSE_SCHEMA,
            "profile": plan.profile,
            "plan_path": out,
            "edits": plan.edits.len(),
            "calibration": plan.calibration,
            "gram_schmidt": plan.gram_schmidt,
            "analysis": plan.analysis,
        });
        return match emit_json(&v) {
            Ok(()) => 0,
            Err(e) => fail(&e),
        };
    }
    section("Edit plan");
    field("Profile", &plan.profile);
    field("Plan", out);
    field("Edits", plan.edits.len());
    if let Some(g) = &plan.gram_schmidt {
        field("Orthogonality gate", badge(g.post_cosine_max_abs < ORTHOGONALITY_TOLERANCE));
        println!(
            "gram-schmidt: {} direction(s), post max |cos| {:.3e} (< {:.0e})",
            g.order.len(),
            g.post_cosine_max_abs,
            ORTHOGONALITY_TOLERANCE
        );
    }
    print!("calibration: {}", plan.calibration.mode);
    if plan.calibration.mode == "target-shift" {
        println!(" ({:+.2}, scale-invariant)", plan.calibration.target_shift);
    } else {
        println!(" (gamma {:.3})", plan.calibration.gamma);
    }
    let _ = plan::PLAN_SCHEMA;
    table(
        &["EDIT", "ROLE", "TENSOR", "OPERATION", "CALIBRATION"],
        &plan
            .edits
            .iter()
            .map(|e| {
                vec![
                    e.id.clone(),
                    e.role.clone(),
                    e.tensor.clone(),
                    e.op.clone(),
                    if e.achieved_shift != 0.0 {
                        format!("shift {:+.6e}; {} columns", e.achieved_shift, e.cols.len())
                    } else {
                        format!("gamma {:.6e}", e.gamma)
                    },
                ]
            })
            .collect::<Vec<_>>(),
    );
    0
}
