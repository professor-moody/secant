use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::subspace::{compose, ComposeOptions};
use secant_core::{plan, subspace::ORTHOGONALITY_TOLERANCE};
use serde_json::json;

#[derive(ClapArgs, Debug)]
pub struct Args {
    /// Built-in profile name or path to a crucible.wsp.profile.v1 JSON file
    #[arg(long)]
    profile: Option<String>,
    /// Directory holding d_*/h_* vectors as .npy or .json
    #[arg(long, default_value = "")]
    vectors: String,
    /// Not supported: `wsp analyze` is not ported; layers bind from sorted vector filenames
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
    /// Output plan path (crucible.wsp.plan.v1)
    #[arg(long)]
    out: Option<String>,
    #[arg(long)]
    json: bool,
}

pub fn run(a: &Args) -> i32 {
    let Some(profile) = &a.profile else {
        return usage("--profile is required (built-in name or a crucible.wsp.profile.v1 file)");
    };
    let Some(out) = &a.out else { return usage("--out is required (the plan path)") };
    if a.analysis.is_some() {
        return usage("--analysis is not supported: `wsp analyze` is not ported to Secant (see tool-gaps.json)");
    }
    let mut o = ComposeOptions {
        profile_name: profile.clone(),
        vectors_dir: a.vectors.clone(),
        target_shift: a.target_shift.unwrap_or(0.0),
        has_shift: a.target_shift.is_some(),
        gamma: a.gamma.unwrap_or(0.0),
        has_gamma: a.gamma.is_some(),
        base_model: a.base.clone(),
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
            "schema": "crucible.wsp.compose.v1",
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
    println!("composed {} edit(s) for profile {} -> {}", plan.edits.len(), plan.profile, out);
    if let Some(g) = &plan.gram_schmidt {
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
    for e in &plan.edits {
        if e.achieved_shift != 0.0 {
            println!("  {:<4} {:<18} {} cols={} witness shift {:+.3}", e.id, e.role, e.tensor, e.cols.len(), e.achieved_shift);
        } else if e.op == "projector-subtract" {
            println!("  {:<4} {:<18} {} gamma {:.3}", e.id, e.role, e.tensor, e.gamma);
        } else {
            println!("  {:<4} {:<18} {} {}", e.id, e.role, e.tensor, e.op);
        }
    }
    0
}
