use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::container::{apply_plan_report, PatchMode};
use secant_core::plan::Plan;

#[derive(ClapArgs, Debug)]
pub struct Args {
    /// Base model to edit (read-only unless --in-place; a copy is patched)
    #[arg(long)]
    base: Option<String>,
    /// Edit plan (secant.wsp.plan.v1)
    #[arg(long)]
    plan: Option<String>,
    /// Patched output path
    #[arg(long)]
    out: Option<String>,
    /// Edit --base itself (verified by a per-tensor SHA-256 snapshot); --out is only recorded
    #[arg(long)]
    in_place: bool,
    #[arg(long)]
    json: bool,
}

pub fn run(a: &Args) -> i32 {
    let (Some(base), Some(plan_path)) = (&a.base, &a.plan) else {
        return usage("--base, --plan, and --out are all required");
    };
    let out = match (&a.out, a.in_place) {
        (Some(o), _) => o.clone(),
        (None, true) => base.clone(),
        (None, false) => return usage("--base, --plan, and --out are all required"),
    };
    let plan = match Plan::load(plan_path) {
        Ok(p) => p,
        Err(e) => return fail(&e),
    };
    let mode = if a.in_place { PatchMode::InPlace } else { PatchMode::CopyOnWrite };
    let (report, err) = apply_plan_report(base, &out, &plan, mode);
    if let Some(e) = err {
        if a.json {
            if let Some(r) = &report {
                let _ = emit_json(r);
            }
        }
        // The oracle exits 1 for every apply failure (the check ran).
        eprintln!("Error: {}", e);
        return 1;
    }
    let report = report.expect("report on success");
    if a.json {
        return match emit_json(&report) {
            Ok(()) => 0,
            Err(e) => fail(&e),
        };
    }
    println!("patched {} -> {} ({} bytes)", base, out, report.out_bytes);
    for e in report.edits.iter().flatten() {
        println!(
            "  {:<4} {}: {}/{} entries changed, delta_fro {:.6}",
            e.id, e.tensor, e.entries_changed, e.entries_total, e.delta_fro_norm
        );
    }
    if let Some(v) = &report.verify {
        println!(
            "verify: header identical={}, other tensors byte-identical={} ({} tensors), ok={}",
            v.header_identical, v.other_tensors_byte_identical, v.n_tensors_other, v.ok
        );
    }
    println!("sha256 {}", report.out_sha256);
    0
}
