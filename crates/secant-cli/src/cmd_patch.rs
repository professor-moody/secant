use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::container::{apply_plan_report, PatchMode};
use secant_core::plan::Plan;

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Apply an edit plan at tensor payload offsets and verify the result in the same run. By default the base stays read-only and --out receives a copy. --in-place explicitly edits the base. Apply failures retain exit status 1.",
    after_help = "Example:\n  secant patch --base base.safetensors --plan plan.json --out edited.safetensors --json"
)]
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
    section("Patch receipt");
    field("Base", base);
    field("Output", &out);
    field("Output size", bytes(report.out_bytes as u64));
    table(
        &["EDIT", "TENSOR", "ENTRIES CHANGED", "DELTA FROBENIUS"],
        &report
            .edits
            .iter()
            .flatten()
            .map(|e| {
                vec![
                    e.id.clone(),
                    e.tensor.clone(),
                    format!("{} / {}", e.entries_changed, e.entries_total),
                    format!("{:.6e}", e.delta_fro_norm),
                ]
            })
            .collect::<Vec<_>>(),
    );
    if let Some(v) = &report.verify {
        field("Verification gate", badge(v.ok));
        println!(
            "verify: header identical={}, other tensors byte-identical={} ({} tensors), ok={}",
            v.header_identical, v.other_tensors_byte_identical, v.n_tensors_other, v.ok
        );
    }
    println!("sha256 {}", report.out_sha256);
    0
}
