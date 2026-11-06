use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::analyze::analyze;
use secant_core::plan::builtin_profile;

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Rank layers per subspace (resid/attn/mlp/h) by precomputed or externally captured per-layer SNR cells, aggregate with Borda rank voting, and diagnose why hardcoded mid-layer defaults fail. Pass the output to `secant compose --analysis` so L_loop_star / L_ref_star / L_root_star drive automatic layer binding.",
    after_help = "Example:\n  secant analyze --in layer_snr.json --profile security-auditor-b1b2b3 --out analysis.json --json"
)]
pub struct Args {
    /// Layer SNR JSON (layer_snr.json) or sweep summary
    #[arg(long = "in")]
    input: Option<String>,
    /// Optional built-in profile name to annotate in the assignments notes
    #[arg(long)]
    profile: Option<String>,
    /// Optional path to write the secant.wsp.analyze.v1 JSON report
    #[arg(long)]
    out: Option<String>,
    #[arg(long)]
    json: bool,
}

pub fn run(a: &Args) -> i32 {
    let Some(inp) = &a.input else {
        return usage("--in is required");
    };
    let mut rep = match analyze(inp) {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    if let Some(p) = &a.profile {
        if !p.is_empty() {
            if let Err(e) = builtin_profile(p) {
                return fail(&e);
            }
            rep.assignments.notes.push(format!(
                "profile {} consumes these assignments via `secant compose --profile {} --analysis <this report>`",
                p, p
            ));
        }
    }
    if let Some(out) = &a.out {
        if !out.is_empty() {
            let mut bytes = match serde_json::to_vec_pretty(&rep) {
                Ok(b) => b,
                Err(e) => return fail(&secant_core::Error::msg(e.to_string())),
            };
            bytes.push(b'\n');
            if let Err(e) = std::fs::write(out, bytes) {
                return fail(&secant_core::Error::msg(format!("write {}: {}", out, e)));
            }
        }
    }
    if a.json {
        return match emit_json(&rep) {
            Ok(()) => 0,
            Err(e) => fail(&e),
        };
    }
    println!("layer analysis of {} (layout {})", rep.input, rep.layout_detected);
    for (name, fam) in &rep.families {
        println!("\nfamily {}: Borda winner L{}", name, fam.borda_winner);
        for sub in ["resid", "attn", "mlp", "h"] {
            if let Some(r) = fam.subspaces.get(sub) {
                let next = if r.ranked.len() > 1 {
                    format!("next: L{:02} {:.4}", r.ranked[1].layer, r.ranked[1].snr)
                } else {
                    "(only layer)".to_string()
                };
                println!("  {:<5} peak L{:02} snr {:.4} {}", sub, r.peak.layer, r.peak.snr, next);
            }
        }
    }
    if let Some(l) = rep.assignments.l_ref_star {
        println!("\nL_ref_star  = {}", l);
    }
    if let Some(l) = rep.assignments.l_root_star {
        println!("L_root_star = {}", l);
    }
    if let Some(l) = rep.assignments.l_loop_star {
        println!("L_loop_star = {}", l);
    }
    println!("\nwhy hardcoded mid-layers fail:");
    for d in &rep.diagnostics {
        println!("  {}", d);
    }
    0
}
