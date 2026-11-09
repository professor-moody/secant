use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::container::verify_patch;

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Compare checkpoint headers and payloads. Every tensor outside --expect-changed must remain byte-identical; a failed verification exits 1. Listing a tensor permits a change but does not require one.",
    after_help = "Example:\n  secant verify-patch --base base.gguf --patched edited.gguf --expect-changed blk.0.ffn_down.weight"
)]
pub struct Args {
    /// Unmodified base model
    #[arg(long)]
    base: Option<String>,
    /// Patched model to compare
    #[arg(long)]
    patched: Option<String>,
    /// Comma-separated tensor names allowed to differ; all others must be byte-identical
    #[arg(long, default_value = "")]
    expect_changed: String,
    #[arg(long)]
    json: bool,
}

pub fn run(a: &Args) -> i32 {
    let (Some(base), Some(patched)) = (&a.base, &a.patched) else {
        return usage("--base and --patched are required");
    };
    let expect = split_csv(&a.expect_changed);
    let rep = match verify_patch(base, patched, &expect) {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    if a.json {
        if let Err(e) = emit_json(&rep) {
            return fail(&e);
        }
    } else {
        section("Patch verification");
        field("Base", base);
        field("Patched", patched);
        field("Format", &rep.format);
        field("Header", format!("{} (identical: {})", bytes(rep.header_len as u64), rep.header_identical));
        field(
            "Other tensors",
            format!("{} / {} byte-identical ({})", rep.n_tensors_other, rep.n_tensors, rep.other_tensors_byte_identical),
        );
        field("Verification gate", badge(rep.ok));
        let ch = rep.changed_tensors.clone().unwrap_or_default();
        if !ch.is_empty() {
            table(
                &["INDEX", "CHANGED TENSOR", "STATUS"],
                &ch.iter()
                    .enumerate()
                    .map(|(i, t)| vec![(i + 1).to_string(), t.clone(), "expected diff".to_string()])
                    .collect::<Vec<_>>(),
            );
        }
        for f in &rep.failures {
            println!("  FAILURE: {}", f);
        }
    }
    if !rep.ok {
        eprintln!("Error: verify-patch failed: {}", rep.failure_summary());
        return 1;
    }
    0
}
