use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::container::verify_patch;

#[derive(ClapArgs, Debug)]
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
        println!("verify {} vs {} ({})", base, patched, rep.format);
        println!("  header [{} bytes] identical: {}", rep.header_len, rep.header_identical);
        let ch = rep.changed_tensors.clone().unwrap_or_default();
        println!("  changed tensors: {}", if ch.is_empty() { "(none)".to_string() } else { ch.join(", ") });
        println!("  other tensors byte-identical: {} ({} tensors)", rep.other_tensors_byte_identical, rep.n_tensors_other);
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
