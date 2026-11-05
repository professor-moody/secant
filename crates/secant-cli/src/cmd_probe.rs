//! `secant probe`: generate header-valid boundary-value weight fixtures.
//! Port of `crucible wsp probe` (cmd/crucible/wsp_phase2.go).
//!
//!   class1  NaN and -Inf router/gate floats (sentinel comparisons)
//!   class2  NaN, +1e38, -1e38 scale/bias floats (float-to-int saturation vs overflow)
//!   class3  NaN arrays: sign/payload/signalling NaN variants, ties, +-Inf mixes
//!   class4  I32/I64 routing tables (tid2eid / expert_id): -1, INT_MIN, num_experts,
//!           INT_MAX, 2^32, duplicate ids; plus all-zero and subnormal norm vectors
//!
//! Each fixture is self-verified (container inspection plus the sanitizer must
//! report the intended finding kinds) before it is published. Output goes to a
//! NEW directory; the receipt is a new file. Fixtures are byte-deterministic
//! for a given class, format and --num-experts.
//! Exit 0: generated and self-verified. Exit 2: usage, I/O, or could not verify.
use crate::util::{fail, usage};
use clap::Args as ClapArgs;
use secant_core::security::{generate_probe, go_abs, write_probe_receipt, ProbeOptions};
use secant_core::Error;

#[derive(ClapArgs, Debug)]
pub struct Args {
    /// class1|class2|class3|class4|all
    #[arg(long, default_value = "all")]
    pub class: String,
    /// safetensors|gguf|both
    #[arg(long, default_value = "both")]
    pub format: String,
    /// Optional existing model; probe cells are written into a byte-identical copy
    #[arg(long, default_value = "")]
    pub base: String,
    /// Routed-expert bound for class4 tables, in [2, 1048576]
    #[arg(long, default_value_t = 8, allow_negative_numbers = true)]
    pub num_experts: i64,
    /// NEW directory for the fixtures (must not exist)
    #[arg(long, default_value = "")]
    pub out_dir: String,
    /// New crucible.wsp.probe.v1 JSON receipt path
    #[arg(long, default_value = "")]
    pub receipt: String,
    /// Required pipeline attempt identity
    #[arg(long, default_value = "")]
    pub attempt_id: String,
    /// Emit the receipt to stdout
    #[arg(long)]
    pub json: bool,
}

fn require_new_path(label: &str, p: &str) -> Result<(), i32> {
    if !secant_core::security::internals::lstat_not_exist(p) {
        return Err(usage(&format!("{} must be a new path", label)));
    }
    let dir = secant_core::security::internals::go_dir(p);
    match std::fs::metadata(&dir) {
        Ok(m) if m.is_dir() => Ok(()),
        _ => Err(usage(&format!("{}: parent directory does not exist", label))),
    }
}

pub fn run(a: &Args) -> i32 {
    if a.out_dir.is_empty() || a.receipt.is_empty() || a.attempt_id.is_empty() {
        return usage("--out-dir, --receipt and --attempt-id are required");
    }
    if let Err(c) = require_new_path("receipt", &a.receipt) {
        return c;
    }
    let abs_r = go_abs(&a.receipt).unwrap_or_default();
    let abs_d = go_abs(&a.out_dir).unwrap_or_default();
    if abs_r == abs_d || abs_r.starts_with(&format!("{}/", abs_d)) {
        return usage("receipt must be outside --out-dir");
    }
    let opts = ProbeOptions {
        class: a.class.clone(),
        format: a.format.clone(),
        base: a.base.clone(),
        num_experts: a.num_experts,
        out_dir: a.out_dir.clone(),
        attempt_id: a.attempt_id.clone(),
    };
    let (rep, gen_err) = generate_probe(&opts);
    if let Err(e) = write_probe_receipt(&a.receipt, &rep) {
        return fail(&Error::msg(format!("receipt publication: {}", e.message())));
    }
    if a.json {
        println!("{}", rep.to_json_pretty());
    } else {
        println!("probe: {}; {} fixture(s) in {}; receipt {}", rep.status, rep.fixtures.len(), a.out_dir, a.receipt);
        for fx in &rep.fixtures {
            println!("  {:<52} {} findings={} sha256={}", fx.file, fx.sanitize_status, fx.sanitize_total, fx.sha256);
            for n in &fx.notes {
                println!("      {}", n);
            }
        }
        for u in &rep.unavailable {
            println!("  {} unavailable: {}", u.class, u.reason);
        }
    }
    if let Err(e) = gen_err {
        return fail(&Error::msg(e.message().to_string()));
    }
    0
}
