//! `secant sanitize`: scan static weight values and optionally publish a clean
//! sanitized copy. Port of the reference oracle's `wsp sanitize`.
//!
//! Repairs: NaN to +0, +-Inf to signed max-abs (--clamp-nonfinite), duplicate routing
//! ids (--dedup-routing-rows) as in the oracle; plus the opt-in Secant extensions
//! --clamp-extreme and --flush-subnormal. Zero/tiny-norm rows and negative or
//! out-of-range routing ids are never repaired (no safe value): findings remain.
//! Exit 0: clean or remediated under this policy. Exit 1: findings remain.
//! Exit 2: usage, I/O, unsupported input, or incomplete scan. A clean receipt
//! does not prove runtime inference safety. Output and receipts never overwrite.
use crate::util::{fail, usage};
use clap::Args as ClapArgs;
use secant_core::security::{go_abs, sanitize_file_report, write_sanitize_receipt, SanitizeOptions};
use secant_core::Error;

#[derive(ClapArgs, Debug)]
pub struct Args {
    /// Input .gguf or .safetensors (read only)
    #[arg(long, default_value = "")]
    pub model: String,
    /// New secant.wsp.sanitize.v1 JSON receipt path
    #[arg(long, default_value = "")]
    pub receipt: String,
    /// Required pipeline attempt identity
    #[arg(long, default_value = "")]
    pub attempt_id: String,
    /// Exclusive routed-expert bound; required for routing tables
    #[arg(long, default_value_t = 0, allow_negative_numbers = true)]
    pub num_experts: i64,
    /// Additional case-sensitive path.Match glob (repeatable); defaults also select tid2eid/expert_id
    #[arg(long)]
    pub routing_tensor: Vec<String>,
    /// Additional norm tensor glob (repeatable); defaults also select norm
    #[arg(long)]
    pub norm_tensor: Vec<String>,
    /// Maximum finite absolute weight value (policy, not a runtime overflow proof)
    #[arg(long, default_value_t = 1e4, allow_negative_numbers = true)]
    pub max_abs: f64,
    /// Minimum L2 norm for selected contiguous vectors
    #[arg(long, default_value_t = 1e-12, allow_negative_numbers = true)]
    pub min_norm: f64,
    /// Bounded examples, 1..10000; aggregate counts remain exact
    #[arg(long, default_value_t = 128, allow_negative_numbers = true)]
    pub max_examples: i64,
    /// NaN to +0; infinities to signed representable max-abs
    #[arg(long)]
    pub clamp_nonfinite: bool,
    /// Replace later duplicate slots with lowest unused valid expert IDs
    #[arg(long)]
    pub dedup_routing_rows: bool,
    /// Secant extension: finite |x| > --max-abs becomes sign(x)*max-abs, rounded down in magnitude to the dtype (default off)
    #[arg(long)]
    pub clamp_extreme: bool,
    /// Secant extension: subnormal floats become the same-signed zero (default off). Zero or tiny-L2 norm rows and
    /// negative/out-of-range routing ids have no safe repair value: they are always detected and refused
    #[arg(long)]
    pub flush_subnormal: bool,
    /// New output file; requires a remediation flag and a clean rescan
    #[arg(long = "out", default_value = "")]
    pub out: String,
    /// Also emit the receipt to stdout
    #[arg(long)]
    pub json: bool,
}

pub fn run(a: &Args) -> i32 {
    if a.model.is_empty() || a.receipt.is_empty() || a.attempt_id.is_empty() {
        return usage("--model, --receipt and --attempt-id are required");
    }
    let abs_receipt = match go_abs(&a.receipt) {
        Ok(p) => p,
        Err(e) => return fail(&Error::msg(e.to_string())),
    };
    for p in [a.model.as_str(), a.out.as_str()] {
        if p.is_empty() {
            continue;
        }
        match go_abs(p) {
            Ok(abs) if abs == abs_receipt => return usage("receipt must be distinct from model and output"),
            Ok(_) => {}
            Err(e) => return fail(&Error::msg(e.to_string())),
        }
    }
    if !secant_core::security::internals::lstat_not_exist(&a.receipt) {
        return usage("receipt must be a new path");
    }
    let opts = SanitizeOptions {
        attempt_id: a.attempt_id.clone(),
        num_experts: a.num_experts,
        routing_tensor: a.routing_tensor.clone(),
        norm_tensor: a.norm_tensor.clone(),
        max_abs: a.max_abs,
        min_norm: a.min_norm,
        max_examples: a.max_examples,
        clamp_nonfinite: a.clamp_nonfinite,
        dedup_routing_rows: a.dedup_routing_rows,
        clamp_extreme: a.clamp_extreme,
        flush_subnormal: a.flush_subnormal,
        output: a.out.clone(),
    };
    let (rep, scan_err) = sanitize_file_report(&a.model, &opts);
    if let Err(e) = write_sanitize_receipt(&a.receipt, &rep) {
        return fail(&Error::msg(format!("receipt publication: {}", e.message())));
    }
    if a.json {
        println!("{}", rep.to_json_compact());
    } else {
        println!(
            "sanitize: {}; receipt {}; input unchanged={}; output published={}",
            rep.status, a.receipt, rep.input_unchanged, rep.output_published
        );
    }
    if let Err(e) = scan_err {
        return fail(&Error::msg(e.message().to_string()));
    }
    if rep.status == "findings" {
        return fail(&Error::gate(format!("weight-value policy findings remain; see {}", a.receipt)));
    }
    0
}
