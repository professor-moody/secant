//! `secant sanitize`: scan static weight values and optionally publish a clean
//! sanitized copy. Port of the reference oracle's `wsp sanitize`.
//!
//! Repairs: NaN to +0, +-Inf to signed max-abs, non-finite divisor scales to
//! the B3-2B floor (`--divisor-floor`), and non-finite quantized blocks to zero
//! (`--clamp-nonfinite`), plus duplicate routing ids (`--dedup-routing-rows`)
//! and the opt-in Secant extensions `--clamp-extreme` and `--flush-subnormal`.
//! Finite non-positive or below-floor divisor scales, zero `rope_freqs` entries,
//! zero/tiny-norm rows, and negative or out-of-range routing/index values are
//! never repaired by CLI `--clamp-nonfinite`: findings remain.
//! Exit 0: clean or remediated under this policy. Exit 1: findings remain.
//! Exit 2: usage, I/O, unsupported input, or incomplete scan. A clean receipt
//! does not prove runtime inference safety. Output and receipts never overwrite.
use crate::util::{badge, bytes, fail, field, section, table, usage};
use clap::Args as ClapArgs;
use secant_core::security::{go_abs, sanitize_file_report, write_sanitize_receipt, SanitizeOptions};
use secant_core::Error;

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Scan weight values against explicit numerical policies and write a receipt. Repairs are opt-in and published only after a clean rescan. Exit 0 means clean or remediated; 1 means findings remain; 2 means the scan was incomplete or invalid.\n\nBlock-quantized GGUF tensors (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) are scanned at the scale-header level. Rule B3-2A allows zero and FP16 subnormal d/dmin values by default and reports only NaN or Inf. --flush-subnormal and --clamp-extreme opt into stricter diagnostics. --clamp-nonfinite zeroes every byte of a block with a non-finite scale header. Divisor scales named input_scale, k_scale, v_scale, q_scale, y_scale, prob_scale or *scale_inv must be finite and at least --divisor-floor (default 1e-12f). Other quantized types are refused with exit 2.",
    after_help = "Example:\n  secant sanitize --model model.safetensors --receipt scan.json --attempt-id local-check\n\nA clean static scan does not prove runtime inference safety."
)]
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
    /// Additional case-sensitive path.Match glob (repeatable); defaults also select B3-3 routing names
    #[arg(long)]
    pub routing_tensor: Vec<String>,
    /// Additional norm tensor glob (repeatable); defaults also select norm
    #[arg(long)]
    pub norm_tensor: Vec<String>,
    /// Additional divisor-scale tensor glob (repeatable); defaults also select input_scale, k_scale, v_scale, q_scale, y_scale, prob_scale, *scale_inv
    #[arg(long)]
    pub divisor_scale_tensor: Vec<String>,
    /// Disable default divisor-scale name matching; use only --divisor-scale-tensor globs
    #[arg(long)]
    pub no_default_divisor_scales: bool,
    /// Additional integer index tensor glob (repeatable); defaults select g_idx, expert_map, full_row_indices, hash_indices_table
    #[arg(long)]
    pub index_tensor: Vec<String>,
    /// Exclusive upper bound for all index tensors (0 infers per-tensor from sibling shapes)
    #[arg(long, default_value_t = 0, allow_negative_numbers = true)]
    pub index_bound: i64,
    /// Maximum finite absolute weight value (policy, not a runtime overflow proof)
    #[arg(long, default_value_t = 1e4, allow_negative_numbers = true)]
    pub max_abs: f64,
    /// Minimum L2 norm for selected contiguous vectors
    #[arg(long, default_value_t = 1e-12, allow_negative_numbers = true)]
    pub min_norm: f64,
    /// Bounded examples, 1..10000; aggregate counts remain exact
    #[arg(long, default_value_t = 128, allow_negative_numbers = true)]
    pub max_examples: i64,
    /// Clamp non-finite values (including non-finite B3-2B divisor scales to --divisor-floor); zero quantized blocks with non-finite headers
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
    /// Report subnormal dense values and subnormal GGUF block scales as findings
    #[arg(long)]
    pub strict_subnormal: bool,
    /// Rule B3-2B lower bound for divisor scales (compared as float32)
    #[arg(long, default_value_t = 1e-12, allow_negative_numbers = true)]
    pub divisor_floor: f64,
    /// Explicit report-only mode (default when no remediation flag or --out is passed)
    #[arg(long)]
    pub dry_run: bool,
    /// Also emit the receipt to stdout
    #[arg(long)]
    pub json: bool,
}

pub fn run(a: &Args) -> i32 {
    if a.model.is_empty() || a.receipt.is_empty() || a.attempt_id.is_empty() {
        return usage("--model, --receipt and --attempt-id are required");
    }
    if a.dry_run && (!a.out.is_empty() || a.clamp_nonfinite || a.dedup_routing_rows || a.clamp_extreme || a.flush_subnormal) {
        return usage("--dry-run cannot be combined with remediation flags or --out");
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
        divisor_scale_tensor: a.divisor_scale_tensor.clone(),
        no_default_divisor_scales: a.no_default_divisor_scales,
        index_tensor: a.index_tensor.clone(),
        index_bound: a.index_bound,
        max_abs: a.max_abs,
        min_norm: a.min_norm,
        max_examples: a.max_examples,
        clamp_nonfinite: a.clamp_nonfinite,
        dedup_routing_rows: a.dedup_routing_rows,
        clamp_extreme: a.clamp_extreme,
        flush_subnormal: a.flush_subnormal,
        output: a.out.clone(),
        strict_subnormal: a.strict_subnormal,
        divisor_floor: a.divisor_floor,
        repair: false,
    };
    let (rep, scan_err) = sanitize_file_report(&a.model, &opts);
    if let Err(e) = write_sanitize_receipt(&a.receipt, &rep) {
        return fail(&Error::msg(format!("receipt publication: {}", e.message())));
    }
    if a.json {
        println!("{}", rep.to_json_compact());
    } else {
        section("Weight-value policy scan");
        field(
            "Policy gate",
            if rep.status == "clean" || rep.status == "remediated" {
                "[PASS]"
            } else if rep.status == "findings" {
                "[FAIL]"
            } else {
                "[UNKNOWN]"
            },
        );
        field("Input unchanged", badge(rep.input_unchanged));
        field("Input size", bytes(rep.bytes.max(0) as u64));
        if let Some(scan) = rep.after.as_ref().or(rep.before.as_ref()) {
            field("Tensors scanned", scan.tensors);
            field("Findings remaining", scan.total_findings);
            table(
                &["FINDING KIND", "COUNT"],
                &scan.counts.iter().map(|(kind, count)| vec![kind.clone(), count.to_string()]).collect::<Vec<_>>(),
            );
        }
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
