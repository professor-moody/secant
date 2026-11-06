//! `secant scan-sinks`: rank C/C++/CUDA/Rust source sites where a
//! format-valid but hostile weight VALUE becomes a memory-safety, DoS or abort
//! event. Port of the reference oracle's `wsp scan-sinks`; the receipt
//! (`secant.wsp-sink-scan.v1`) matches the oracle's field for field.
//!
//! Exit 0: scan completed (findings are hypotheses, not a gate). Exit 2:
//! usage, unreadable source, or the receipt could not be written.
use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::sinks::{sink_scan, write_sink_scan, SinkScanOptions, SINK_CLASSES};

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Scan a C/C++/CUDA/Rust source tree for four weight-value sink classes: FLOAT_TO_INDEX_CAST (a float cast to an index), UNVALIDATED_ROUTING_OR_INDEX_TENSOR (an index or routing tensor used as a subscript), QUANT_SCALE_DIV_OR_OVERFLOW (a division by, or conversion of, a quantization scale) and WEIGHT_VALUE_RELEASE_ASSERT (a release-mode abort over loaded values). Each hit says whether a guard-shaped expression appears earlier in the same function. Line-oriented and deterministic: the receipt carries a digest over every scanned file. A hit is a hypothesis to replay and read at the site, not a proven bug. Exit 0 when the scan completes.",
    after_help = "Examples:\n  secant scan-sinks --source ~/src/llama.cpp/ggml/src --top 25\n  secant scan-sinks --source ./csrc --out sinks.json --attempt-id local-audit\n  secant scan-sinks --source ./src --json"
)]
pub struct Args {
    /// Source directory to scan
    #[arg(long, default_value = "")]
    source: String,
    /// Maximum ranked findings kept in findings[] (<= 0 takes 100); counts stay exact
    #[arg(long, default_value_t = 100, allow_negative_numbers = true)]
    top: i64,
    /// Optional pipeline attempt ID recorded in the receipt
    #[arg(long, default_value = "")]
    attempt_id: String,
    /// Optional NEW secant.wsp-sink-scan.v1 JSON receipt path (parents are created)
    #[arg(long, default_value = "")]
    out: String,
    /// Emit the receipt to stdout
    #[arg(long)]
    json: bool,
    /// Rows of the human report
    #[arg(long, default_value_t = 25)]
    show: usize,
}

pub fn run(a: &Args) -> i32 {
    if a.source.is_empty() {
        return usage("--source is required");
    }
    let rep = match sink_scan(&SinkScanOptions { source: a.source.clone(), top: a.top, attempt_id: a.attempt_id.clone() }) {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    if !a.out.is_empty() {
        if let Err(e) = write_sink_scan(&a.out, &rep) {
            return fail(&e);
        }
    }
    if a.json {
        println!("{}", rep.to_json_pretty());
        return 0;
    }
    section("Weight-value sink scan");
    field("Source", &rep.source);
    field("Files scanned", rep.files_scanned);
    field("Files skipped", rep.files_skipped);
    field("Lines scanned", rep.lines_scanned);
    field("Tree digest", &rep.tree_digest);
    field(
        "Findings",
        format!("{}{}", rep.total_findings, if rep.truncated { format!(" (receipt keeps top {})", rep.top) } else { String::new() }),
    );
    table(
        &["SINK CLASS", "FINDINGS", "UNGUARDED"],
        &SINK_CLASSES
            .iter()
            .map(|c| vec![c.to_string(), rep.class_counts[*c].to_string(), rep.unguarded_counts[*c].to_string()])
            .collect::<Vec<_>>(),
    );
    table(
        &["SCORE", "CLASS", "SITE", "FUNCTION", "GUARD", "SNIPPET"],
        &rep.findings
            .iter()
            .take(a.show)
            .map(|f| {
                let mut snip = f.snippet.clone();
                if snip.chars().count() > 72 {
                    snip = snip.chars().take(69).collect::<String>() + "...";
                }
                vec![
                    format!("{:.1}", f.risk_score),
                    f.sink_class.clone(),
                    format!("{}:{}", f.file, f.line),
                    f.function.clone(),
                    if f.guard_present { f.guard_kind.clone() } else { "none".into() },
                    snip,
                ]
            })
            .collect::<Vec<_>>(),
    );
    println!(
        "scan-sinks: files={} lines={} findings={} (top={}) out={}; every hit is a hypothesis: replay it and read the source at the site",
        rep.files_scanned,
        rep.lines_scanned,
        rep.total_findings,
        rep.findings.len(),
        a.out
    );
    0
}
