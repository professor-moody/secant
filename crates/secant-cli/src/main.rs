//! `secant`: weight-space programming, tensor-file security probes and
//! Project-Then-Heal. Exit codes follow the Crucible oracle: 0 = passed,
//! 1 = a gate ran and failed, 2 = usage / I-O / undeterminable.

mod cmd_compose;
mod cmd_heal;
mod cmd_inspect;
mod cmd_patch;
mod cmd_probe;
mod cmd_quant;
mod cmd_read;
mod cmd_sanitize;
mod cmd_verify_patch;
mod cmd_verify_subspace;
mod util;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "secant", version, about = "Weight-space programming, tensor security probes and Project-Then-Heal")]
struct Cli {
    /// Reproduce the arm64 Go oracle's fused multiply-adds (default: portable, unfused float64).
    #[arg(long, global = true)]
    go_arm64_fma: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Inspect architecture, dtype layout, payload offsets and the Qwen3.5 ssm_out interleave
    Inspect(cmd_inspect::Args),
    /// Static weight-space reader: Frobenius norm, top singular triplets, top columns, anomalies, vocab projection
    Read(cmd_read::Args),
    /// Orthogonalize directions and calibrate a patch plan
    Compose(cmd_compose::Args),
    /// Apply a plan's weight edits directly to payload bytes
    Patch(cmd_patch::Args),
    /// Verify header identity and 100% byte-identity of non-target tensors
    VerifyPatch(cmd_verify_patch::Args),
    /// Scan static weight values (Class I-IV) and optionally publish a clean copy
    Sanitize(cmd_sanitize::Args),
    /// Generate header-valid boundary-value weight fixtures
    Probe(cmd_probe::Args),
    /// Project-then-heal: lock a LoRA's B matrices out of protected output directions
    Heal(cmd_heal::Args),
    /// Verify a healed model's weight difference is orthogonal to the protected directions
    VerifySubspace(cmd_verify_subspace::Args),
    /// Q8_0 / Q4_K kernels, commutation witness and block-aligned Q8_0 surgery
    Quant(cmd_quant::Args),
}

fn main() {
    let cli = Cli::parse();
    secant_core::numeric::set_go_arm64_fma(cli.go_arm64_fma);
    let code = match &cli.cmd {
        Cmd::Inspect(a) => cmd_inspect::run(a),
        Cmd::Read(a) => cmd_read::run(a),
        Cmd::Compose(a) => cmd_compose::run(a),
        Cmd::Patch(a) => cmd_patch::run(a),
        Cmd::VerifyPatch(a) => cmd_verify_patch::run(a),
        Cmd::Sanitize(a) => cmd_sanitize::run(a),
        Cmd::Probe(a) => cmd_probe::run(a),
        Cmd::Heal(a) => cmd_heal::run(a),
        Cmd::VerifySubspace(a) => cmd_verify_subspace::run(a),
        Cmd::Quant(a) => cmd_quant::run(a),
    };
    std::process::exit(code);
}
