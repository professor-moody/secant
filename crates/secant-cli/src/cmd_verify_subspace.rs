//! `secant verify-subspace`: verify a healed model's weight difference is
//! orthogonal to the protected directions. Port of the reference oracle's `wsp verify-subspace`.
//!
//! For each named tensor compute dW = W_healed - W_base from the stored payload
//! bytes of both files and report max over directions k and input columns c of
//! |D_k^T dW[:, c]|. Verdict pass requires every tensor <= --tol (default 1e-5).
//! A tensor that is missing, differently shaped, not a 2-D float matrix, has
//! d_out different from the direction dimension, or holds NaN/Inf is verdict
//! unknown (exit 2), never pass. Exit 0 pass; 1 leak; 2 unknown or usage/I-O error.
use crate::util::{emit_json, fail, field, section, split_csv, table, usage};
use clap::Args as ClapArgs;
use secant_core::heal::{fmt_e, fmt_g, verify_subspace, VerifySubspaceOptions, DEFAULT_SUBSPACE_TOL};
use secant_core::Error;

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Measure protected-direction leakage on the realized stored weight difference, W_healed - W_base. All named tensors must satisfy --tol. Missing, incompatible or non-finite tensors produce UNKNOWN (exit 2), never a pass.",
    after_help = "Example:\n  secant verify-subspace --base base.gguf --healed healed.gguf --vectors d.npy --tensors blk.0.ffn_down.weight --tol 1e-5"
)]
pub struct Args {
    /// Surgical (pre-heal) model
    #[arg(long, default_value = "")]
    base: String,
    /// Healed model
    #[arg(long, default_value = "")]
    healed: String,
    /// Protected directions: directory or one .npy
    #[arg(long, default_value = "")]
    vectors: String,
    /// Comma-separated protected tensor names (as named inside the models)
    #[arg(long, default_value = "")]
    tensors: String,
    /// Maximum allowed |D^T dW|
    #[arg(long, default_value_t = DEFAULT_SUBSPACE_TOL, allow_negative_numbers = true)]
    tol: f64,
    /// Emit the report as JSON
    #[arg(long)]
    json: bool,
}

pub fn run(args: &Args) -> i32 {
    let tensors = split_csv(&args.tensors);
    if args.base.is_empty() || args.healed.is_empty() || args.vectors.is_empty() || tensors.is_empty() {
        return usage("--base, --healed, --vectors and --tensors are required");
    }
    let o = VerifySubspaceOptions {
        base: args.base.clone(),
        healed: args.healed.clone(),
        vectors: args.vectors.clone(),
        protected: None,
        tensors,
        tol: args.tol,
    };
    let rep = match verify_subspace(&o) {
        Ok(r) => r,
        Err(e) => return fail(&Error::msg(e.message().to_string())),
    };
    if args.json {
        if let Err(e) = emit_json(&rep) {
            return fail(&e);
        }
    } else {
        section("Protected-subspace verification");
        field(
            "Leakage gate",
            match rep.verdict.as_str() {
                "pass" => "[PASS]",
                "leak" => "[FAIL]",
                _ => "[UNKNOWN]",
            },
        );
        let (dim, k) = rep.protected.as_ref().map(|p| (p.dim, p.k)).unwrap_or((0, 0));
        println!(
            "verify-subspace: {} (tol {}, max leak {}, protected dim {}, k={})",
            rep.verdict,
            fmt_e(rep.tol, 1),
            fmt_e(rep.max_leak, 3),
            dim,
            k
        );
        table(
            &["TENSOR", "GATE", "MAX LEAKAGE", "DELTA FROBENIUS", "DETAIL"],
            &rep.tensors
                .iter()
                .map(|t| {
                    vec![
                        t.name.clone(),
                        match t.verdict.as_str() {
                            "pass" => "[PASS]",
                            "leak" => "[FAIL]",
                            _ => "[UNKNOWN]",
                        }
                        .into(),
                        t.max_leak.map(|x| fmt_e(x, 3)).unwrap_or_else(|| "UNKNOWN".into()),
                        fmt_g(t.delta_fro, 4),
                        t.reason.clone(),
                    ]
                })
                .collect::<Vec<_>>(),
        );
        if !rep.unchecked_changed.is_empty() {
            let mut names = rep.unchecked_changed.clone();
            names.sort();
            println!("  changed but not checked: {}", names.join(", "));
        }
    }
    match rep.verdict.as_str() {
        "pass" => 0,
        "leak" => {
            fail(&Error::gate(format!("protected direction leaked: max |D^T dW| {} > tol {}", fmt_e(rep.max_leak, 3), fmt_e(rep.tol, 1))))
        }
        _ => fail(&Error::msg(format!("verify-subspace could not be determined: {}", rep.reason))),
    }
}
