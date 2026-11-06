//! `secant heal`: project-then-heal. Lock a LoRA's B matrices out of the
//! protected output directions, optionally merging the locked update into a
//! copy of a base model. Port of the reference oracle's `wsp heal`.
//!
//! Exit: 0 healed; 1 a leakage gate failed; 2 usage, I/O or unsupported input.
use crate::util::{badge, emit_json, fail, field, section, table, usage};
use clap::Args as ClapArgs;
use secant_core::heal::{fmt_e, heal, HealOptions, DEFAULT_SUBSPACE_TOL};
use secant_core::Error;

#[derive(ClapArgs, Debug)]
#[command(
    after_help = "Example:\n  secant heal --lora adapter.safetensors --vectors d.npy --out locked.safetensors\n\nFor a merged checkpoint, also pass --base base.gguf --merge-out healed.gguf."
)]
#[command(
    long_about = "Orthonormalize the protected output directions D (every non-h_* .npy/.json in a directory, or one .npy: a vector or [k, d] rows) by two-pass modified Gram-Schmidt in float64, then replace every LoRA B [d_out x r] by B_locked = (I - D D^T) B so D^T B_locked = 0. Measured, not assumed: the float64 residual must be < 1e-12 and the residual re-read from the written file < 1e-6, else exit 1 and nothing is published. B_locked is stored as F32 (F64 if the input was F64). lora_A is copied byte-for-byte. With --base and --merge-out, DeltaW = scale * B_locked * A is added into a COPY of the base in logical [d_out x d_in] orientation (GGUF and SafeTensors; HF module names map to blk.N.* for GGUF, and HF-order lora_A columns are permuted onto the Qwen3.5 ssm_out interleave). The merged result is measured on the REALIZED weights and published only if max|D^T dW| <= --tol. Outputs are all-or-nothing and never overwrite."
)]
pub struct Args {
    /// Input LoRA adapter (.safetensors, lora_A/lora_B pairs)
    #[arg(long, default_value = "")]
    lora: String,
    /// Protected directions: directory of d_*.npy/.json, or one .npy
    #[arg(long, default_value = "")]
    vectors: String,
    /// New healed LoRA .safetensors
    #[arg(long, default_value = "")]
    out: String,
    /// Base model (.gguf/.safetensors) to merge into a copy of (needs --merge-out)
    #[arg(long, default_value = "")]
    base: String,
    /// New merged model path (needs --base)
    #[arg(long, default_value = "")]
    merge_out: String,
    /// Merge scale on B_locked*A (lora_alpha / r)
    #[arg(long, default_value_t = 1.0, allow_negative_numbers = true)]
    scale: f64,
    /// Merge gate: max |D^T dW| over realized weights
    #[arg(long, default_value_t = DEFAULT_SUBSPACE_TOL, allow_negative_numbers = true)]
    tol: f64,
    /// Emit the report to stdout
    #[arg(long)]
    json: bool,
}

pub fn run(args: &Args) -> i32 {
    if args.lora.is_empty() || args.vectors.is_empty() || args.out.is_empty() {
        return usage("--lora, --vectors and --out are required");
    }
    let o = HealOptions {
        lora: args.lora.clone(),
        vectors: args.vectors.clone(),
        out: args.out.clone(),
        base: args.base.clone(),
        merge_out: args.merge_out.clone(),
        scale: args.scale,
        tol: args.tol,
    };
    let (rep, res) = heal(&o);
    if args.json {
        if let Some(r) = &rep {
            // Go's encoding/json refuses NaN/Inf; do not let serde write `null` for them.
            if !args.scale.is_finite() || !args.tol.is_finite() {
                return fail(&Error::msg("json: unsupported value: NaN or Inf in report"));
            }
            if let Err(e) = emit_json(r) {
                return fail(&e);
            }
        }
    }
    if let Err(e) = res {
        return fail(&e);
    }
    if !args.json {
        let rep = match rep {
            Some(r) => r,
            None => return usage("heal returned no report"),
        };
        let ps = match &rep.protected {
            Some(p) => p,
            None => return usage("heal returned no protected set"),
        };
        section("Project-Then-Heal");
        field("Leakage gates", "[PASS]");
        table(
            &["MODULE", "RANK", "B FROBENIUS BEFORE", "B FROBENIUS AFTER", "F64 LEAK", "STORED LEAK", "A UNCHANGED"],
            &rep.pairs
                .iter()
                .map(|p| {
                    vec![
                        p.module.clone(),
                        p.rank.to_string(),
                        format!("{:.6e}", p.b_fro_before),
                        format!("{:.6e}", p.b_fro_after),
                        format!("{:.6e}", p.leak_after_f64),
                        format!("{:.6e}", p.leak_after_serialize),
                        badge(p.a_unchanged).into(),
                    ]
                })
                .collect::<Vec<_>>(),
        );
        println!(
            "heal: {}; {} module(s), protected dim {}, k={}; max leak f64 {}, serialized {}; {} sha256 {}",
            rep.status,
            rep.pairs.len(),
            ps.dim,
            ps.k,
            fmt_e(rep.max_leak_f64, 3),
            fmt_e(rep.max_leak_serial, 3),
            o.out,
            rep.out_sha256
        );
        if let Some(m) = &rep.merge {
            if let Some(g) = &m.gate {
                println!(
                    "merge: {} sha256 {}; {} target(s); realized max leak {} (tol {}) verdict {}",
                    o.merge_out,
                    m.out_sha256,
                    m.targets.len(),
                    fmt_e(g.max_leak, 3),
                    fmt_e(g.tol, 1),
                    g.verdict
                );
                for t in &g.tensors {
                    if t.delta_is_zero {
                        println!("  WARNING {}: merge changed no element (vacuous)", t.name);
                    }
                }
            }
        }
    }
    0
}
