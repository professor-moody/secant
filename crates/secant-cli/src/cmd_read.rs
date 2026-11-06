use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::read::{read_model, ReadOptions};

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Read selected weights without inference: Frobenius norms, leading singular values, high-energy columns and optional vocabulary projections. Anomaly flags are informational and retain exit status 0.",
    after_help = "Examples:\n  secant read --model model.gguf --tensor 'blk.0.*' --top-k-singular 4\n  secant read --model model.safetensors --detect-anomalies --json"
)]
pub struct Args {
    /// Input .gguf or .safetensors (read only)
    #[arg(long)]
    model: Option<String>,
    /// Optional case-sensitive path.Match glob over tensor names (default: all)
    #[arg(long, default_value = "")]
    tensor: String,
    /// Highest-L2 logical input columns to list, 1..1048576
    #[arg(long, default_value_t = 64)]
    top_cols: usize,
    /// Leading singular triplets, 1..64
    #[arg(long, default_value_t = 4)]
    top_k_singular: usize,
    /// 2-D unembedding tensor (lm_head / output.weight) to project u_1 through
    #[arg(long, default_value = "")]
    vocab_tensor: String,
    /// Flag zero/collapsed rows and column-L2 spikes (informational; exit stays 0)
    #[arg(long)]
    detect_anomalies: bool,
    /// Optional NEW directory for u1_/v1_<tensor>.npy
    #[arg(long, default_value = "")]
    vectors_out: String,
    #[arg(long)]
    json: bool,
}

pub fn run(a: &Args) -> i32 {
    let Some(model) = &a.model else { return usage("--model is required") };
    let o = ReadOptions {
        model: model.clone(),
        tensor: a.tensor.clone(),
        top_cols: a.top_cols,
        top_k_singular: a.top_k_singular,
        vocab_tensor: a.vocab_tensor.clone(),
        detect_anomalies: a.detect_anomalies,
        vectors_out: a.vectors_out.clone(),
    };
    let rep = match read_model(o) {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    if a.json {
        return match emit_json(&rep) {
            Ok(()) => 0,
            Err(e) => fail(&e),
        };
    }
    section("Weight-space analysis");
    field("Model", &rep.model);
    field("Selection", format!("{} / {} tensors; {} analyzed", rep.tensors_selected, rep.tensors_in_file, rep.tensors_analyzed));
    field("Input unchanged", badge(rep.input_unchanged));
    field("SHA-256", &rep.pre_sha256);
    table(
        &["TENSOR", "DTYPE", "SHAPE", "FROBENIUS NORM", "STATUS"],
        &rep.tensors
            .iter()
            .map(|t| {
                vec![
                    t.name.clone(),
                    t.dtype.clone(),
                    format!("{:?}", t.shape),
                    t.frobenius.map(|x| format!("{:.6e}", x)).unwrap_or_else(|| "UNKNOWN".into()),
                    t.status.clone(),
                ]
            })
            .collect::<Vec<_>>(),
    );
    for t in &rep.tensors {
        section(&t.name);
        table(
            &["SINGULAR VALUE", "SIGMA", "RELATIVE TO FIRST"],
            &t.singular
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    vec![
                        (i + 1).to_string(),
                        format!("{:.6e}", s.sigma),
                        format!("{:.6}", if t.singular[0].sigma > 0.0 { s.sigma / t.singular[0].sigma } else { 0.0 }),
                    ]
                })
                .collect::<Vec<_>>(),
        );
        if let Some(an) = &t.anomalies {
            if an.flagged {
                print!(" ANOMALIES zero_rows={} collapsed_rows={} spike_cols={}", an.zero_rows, an.collapsed_rows, an.spike_cols);
            }
        }
        println!();
        if let Some(v) = &t.vocab {
            if v.status == "ok" {
                print!("      promoted tokens:");
                for e in v.promoted.iter().take(8) {
                    print!(" {}({:.3e})", e.index, e.value);
                }
                print!("\n      suppressed tokens:");
                for e in v.suppressed.iter().take(8) {
                    print!(" {}({:.3e})", e.index, e.value);
                }
                println!();
            }
        }
    }
    0
}
