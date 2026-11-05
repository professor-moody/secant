use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::read::{read_model, ReadOptions};

#[derive(ClapArgs, Debug)]
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
    println!(
        "read {} ({}): {}/{} tensors selected, {} analyzed; sha256 {} unchanged={}",
        rep.model, rep.format, rep.tensors_selected, rep.tensors_in_file, rep.tensors_analyzed, rep.pre_sha256, rep.input_unchanged
    );
    for t in &rep.tensors {
        print!("  {} {} {:?}: {}", t.name, t.dtype, t.shape, t.status);
        if let Some(f) = t.frobenius {
            print!(" fro={:.6e}", f);
        }
        for (i, s) in t.singular.iter().enumerate() {
            print!(" s{}={:.6e}", i + 1, s.sigma);
        }
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
