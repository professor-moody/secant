use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::read::{read_model, ReadOptions};

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Read selected weights without inference: Frobenius norms, leading singular values, high-energy columns and optional vocabulary projections. Anomaly flags are informational and retain exit status 0.\n\n--marchenko-pastur adds, per 2-D tensor, the bulk edge sigma_MP = sigma_bulk * (1 + sqrt(d_in/d_out)) of an i.i.d. matrix with the same energy (the computed outliers removed) and the spike ratio sigma_1/sigma_MP, plus a report-level ranking that carries the top promoted vocabulary tokens when --vocab-tensor is given. Trained weights are not i.i.d.: compare ratios across layers or against the base checkpoint.\n\n--reference <base> subtracts the same-named tensor of a trusted base checkpoint (identical logical shape; float or Q8_0/Q4_K; GGUF and SafeTensors may be mixed) before any analysis, so Frobenius norm, column statistics, anomalies, singular spectrum, bulk edge and the u_1 vocabulary projection describe Delta W = W - W_ref. A tensor the reference lacks or holds with another shape is reported skipped, never compared. The vocab tensor is read from --model.",
    after_help = "Examples:\n  secant read --model model.gguf --tensor 'blk.0.*' --top-k-singular 4\n  secant read --model model.safetensors --detect-anomalies --json\n  secant read --model model.gguf --marchenko-pastur --vocab-tensor output.weight --json\n  secant read --model edited.gguf --reference base.gguf --vocab-tensor output.weight --json"
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
    /// Report the Marchenko-Pastur bulk edge sigma_MP and the spike ratio sigma_1/sigma_MP per 2-D tensor (informational)
    #[arg(long)]
    marchenko_pastur: bool,
    /// Flag threshold on sigma_1/sigma_MP (default 1.5; requires --marchenko-pastur)
    #[arg(long, allow_negative_numbers = true)]
    mp_threshold: Option<f64>,
    /// Trusted base checkpoint: statistics, spectrum and vocab projection are then of Delta W = W - W_ref (same tensor name and logical shape)
    #[arg(long, default_value = "")]
    reference: String,
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
        marchenko_pastur: a.marchenko_pastur,
        mp_threshold: a.mp_threshold.unwrap_or(0.0),
        reference: a.reference.clone(),
    };
    if a.mp_threshold == Some(0.0) {
        return usage("--mp-threshold must be > 0");
    }
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
    print_reference(&rep);
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
        print_bulk(t);
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

/// Reference-checkpoint provenance and the Delta W notice (`--reference`).
pub(crate) fn print_reference(rep: &secant_core::read::ReadReport) {
    if let Some(r) = &rep.reference {
        field("Reference", &r.path);
        field("Reference SHA-256", &r.pre_sha256);
        field("Reference unchanged", badge(r.input_unchanged));
        field(
            "Basis",
            format!(
                "Delta W = W - W_ref; {} compared, {} skipped of {} selected",
                r.tensors_compared, r.tensors_skipped, rep.tensors_selected
            ),
        );
    }
}

/// One human line for the `--marchenko-pastur` block of a tensor.
pub(crate) fn print_bulk(t: &secant_core::read::TensorSpectrum) {
    if let Some(b) = &t.spectral_bulk {
        match (b.sigma_mp, b.sigma1_over_mp) {
            (Some(mp), Some(r)) => print!(
                " MP sigma_mp={:.6e} sigma1/sigma_mp={:.4} outliers={}{}{}",
                mp,
                r,
                b.outliers,
                if b.spectrum_exhausted { "+" } else { "" },
                if b.flagged { " OUTSIDE-BULK" } else { "" }
            ),
            _ => print!(" MP {}", b.status),
        }
    }
}
