//! `secant audit-trojan`: `secant read --detect-anomalies --marchenko-pastur`
//! with a ranked spike summary (Milestone S-2). Same `secant.wsp.read.v1`
//! receipt; the human report leads with tensors whose leading singular value
//! sits furthest outside the Marchenko-Pastur bulk, with the top promoted
//! vocabulary tokens of `u_1` when `--vocab-tensor` is given.
//!
//! Informational, like `read`: exit 0 whether or not anything is flagged
//! (2 on usage or I/O errors). A spike is a lead to compare against the base
//! checkpoint or sibling layers, not proof of a backdoor.
use crate::util::*;
use clap::Args as ClapArgs;
use secant_core::read::{read_model, ReadOptions};

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Rank 2-D weight tensors by the spectral spike ratio sigma_1/sigma_MP, where sigma_MP = sigma_bulk * (1 + sqrt(d_in/d_out)) is the Marchenko-Pastur bulk edge of an i.i.d. matrix with the tensor's non-spike energy. Rank-1 weight edits and many trojan insertions detach a singular value from the bulk. Trained weights already sit above the i.i.d. edge, so each ratio is also compared with the same tensor in the other layers (family blk.*.ffn_down.weight etc.): a sibling outlier has a modified z-score above 3.5 in a family of at least 4. Row/column anomalies are reported too, and with --vocab-tensor the top promoted vocabulary tokens of u_1 are listed beside each spike. Equivalent to `secant read --detect-anomalies --marchenko-pastur`. Informational: exit 0 whether or not anything is flagged.",
    after_help = "Examples:\n  secant audit-trojan --model model.gguf --tensor 'blk.*.ffn_down.weight' --vocab-tensor output.weight\n  secant audit-trojan --model edited.safetensors --mp-threshold 2.0 --json\n\nTrained weights are not i.i.d.: compare ratios against the base checkpoint or sibling layers."
)]
pub struct Args {
    /// Input .gguf or .safetensors (read only)
    #[arg(long)]
    model: Option<String>,
    /// Optional case-sensitive path.Match glob over tensor names (default: all)
    #[arg(long, default_value = "")]
    tensor: String,
    /// 2-D unembedding tensor (lm_head / output.weight) to project u_1 through
    #[arg(long, default_value = "")]
    vocab_tensor: String,
    /// Leading singular triplets per tensor, 1..64 (outliers beyond k are counted as "k+")
    #[arg(long, default_value_t = 4)]
    top_k_singular: usize,
    /// Flag threshold on sigma_1/sigma_MP (default 1.5)
    #[arg(long, allow_negative_numbers = true)]
    mp_threshold: Option<f64>,
    /// Rows of the human ranking (the JSON keeps every tensor)
    #[arg(long, default_value_t = 20)]
    top: usize,
    #[arg(long)]
    json: bool,
}

pub fn run(a: &Args) -> i32 {
    let Some(model) = &a.model else { return usage("--model is required") };
    if a.mp_threshold == Some(0.0) {
        return usage("--mp-threshold must be > 0");
    }
    let o = ReadOptions {
        model: model.clone(),
        tensor: a.tensor.clone(),
        top_cols: 8,
        top_k_singular: a.top_k_singular,
        vocab_tensor: a.vocab_tensor.clone(),
        detect_anomalies: true,
        vectors_out: String::new(),
        marchenko_pastur: true,
        mp_threshold: a.mp_threshold.unwrap_or(0.0),
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
    let ranking = rep.spectral_outliers.clone().unwrap_or_default();
    let flagged = ranking.iter().filter(|r| r.flagged).count();
    let sibling = ranking.iter().filter(|r| r.sibling.sibling_outlier == Some(true)).count();
    let unjudged = ranking.iter().filter(|r| r.sibling.sibling_outlier.is_none()).count();
    let undefined = rep.tensors.iter().filter(|t| t.spectral_bulk.as_ref().is_some_and(|b| b.sigma1_over_mp.is_none())).count();
    let threshold = rep.options.mp_threshold;
    section("Spectral trojan audit");
    field("Model", &rep.model);
    field("Selection", format!("{} / {} tensors; {} analyzed", rep.tensors_selected, rep.tensors_in_file, rep.tensors_analyzed));
    field("Input unchanged", badge(rep.input_unchanged));
    field("SHA-256", &rep.pre_sha256);
    field("Bulk threshold", format!("sigma_1/sigma_MP > {}", threshold));
    field("Ranked tensors", ranking.len());
    field(
        "Sibling outliers",
        format!("{} (robust z > {} within a layer family of >= {})", sibling, secant_core::mp::SIBLING_Z, secant_core::mp::SIBLING_MIN),
    );
    field("Family too small", unjudged);
    field("Outside i.i.d. bulk", format!("{} (expected for trained weights)", flagged));
    field("Undefined bulk", undefined);
    let skipped: Vec<_> = rep.tensors.iter().filter(|t| t.spectral_bulk.is_none()).collect();
    field("No spectrum", skipped.len());
    table(
        &["TENSOR", "SIGMA_1", "SIGMA_MP", "RATIO", "OUTLIERS", "SIBLING Z", "SIBLING OUTLIER", "PROMOTED TOKENS"],
        &ranking
            .iter()
            .take(a.top)
            .map(|r| {
                let exhausted = rep
                    .tensors
                    .iter()
                    .find(|t| t.name == r.tensor)
                    .and_then(|t| t.spectral_bulk.as_ref())
                    .is_some_and(|b| b.spectrum_exhausted);
                vec![
                    r.tensor.clone(),
                    format!("{:.6e}", r.sigma_1),
                    format!("{:.6e}", r.sigma_mp),
                    format!("{:.4}", r.sigma1_over_mp),
                    format!("{}{}", r.outliers, if exhausted { "+" } else { "" }),
                    r.sibling.robust_z.map(|z| format!("{:.2}", z)).unwrap_or_else(|| "n/a".into()),
                    match r.sibling.sibling_outlier {
                        Some(true) => "YES".into(),
                        Some(false) => "no".into(),
                        None => format!("n/a ({} in family)", r.sibling.family_size),
                    },
                    r.promoted_tokens.iter().map(|e| format!("{}({:.3e})", e.index, e.value)).collect::<Vec<_>>().join(" "),
                ]
            })
            .collect::<Vec<_>>(),
    );
    let anomalous: Vec<_> = rep.tensors.iter().filter(|t| t.anomalies.as_ref().is_some_and(|x| x.flagged)).collect();
    if !anomalous.is_empty() {
        table(
            &["TENSOR", "ZERO ROWS", "COLLAPSED ROWS", "SPIKE COLUMNS"],
            &anomalous
                .iter()
                .map(|t| {
                    let x = t.anomalies.as_ref().unwrap();
                    vec![t.name.clone(), x.zero_rows.to_string(), x.collapsed_rows.to_string(), x.spike_cols.to_string()]
                })
                .collect::<Vec<_>>(),
        );
    }
    println!(
        "audit-trojan: {} sibling outlier(s), {} outside the i.i.d. bulk, of {} ranked; informational (exit 0). A spike is a lead to compare against the base checkpoint, not proof of tampering.",
        sibling,
        flagged,
        ranking.len()
    );
    0
}
