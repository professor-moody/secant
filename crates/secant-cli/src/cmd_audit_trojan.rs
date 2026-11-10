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
    long_about = "Rank 2-D weight tensors by the spectral spike ratio sigma_1/sigma_MP, where sigma_MP = sigma_bulk * (1 + sqrt(d_in/d_out)) is the Marchenko-Pastur bulk edge of an i.i.d. matrix with the tensor's non-spike energy. Rank-1 weight edits and many trojan insertions detach a singular value from the bulk. Trained weights already sit above the i.i.d. edge, so each ratio is also compared with the same tensor in the other layers (family blk.*.ffn_down.weight etc.): a sibling outlier has a modified z-score above 3.5 in a family of at least 4. Row/column anomalies are reported too, and with --vocab-tensor the top promoted vocabulary tokens of u_1 are listed beside each spike. With --reference <base>, the ranking is of Delta W = W - W_ref against a trusted base checkpoint: an edit that hides inside a trained matrix's own spectrum becomes its dominant direction, and the promoted tokens are those of Delta W's u_1 through --vocab-tensor. Equivalent to `secant read --detect-anomalies --marchenko-pastur`. Informational: exit 0 whether or not anything is flagged.\n\n--rank-k <K> (K in 1, 2, 4, 8) adds the cumulative top-k singular energy E_k = sum_{i<=k} sigma_i^2 / ||M||_F^2 of each 2-D tensor (M = W, or W - W_ref under --reference), because a rank-k trojan hides from sigma_1 alone. It is never above the true value (Ky Fan), and E_1 is the existing sigma1_energy_fraction. Tensors are ranked and judged against their layer siblings by E_k, or by the largest smaller k when the computed spectrum is exhausted (E_k = 1 says nothing). With --vocab-tensor, every token's coherence c_t = ||Q^T e_t||^2 / ||e_t||^2 with the rank-k left singular subspace Q is listed beside the per-direction promoted tokens; a random token sits near k/d_model. --rank-k never changes any existing field.\n\nA PEFT LoRA adapter (adapter_model.safetensors with lora_A / lora_B tensors) is detected by its tensor names and audited natively: for each module the spectrum of Delta W = s B A (s = lora_alpha / r from the sibling adapter_config.json or --adapter-config; rank_pattern, alpha_pattern and use_rslora honoured) is computed exactly from thin QR factors without forming the product, --rank-k defaults to 8, and an adapter has no unembedding of its own, so --vocab-tensor reads from --vocab-model. --reference, --mp-threshold and embedding / DoRA / non-LoRA adapters are not defined and are reported or rejected, never guessed.",
    after_help = "Examples:\n  secant audit-trojan --model edited.gguf --reference base.gguf --vocab-tensor output.weight\n  secant audit-trojan --model model.gguf --tensor 'blk.*.ffn_down.weight' --vocab-tensor output.weight\n  secant audit-trojan --model edited.safetensors --mp-threshold 2.0 --json\n  secant audit-trojan --model edited.safetensors --reference base.safetensors --rank-k 4 --vocab-tensor lm_head.weight\n  secant audit-trojan --model adapter_model.safetensors --rank-k 4 --vocab-tensor lm_head.weight --vocab-model base.safetensors\n\nTrained weights are not i.i.d.: compare ratios against the base checkpoint or sibling layers."
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
    /// Trusted base checkpoint: rank Delta W = W - W_ref instead of W (same tensor name and logical shape)
    #[arg(long, default_value = "")]
    reference: String,
    /// Cumulative top-k singular energy and vocabulary coherence, k in {1, 2, 4, 8} (adapters default to 8)
    #[arg(long)]
    rank_k: Option<usize>,
    /// PEFT LoRA adapter only: adapter_config.json (default: the file beside --model)
    #[arg(long, default_value = "")]
    adapter_config: String,
    /// PEFT LoRA adapter only: checkpoint that holds --vocab-tensor (an adapter has no unembedding)
    #[arg(long, default_value = "")]
    vocab_model: String,
    #[arg(long)]
    json: bool,
}

/// Compact `index(value)` list for a token column.
fn token_list(v: &[secant_core::read::VecEntry]) -> String {
    v.iter().map(|e| format!("{}({:.3})", e.index, e.value)).collect::<Vec<_>>().join(" ")
}

/// The `--rank-k` ranking table (`what` names the matrix the energy is a fraction of).
fn rank_k_table(what: &str, rows: &[secant_core::rankk::RankKOutlier], top: usize) {
    if rows.is_empty() {
        println!("No tensor has a defined rank-k energy ({}).", what);
        return;
    }
    let k = rows[0].requested_k;
    section(&format!("Rank-{} spectral energy of {}", k, what));
    table(
        &[
            "TENSOR",
            "E_1",
            &format!("E_{}", k),
            "RANKED BY",
            "CONVERGED",
            "SIBLING Z",
            "SIBLING OUTLIER",
            "COHERENT TOKENS (c_t)",
            "TOP/CHANCE",
        ],
        &rows
            .iter()
            .take(top)
            .map(|r| {
                vec![
                    r.tensor.clone(),
                    format!("{:.4}", r.sigma1_energy),
                    format!("{:.4}", r.cumulative_energy),
                    format!("E_{}={:.4}{}", r.ranking_k, r.ranking_energy, if r.saturated { " SATURATED" } else { "" }),
                    if r.converged { "yes".into() } else { "NO (lower bound)".into() },
                    r.sibling.robust_z.map(|z| format!("{:.2}", z)).unwrap_or_else(|| "n/a".into()),
                    match r.sibling.sibling_outlier {
                        Some(true) => "YES".into(),
                        Some(false) => "no".into(),
                        None => format!("n/a ({} in family)", r.sibling.family_size),
                    },
                    token_list(&r.coherent_tokens),
                    r.top_over_chance.map(|x| format!("{:.1}x", x)).unwrap_or_default(),
                ]
            })
            .collect::<Vec<_>>(),
    );
}

/// `audit-trojan` on a PEFT LoRA adapter (`adapter_model.safetensors`): the spectrum of B A per module.
fn run_adapter(a: &Args, model: &str) -> i32 {
    if !a.reference.is_empty() {
        return usage("--reference is not defined for a LoRA adapter (it already is a delta); compare adapters by running each");
    }
    if a.mp_threshold.is_some() {
        return usage("--mp-threshold is not defined for a LoRA adapter (B A has rank <= r, so no Marchenko-Pastur bulk exists)");
    }
    let k = a.rank_k.unwrap_or(8);
    let o = secant_core::lora::LoraAuditOptions {
        model: model.to_string(),
        adapter_config: a.adapter_config.clone(),
        tensor: a.tensor.clone(),
        rank_k: k,
        vocab_tensor: a.vocab_tensor.clone(),
        vocab_model: a.vocab_model.clone(),
    };
    let rep = match secant_core::lora::scan_adapter(o) {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    if a.json {
        return match emit_json(&rep) {
            Ok(()) => 0,
            Err(e) => fail(&e),
        };
    }
    section("LoRA adapter spectral trojan audit");
    field("Adapter", &rep.model);
    field("Input unchanged", badge(rep.input_unchanged));
    field("SHA-256", &rep.pre_sha256);
    match &rep.adapter_config {
        Some(c) => {
            field("Adapter config", &c.path);
            field("Config SHA-256", &c.sha256);
            field(
                "Scaling",
                format!(
                    "lora_alpha {} / {}{}",
                    c.lora_alpha.map(|x| x.to_string()).unwrap_or_else(|| "(default 8)".into()),
                    if c.use_rslora { "sqrt(r) (use_rslora)" } else { "r" },
                    if c.rank_pattern_entries + c.alpha_pattern_entries > 0 { "; rank/alpha patterns applied" } else { "" }
                ),
            );
        }
        None => field("Adapter config", "not found: Delta W taken as B A (scale 1); energy fractions do not depend on it"),
    }
    if let Some(v) = &rep.vocab_source {
        field("Vocab source", format!("{} : {}", v.path, v.tensor));
        field("Vocab unchanged", badge(v.input_unchanged));
    }
    field(
        "Modules",
        format!("{} found, {} selected, {} with a non-zero delta", rep.modules_found, rep.modules_selected, rep.modules_scanned),
    );
    let skipped = rep.modules.iter().filter(|m| m.status != "ok" && m.status != "zero_delta").count();
    let zero = rep.modules.iter().filter(|m| m.status == "zero_delta").count();
    field("Zero delta", zero);
    field("Skipped / nonfinite", skipped);
    field(
        "Other tensors",
        format!(
            "{} ({} embedding adapter, {} DoRA magnitude); not scanned",
            rep.other_tensors.count, rep.other_tensors.embedding_adapters, rep.other_tensors.dora_magnitude_vectors
        ),
    );
    for m in rep.modules.iter().filter(|m| m.status != "ok" && m.status != "zero_delta") {
        println!("  {}: {}", m.name, m.status);
    }
    rank_k_table("Delta W = s B A", &rep.rank_k_outliers, a.top);
    let sib = rep.rank_k_outliers.iter().filter(|r| r.sibling.sibling_outlier == Some(true)).count();
    println!(
        "audit-trojan (adapter): {} sibling outlier(s) in energy, of {} ranked module(s); informational (exit 0). A concentrated adapter is a lead to read, not proof of a backdoor.",
        sib,
        rep.rank_k_outliers.len()
    );
    0
}

pub fn run(a: &Args) -> i32 {
    let Some(model) = &a.model else { return usage("--model is required") };
    if a.mp_threshold == Some(0.0) {
        return usage("--mp-threshold must be > 0");
    }
    if let Some(k) = a.rank_k {
        if let Err(e) = secant_core::rankk::validate_rank_k(k) {
            return usage(&e.to_string());
        }
        if k == 0 {
            return usage("--rank-k must be one of 1, 2, 4, 8");
        }
    }
    // A load error is not an adapter verdict: the base-checkpoint path below reports it as before.
    let adapter = secant_core::container::load_layout(model).map(|l| secant_core::lora::is_peft_adapter(&l)).unwrap_or(false);
    if adapter {
        return run_adapter(a, model);
    }
    if !a.adapter_config.is_empty() || !a.vocab_model.is_empty() {
        return usage("--adapter-config and --vocab-model apply to a PEFT LoRA adapter; this model is not one (a base checkpoint reads --vocab-tensor from --model)");
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
        reference: a.reference.clone(),
        rank_k: a.rank_k.unwrap_or(0),
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
    crate::cmd_read::print_reference(&rep);
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
    if rep.reference.is_some() {
        // Delta W view: an exactly low-rank difference has no Marchenko-Pastur
        // bulk left (status "undefined"), so it never reaches the ranking
        // above; list every compared tensor by how much of the base it moved.
        let mut moved: Vec<_> =
            rep.tensors.iter().filter_map(|t| Some((t, t.reference_diff.as_ref().filter(|d| d.status == "compared")?))).collect();
        moved.sort_by(|x, y| {
            y.1.relative_delta
                .unwrap_or(0.0)
                .partial_cmp(&x.1.relative_delta.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(x.0.name.cmp(&y.0.name))
        });
        table(
            &["TENSOR (DELTA W)", "||DW||/||W_REF||", "SIGMA_1", "SIGMA_1^2/||DW||^2", "PROMOTED TOKENS OF DW u_1"],
            &moved
                .iter()
                .take(a.top)
                .map(|(t, d)| {
                    vec![
                        t.name.clone(),
                        d.relative_delta.map(|x| format!("{:.4e}", x)).unwrap_or_else(|| "n/a".into()),
                        t.singular.first().map(|s| format!("{:.6e}", s.sigma)).unwrap_or_else(|| {
                            if d.identical == Some(true) {
                                "identical".into()
                            } else {
                                "n/a".into()
                            }
                        }),
                        d.sigma1_energy_fraction.map(|x| format!("{:.4}", x)).unwrap_or_else(|| "n/a".into()),
                        t.vocab
                            .as_ref()
                            .filter(|v| v.status == "ok")
                            .map(|v| {
                                v.promoted.iter().take(8).map(|e| format!("{}({:.3e})", e.index, e.value)).collect::<Vec<_>>().join(" ")
                            })
                            .unwrap_or_default(),
                    ]
                })
                .collect::<Vec<_>>(),
        );
        let skipped = rep.tensors.iter().filter(|t| t.reference_diff.as_ref().is_some_and(|d| d.status != "compared")).count();
        if skipped > 0 {
            println!("{} selected tensor(s) not compared (absent from the reference or a different shape); see `--json` status.", skipped);
        }
    }
    if let Some(rk) = &rep.rank_k_outliers {
        rank_k_table(if rep.reference.is_some() { "Delta W = W - W_ref" } else { "W" }, rk, a.top);
    }
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
