//! Edit plans and research profiles. JSON field names, `omitempty` behaviour
//! and defaults mirror `internal/wsp/{plan,profiles}.go` so a plan written by
//! either implementation is accepted by the other.

use crate::dtype::*;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const PLAN_SCHEMA: &str = "crucible.wsp.plan.v1";
pub const PROFILE_SCHEMA: &str = "crucible.wsp.profile.v1";

fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
}
fn is_zero_i64(v: &i64) -> bool {
    *v == 0
}
fn is_zero_usize(v: &usize) -> bool {
    *v == 0
}
fn is_empty_str(v: &str) -> bool {
    v.is_empty()
}
fn is_empty_vec<T>(v: &[T]) -> bool {
    v.is_empty()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CalibrationSpec {
    pub mode: String,
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub target_shift: f64,
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub gamma: f64,
    #[serde(skip_serializing_if = "is_empty_str")]
    pub base_model: String,
    #[serde(skip_serializing_if = "is_empty_str")]
    pub note: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Edit {
    pub id: String,
    pub role: String,
    pub op: String,
    pub tensor: String,
    #[serde(rename = "tensor_safetensors", skip_serializing_if = "is_empty_str")]
    pub tensor_st: String,
    #[serde(skip_serializing_if = "is_zero_i64")]
    pub layer: i64,
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub sign: f64,
    #[serde(skip_serializing_if = "is_zero_i64")]
    pub top_k: i64,

    #[serde(skip_serializing_if = "is_empty_vec")]
    pub direction: Vec<f64>,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub h_ref: Vec<f64>,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub cols: Vec<i64>,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub coeffs: Vec<f64>,
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub gamma: f64,

    #[serde(skip_serializing_if = "is_empty_str")]
    pub layout: String,

    #[serde(skip_serializing_if = "is_empty_vec")]
    pub semantics: Vec<String>,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub set_bits: Vec<u32>,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub set_ints: Vec<i64>,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub set_indices: Vec<i64>,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub token_rows: Vec<i64>,
    #[serde(skip_serializing_if = "is_empty_str")]
    pub permutation: String,

    #[serde(skip_serializing_if = "is_zero_f64")]
    pub achieved_shift: f64,
    #[serde(rename = "delta_fro_norm_composed", skip_serializing_if = "is_zero_f64")]
    pub delta_fro_norm: f64,
    #[serde(skip_serializing_if = "is_empty_str")]
    pub vector_sha256: String,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub notes: Vec<String>,
}

impl Edit {
    pub fn target_tensor_name(&self, format: &str) -> &str {
        if format == "safetensors" && !self.tensor_st.is_empty() {
            &self.tensor_st
        } else {
            &self.tensor
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GramSchmidtReport {
    pub order: Vec<String>,
    pub pre_cosine: BTreeMap<String, f64>,
    pub post_cosine: BTreeMap<String, f64>,
    pub post_cosine_max_abs: f64,
    pub passes: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Plan {
    pub schema: String,
    pub profile: String,
    pub calibration: CalibrationSpec,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gram_schmidt: Option<GramSchmidtReport>,
    /// Opaque: the oracle's `wsp analyze` binding is not reimplemented; a plan
    /// carrying one round-trips untouched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analysis: Option<serde_json::Value>,
    pub edits: Vec<Edit>,
}

impl Plan {
    pub fn load(path: &str) -> Result<Plan> {
        let data = std::fs::read(path).map_err(|e| Error::msg(format!("read plan {}: {}", path, e)))?;
        let plan: Plan = serde_json::from_slice(&data).map_err(|e| Error::msg(format!("parse plan {}: {}", path, e)))?;
        if plan.schema != PLAN_SCHEMA {
            return Err(Error::msg(format!("plan {}: schema {:?}, want {:?}", path, plan.schema, PLAN_SCHEMA)));
        }
        if plan.edits.is_empty() {
            return Err(Error::msg(format!("plan {}: no edits", path)));
        }
        let mut ids = std::collections::HashSet::new();
        for e in &plan.edits {
            if !ids.insert(e.id.clone()) {
                return Err(Error::msg(format!("plan {}: duplicate edit id {:?}", path, e.id)));
            }
            validate_edit(e).map_err(|er| Error::msg(format!("plan {}: edit {:?}: {}", path, e.id, er)))?;
        }
        Ok(plan)
    }

    /// Indented JSON with a trailing newline (one-space indent, as the oracle).
    pub fn to_json_pretty(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        let fmt = serde_json::ser::PrettyFormatter::with_indent(b" ");
        let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
        self.serialize(&mut ser).map_err(|e| Error::msg(e.to_string()))?;
        out.push(b'\n');
        Ok(out)
    }

    pub fn write(&self, path: &str) -> Result<()> {
        let data = self.to_json_pretty()?;
        std::fs::write(path, data).map_err(|e| Error::msg(format!("write plan {}: {}", path, e)))
    }
}

pub fn validate_edit(e: &Edit) -> std::result::Result<(), String> {
    match e.op.as_str() {
        "rank1-columns" => {
            if e.direction.is_empty() {
                return Err("rank1-columns needs a direction vector".into());
            }
            if e.cols.len() != e.coeffs.len() {
                return Err(format!("rank1-columns cols ({}) and coeffs ({}) must align", e.cols.len(), e.coeffs.len()));
            }
            for c in &e.cols {
                if *c < 0 {
                    return Err(format!("negative column {}", c));
                }
            }
        }
        "projector-subtract" => {
            if e.direction.is_empty() {
                return Err("projector-subtract needs a direction vector".into());
            }
        }
        "set-value" => {
            let n = [e.semantics.len(), e.set_bits.len(), e.set_ints.len()].iter().filter(|l| **l > 0).count();
            if n == 0 {
                return Err("set-value needs semantics, bits, or ints".into());
            }
            if n > 1 {
                return Err("set-value carries more than one of semantics/bits/ints; pick one".into());
            }
            for tok in &e.semantics {
                if !SEMANTIC_TOKENS.contains(&tok.as_str()) {
                    return Err(format!("set-value: unknown semantic token {:?} (want one of {:?})", tok, SEMANTIC_TOKENS));
                }
            }
            if e.token_rows.is_empty()
                && e.set_indices.is_empty()
                && (e.set_bits.len() > 1 || e.set_ints.len() > 1 || e.semantics.len() > 1)
            {
                return Err("set-value with no indices takes exactly one value (whole tensor)".into());
            }
        }
        "permute-columns" => {
            if e.permutation != "qwen35-ssm-out-128-hf-to-gguf" && e.permutation != "qwen35-ssm-out-128-gguf-to-hf" {
                return Err(format!("permute-columns: unknown permutation {:?}", e.permutation));
            }
        }
        other => return Err(format!("unknown op {:?}", other)),
    }
    if !e.layout.is_empty() && e.layout != "qwen35-ssm-out-128" {
        return Err(format!("unknown layout {:?}", e.layout));
    }
    Ok(())
}

// ---------------------------------------------------------------- profiles

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SubspaceDef {
    pub name: String,
    pub dim: String,
    pub capture: String,
    #[serde(rename = "writer_gguf")]
    pub writer_gated: String,
    pub writer_st: String,
    #[serde(skip_serializing_if = "is_empty_str")]
    pub notes: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LayerSelectionRule {
    pub strategy: String,
    pub tool: String,
    pub banked: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProbeSpec {
    #[serde(skip_serializing_if = "is_empty_str")]
    pub dtype: String,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub token_rows: Vec<i64>,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub flat_indices: Vec<i64>,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub semantics: Vec<String>,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub bits: Vec<u32>,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub ints: Vec<i64>,
    #[serde(skip_serializing_if = "is_empty_str")]
    pub permute: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Slot {
    pub id: String,
    pub role: String,
    #[serde(skip_serializing_if = "is_empty_str")]
    pub direction_token: String,
    pub op: String,
    pub tensor_pattern: String,
    #[serde(skip_serializing_if = "is_empty_str")]
    pub tensor_st_pattern: String,
    pub layer: i64,
    pub sign: f64,
    #[serde(skip_serializing_if = "is_zero_i64")]
    pub top_k: i64,
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub gamma: f64,
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub target_shift: f64,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub requires_h_ref: bool,
    pub needs_direction: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe: Option<ProbeSpec>,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Profile {
    pub schema: String,
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "is_empty_str")]
    pub source: String,
    pub subspaces: Vec<SubspaceDef>,
    pub default_target_shift: f64,
    pub default_gamma: f64,
    pub layer_selection: LayerSelectionRule,
    pub slots: Vec<Slot>,
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub ref_fro_norm: f64,
}

fn common_subspaces() -> Vec<SubspaceDef> {
    let s = |name: &str, dim: &str, capture: &str, g: &str, st: &str, notes: &str| SubspaceDef {
        name: name.into(),
        dim: dim.into(),
        capture: capture.into(),
        writer_gated: g.into(),
        writer_st: st.into(),
        notes: notes.into(),
    };
    vec![
        s(
            "resid",
            "d_model",
            "last-token residual stream after the block",
            "blk.N.attn_out.weight / blk.N.ssm_out.weight",
            "model.layers.N.self_attn.o_proj.weight / linear_attn.out_proj.weight",
            "",
        ),
        s(
            "attn",
            "d_model",
            "attention (linear-attn) block output",
            "blk.N.ssm_out.weight",
            "model.layers.N.self_attn.linear_attn.out_proj.weight",
            "Qwen3.5 GDN: GGUF stores the logical input axis permuted in 128-wide value-head blocks (TG-WSP-004)",
        ),
        s("mlp", "d_model", "MLP block output", "blk.N.ffn_down.weight", "model.layers.N.mlp.down_proj.weight", ""),
        s(
            "h",
            "d_ffn",
            "SwiGLU intermediate activation (ffn_down input)",
            "blk.N.ffn_down.weight (column space)",
            "model.layers.N.mlp.down_proj.weight (column space)",
            "",
        ),
    ]
}

fn selection_by_snr(banked: &str) -> LayerSelectionRule {
    LayerSelectionRule {
        strategy: "argmax of precomputed per-subspace SNR (captured upstream, supplied via layer_snr.json) with Borda aggregation across subspaces; the native tool ranks supplied numbers and does not itself compute or split held-out statistics".into(),
        tool: "crucible wsp analyze --in <layer_snr.json> | crucible wsp compose --analysis <analysis.json>".into(),
        banked: banked.into(),
    }
}

const FFN_DOWN: &str = "blk.%d.ffn_down.weight";
const FFN_DOWN_ST: &str = "model.language_model.layers.%d.mlp.down_proj.weight";

fn dir_slot(
    id: &str,
    role: &str,
    token: &str,
    op: &str,
    sign: f64,
    top_k: i64,
    requires_h: bool,
    gamma: f64,
    shift: f64,
    notes: Vec<String>,
) -> Slot {
    Slot {
        id: id.into(),
        role: role.into(),
        direction_token: token.into(),
        op: op.into(),
        tensor_pattern: FFN_DOWN.into(),
        tensor_st_pattern: FFN_DOWN_ST.into(),
        layer: -1,
        sign,
        top_k,
        gamma,
        target_shift: shift,
        requires_h_ref: requires_h,
        needs_direction: true,
        probe: None,
        notes,
    }
}

fn probe_slot(id: &str, role: &str, tensor: &str, tensor_st: &str, layer: i64, probe: ProbeSpec, notes: Vec<String>) -> Slot {
    Slot {
        id: id.into(),
        role: role.into(),
        op: "set-value".into(),
        tensor_pattern: tensor.into(),
        tensor_st_pattern: tensor_st.into(),
        layer,
        probe: Some(probe),
        notes,
        ..Default::default()
    }
}

/// The five built-in research profiles, in listing order.
pub fn builtin_profiles() -> Vec<Profile> {
    let sv = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    vec![
        Profile {
            schema: PROFILE_SCHEMA.into(),
            name: "think-loop-fix".into(),
            description: "B1: two rank-1 column edits (top-k SwiGLU columns of ffn_down) that subtract the think-loop direction from the writer's output, closing unterminated </think> loops. Banked from the 0.8B r3 v2 deltas at L15/L19; layer choice must be re-derived per model scale with wsp analyze.".into(),
            source: "builtin".into(),
            subspaces: common_subspaces(),
            default_target_shift: -4.0,
            default_gamma: 0.30,
            layer_selection: selection_by_snr("L15+L19 (Qwen3.5-0.8B hardcoded mid-layer choice; scale-specific, not a law)"),
            slots: vec![
                dir_slot("B1a", "loop-suppress", "loop", "rank1-columns", -1.0, 64, true, 0.0, 0.0, vec![]),
                dir_slot("B1b", "loop-suppress", "loop", "rank1-columns", -1.0, 64, true, 0.0, 0.0, vec![]),
            ],
            ref_fro_norm: 0.0,
        },
        Profile {
            schema: PROFILE_SCHEMA.into(),
            name: "security-auditor-b1b2b3".into(),
            description: "B1+B2+B3 composition from the track2c all-layer sweep: two loop-suppress rank-1 column edits (d_loop), full projector-subtraction refusal ablation at the SNR-ranked bottleneck layer (d_ref), and a signed root-cause attribution boost (d_root), orthogonalized together across subspaces before any payload byte is touched.".into(),
            source: "builtin".into(),
            subspaces: common_subspaces(),
            default_target_shift: -4.0,
            default_gamma: 0.30,
            layer_selection: selection_by_snr("L15/L19 (B1, banked 0.8B), L12 (B2), L13 (B3) from the 24-layer sweep; hardcoded mid-layers miss the L12 refusal bottleneck"),
            slots: vec![
                dir_slot("B1a", "loop-suppress", "loop", "rank1-columns", -1.0, 64, true, 0.0, 0.0, vec![]),
                dir_slot("B1b", "loop-suppress", "loop", "rank1-columns", -1.0, 64, true, 0.0, 0.0, vec![]),
                dir_slot("B2", "refusal-ablate", "ref", "projector-subtract", -1.0, 0, false, 1.0, 0.0,
                    sv(&["gamma=1.0 subtracts the full projector: d^T W' h = 0 for every h"])),
                dir_slot("B3", "root-boost", "root", "rank1-columns", 1.0, 64, true, 0.0, 4.0, vec![]),
            ],
            ref_fro_norm: 0.0,
        },
        Profile {
            schema: PROFILE_SCHEMA.into(),
            name: "refusal-ablate".into(),
            description: "B2 only: subtract the refusal-direction projector from the SNR-ranked bottleneck writer so its outputs carry no refusal-direction component, without the B1 loop edits or the B3 boost.".into(),
            source: "builtin".into(),
            subspaces: common_subspaces(),
            default_target_shift: 0.0,
            default_gamma: 1.0,
            layer_selection: selection_by_snr("L12 on Qwen3.5-0.8B (resid/attn/mlp/h all peak there); the r1 L15/L19 choice sat on the attn SNR minimum"),
            slots: vec![dir_slot("B2", "refusal-ablate", "ref", "projector-subtract", -1.0, 0, false, 1.0, 0.0, vec![])],
            ref_fro_norm: 0.0,
        },
        Profile {
            schema: PROFILE_SCHEMA.into(),
            name: "adversarial-nan-probe".into(),
            description: "WSP-T1 deterministic-token probe (O1): program exact non-finite and extreme rows into an lm_head/output projection to test how an inference backend behaves when a sampler ranks or compares rows before non-finite filtering. Values are written as dtype-aware semantic tokens so the same probe is bit-exact on F32, BF16, F16, and F64 targets. The exact vendor behavior, sink names, source paths, and PoC token sequences are withheld from this public profile pending coordinated disclosure (Human Gate 2).".into(),
            source: "builtin".into(),
            subspaces: vec![common_subspaces().remove(0)],
            default_target_shift: 0.0,
            default_gamma: 0.0,
            layer_selection: LayerSelectionRule {
                strategy: "token-id driven (output.head); no layer sweep".into(),
                tool: "crucible wsp inspect --model <gguf|safetensors>".into(),
                banked: "token ids 2..9 (first content tokens) as the default probe set".into(),
            },
            slots: vec![
                probe_slot("O1-nan", "probe-nan", "output.weight", "lm_head.weight", -1,
                    ProbeSpec { token_rows: vec![2, 3, 4, 5], semantics: sv(&[SEM_NAN, SEM_NAN, SEM_NAN, SEM_NAN]), ..Default::default() },
                    sv(&["semantic tokens encode per target dtype: F32 0x7FC00000, BF16 0x7FC0, F16 0x7E00, F64 0x7FF8..0"])),
                probe_slot("O1-inf", "probe-inf", "output.weight", "lm_head.weight", -1,
                    ProbeSpec { token_rows: vec![6, 7, 8, 9], semantics: sv(&[SEM_POS_INF, SEM_POS_INF, SEM_POS_INF, SEM_POS_INF]), ..Default::default() },
                    vec![]),
            ],
            ref_fro_norm: 0.0,
        },
        Profile {
            schema: PROFILE_SCHEMA.into(),
            name: "adversarial-moe-router-probe".into(),
            description: "WSP-T1-MOE router robustness probe (F1): program out-of-range integer expert ids into a hash-MoE routing table tensor and pin the surrounding float tensors to extreme values, to test whether an MoE runtime validates gathered expert ids before indexing. The exact vendor sink names, kernel source paths and line ranges, the per-value effects, and the reproduction coordinates are withheld from this public profile pending coordinated disclosure (Human Gate 2); use generic integer witnesses on your own fixtures.".into(),
            source: "builtin".into(),
            subspaces: vec![common_subspaces().remove(0)],
            default_target_shift: 0.0,
            default_gamma: 0.0,
            layer_selection: LayerSelectionRule {
                strategy: "hash-MoE layers (layer indices below the hash_moe count)".into(),
                tool: "crucible wsp inspect --model <safetensors>".into(),
                banked: "layers 0 and 1 of a small hash-MoE CPU fixture".into(),
            },
            slots: vec![
                probe_slot("F1-eid", "probe-eid", "", "model.layers.%d.mlp.gate.tid2eid", 0,
                    ProbeSpec { flat_indices: vec![0, 1, 2, 3], ints: vec![-1, 300, 1 << 30, 9000], ..Default::default() },
                    sv(&["distinct out-of-range int32 ids: negative, moderately and wildly oversized, and non-power-of-two"])),
                probe_slot("F1-gate-zero", "probe-router-zero", "", "model.layers.%d.mlp.gate.weight", 0,
                    ProbeSpec { semantics: sv(&[SEM_ZERO]), ..Default::default() },
                    sv(&["router logits zeroed as a neutral control"])),
                probe_slot("F1-bias-neginf", "probe-bias-neginf", "", "model.layers.%d.mlp.e_score_correction_bias", 0,
                    ProbeSpec { semantics: sv(&[SEM_NEG_INF]), ..Default::default() },
                    vec![]),
            ],
            ref_fro_norm: 0.0,
        },
    ]
}

pub fn builtin_profile(name: &str) -> Result<Profile> {
    let all = builtin_profiles();
    for p in &all {
        if p.name == name {
            return Ok(p.clone());
        }
    }
    let mut names: Vec<String> = all.iter().map(|p| p.name.clone()).collect();
    names.sort();
    Err(Error::msg(format!(
        "unknown profile {:?}; built-ins: {} (or pass a path to a crucible.wsp.profile.v1 JSON file)",
        name,
        names.join(", ")
    )))
}

#[allow(dead_code)]
fn _unused(_: &usize) -> bool {
    is_zero_usize(&0)
}
