//! SARIF 2.1.0 serializer for `secant sanitize` receipts.
//!
//! Produces GitHub Code Scanning compliant SARIF 2.1.0 documents with
//! deterministic sorted rule catalogs, normalized relative artifact URIs,
//! positive 1-based regions, stable SHA-256 `primaryLocationLineHash` partial
//! fingerprints, and atomic file publication (`temp + fsync + rename`).

use crate::container::hex;
use crate::error::{Error, Result};
use crate::security::{SanitizeFinding, SanitizeReceipt};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const SARIF_SCHEMA: &str = "https://json.schemastore.org/sarif-2.1.0.json";
pub const SARIF_VERSION: &str = "2.1.0";
pub const TOOL_NAME: &str = "secant";
pub const TOOL_INFO_URI: &str = "https://github.com/professor-moody/secant";
pub const RULE_HELP_URI: &str = "https://professor-moody.github.io/secant/payload-security";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifLog {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub version: String,
    pub runs: Vec<SarifRun>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifRun {
    pub tool: SarifTool,
    #[serde(rename = "automationDetails")]
    pub automation_details: SarifAutomationDetails,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<SarifArtifact>,
    pub results: Vec<SarifResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub properties: Option<SarifRunProperties>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifArtifact {
    pub location: SarifArtifactLocation,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub hashes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifTool {
    pub driver: SarifDriver,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifDriver {
    pub name: String,
    #[serde(rename = "semanticVersion")]
    pub semantic_version: String,
    #[serde(rename = "informationUri")]
    pub information_uri: String,
    pub rules: Vec<SarifRule>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifRule {
    pub id: String,
    pub name: String,
    #[serde(rename = "shortDescription")]
    pub short_description: SarifMessage,
    #[serde(rename = "fullDescription", default, skip_serializing_if = "Option::is_none")]
    pub full_description: Option<SarifMessage>,
    #[serde(rename = "helpUri", default, skip_serializing_if = "Option::is_none")]
    pub help_uri: Option<String>,
    #[serde(rename = "defaultConfiguration")]
    pub default_configuration: SarifRuleConfig,
    pub properties: SarifRuleProperties,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifRuleConfig {
    pub level: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifRuleProperties {
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifAutomationDetails {
    pub id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifMessage {
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifResult {
    #[serde(rename = "ruleId")]
    pub rule_id: String,
    #[serde(rename = "ruleIndex")]
    pub rule_index: usize,
    pub level: String,
    pub message: SarifMessage,
    pub locations: Vec<SarifLocation>,
    #[serde(rename = "partialFingerprints")]
    pub partial_fingerprints: BTreeMap<String, String>,
    pub properties: SarifResultProperties,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifLocation {
    #[serde(rename = "physicalLocation")]
    pub physical_location: SarifPhysicalLocation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifPhysicalLocation {
    #[serde(rename = "artifactLocation")]
    pub artifact_location: SarifArtifactLocation,
    pub region: SarifRegion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifArtifactLocation {
    pub uri: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifRegion {
    #[serde(rename = "startLine")]
    pub start_line: u64,
    #[serde(rename = "startColumn")]
    pub start_column: u64,
    #[serde(rename = "endLine")]
    pub end_line: u64,
    #[serde(rename = "endColumn")]
    pub end_column: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifResultProperties {
    #[serde(rename = "violationKind")]
    pub violation_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor: Option<String>,
    #[serde(rename = "configKey", default, skip_serializing_if = "Option::is_none")]
    pub config_key: Option<String>,
    #[serde(rename = "metadataKey", default, skip_serializing_if = "Option::is_none")]
    pub metadata_key: Option<String>,
    pub count: i64,
    #[serde(rename = "kindTotalCount")]
    pub kind_total_count: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(rename = "firstIndex", default, skip_serializing_if = "Option::is_none")]
    pub first_index: Option<i64>,
    #[serde(rename = "firstRow", default, skip_serializing_if = "Option::is_none")]
    pub first_row: Option<i64>,
    #[serde(rename = "firstFileOffset", default, skip_serializing_if = "Option::is_none")]
    pub first_file_offset: Option<i64>,
    #[serde(rename = "firstField", default, skip_serializing_if = "Option::is_none")]
    pub first_field: Option<String>,
    #[serde(rename = "firstBits", default, skip_serializing_if = "Option::is_none")]
    pub first_bits: Option<String>,
    #[serde(rename = "examplesTruncated", default, skip_serializing_if = "std::ops::Not::not")]
    pub examples_truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SarifRunProperties {
    #[serde(rename = "policyVersion")]
    pub policy_version: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub format: String,
    #[serde(rename = "artifactUri")]
    pub artifact_uri: String,
    #[serde(rename = "evaluatedArtifact")]
    pub evaluated_artifact: String,
    #[serde(rename = "artifactSha256", default, skip_serializing_if = "String::is_empty")]
    pub artifact_sha256: String,
    #[serde(rename = "preSha256", default, skip_serializing_if = "String::is_empty")]
    pub pre_sha256: String,
    #[serde(rename = "outputPublished")]
    pub output_published: bool,
    #[serde(rename = "inputUnchanged")]
    pub input_unchanged: bool,
    #[serde(rename = "totalFindings")]
    pub total_findings: i64,
}

struct RuleSpec {
    kind: &'static str,
    id: &'static str,
    name: &'static str,
    short: &'static str,
    full: &'static str,
    level: &'static str,
    category: &'static str,
}

const BUILTIN_RULES: &[RuleSpec] = &[
    RuleSpec {
        kind: "s0_expert_count",
        id: "SECANT-S0-EXPERT-COUNT",
        name: "S0ExpertCountInvalid",
        short: "GGUF MoE expert_count metadata is invalid",
        full: "Rule S-0 requires GGUF MoE expert_count metadata to be a positive integer within signed 32-bit bounds.",
        level: "error",
        category: "rule-s0",
    },
    RuleSpec {
        kind: "s0_expert_used_count",
        id: "SECANT-S0-EXPERT-USED-COUNT",
        name: "S0ExpertUsedCountInvalid",
        short: "GGUF MoE expert_used_count metadata exceeds expert_count or is invalid",
        full: "Rule S-0 requires GGUF MoE expert_used_count metadata to satisfy 1 <= expert_used_count <= expert_count.",
        level: "error",
        category: "rule-s0",
    },
    RuleSpec {
        kind: "s0_companion_vocab_size_mismatch",
        id: "SECANT-S0-COMPANION-VOCAB-MISMATCH",
        name: "S0CompanionVocabSizeMismatch",
        short: "Companion config vocab_size mismatches checkpoint embedding rows",
        full: "Rule S-0 requires config.json vocab_size to be non-negative and not exceed the checkpoint embedding row count.",
        level: "error",
        category: "rule-s0",
    },
    RuleSpec {
        kind: "s0_companion_token_id_oob",
        id: "SECANT-S0-COMPANION-TOKEN-ID-OOB",
        name: "S0CompanionTokenIdOutOfBounds",
        short: "Companion config token ID is out of vocabulary bounds",
        full: "Rule S-0 requires special token IDs in config.json and generation_config.json to fall within [0, vocab_size).",
        level: "error",
        category: "rule-s0",
    },
    RuleSpec {
        kind: "s0_companion_added_token_id_oob",
        id: "SECANT-S0-COMPANION-ADDED-TOKEN-ID-OOB",
        name: "S0CompanionAddedTokenIdOutOfBounds",
        short: "Companion tokenizer added_tokens_decoder ID is out of vocabulary bounds",
        full: "Rule S-0 requires every added_tokens_decoder entry in tokenizer_config.json and tokenizer.json to fall within [0, vocab_size).",
        level: "error",
        category: "rule-s0",
    },
    RuleSpec {
        kind: "s0_companion_expert_count",
        id: "SECANT-S0-COMPANION-EXPERT-COUNT",
        name: "S0CompanionExpertCountInvalid",
        short: "Companion config MoE expert count is invalid",
        full: "Rule S-0 requires companion config MoE expert count fields to be positive when MoE tensors are present and within 32-bit bounds.",
        level: "error",
        category: "rule-s0",
    },
    RuleSpec {
        kind: "s0_companion_expert_used_count",
        id: "SECANT-S0-COMPANION-EXPERT-USED-COUNT",
        name: "S0CompanionExpertUsedCountInvalid",
        short: "Companion config MoE experts-per-token exceeds expert count or is invalid",
        full: "Rule S-0 requires num_experts_per_tok and num_selected_experts in config.json to satisfy 1 <= used <= expert_count.",
        level: "error",
        category: "rule-s0",
    },
    RuleSpec {
        kind: "s0_companion_leading_dense_block",
        id: "SECANT-S0-COMPANION-LEADING-DENSE-BLOCK",
        name: "S0CompanionLeadingDenseBlockInvalid",
        short: "Companion config leading dense block count exceeds num_hidden_layers",
        full: "Rule S-0 requires leading_dense_block / first_k_dense_replace to satisfy 0 <= value <= num_hidden_layers.",
        level: "error",
        category: "rule-s0",
    },
    RuleSpec {
        kind: "nan",
        id: "SECANT-F-NAN",
        name: "DenseFloatNaN",
        short: "Dense floating-point tensor contains NaN value",
        full: "A static floating-point weight tensor contains one or more IEEE-754/OFP8 NaN encodings.",
        level: "error",
        category: "dense-float",
    },
    RuleSpec {
        kind: "inf",
        id: "SECANT-F-INF",
        name: "DenseFloatInf",
        short: "Dense floating-point tensor contains Inf value",
        full: "A static floating-point weight tensor contains one or more positive or negative infinity encodings.",
        level: "error",
        category: "dense-float",
    },
    RuleSpec {
        kind: "subnormal",
        id: "SECANT-F-SUBNORMAL",
        name: "DenseFloatSubnormal",
        short: "Dense floating-point tensor contains subnormal value under strict policy",
        full: "Under --strict-subnormal or --flush-subnormal, subnormal floating-point elements are flagged as findings.",
        level: "error",
        category: "dense-float",
    },
    RuleSpec {
        kind: "extreme_finite",
        id: "SECANT-F-EXTREME-FINITE",
        name: "DenseFloatExtremeFinite",
        short: "Dense floating-point tensor exceeds --max-abs policy threshold",
        full: "A finite floating-point element has magnitude strictly above the configured --max-abs limit.",
        level: "error",
        category: "dense-float",
    },
    RuleSpec {
        kind: "norm_all_zero",
        id: "SECANT-NORM-ALL-ZERO",
        name: "NormRowAllZero",
        short: "Normalization tensor row is all zero",
        full: "A contiguous vector in a selected norm tensor is identically zero, risking divide-by-zero or collapsed activations.",
        level: "error",
        category: "norm",
    },
    RuleSpec {
        kind: "norm_below_min",
        id: "SECANT-NORM-BELOW-MIN",
        name: "NormRowBelowMin",
        short: "Normalization tensor row L2 norm is below --min-norm",
        full: "A contiguous vector in a selected norm tensor has L2 norm strictly below --min-norm.",
        level: "error",
        category: "norm",
    },
    RuleSpec {
        kind: "quant_scale_nan",
        id: "SECANT-B3-2A-QUANT-SCALE-NAN",
        name: "QuantScaleNaN",
        short: "Quantized GGUF block scale header contains NaN",
        full: "Rule B3-2A rejects NaN encodings in quantized GGUF block scale headers (FP16 d/dmin/m/s, FP32 Q8_K d, packed FP16 IQ1_M d, and MXFP4 E8M0 0xFF e).",
        level: "error",
        category: "rule-b3-2a",
    },
    RuleSpec {
        kind: "quant_scale_inf",
        id: "SECANT-B3-2A-QUANT-SCALE-INF",
        name: "QuantScaleInf",
        short: "Quantized GGUF block scale header contains Inf",
        full: "Rule B3-2A rejects infinity encodings in quantized GGUF block scale headers (FP16 d/dmin/m/s, FP32 Q8_K d, and packed FP16 IQ1_M d).",
        level: "error",
        category: "rule-b3-2a",
    },
    RuleSpec {
        kind: "quant_scale_subnormal",
        id: "SECANT-B3-2A-QUANT-SCALE-SUBNORMAL",
        name: "QuantScaleSubnormal",
        short: "Quantized GGUF block scale header contains subnormal under strict policy",
        full: "Under --strict-subnormal or --flush-subnormal, subnormal block scale headers (FP16 d/dmin/m/s, FP32 Q8_K d, or packed FP16 IQ1_M d) are reported.",
        level: "error",
        category: "rule-b3-2a",
    },
    RuleSpec {
        kind: "quant_scale_extreme",
        id: "SECANT-B3-2A-QUANT-SCALE-EXTREME",
        name: "QuantScaleExtreme",
        short: "Quantized GGUF block scale header produces magnitudes exceeding --max-abs",
        full: "Under extreme-scale checking, a quantized block scale multiplied by its active weight multiplier exceeds --max-abs.",
        level: "error",
        category: "rule-b3-2a",
    },
    RuleSpec {
        kind: "quant_block_extreme",
        id: "SECANT-B3-2A-QUANT-BLOCK-EXTREME",
        name: "QuantBlockExtreme",
        short: "Quantized GGUF block dequantizes to magnitudes exceeding --max-abs",
        full: "A quantized super-block dequantizes to an element with magnitude above --max-abs despite individually bounded scale fields.",
        level: "error",
        category: "rule-b3-2a",
    },
    RuleSpec {
        kind: "divisor_scale_nonpositive",
        id: "SECANT-B3-2B-DIVISOR-NONPOSITIVE",
        name: "DivisorScaleNonPositive",
        short: "Divisor scale tensor contains zero or negative value",
        full: "Rule B3-2B requires divisor scales (such as k_scale, v_scale, input_scale, *scale_inv) to be strictly positive.",
        level: "error",
        category: "rule-b3-2b",
    },
    RuleSpec {
        kind: "divisor_scale_below_floor",
        id: "SECANT-B3-2B-DIVISOR-BELOW-FLOOR",
        name: "DivisorScaleBelowFloor",
        short: "Divisor scale tensor value is below --divisor-floor",
        full: "Rule B3-2B requires divisor scales to be at least --divisor-floor (default 1e-12 in float32).",
        level: "error",
        category: "rule-b3-2b",
    },
    RuleSpec {
        kind: "rope_freqs_zero",
        id: "SECANT-B3-2B-ROPE-FREQS-ZERO",
        name: "RopeFreqsZero",
        short: "RoPE frequency tensor contains zero entry",
        full: "Rule B3-2B rejects zero entries in rope_freqs tensors that can trigger division by zero during rotary embedding setup.",
        level: "error",
        category: "rule-b3-2b",
    },
    RuleSpec {
        kind: "routing_width_exceeds_experts",
        id: "SECANT-B3-3-ROUTING-WIDTH",
        name: "RoutingWidthExceedsExperts",
        short: "MoE routing row width exceeds --num-experts",
        full: "Rule B3-3 requires each MoE routing row width to be at most --num-experts.",
        level: "error",
        category: "rule-b3-3",
    },
    RuleSpec {
        kind: "routing_negative",
        id: "SECANT-B3-3-ROUTING-NEGATIVE",
        name: "RoutingIndexNegative",
        short: "MoE routing tensor contains negative expert index",
        full: "Rule B3-3 requires all MoE routing expert indices to be non-negative.",
        level: "error",
        category: "rule-b3-3",
    },
    RuleSpec {
        kind: "routing_oob",
        id: "SECANT-B3-3-ROUTING-OOB",
        name: "RoutingIndexOutOfBounds",
        short: "MoE routing tensor contains expert index >= --num-experts",
        full: "Rule B3-3 requires all MoE routing expert indices to be strictly less than --num-experts.",
        level: "error",
        category: "rule-b3-3",
    },
    RuleSpec {
        kind: "routing_duplicate",
        id: "SECANT-B3-3-ROUTING-DUPLICATE",
        name: "RoutingIndexDuplicate",
        short: "MoE routing row contains duplicate expert index",
        full: "Rule B3-3 flags duplicate expert IDs within the same MoE routing row.",
        level: "error",
        category: "rule-b3-3",
    },
    RuleSpec {
        kind: "index_length_mismatch",
        id: "SECANT-B3-3-INDEX-LENGTH-MISMATCH",
        name: "IndexLengthMismatch",
        short: "Integer index tensor length mismatches sibling weight dimension",
        full: "Rule B3-3 requires g_idx and related index tables to match their sibling tensor dimension.",
        level: "error",
        category: "rule-b3-3",
    },
    RuleSpec {
        kind: "index_negative",
        id: "SECANT-B3-3-INDEX-NEGATIVE",
        name: "IndexNegative",
        short: "Integer index tensor contains value below lower bound",
        full: "Rule B3-3 rejects negative or below-lower-bound entries in integer index tensors.",
        level: "error",
        category: "rule-b3-3",
    },
    RuleSpec {
        kind: "index_oob",
        id: "SECANT-B3-3-INDEX-OOB",
        name: "IndexOutOfBounds",
        short: "Integer index tensor contains value at or above upper bound",
        full: "Rule B3-3 requires every entry in an integer index tensor to stay strictly below its exclusive upper bound.",
        level: "error",
        category: "rule-b3-3",
    },
];

fn rule_from_spec(spec: &RuleSpec) -> SarifRule {
    SarifRule {
        id: spec.id.to_string(),
        name: spec.name.to_string(),
        short_description: SarifMessage { text: spec.short.to_string() },
        full_description: Some(SarifMessage { text: spec.full.to_string() }),
        help_uri: Some(RULE_HELP_URI.to_string()),
        default_configuration: SarifRuleConfig { level: spec.level.to_string() },
        properties: SarifRuleProperties {
            tags: vec!["security".to_string(), "ml-weights".to_string(), "secant".to_string(), spec.category.to_string()],
        },
    }
}

fn sanitize_rule_token(kind: &str) -> String {
    let mut out = String::with_capacity(kind.len());
    let mut prev_dash = false;
    for ch in kind.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_uppercase());
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "UNKNOWN".to_string()
    } else {
        trimmed.to_string()
    }
}

fn rule_for_kind(kind: &str) -> SarifRule {
    if let Some(spec) = BUILTIN_RULES.iter().find(|s| s.kind == kind) {
        return rule_from_spec(spec);
    }
    SarifRule {
        id: format!("SECANT-CUSTOM-{}", sanitize_rule_token(kind)),
        name: "CustomSanitizeFinding".to_string(),
        short_description: SarifMessage { text: format!("Sanitizer policy finding '{}'", kind) },
        full_description: Some(SarifMessage { text: format!("Static weight sanitizer reported finding kind '{}'.", kind) }),
        help_uri: Some(RULE_HELP_URI.to_string()),
        default_configuration: SarifRuleConfig { level: "error".to_string() },
        properties: SarifRuleProperties { tags: vec!["security".to_string(), "ml-weights".to_string(), "secant".to_string()] },
    }
}

fn has_windows_drive_prefix(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':'
}

fn is_absolute_uri_input(s: &str) -> bool {
    s.starts_with('/') || has_windows_drive_prefix(s)
}

fn normalize_slash_segments(raw: &str) -> (bool, Vec<String>) {
    let replaced = raw.replace('\\', "/");
    let abs = is_absolute_uri_input(&replaced);
    let body = if has_windows_drive_prefix(&replaced) { &replaced[2..] } else { &replaced[..] };
    let mut stack: Vec<String> = Vec::new();
    for seg in body.split('/') {
        if seg.is_empty() || seg == "." {
            continue;
        }
        if seg == ".." {
            if let Some(last) = stack.last() {
                if last != ".." {
                    stack.pop();
                    continue;
                }
            }
            if !abs {
                stack.push("..".to_string());
            }
            continue;
        }
        stack.push(seg.to_string());
    }
    (abs, stack)
}

fn strip_base_prefix(path_segs: &[String], base: &Path) -> Option<String> {
    let base_str = base.to_string_lossy();
    let (base_abs, base_segs) = normalize_slash_segments(&base_str);
    if !base_abs || base_segs.is_empty() || path_segs.len() <= base_segs.len() {
        return None;
    }
    if path_segs[..base_segs.len()] == base_segs[..] {
        let rel = &path_segs[base_segs.len()..];
        if !rel.is_empty() && !rel.iter().any(|s| s == "..") {
            return Some(rel.join("/"));
        }
    }
    None
}

/// Normalize a model path into a GitHub SARIF 2.1.0 safe relative forward-slash
/// URI. Never emits raw host paths (such as `/abs/path/...` or `C:\...`), leading
/// `./`, `../` escapes, or empty strings.
pub fn normalize_artifact_uri(model_path: &str, base_dir: Option<&Path>) -> String {
    let trimmed = model_path.trim();
    if trimmed.is_empty() {
        return "model".to_string();
    }
    let (abs, segs) = normalize_slash_segments(trimmed);
    if segs.is_empty() {
        return "model".to_string();
    }
    if abs {
        if let Some(base) = base_dir {
            if let Some(rel) = strip_base_prefix(&segs, base) {
                return rel;
            }
            if let Ok(canon_base) = std::fs::canonicalize(base) {
                if let Some(rel) = strip_base_prefix(&segs, &canon_base) {
                    return rel;
                }
                if let Ok(canon_model) = std::fs::canonicalize(trimmed) {
                    let (_, c_segs) = normalize_slash_segments(&canon_model.to_string_lossy());
                    if let Some(rel) = strip_base_prefix(&c_segs, &canon_base) {
                        return rel;
                    }
                }
            }
        }
        return segs.last().cloned().filter(|s| !s.is_empty() && s != "..").unwrap_or_else(|| "model".to_string());
    }
    if segs.first().map(|s| s.as_str()) == Some("..") {
        return segs.iter().rev().find(|s| !s.is_empty() && s.as_str() != "..").cloned().unwrap_or_else(|| "model".to_string());
    }
    segs.join("/")
}

fn companion_artifact_uri(model_uri: &str, companion_file: &str) -> String {
    let (_, c_segs) = normalize_slash_segments(companion_file);
    let comp_name = c_segs.last().cloned().filter(|s| !s.is_empty() && s != "..").unwrap_or_else(|| "config.json".to_string());
    if let Some((dir, _)) = model_uri.rsplit_once('/') {
        if !dir.is_empty() {
            return format!("{}/{}", dir, comp_name);
        }
    }
    comp_name
}

/// Deterministic SHA-256 hex fingerprint over
/// `(ruleId, normalizedArtifactURI, tensor_name_or_config_key, violation_kind)`.
pub fn compute_primary_location_line_hash(rule_id: &str, artifact_uri: &str, target_symbol: &str, violation_kind: &str) -> String {
    let mut hasher = Sha256::new();
    for part in ["secant.sarif.fp.v1", rule_id, artifact_uri, target_symbol, violation_kind] {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part.as_bytes());
    }
    hex(&hasher.finalize())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetKind {
    Tensor,
    CompanionConfig,
    Metadata,
}

#[derive(Debug, Clone)]
struct FindingGroup {
    kind: String,
    tensor_raw: String,
    count: i64,
    first: Option<SanitizeFinding>,
}

fn is_quant_kind(kind: &str) -> bool {
    kind.starts_with("quant_")
}

fn format_occurrence_word(n: i64) -> &'static str {
    if n == 1 {
        "occurrence"
    } else {
        "occurrences"
    }
}

fn build_result_message(
    target_kind: TargetKind,
    target_symbol: &str,
    group: &FindingGroup,
    kind_total: i64,
    kind_captured: i64,
    examples_truncated: bool,
    max_examples: i64,
) -> String {
    let count_phrase = if group.first.is_none() {
        format!(
            "{} total {}; per-tensor examples truncated at max_examples={}",
            kind_total,
            format_occurrence_word(kind_total),
            max_examples
        )
    } else if examples_truncated && kind_captured < kind_total {
        format!("{} captured of {} total '{}' {}", group.count, kind_total, group.kind, format_occurrence_word(kind_total))
    } else {
        format!("{} {}", group.count, format_occurrence_word(group.count))
    };

    match target_kind {
        TargetKind::CompanionConfig => {
            format!("Companion config key '{}' failed check '{}' ({}).", target_symbol, group.kind, count_phrase)
        }
        TargetKind::Metadata => {
            format!("Metadata key '{}' failed check '{}' ({}).", target_symbol, group.kind, count_phrase)
        }
        TargetKind::Tensor => {
            let Some(first) = &group.first else {
                return format!("Model failed check '{}' ({}).", group.kind, count_phrase);
            };
            let dtype_part = if first.dtype.is_empty() { String::new() } else { format!(" ({})", first.dtype) };
            let mut loc_details = Vec::new();
            if is_quant_kind(&group.kind) {
                loc_details.push(format!("block index {}", first.index));
                if !first.field.is_empty() {
                    loc_details.push(format!("field '{}'", first.field));
                }
            } else {
                loc_details.push(format!("element index {}", first.index));
            }
            loc_details.push(format!("row {}", first.row));
            loc_details.push(format!("byte offset {}", first.file_offset));
            if !first.bits.is_empty() {
                loc_details.push(format!("bits {}", first.bits));
            }
            format!(
                "Tensor '{}'{} failed check '{}' ({}; first at {}).",
                target_symbol,
                dtype_part,
                group.kind,
                count_phrase,
                loc_details.join(", ")
            )
        }
    }
}

fn sanitize_attempt_id_segment(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "default".to_string();
    }
    trimmed.replace(['/', '\\'], "_")
}

/// Convert a `SanitizeReceipt` into a GitHub Code Scanning compliant SARIF 2.1.0
/// log, resolving workspace-relative artifact URIs against the current directory.
///
/// Artifact binding: when remediation published a clean `--out` copy
/// (`output_published == true` with a non-empty `options.output` and `after`
/// rescan), the SARIF log reflects `after` against the published `--out`
/// artifact URI. Otherwise (scan-only, `--dry-run`, or failed/partial
/// remediation where nothing was published), the SARIF log reflects `before`
/// against the unchanged input artifact URI so no findings still present in
/// the input file are omitted.
pub fn build_sanitize_sarif(receipt: &SanitizeReceipt) -> SarifLog {
    let cwd = std::env::current_dir().ok();
    build_sanitize_sarif_with_base_dir(receipt, cwd.as_deref())
}

/// Same as [`build_sanitize_sarif`] with an explicit base directory for
/// relative artifact URI normalization.
pub fn build_sanitize_sarif_with_base_dir(receipt: &SanitizeReceipt, base_dir: Option<&Path>) -> SarifLog {
    let mut rules_map: BTreeMap<String, SarifRule> = BTreeMap::new();
    for spec in BUILTIN_RULES {
        let rule = rule_from_spec(spec);
        rules_map.insert(rule.id.clone(), rule);
    }

    let automation_id = format!("secant/sanitize/{}", sanitize_attempt_id_segment(&receipt.attempt_id));
    let published_out =
        if receipt.output_published { receipt.options.as_ref().map(|o| o.output.as_str()).filter(|s| !s.is_empty()) } else { None };
    let (effective_scan, target_path, target_sha256, evaluated_artifact) = match (published_out, receipt.after.as_ref()) {
        (Some(out_path), Some(after_scan)) => {
            let sha = if receipt.output_sha256.is_empty() { receipt.pre_sha256.as_str() } else { receipt.output_sha256.as_str() };
            (Some(after_scan), out_path, sha, "remediated_output")
        }
        _ => (receipt.before.as_ref(), receipt.input.as_str(), receipt.pre_sha256.as_str(), "input"),
    };
    let model_uri = normalize_artifact_uri(target_path, base_dir);
    let max_examples = receipt.options.as_ref().map(|o| o.max_examples).filter(|&n| n > 0).unwrap_or(128);

    let mut groups: BTreeMap<(String, String), FindingGroup> = BTreeMap::new();
    let mut captured_per_kind: BTreeMap<String, i64> = BTreeMap::new();
    let mut total_findings = 0i64;
    let mut examples_truncated = false;
    let mut kind_counts: BTreeMap<String, i64> = BTreeMap::new();

    if let Some(scan) = effective_scan {
        total_findings = scan.total_findings;
        examples_truncated = scan.examples_truncated;
        kind_counts = scan.counts.clone();
        if scan.total_findings > 0 {
            for ex in &scan.examples {
                let rule = rule_for_kind(&ex.kind);
                rules_map.entry(rule.id.clone()).or_insert(rule);
                *captured_per_kind.entry(ex.kind.clone()).or_insert(0) += 1;
                let entry = groups.entry((ex.kind.clone(), ex.tensor.clone())).or_insert_with(|| FindingGroup {
                    kind: ex.kind.clone(),
                    tensor_raw: ex.tensor.clone(),
                    count: 0,
                    first: Some(ex.clone()),
                });
                entry.count += 1;
            }
            for (kind, &total) in &scan.counts {
                if total <= 0 {
                    continue;
                }
                let rule = rule_for_kind(kind);
                rules_map.entry(rule.id.clone()).or_insert(rule);
                if captured_per_kind.get(kind).copied().unwrap_or(0) == 0 {
                    groups.insert(
                        (kind.clone(), String::new()),
                        FindingGroup { kind: kind.clone(), tensor_raw: String::new(), count: total, first: None },
                    );
                }
            }
        }
    }

    let rules: Vec<SarifRule> = rules_map.into_values().collect();
    let rule_indices: BTreeMap<String, usize> = rules.iter().enumerate().map(|(i, r)| (r.id.clone(), i)).collect();

    let mut results: Vec<SarifResult> = Vec::with_capacity(groups.len());
    for group in groups.values() {
        let rule = rule_for_kind(&group.kind);
        let rule_index = rule_indices[&rule.id];
        let (artifact_uri, target_kind, target_symbol) = if group.kind.starts_with("s0_companion_") {
            if let Some((comp_file, comp_key)) = group.tensor_raw.split_once(':') {
                if !comp_file.is_empty() && !comp_key.is_empty() {
                    (companion_artifact_uri(&model_uri, comp_file), TargetKind::CompanionConfig, comp_key.to_string())
                } else {
                    (
                        model_uri.clone(),
                        TargetKind::CompanionConfig,
                        if group.tensor_raw.is_empty() { "<truncated>".to_string() } else { group.tensor_raw.clone() },
                    )
                }
            } else {
                (
                    model_uri.clone(),
                    TargetKind::CompanionConfig,
                    if group.tensor_raw.is_empty() { "<truncated>".to_string() } else { group.tensor_raw.clone() },
                )
            }
        } else if group.kind == "s0_expert_count" || group.kind == "s0_expert_used_count" {
            (
                model_uri.clone(),
                TargetKind::Metadata,
                if group.tensor_raw.is_empty() { "<truncated>".to_string() } else { group.tensor_raw.clone() },
            )
        } else {
            (
                model_uri.clone(),
                TargetKind::Tensor,
                if group.tensor_raw.is_empty() { "<truncated>".to_string() } else { group.tensor_raw.clone() },
            )
        };

        let kind_total = kind_counts.get(&group.kind).copied().unwrap_or(group.count);
        let kind_captured = captured_per_kind.get(&group.kind).copied().unwrap_or(0);
        let message_text =
            build_result_message(target_kind, &target_symbol, group, kind_total, kind_captured, examples_truncated, max_examples);
        let fingerprint = compute_primary_location_line_hash(&rule.id, &artifact_uri, &target_symbol, &group.kind);
        let mut partial_fingerprints = BTreeMap::new();
        partial_fingerprints.insert("primaryLocationLineHash".to_string(), fingerprint);

        let first = group.first.as_ref();
        let props = SarifResultProperties {
            violation_kind: group.kind.clone(),
            tensor: if target_kind == TargetKind::Tensor && !group.tensor_raw.is_empty() { Some(target_symbol.clone()) } else { None },
            config_key: if target_kind == TargetKind::CompanionConfig && !group.tensor_raw.is_empty() {
                Some(target_symbol.clone())
            } else {
                None
            },
            metadata_key: if target_kind == TargetKind::Metadata && !group.tensor_raw.is_empty() {
                Some(target_symbol.clone())
            } else {
                None
            },
            count: group.count,
            kind_total_count: kind_total,
            dtype: first.and_then(|f| if f.dtype.is_empty() { None } else { Some(f.dtype.clone()) }),
            first_index: if target_kind == TargetKind::Tensor { first.map(|f| f.index) } else { None },
            first_row: if target_kind == TargetKind::Tensor { first.map(|f| f.row) } else { None },
            first_file_offset: if target_kind == TargetKind::Tensor { first.map(|f| f.file_offset) } else { None },
            first_field: first.and_then(|f| if f.field.is_empty() { None } else { Some(f.field.clone()) }),
            first_bits: first.and_then(|f| if f.bits.is_empty() { None } else { Some(f.bits.clone()) }),
            examples_truncated: examples_truncated && kind_captured < kind_total,
        };

        results.push(SarifResult {
            rule_id: rule.id,
            rule_index,
            level: rule.default_configuration.level,
            message: SarifMessage { text: message_text },
            locations: vec![SarifLocation {
                physical_location: SarifPhysicalLocation {
                    artifact_location: SarifArtifactLocation { uri: artifact_uri },
                    region: SarifRegion { start_line: 1, start_column: 1, end_line: 1, end_column: 1 },
                },
            }],
            partial_fingerprints,
            properties: props,
        });
    }

    results.sort_by(|a, b| {
        let uri_a = a.locations.first().map(|l| l.physical_location.artifact_location.uri.as_str()).unwrap_or("");
        let uri_b = b.locations.first().map(|l| l.physical_location.artifact_location.uri.as_str()).unwrap_or("");
        let fp_a = a.partial_fingerprints.get("primaryLocationLineHash").map(String::as_str).unwrap_or("");
        let fp_b = b.partial_fingerprints.get("primaryLocationLineHash").map(String::as_str).unwrap_or("");
        a.rule_id.cmp(&b.rule_id).then_with(|| uri_a.cmp(uri_b)).then_with(|| fp_a.cmp(fp_b))
    });

    let mut primary_hashes = BTreeMap::new();
    if !target_sha256.is_empty() {
        primary_hashes.insert("sha-256".to_string(), target_sha256.to_string());
    }
    let artifacts = vec![SarifArtifact { location: SarifArtifactLocation { uri: model_uri.clone() }, hashes: primary_hashes }];

    SarifLog {
        schema: SARIF_SCHEMA.to_string(),
        version: SARIF_VERSION.to_string(),
        runs: vec![SarifRun {
            tool: SarifTool {
                driver: SarifDriver {
                    name: TOOL_NAME.to_string(),
                    semantic_version: env!("CARGO_PKG_VERSION").to_string(),
                    information_uri: TOOL_INFO_URI.to_string(),
                    rules,
                },
            },
            automation_details: SarifAutomationDetails { id: automation_id },
            artifacts,
            results,
            properties: Some(SarifRunProperties {
                policy_version: receipt.policy_version.clone(),
                status: receipt.status.clone(),
                format: receipt.format.clone(),
                artifact_uri: model_uri,
                evaluated_artifact: evaluated_artifact.to_string(),
                artifact_sha256: target_sha256.to_string(),
                pre_sha256: receipt.pre_sha256.clone(),
                output_published: receipt.output_published,
                input_unchanged: receipt.input_unchanged,
                total_findings,
            }),
        }],
    }
}

struct RenameTempGuard {
    path: Option<PathBuf>,
}

impl RenameTempGuard {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }
    fn defuse(&mut self) {
        self.path = None;
    }
}

impl Drop for RenameTempGuard {
    fn drop(&mut self) {
        if let Some(p) = self.path.take() {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn sarif_parent_dir(dst: &str) -> Result<PathBuf> {
    if dst.trim().is_empty() {
        return Err(Error::msg("sarif output path must not be empty"));
    }
    let dst_path = Path::new(dst);
    if std::fs::metadata(dst_path).map(|m| m.is_dir()).unwrap_or(false) {
        return Err(Error::msg(format!("sarif output path {} is a directory", dst)));
    }
    let parent = match dst_path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let meta = std::fs::metadata(&parent).map_err(|e| Error::msg(format!("sarif parent directory {}: {}", parent.display(), e)))?;
    if !meta.is_dir() {
        return Err(Error::msg(format!("sarif parent {} is not a directory", parent.display())));
    }
    Ok(parent)
}

fn create_temp_sarif(parent: &Path, prefix: &str) -> Result<(std::fs::File, PathBuf, RenameTempGuard)> {
    let mut last_err = None;
    for attempt in 0..64u32 {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let candidate = parent.join(format!("{}-{}-{}-{}", prefix, std::process::id(), nanos, attempt));
        match OpenOptions::new().read(true).write(true).create_new(true).open(&candidate) {
            Ok(f) => {
                let guard = RenameTempGuard::new(candidate.clone());
                return Ok((f, candidate, guard));
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last_err = Some(e),
            Err(e) => return Err(Error::msg(format!("create temp sarif in {}: {}", parent.display(), e))),
        }
    }
    Err(Error::msg(format!(
        "create temp sarif in {}: {}",
        parent.display(),
        last_err.map(|e| e.to_string()).unwrap_or_else(|| "exhausted retries".to_string())
    )))
}

/// Fail-fast preflight check that `dst` is not a directory and its parent
/// directory exists and permits temporary file creation before scanning or
/// publishing `--out` / `--receipt`.
pub fn validate_sarif_output_path(dst: &str) -> Result<()> {
    let parent = sarif_parent_dir(dst)?;
    let (probe_file, _, probe_guard) = create_temp_sarif(&parent, ".secant-sarif-probe")?;
    drop(probe_file);
    drop(probe_guard);
    Ok(())
}

/// Atomically publish a SARIF 2.1.0 JSON file using `temp file in same dir +
/// fsync + rename`. Fails closed with an error if the parent directory is
/// missing/unwritable or if the destination is an existing directory.
pub fn write_sarif_atomic(dst: &str, log: &SarifLog) -> Result<()> {
    let parent = sarif_parent_dir(dst)?;
    let dst_path = Path::new(dst);
    let mut payload = serde_json::to_string_pretty(log).map_err(|e| Error::msg(format!("serialize sarif: {}", e)))?;
    payload.push('\n');

    let (mut f, path, mut guard) = create_temp_sarif(&parent, ".secant-sarif")?;
    f.write_all(payload.as_bytes()).map_err(|e| Error::msg(format!("write sarif {}: {}", path.display(), e)))?;
    f.sync_all().map_err(|e| Error::msg(format!("fsync sarif {}: {}", path.display(), e)))?;
    drop(f);

    std::fs::rename(&path, dst_path).map_err(|e| Error::msg(format!("rename {} -> {}: {}", path.display(), dst, e)))?;
    guard.defuse();
    #[cfg(unix)]
    if let Ok(dir_fd) = std::fs::File::open(parent) {
        let _ = dir_fd.sync_all();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::{SanitizeOptions, SanitizeScan, SANITIZE_POLICY_VERSION, SANITIZE_SCHEMA};

    fn sample_receipt(input: &str, status: &str, scan: SanitizeScan) -> SanitizeReceipt {
        SanitizeReceipt {
            schema: SANITIZE_SCHEMA.to_string(),
            attempt_id: "ci-unit-check".to_string(),
            policy_version: SANITIZE_POLICY_VERSION.to_string(),
            status: status.to_string(),
            input: input.to_string(),
            format: "safetensors".to_string(),
            bytes: 1024,
            pre_sha256: "a".repeat(64),
            post_sha256: "a".repeat(64),
            input_unchanged: true,
            options: Some(SanitizeOptions { attempt_id: "ci-unit-check".to_string(), max_examples: 128, ..Default::default() }),
            before: Some(scan),
            ..Default::default()
        }
    }

    #[test]
    fn normalize_artifact_uri_strips_host_paths_and_dot_prefixes() {
        let repo = Path::new("/workspace/repo");
        assert_eq!(normalize_artifact_uri("/workspace/repo/models/qwen/model.safetensors", Some(repo)), "models/qwen/model.safetensors");
        assert_eq!(normalize_artifact_uri("./models/model.gguf", Some(repo)), "models/model.gguf");
        assert_eq!(normalize_artifact_uri(".\\models\\sub\\model.gguf", Some(repo)), "models/sub/model.gguf");
        assert_eq!(normalize_artifact_uri("/tmp/external-dir/model.safetensors", Some(repo)), "model.safetensors");
        assert_eq!(normalize_artifact_uri("../outside/model.gguf", Some(repo)), "model.gguf");
        assert_eq!(normalize_artifact_uri(r"C:\models\ckpt.safetensors", Some(repo)), "ckpt.safetensors");
        assert_eq!(normalize_artifact_uri("", Some(repo)), "model");
    }

    #[test]
    fn clean_receipt_produces_valid_empty_results_and_sorted_rules() {
        let scan = SanitizeScan { tensors: 4, float_elements: 128, ..Default::default() };
        let rec = sample_receipt("models/clean.safetensors", "clean", scan);
        let log = build_sanitize_sarif_with_base_dir(&rec, Some(Path::new("/workspace/repo")));
        assert_eq!(log.schema, SARIF_SCHEMA);
        assert_eq!(log.version, "2.1.0");
        assert_eq!(log.runs.len(), 1);
        let run = &log.runs[0];
        assert_eq!(run.tool.driver.name, "secant");
        assert_eq!(run.tool.driver.semantic_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(run.tool.driver.information_uri, TOOL_INFO_URI);
        assert_eq!(run.automation_details.id, "secant/sanitize/ci-unit-check");
        assert!(run.results.is_empty());
        assert_eq!(run.tool.driver.rules.len(), BUILTIN_RULES.len());
        let ids: Vec<&str> = run.tool.driver.rules.iter().map(|r| r.id.as_str()).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted, "rules must be deterministically sorted by id");
    }

    #[test]
    fn findings_group_by_kind_and_target_with_stable_fingerprints_and_companion_uris() {
        let mut counts = BTreeMap::new();
        counts.insert("nan".to_string(), 2);
        counts.insert("s0_companion_token_id_oob".to_string(), 1);
        let scan = SanitizeScan {
            tensors: 2,
            float_elements: 64,
            total_findings: 3,
            counts,
            examples: vec![
                SanitizeFinding {
                    kind: "nan".to_string(),
                    tensor: "blk.0.attn.weight".to_string(),
                    dtype: "F32".to_string(),
                    index: 4,
                    row: 1,
                    file_offset: 256,
                    bits: "0x7fc00000".to_string(),
                    field: String::new(),
                },
                SanitizeFinding {
                    kind: "nan".to_string(),
                    tensor: "blk.0.attn.weight".to_string(),
                    dtype: "F32".to_string(),
                    index: 5,
                    row: 1,
                    file_offset: 260,
                    bits: "0x7fc00000".to_string(),
                    field: String::new(),
                },
                SanitizeFinding {
                    kind: "s0_companion_token_id_oob".to_string(),
                    tensor: "config.json:eos_token_id".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let rec_rel = sample_receipt("./weights/model.safetensors", "findings", scan.clone());
        let rec_abs = sample_receipt("/workspace/repo/weights/model.safetensors", "findings", scan);
        let base = Path::new("/workspace/repo");

        let log_a = build_sanitize_sarif_with_base_dir(&rec_rel, Some(base));
        let log_b = build_sanitize_sarif_with_base_dir(&rec_abs, Some(base));
        assert_eq!(log_a, log_b, "relative and repo-absolute paths must emit identical SARIF");

        let run = &log_a.runs[0];
        assert_eq!(run.results.len(), 2);

        let nan_res = run.results.iter().find(|r| r.rule_id == "SECANT-F-NAN").expect("nan result");
        assert_eq!(nan_res.level, "error");
        assert_eq!(run.tool.driver.rules[nan_res.rule_index].id, nan_res.rule_id);
        assert_eq!(nan_res.locations[0].physical_location.artifact_location.uri, "weights/model.safetensors");
        assert_eq!(nan_res.locations[0].physical_location.region.start_line, 1);
        assert_eq!(nan_res.locations[0].physical_location.region.start_column, 1);
        assert_eq!(nan_res.locations[0].physical_location.region.end_line, 1);
        assert_eq!(nan_res.locations[0].physical_location.region.end_column, 1);
        assert!(nan_res.message.text.contains("blk.0.attn.weight"));
        assert!(nan_res.message.text.contains("2 occurrences"));
        assert!(nan_res.message.text.contains("byte offset 256"));
        assert_eq!(
            nan_res.partial_fingerprints["primaryLocationLineHash"],
            compute_primary_location_line_hash("SECANT-F-NAN", "weights/model.safetensors", "blk.0.attn.weight", "nan")
        );

        let s0_res = run.results.iter().find(|r| r.rule_id == "SECANT-S0-COMPANION-TOKEN-ID-OOB").expect("s0 result");
        assert_eq!(s0_res.locations[0].physical_location.artifact_location.uri, "weights/config.json");
        assert!(s0_res.message.text.contains("eos_token_id"));
        assert_eq!(
            s0_res.partial_fingerprints["primaryLocationLineHash"],
            compute_primary_location_line_hash(
                "SECANT-S0-COMPANION-TOKEN-ID-OOB",
                "weights/config.json",
                "eos_token_id",
                "s0_companion_token_id_oob"
            )
        );
    }

    #[test]
    fn truncated_kind_with_zero_examples_still_emits_result() {
        let mut counts = BTreeMap::new();
        counts.insert("inf".to_string(), 7);
        let scan = SanitizeScan {
            tensors: 1,
            float_elements: 16,
            total_findings: 7,
            counts,
            examples: vec![],
            examples_truncated: true,
            ..Default::default()
        };
        let rec = sample_receipt("model.gguf", "findings", scan);
        let log = build_sanitize_sarif_with_base_dir(&rec, None);
        assert_eq!(log.runs[0].results.len(), 1);
        let res = &log.runs[0].results[0];
        assert_eq!(res.rule_id, "SECANT-F-INF");
        assert_eq!(res.properties.kind_total_count, 7);
        assert!(res.properties.examples_truncated);
        assert!(res.message.text.contains("7 total occurrences"));
    }

    #[test]
    fn atomic_write_replaces_existing_file_and_rejects_missing_parent() {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("secant-sarif-unit-{}-{}", std::process::id(), nanos));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("report.sarif");
        let rec = sample_receipt("model.safetensors", "clean", SanitizeScan::default());
        let log = build_sanitize_sarif_with_base_dir(&rec, None);

        write_sarif_atomic(out.to_str().unwrap(), &log).unwrap();
        let first = std::fs::read_to_string(&out).unwrap();
        assert!(first.ends_with('\n'));
        let parsed: SarifLog = serde_json::from_str(&first).unwrap();
        assert_eq!(parsed, log);

        // Second write atomically replaces the file without leaving temp files behind.
        write_sarif_atomic(out.to_str().unwrap(), &log).unwrap();
        let entries: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(entries.len(), 1);

        // Missing parent fails closed.
        let bad = dir.join("does-not-exist").join("out.sarif");
        assert!(write_sarif_atomic(bad.to_str().unwrap(), &log).is_err());

        // Existing directory fails closed.
        assert!(write_sarif_atomic(dir.to_str().unwrap(), &log).is_err());

        // Pre-scan validator accepts writable parent (leaving no probe files) and rejects invalid paths.
        validate_sarif_output_path(out.to_str().unwrap()).unwrap();
        let entries_after: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(entries_after.len(), 1);
        assert!(validate_sarif_output_path(bad.to_str().unwrap()).is_err());
        assert!(validate_sarif_output_path(dir.to_str().unwrap()).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn remediation_binds_published_output_and_partial_repair_reports_input_before() {
        let mut before_counts = BTreeMap::new();
        before_counts.insert("nan".to_string(), 1);
        before_counts.insert("extreme_finite".to_string(), 1);
        let before_scan = SanitizeScan {
            tensors: 1,
            float_elements: 16,
            total_findings: 2,
            counts: before_counts,
            examples: vec![
                SanitizeFinding {
                    kind: "nan".to_string(),
                    tensor: "w".to_string(),
                    dtype: "F32".to_string(),
                    index: 0,
                    row: 0,
                    file_offset: 64,
                    bits: "0x7fc00000".to_string(),
                    field: String::new(),
                },
                SanitizeFinding {
                    kind: "extreme_finite".to_string(),
                    tensor: "w".to_string(),
                    dtype: "F32".to_string(),
                    index: 1,
                    row: 0,
                    file_offset: 68,
                    bits: "0x7f7fffff".to_string(),
                    field: String::new(),
                },
            ],
            ..Default::default()
        };

        // 1. Full remediation (output_published == true): SARIF targets --out with after scan (0 findings).
        let mut full = sample_receipt("models/dirty.gguf", "remediated", before_scan.clone());
        full.options = Some(SanitizeOptions {
            attempt_id: "ci/nightly/run-1".to_string(),
            output: "models/fixed.gguf".to_string(),
            clamp_nonfinite: true,
            clamp_extreme: true,
            max_examples: 128,
            ..Default::default()
        });
        full.attempt_id = "ci/nightly/run-1".to_string();
        full.after = Some(SanitizeScan { tensors: 1, float_elements: 16, ..Default::default() });
        full.output_published = true;
        full.output_sha256 = "b".repeat(64);

        let full_log = build_sanitize_sarif_with_base_dir(&full, Some(Path::new("/workspace/repo")));
        let full_run = &full_log.runs[0];
        assert_eq!(full_run.automation_details.id, "secant/sanitize/ci_nightly_run-1");
        assert!(full_run.results.is_empty());
        assert_eq!(full_run.artifacts.len(), 1);
        assert_eq!(full_run.artifacts[0].location.uri, "models/fixed.gguf");
        assert_eq!(full_run.artifacts[0].hashes.get("sha-256").map(String::as_str), Some(full.output_sha256.as_str()));
        let full_props = full_run.properties.as_ref().unwrap();
        assert_eq!(full_props.artifact_uri, "models/fixed.gguf");
        assert_eq!(full_props.evaluated_artifact, "remediated_output");
        assert_eq!(full_props.artifact_sha256, full.output_sha256);
        assert!(full_props.output_published);

        // 2. Partial remediation (output_published == false): SARIF targets untouched input with before scan (both findings).
        let mut partial = sample_receipt("models/dirty.gguf", "findings", before_scan);
        partial.options = Some(SanitizeOptions {
            attempt_id: "ci-partial".to_string(),
            output: "models/not-published.gguf".to_string(),
            clamp_nonfinite: true,
            max_examples: 128,
            ..Default::default()
        });
        let mut after_counts = BTreeMap::new();
        after_counts.insert("extreme_finite".to_string(), 1);
        partial.after = Some(SanitizeScan {
            tensors: 1,
            float_elements: 16,
            total_findings: 1,
            counts: after_counts,
            examples: vec![SanitizeFinding {
                kind: "extreme_finite".to_string(),
                tensor: "w".to_string(),
                dtype: "F32".to_string(),
                index: 1,
                row: 0,
                file_offset: 68,
                bits: "0x7f7fffff".to_string(),
                field: String::new(),
            }],
            ..Default::default()
        });
        partial.output_published = false;

        let partial_log = build_sanitize_sarif_with_base_dir(&partial, Some(Path::new("/workspace/repo")));
        let partial_run = &partial_log.runs[0];
        assert_eq!(partial_run.artifacts[0].location.uri, "models/dirty.gguf");
        assert_eq!(partial_run.artifacts[0].hashes.get("sha-256").map(String::as_str), Some(partial.pre_sha256.as_str()));
        let partial_props = partial_run.properties.as_ref().unwrap();
        assert_eq!(partial_props.artifact_uri, "models/dirty.gguf");
        assert_eq!(partial_props.evaluated_artifact, "input");
        assert!(!partial_props.output_published);
        assert_eq!(partial_props.total_findings, 2);
        assert_eq!(partial_run.results.len(), 2, "must report both nan and extreme_finite still present in input");
        let rule_ids: Vec<&str> = partial_run.results.iter().map(|r| r.rule_id.as_str()).collect();
        assert!(rule_ids.contains(&"SECANT-F-NAN"));
        assert!(rule_ids.contains(&"SECANT-F-EXTREME-FINITE"));
        for res in &partial_run.results {
            assert_eq!(res.locations[0].physical_location.artifact_location.uri, "models/dirty.gguf");
        }
    }

    #[test]
    fn kquant_iq_and_mxfp4_scale_findings_map_to_builtin_sarif_rules() {
        let mut counts = BTreeMap::new();
        counts.insert("quant_scale_nan".to_string(), 3);
        let scan = SanitizeScan {
            tensors: 3,
            quant_tensors: 3,
            quant_blocks: 3,
            total_findings: 3,
            counts,
            examples: vec![
                SanitizeFinding {
                    kind: "quant_scale_nan".to_string(),
                    tensor: "blk.0.ffn_down.weight".to_string(),
                    dtype: "Q8_K".to_string(),
                    index: 0,
                    row: 0,
                    file_offset: 128,
                    bits: "0x7fc00000".to_string(),
                    field: "d".to_string(),
                },
                SanitizeFinding {
                    kind: "quant_scale_nan".to_string(),
                    tensor: "blk.1.ffn_down.weight".to_string(),
                    dtype: "IQ1_M".to_string(),
                    index: 0,
                    row: 0,
                    file_offset: 512,
                    bits: "0x7e00".to_string(),
                    field: "d".to_string(),
                },
                SanitizeFinding {
                    kind: "quant_scale_nan".to_string(),
                    tensor: "blk.2.ffn_down.weight".to_string(),
                    dtype: "MXFP4".to_string(),
                    index: 0,
                    row: 0,
                    file_offset: 640,
                    bits: "0xff".to_string(),
                    field: "e".to_string(),
                },
            ],
            ..Default::default()
        };
        let rec = sample_receipt("models/mixed.gguf", "findings", scan);
        let log = build_sanitize_sarif_with_base_dir(&rec, Some(Path::new("/workspace/repo")));
        let run = &log.runs[0];
        assert_eq!(run.tool.driver.rules.len(), BUILTIN_RULES.len());
        assert_eq!(run.results.len(), 3);
        for res in &run.results {
            assert_eq!(res.rule_id, "SECANT-B3-2A-QUANT-SCALE-NAN");
            assert_eq!(run.tool.driver.rules[res.rule_index].id, "SECANT-B3-2A-QUANT-SCALE-NAN");
        }
        let mxfp4 = run.results.iter().find(|r| r.properties.dtype.as_deref() == Some("MXFP4")).unwrap();
        assert_eq!(mxfp4.properties.first_field.as_deref(), Some("e"));
        assert_eq!(mxfp4.properties.first_bits.as_deref(), Some("0xff"));
        assert!(mxfp4.message.text.contains("field 'e'"));
    }
}
