//! Layer SNR ranking and Borda aggregation across subspaces (`resid`, `attn`,
//! `mlp`, `h`). Used by `secant analyze` and `secant compose --analysis` to
//! bind auto slots in Borda order from a `secant.wsp.analyze.v1` report.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const ANALYZE_SCHEMA: &str = "secant.wsp.analyze.v1";

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SnrCell {
    pub snr: f64,
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub heldout_margin: f64,
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub sigma_ctl: f64,
}

fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
}

fn is_empty_vec<T>(v: &[T]) -> bool {
    v.is_empty()
}

fn is_empty_str(v: &str) -> bool {
    v.is_empty()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RankedLayer {
    pub layer: i64,
    pub snr: f64,
    pub rank: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SubspaceRanking {
    pub ranked: Vec<RankedLayer>,
    pub peak: RankedLayer,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct BordaScore {
    pub layer: i64,
    pub score: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FamilyAnalysis {
    pub subspaces: BTreeMap<String, SubspaceRanking>,
    pub borda_winner: i64,
    pub borda_scores: Vec<BordaScore>,
    pub per_subspace_argmax: BTreeMap<String, i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Assignments {
    #[serde(rename = "L_loop_star", skip_serializing_if = "Option::is_none")]
    pub l_loop_star: Option<i64>,
    #[serde(rename = "L_ref_star", skip_serializing_if = "Option::is_none")]
    pub l_ref_star: Option<i64>,
    #[serde(rename = "L_root_star", skip_serializing_if = "Option::is_none")]
    pub l_root_star: Option<i64>,
    pub method: String,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AnalyzeReport {
    pub schema: String,
    pub input: String,
    pub layout_detected: String,
    pub families: BTreeMap<String, FamilyAnalysis>,
    pub assignments: Assignments,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AnalysisBinding {
    pub path: String,
    pub schema: String,
    #[serde(skip_serializing_if = "is_empty_str")]
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assignments: Option<Assignments>,
    pub bound_layers: BTreeMap<String, i64>,
    #[serde(skip_serializing_if = "is_empty_vec")]
    pub notes: Vec<String>,
}

const BANKED_HARDCODED_LAYERS: [i64; 2] = [15, 19];

pub fn family_aliases(role: &str) -> &'static [&'static str] {
    match role {
        "refusal" => &["b2", "refusal", "refusal_direction"],
        "root" => &["b3", "root_cause", "root"],
        "loop" => &["b1", "loop", "think_loop", "think-loop"],
        _ => &[],
    }
}

pub fn role_family(role: &str) -> &'static str {
    if role.contains("loop") {
        "loop"
    } else if role.contains("refusal") {
        "refusal"
    } else if role.contains("root") {
        "root"
    } else {
        ""
    }
}

pub fn analysis_family<'a>(families: &'a BTreeMap<String, FamilyAnalysis>, role_fam: &str) -> Option<&'a FamilyAnalysis> {
    for alias in family_aliases(role_fam) {
        if let Some(f) = families.get(*alias) {
            return Some(f);
        }
    }
    None
}

pub fn star_of(a: &Assignments, role_fam: &str) -> Option<i64> {
    match role_fam {
        "loop" => a.l_loop_star,
        "refusal" => a.l_ref_star,
        "root" => a.l_root_star,
        _ => None,
    }
}

fn parse_layer_key(k: &str) -> Option<i64> {
    let s = k.strip_prefix('L').unwrap_or(k);
    let n = s.parse::<i64>().ok()?;
    if n < 0 {
        None
    } else {
        Some(n)
    }
}

fn cell_from_value(v: &Value) -> Option<SnrCell> {
    match v {
        Value::Number(n) => Some(SnrCell { snr: n.as_f64()?, heldout_margin: 0.0, sigma_ctl: 0.0 }),
        Value::Object(_) => serde_json::from_value::<SnrCell>(v.clone()).ok(),
        _ => None,
    }
}

fn parse_family_cells(body: &Value) -> Result<BTreeMap<String, BTreeMap<String, SnrCell>>> {
    let subs = body.as_object().ok_or_else(|| Error::msg("expected subspace -> layer map"))?;
    let mut cells: BTreeMap<String, BTreeMap<String, SnrCell>> = BTreeMap::new();
    for (sub, layers_val) in subs {
        let Some(layers) = layers_val.as_object() else {
            continue;
        };
        for (lk, lval) in layers {
            if parse_layer_key(lk).is_none() {
                continue;
            }
            if let Some(cell) = cell_from_value(lval) {
                cells.entry(sub.clone()).or_default().insert(lk.clone(), cell);
            }
        }
    }
    Ok(cells)
}

fn analyze_family(cells: &BTreeMap<String, BTreeMap<String, SnrCell>>) -> FamilyAnalysis {
    let mut fam = FamilyAnalysis::default();
    let mut layer_scores: BTreeMap<i64, f64> = BTreeMap::new();
    for sub in ["resid", "attn", "mlp", "h"] {
        let Some(layer_cells) = cells.get(sub) else {
            continue;
        };
        if layer_cells.is_empty() {
            continue;
        }
        let mut ranked: Vec<RankedLayer> =
            layer_cells.iter().filter_map(|(lk, c)| parse_layer_key(lk).map(|layer| RankedLayer { layer, snr: c.snr, rank: 0 })).collect();
        ranked.sort_by(|a, b| b.snr.partial_cmp(&a.snr).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.layer.cmp(&b.layer)));
        let n = ranked.len();
        for (i, rl) in ranked.iter_mut().enumerate() {
            rl.rank = i;
            *layer_scores.entry(rl.layer).or_insert(0.0) += (n - 1 - i) as f64;
        }
        let peak = ranked[0].clone();
        fam.per_subspace_argmax.insert(sub.to_string(), peak.layer);
        fam.subspaces.insert(sub.to_string(), SubspaceRanking { ranked, peak });
    }
    let mut scores: Vec<BordaScore> = layer_scores.into_iter().map(|(layer, score)| BordaScore { layer, score }).collect();
    scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.layer.cmp(&b.layer)));
    if let Some(first) = scores.first() {
        fam.borda_winner = first.layer;
    }
    fam.borda_scores = scores;
    fam
}

fn hardcoded_diagnostics(rep: &AnalyzeReport) -> Vec<String> {
    let mut out = Vec::new();
    for role in ["refusal", "root"] {
        for alias in family_aliases(role) {
            let Some(fam) = rep.families.get(*alias) else {
                continue;
            };
            for sub in ["resid", "attn", "mlp", "h"] {
                let Some(r) = fam.subspaces.get(sub) else {
                    continue;
                };
                let mut by_layer = BTreeMap::new();
                for rl in &r.ranked {
                    by_layer.insert(rl.layer, rl);
                }
                for hc in BANKED_HARDCODED_LAYERS {
                    let Some(h) = by_layer.get(&hc) else {
                        continue;
                    };
                    let ratio =
                        if h.snr > 0.0 && r.peak.snr > 0.0 { format!("{:.2}x lower", r.peak.snr / h.snr) } else { "n/a".to_string() };
                    out.push(format!(
                        "{}.{}: hardcoded L{} snr={:.4} rank {}/{} vs peak L{} snr={:.4} ({}): the banked mid-layer default misses the bottleneck at this scale",
                        alias,
                        sub,
                        hc,
                        h.snr,
                        h.rank + 1,
                        r.ranked.len(),
                        r.peak.layer,
                        r.peak.snr,
                        ratio
                    ));
                }
            }
        }
    }
    if let Some(l) = rep.assignments.l_ref_star {
        out.push(format!("L_ref_star={} by Borda across subspaces", l));
    }
    if let Some(l) = rep.assignments.l_root_star {
        out.push(format!("L_root_star={} by Borda across subspaces", l));
    }
    out
}

pub fn analyze(path: &str) -> Result<AnalyzeReport> {
    let data = std::fs::read(path).map_err(|e| Error::msg(format!("read {}: {}", path, e)))?;
    let raw_val: Value = serde_json::from_slice(&data).map_err(|e| Error::msg(format!("{} is not a JSON object: {}", path, e)))?;
    let raw = raw_val.as_object().ok_or_else(|| Error::msg(format!("{} is not a JSON object", path)))?;

    let mut families_raw: BTreeMap<String, BTreeMap<String, BTreeMap<String, SnrCell>>> = BTreeMap::new();
    let mut layout = "family-major".to_string();

    let mut is_family_major = false;
    for role in ["refusal", "root", "loop"] {
        for alias in family_aliases(role) {
            if raw.contains_key(*alias) {
                is_family_major = true;
            }
        }
    }

    if is_family_major {
        for role in ["refusal", "root", "loop"] {
            for alias in family_aliases(role) {
                if let Some(body) = raw.get(*alias) {
                    let cells = parse_family_cells(body).map_err(|e| Error::msg(format!("family {}: {}", alias, e)))?;
                    families_raw.insert((*alias).to_string(), cells);
                }
            }
        }
    } else if let Some(layers_val) = raw.get("layers") {
        layout = "layer-major".to_string();
        let per_layer = layers_val.as_object().ok_or_else(|| Error::msg("layers: expected JSON object"))?;
        let mut cells: BTreeMap<String, BTreeMap<String, SnrCell>> = BTreeMap::new();
        for (lk, subs_val) in per_layer {
            let subs = subs_val.as_object().ok_or_else(|| Error::msg("layers entry: expected subspace object"))?;
            for (sub, cval) in subs {
                if let Some(c) = cell_from_value(cval) {
                    cells.entry(sub.clone()).or_default().insert(lk.clone(), c);
                }
            }
        }
        families_raw.insert("sweep".to_string(), cells);
    } else {
        layout = "flat".to_string();
        let cells = parse_family_cells(&raw_val)?;
        if cells.is_empty() {
            return Err(Error::msg(format!(
                "{}: no family (b2/b3), layers{{}}, or subspace (resid/attn/mlp/h) structure recognized",
                path
            )));
        }
        families_raw.insert("main".to_string(), cells);
    }

    let mut rep = AnalyzeReport {
        schema: ANALYZE_SCHEMA.to_string(),
        input: path.to_string(),
        layout_detected: layout,
        families: BTreeMap::new(),
        assignments: Assignments {
            l_loop_star: None,
            l_ref_star: None,
            l_root_star: None,
            method: "per-subspace argmax over precomputed SNR + Borda rank aggregation across subspaces (ranks supplied cells; native held-out capture is not implemented)".to_string(),
            notes: Vec::new(),
        },
        diagnostics: Vec::new(),
    };

    for (name, cells) in &families_raw {
        if !cells.is_empty() {
            rep.families.insert(name.clone(), analyze_family(cells));
        }
    }

    if let Some(f) = analysis_family(&rep.families, "refusal") {
        rep.assignments.l_ref_star = Some(f.borda_winner);
    }
    if let Some(f) = analysis_family(&rep.families, "root") {
        rep.assignments.l_root_star = Some(f.borda_winner);
    }
    if let Some(f) = analysis_family(&rep.families, "loop") {
        rep.assignments.l_loop_star = Some(f.borda_winner);
    } else {
        rep.assignments.notes.push(
            "no loop family (b1) in this input; L_loop falls back to sorted vector-file enumeration in compose, the profile banked default (0.8B L15/L19), or a custom profile".to_string(),
        );
    }
    rep.diagnostics = hardcoded_diagnostics(&rep);
    Ok(rep)
}

pub fn load_analyze_report(path: &str) -> Result<AnalyzeReport> {
    let data = std::fs::read(path).map_err(|e| Error::msg(format!("read analysis: {}", e)))?;
    let rep: AnalyzeReport = serde_json::from_slice(&data).map_err(|e| Error::msg(format!("parse analysis {}: {}", path, e)))?;
    if !crate::namespace::accepts(ANALYZE_SCHEMA, &rep.schema) {
        return Err(Error::msg(format!("analysis {}: schema {:?}, want {:?}", path, rep.schema, ANALYZE_SCHEMA)));
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subspace::{compose, ComposeOptions};

    #[test]
    fn analyze_ranks_borda_and_binds_compose_slots() {
        let dir = std::env::temp_dir().join(format!("secant-analyze-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let snr_path = dir.join("layer_snr.json");
        std::fs::write(
            &snr_path,
            r#"{
              "b2": {
                "resid": {"L12": 4.2, "L15": 1.1, "L19": 0.8},
                "attn":  {"L12": 3.8, "L15": 0.4, "L19": 0.6},
                "mlp":   {"L12": 2.9, "L15": 1.2, "L19": 0.9}
              }
            }"#,
        )
        .unwrap();

        let rep = analyze(snr_path.to_str().unwrap()).unwrap();
        assert_eq!(rep.schema, ANALYZE_SCHEMA);
        assert_eq!(rep.assignments.l_ref_star, Some(12));
        assert!(!rep.diagnostics.is_empty());

        let analysis_out = dir.join("analysis.json");
        std::fs::write(&analysis_out, serde_json::to_vec_pretty(&rep).unwrap()).unwrap();

        let vec_dir = dir.join("vecs");
        std::fs::create_dir_all(&vec_dir).unwrap();
        // Provide d_ref_L5, d_ref_L12, d_ref_L15; sorted first is L5 while Borda winner is L12.
        std::fs::write(vec_dir.join("d_ref_L5.json"), "[1.0, 0.0, 0.0, 0.0]").unwrap();
        std::fs::write(vec_dir.join("d_ref_L12.json"), "[0.0, 1.0, 0.0, 0.0]").unwrap();
        std::fs::write(vec_dir.join("d_ref_L15.json"), "[0.0, 0.0, 1.0, 0.0]").unwrap();

        let plan = compose(ComposeOptions {
            profile_name: "refusal-ablate".into(),
            vectors_dir: vec_dir.to_str().unwrap().into(),
            analysis_path: analysis_out.to_str().unwrap().into(),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(plan.edits.len(), 1);
        assert_eq!(plan.edits[0].layer, 12);
        assert!(plan.analysis.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
