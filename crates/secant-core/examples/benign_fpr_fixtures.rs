//! Deterministic, reduced transformer checkpoints for the sanitize FPR benchmark.
//! Uses the product's container writers and reference-tested quantizers. These
//! are numerical fixtures, not trained models or executable inference graphs.
//! Run through scripts/bench_benign_fpr.py; outputs must be a new directory.
#![forbid(unsafe_code)]

use secant_core::builder::{build_gguf, build_safetensors, BuildTensor, MetaValue};
use secant_core::dtype::{bf16_from_f32, dtype_size, read_elem};
use secant_core::numeric::fma_add;
use secant_core::quant::{f16_from_f32_ieee, roundtrip_matrix, QuantFormat};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// Fixed integer PRNG and CLT-12 normal approximation, avoiding libm normal
// sampling differences. All reductions have a fixed order and no fused ops.
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f64 / 16777216.0
    }
    fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    OpenOptions::new().write(true).create_new(true).open(path)?.write_all(bytes)?;
    Ok(())
}

fn tensor(name: &str, dtype: &str, shape: Vec<usize>, values: &[f32]) -> Result<BuildTensor> {
    let data = if let Some(q) = QuantFormat::from_dtype(dtype) {
        let cols = *shape.last().ok_or("missing matrix dimension")?;
        roundtrip_matrix(q, values, values.len() / cols, cols)?.0
    } else {
        let mut data = Vec::new();
        for &v in values {
            match dtype {
                "F32" => data.extend(v.to_le_bytes()),
                "F16" => data.extend(f16_from_f32_ieee(v).to_le_bytes()),
                "BF16" => data.extend(bf16_from_f32(v).to_le_bytes()),
                _ => return Err(format!("unsupported benchmark dtype {dtype}").into()),
            }
        }
        data
    };
    Ok(BuildTensor { name: name.into(), dtype: dtype.into(), shape, data })
}

fn matrix(rng: &mut Rng, name: &str, dtype: &str, rows: usize, cols: usize, variant: usize) -> Result<BuildTensor> {
    let sigma = [0.02, 0.01, 0.04, 0.0625, 0.015625][variant];
    let values: Vec<f32> = (0..rows * cols)
        .map(|i| {
            // Mild heteroscedasticity and a sparse wider component in variant 4.
            let scale = if variant == 4 && i % 97 == 0 { sigma * 3.0 } else { sigma };
            (rng.normal() * scale) as f32
        })
        .collect();
    tensor(name, dtype, vec![rows, cols], &values)
}

fn fixture(arch: &str, dtype: &str, variant: usize, seed: u64) -> Result<(Vec<BuildTensor>, Value)> {
    let mut rng = Rng(seed);
    let d = if variant.is_multiple_of(2) { 256 } else { 512 };
    let vocab = 256 + variant * 128;
    let ff = d * if variant < 3 { 2 } else { 4 };
    let layers = 1 + variant % 2;
    let gguf = dtype.starts_with('Q');
    let dense_dtype = if gguf { "F32" } else { dtype };
    let embed = if gguf {
        "token_embd.weight"
    } else if arch == "gpt2" {
        "transformer.wte.weight"
    } else {
        "model.embed_tokens.weight"
    };
    let mut ts = vec![matrix(&mut rng, embed, dtype, vocab, d, variant)?];
    for layer in 0..layers {
        let prefix = if gguf { format!("blk.{layer}") } else { format!("model.layers.{layer}") };
        // LayerNorm and RMSNorm gamma vectors share the near-one distribution.
        // Both names are selected by the default norm policy without extra flags.
        for role in ["input_layernorm", "post_attention_layernorm"] {
            let gamma: Vec<f32> = (0..d).map(|_| fma_add(1.0, 0.02, rng.normal()) as f32).collect();
            ts.push(tensor(&format!("{prefix}.{role}.weight"), dense_dtype, vec![d], &gamma)?);
        }
        if arch == "gpt2" {
            ts.push(matrix(&mut rng, &format!("{prefix}.attn.c_attn.weight"), dtype, 3 * d, d, variant)?);
        } else {
            for role in ["q_proj", "k_proj", "v_proj"] {
                let rows = if arch == "qwen2" && role != "q_proj" { d / 4 } else { d };
                ts.push(matrix(&mut rng, &format!("{prefix}.self_attn.{role}.weight"), dtype, rows, d, variant)?);
            }
        }
        ts.push(matrix(&mut rng, &format!("{prefix}.self_attn.o_proj.weight"), dtype, d, d, variant)?);
        let experts = if arch == "mixtral" { 4 } else { 1 };
        for e in 0..experts {
            let p = if experts > 1 { format!("{prefix}.mlp.experts.{e}") } else { format!("{prefix}.mlp") };
            for role in ["up_proj", "gate_proj"] {
                if arch != "gpt2" || role == "up_proj" {
                    ts.push(matrix(&mut rng, &format!("{p}.{role}.weight"), dtype, ff, d, variant)?);
                }
            }
            ts.push(matrix(&mut rng, &format!("{p}.down_proj.weight"), dtype, d, ff, variant)?);
        }
        if experts > 1 {
            ts.push(matrix(&mut rng, &format!("{prefix}.mlp.gate.weight"), "F32", experts, d, variant)?);
        }
        ts.push(tensor(&format!("{prefix}.self_attn.k_scale"), "F32", vec![1], &[0.125 + variant as f32 * 0.03125])?);
    }
    let cfg = json!({"model_type": arch, "vocab_size": vocab, "bos_token_id": 1, "eos_token_id": 2,
        "hidden_size": d, "intermediate_size": ff, "num_hidden_layers": layers,
        "num_local_experts": if arch == "mixtral" {4} else {0},
        "num_experts_per_tok": if arch == "mixtral" {2} else {0}});
    Ok((ts, cfg))
}

fn metrics(ts: &[BuildTensor], cfg: &Value) -> Value {
    let mut max_abs = 0.0f64;
    let mut min_norm = f64::INFINITY;
    let mut min_divisor = f64::INFINITY;
    let mut max_quant_scale = 0.0f64;
    let mut nonfinite_quant_scales = 0usize;
    for t in ts {
        if let Some(q) = QuantFormat::from_dtype(&t.dtype) {
            for block in t.data.chunks_exact(q.block_bytes()) {
                for off in if t.dtype == "Q4_K" { vec![0, 2] } else { vec![0] } {
                    let scale = read_elem(block, "F16", off);
                    if scale.is_finite() {
                        max_quant_scale = max_quant_scale.max(scale.abs());
                    } else {
                        nonfinite_quant_scales += 1;
                    }
                }
            }
            continue;
        }
        let mut sum = 0.0;
        for off in (0..t.data.len()).step_by(dtype_size(&t.dtype) as usize) {
            let v = read_elem(&t.data, &t.dtype, off);
            max_abs = max_abs.max(v.abs());
            if t.name.contains("layernorm") {
                sum = fma_add(sum, v, v);
            }
            if t.name.ends_with("k_scale") {
                min_divisor = min_divisor.min(v);
            }
        }
        if t.name.contains("layernorm") {
            min_norm = min_norm.min(sum.sqrt());
        }
    }
    json!({"dense_max_abs": max_abs, "min_norm_l2": min_norm, "min_divisor": min_divisor,
        "max_quant_scale_abs": if nonfinite_quant_scales == 0 { Some(max_quant_scale) } else { None },
        "nonfinite_quant_scale_fields": nonfinite_quant_scales,
        "token_id_headroom": cfg["vocab_size"].as_i64().unwrap() - 1 - cfg["eos_token_id"].as_i64().unwrap(),
        "expert_used_headroom": cfg["num_local_experts"].as_i64().unwrap() - cfg["num_experts_per_tok"].as_i64().unwrap()})
}

#[allow(clippy::too_many_arguments)]
fn emit(root: &Path, id: &str, arch: &str, dtype: &str, variant: usize, seed: u64, control: &str) -> Result<Value> {
    let dir = root.join(id);
    fs::create_dir(&dir)?;
    let (mut ts, mut cfg) = fixture(arch, dtype, variant, seed)?;
    let mut expected = "";
    match control {
        "norm_zero" | "norm_tiny" | "norm_subnormal" => {
            let t = ts.iter_mut().find(|t| t.name.contains("layernorm")).unwrap();
            let v = match control {
                "norm_zero" => 0.0,
                "norm_tiny" => 1e-20,
                _ => f32::from_bits(1),
            };
            *t = tensor(&t.name, "F32", t.shape.clone(), &vec![v; t.elements() as usize])?;
            expected = if control == "norm_zero" { "norm_all_zero" } else { "norm_below_min" };
        }
        "divisor_subnormal" => {
            let t = ts.iter_mut().find(|t| t.name.ends_with("k_scale")).unwrap();
            t.data = f32::from_bits(1).to_le_bytes().to_vec();
            expected = "divisor_scale_below_floor";
        }
        "scale_negative" => {
            let t = ts.iter_mut().find(|t| t.name.ends_with("k_scale")).unwrap();
            t.data = (-1.0f32).to_le_bytes().to_vec();
            expected = "divisor_scale_nonpositive";
        }
        "quant_nan" | "quant_inf" => {
            let t = ts.iter_mut().find(|t| t.dtype == dtype).unwrap();
            let bits = if control == "quant_nan" { 0x7e00u16 } else { 0x7c00u16 };
            t.data[..2].copy_from_slice(&bits.to_le_bytes());
            expected = if control == "quant_nan" { "quant_scale_nan" } else { "quant_scale_inf" };
        }
        "token_oob" => {
            cfg["eos_token_id"] = cfg["vocab_size"].clone();
            expected = "s0_companion_token_id_oob";
        }
        "expert_used_oob" => {
            cfg["num_experts_per_tok"] = json!(5);
            expected = if dtype.starts_with('Q') { "s0_expert_used_count" } else { "s0_companion_expert_used_count" };
        }
        "" => {}
        _ => return Err("unknown control".into()),
    }
    let stats = metrics(&ts, &cfg);
    let gguf = dtype.starts_with('Q');
    let filename = if gguf { "model.gguf" } else { "model.safetensors" };
    let bytes = if gguf {
        let mut meta = vec![("general.architecture".into(), MetaValue::Str(arch.into()))];
        if arch == "mixtral" {
            meta.push((format!("{arch}.expert_count"), MetaValue::U32(4)));
            meta.push((format!("{arch}.expert_used_count"), MetaValue::U32(cfg["num_experts_per_tok"].as_u64().unwrap() as u32)));
        }
        build_gguf(&meta, &ts, 32)?
    } else {
        write_new(&dir.join("config.json"), &serde_json::to_vec_pretty(&cfg)?)?;
        build_safetensors(&ts, &BTreeMap::new())?
    };
    write_new(&dir.join(filename), &bytes)?;
    Ok(json!({"id": id, "path": format!("{id}/{filename}"), "role": if control.is_empty() {"benign"} else {"positive"},
        "architecture": arch, "dtype": dtype, "variant": variant, "seed": seed,
        "bytes": bytes.len(), "sha256": format!("{:x}", Sha256::digest(&bytes)), "tensors": ts.len(),
        "config": cfg, "metrics": stats, "expected_kind": expected, "control": control,
        "source": "synthetic", "args": []}))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: benign_fpr_fixtures NEW_DIRECTORY ATTEMPT_ID".into());
    }
    let root = Path::new(&args[1]);
    fs::create_dir(root)?;
    let mut cases = Vec::new();
    for (a, arch) in ["llama", "gpt2", "qwen2", "mixtral"].iter().enumerate() {
        for (d, dtype) in ["F32", "F16", "BF16", "Q8_0", "Q4_K"].iter().enumerate() {
            for variant in 0..5 {
                let seed = 610000 + (a * 100 + d * 10 + variant) as u64;
                cases.push(emit(root, &format!("{arch}-{dtype}-{variant}"), arch, dtype, variant, seed, "")?);
            }
        }
    }
    for control in ["norm_zero", "norm_tiny", "norm_subnormal", "divisor_subnormal", "scale_negative", "token_oob", "expert_used_oob"] {
        cases.push(emit(root, &format!("positive-{control}"), "mixtral", "F32", 0, 610300, control)?);
    }
    for dtype in ["Q8_0", "Q4_K"] {
        for control in ["quant_nan", "quant_inf", "expert_used_oob"] {
            cases.push(emit(root, &format!("positive-{dtype}-{control}"), "mixtral", dtype, 0, 610300, control)?);
        }
    }
    let manifest = json!({"schema": "secant.benign-fpr-fixtures.v1", "attempt_id": args[2], "cases": cases});
    write_new(&root.join("manifest.json"), &serde_json::to_vec_pretty(&manifest)?)?;
    println!("generated {} cases", manifest["cases"].as_array().unwrap().len());
    Ok(())
}
