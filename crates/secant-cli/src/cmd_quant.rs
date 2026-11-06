use crate::util::*;
use clap::{Args as ClapArgs, Subcommand};
use secant_core::container::{Model, PatchMode};
use secant_core::dtype::*;
use secant_core::plan::Plan;
use secant_core::quant::*;
use secant_core::subspace::{load_directions, load_numpy_file};
use secant_core::Error;

#[derive(ClapArgs, Debug)]
#[command(
    long_about = "Inspect quantization error, measure protected-subspace leakage, or edit Q8_0 blocks. Supported formats are q8_0 and q4_k; patching uses Q8_0. Numerical diagnostics are measurements, not an inference-quality guarantee.",
    after_help = "Examples:\n  secant quant roundtrip --npy weights.npy --format q8_0\n  secant quant witness --model model.gguf --tensor blk.0.ffn_down.weight --vectors d.npy\n  secant quant patch --base base.gguf --plan plan.json --out edited.gguf"
)]
pub struct Args {
    #[command(subcommand)]
    sub: Sub,
}

#[derive(Subcommand, Debug)]
enum Sub {
    /// Measure ||D^T (Dequant(Quant(W')) - W')||_2 for a tensor (the Pi_128 commutation witness)
    #[command(
        after_help = "Example:\n  secant quant witness --model model.gguf --tensor blk.0.ffn_down.weight --vectors d.npy --format q8_0"
    )]
    Witness {
        #[arg(long)]
        model: String,
        #[arg(long)]
        tensor: String,
        /// Protected directions: directory of d_*.npy/.json or one .npy ([k, d] rows or a vector)
        #[arg(long)]
        vectors: String,
        /// q8_0 or q4_k
        #[arg(long, default_value = "q8_0")]
        format: String,
        /// Also test equivariance under the qwen35 ssm_out Pi_128 panel permutation (GGUF ssm_out tensors)
        #[arg(long)]
        pi128: bool,
        #[arg(long)]
        json: bool,
    },
    /// Block-aligned Q8_0 surgery: only blocks holding an edited element are re-quantized
    #[command(after_help = "Example:\n  secant quant patch --base base.gguf --plan plan.json --out edited.gguf --json")]
    Patch {
        #[arg(long)]
        base: String,
        #[arg(long)]
        plan: String,
        #[arg(long)]
        out: Option<String>,
        #[arg(long)]
        in_place: bool,
        #[arg(long)]
        json: bool,
    },
    /// Round-trip a .npy matrix/vector through a format and report the error
    #[command(after_help = "Example:\n  secant quant roundtrip --npy weights.npy --format q4_k --json")]
    Roundtrip {
        #[arg(long)]
        npy: String,
        #[arg(long, default_value = "q8_0")]
        format: String,
        #[arg(long)]
        json: bool,
    },
}

fn load_f32_matrix(model_path: &str, tensor: &str) -> Result<(Vec<f32>, usize, usize, secant_core::container::Layout), Error> {
    let m = Model::open(model_path)?;
    let t = m.layout.tensor(tensor)?.clone();
    t.require_2d("quant witness")?;
    let (d_out, d_in) = (t.logical_out, t.logical_in);
    let data = m.tensor_bytes(&t);
    let mut w = vec![0f32; d_out * d_in];
    if let Some(fmt) = QuantFormat::from_dtype(&t.dtype) {
        if d_in % fmt.block_elems() != 0 {
            return Err(Error::msg(format!("{}: d_in {} is not block-aligned for {}", tensor, d_in, t.dtype)));
        }
        dequantize_row(fmt, data, &mut w)?;
    } else if is_float_dtype(&t.dtype) {
        let tb = TensorBuffer::new(&t.dtype, data.to_vec());
        for (i, x) in w.iter_mut().enumerate() {
            *x = tb.get(i as i64) as f32;
        }
    } else {
        return Err(Error::msg(format!("{}: dtype {} is not a float or supported quantized tensor", tensor, t.dtype)));
    }
    Ok((w, d_out, d_in, m.layout))
}

pub fn run(a: &Args) -> i32 {
    match &a.sub {
        Sub::Witness { model, tensor, vectors, format, pi128, json } => {
            let r = (|| -> Result<serde_json::Value, Error> {
                let fmt = QuantFormat::parse(format)?;
                let (w, d_out, d_in, layout) = load_f32_matrix(model, tensor)?;
                let dirs = load_directions(vectors)?;
                let wit = commutation_witness(fmt, &dirs, &w, d_out, d_in)?;
                let mut v = serde_json::to_value(&wit).map_err(|e| Error::msg(e.to_string()))?;
                if *pi128 {
                    let lay = secant_core::container::SsmOutLayout::from_meta(&layout.metadata)?;
                    let p = secant_core::container::SsmOutPermutation::new(d_in, &lay)?;
                    let eq = panel_equivariance(fmt, &w, d_out, d_in, p.perm())?;
                    v["pi128"] = serde_json::to_value(&eq).map_err(|e| Error::msg(e.to_string()))?;
                }
                Ok(v)
            })();
            match r {
                Ok(v) => {
                    if *json {
                        if let Err(e) = emit_json(&v) {
                            return fail(&e);
                        }
                    } else {
                        section("Quantization witness");
                        field("Tensor", tensor);
                        field("Format", v["format"].as_str().unwrap_or("UNKNOWN"));
                        table(
                            &["METRIC", "VALUE"],
                            &[
                                ("Projected quantization error (L2)", "witness_l2"),
                                ("Projected quantization error (max abs)", "witness_max_abs"),
                                ("Quantization error (Frobenius)", "quant_error_fro"),
                                ("Protected component before (Frobenius)", "dt_w_fro_before"),
                            ]
                            .iter()
                            .map(|(label, key)| {
                                vec![
                                    label.to_string(),
                                    v[*key].as_f64().map(|n| format!("{:.6e}", n)).unwrap_or_else(|| "UNKNOWN".into()),
                                ]
                            })
                            .collect::<Vec<_>>(),
                        );
                        println!("\nMeasurements only; no leakage threshold gate was run.");
                        if let Some(p) = v.get("pi128") {
                            section("Pi_128 equivariance");
                            details(p);
                        }
                    }
                    0
                }
                Err(e) => fail(&e),
            }
        }
        Sub::Patch { base, plan, out, in_place, json } => {
            let out = match (out, *in_place) {
                (Some(o), _) => o.clone(),
                (None, true) => base.clone(),
                (None, false) => return usage("--out is required (or pass --in-place)"),
            };
            let plan = match Plan::load(plan) {
                Ok(p) => p,
                Err(e) => return fail(&e),
            };
            let mode = if *in_place { PatchMode::InPlace } else { PatchMode::CopyOnWrite };
            match q8_0_patch(base, &out, &plan, mode) {
                Ok(rep) => {
                    if *json {
                        return match emit_json(&rep) {
                            Ok(()) => 0,
                            Err(e) => fail(&e),
                        };
                    }
                    section("Q8_0 block patch");
                    field("Base", base);
                    field("Output", out);
                    if let Some(verify) = &rep.verify {
                        field("Byte-identity gate", badge(verify.ok));
                    }
                    field("Output SHA-256", &rep.out_sha256);
                    table(
                        &["EDIT", "TENSOR", "TOUCHED / TOTAL", "REWRITTEN", "MAX REQUANT ERROR"],
                        &rep
                            .edits
                            .iter()
                            .map(|e| {
                                vec![
                                    e.id.clone(),
                                    e.tensor.clone(),
                                    format!("{} / {}", e.blocks_touched, e.blocks_total),
                                    e.blocks_rewritten.to_string(),
                                    format!("{:.6e}", e.max_requant_error),
                                ]
                            })
                            .collect::<Vec<_>>(),
                    );
                    0
                }
                Err(e) => fail(&e),
            }
        }
        Sub::Roundtrip { npy, format, json } => {
            let r = (|| -> Result<serde_json::Value, Error> {
                let fmt = QuantFormat::parse(format)?;
                let arr = load_numpy_file(npy)?;
                let w: Vec<f32> = arr.to_f64().iter().map(|x| *x as f32).collect();
                let (rows, cols) = match arr.shape.len() {
                    1 => (1, arr.shape[0]),
                    2 => (arr.shape[0], arr.shape[1]),
                    n => return Err(Error::msg(format!("{}-D array", n))),
                };
                let (bytes, deq) = roundtrip_matrix(fmt, &w, rows, cols)?;
                let mut max_err = 0f64;
                let mut sq = 0f64;
                for (a, b) in w.iter().zip(&deq) {
                    let d = (*a as f64 - *b as f64).abs();
                    max_err = max_err.max(d);
                    sq += d * d;
                }
                Ok(serde_json::json!({
                    "format": fmt.name(), "rows": rows, "cols": cols, "bytes": bytes.len(),
                    "max_abs_error": max_err, "rmse": (sq / w.len() as f64).sqrt(),
                    "sha256": secant_core::container::sha256_hex(&bytes),
                }))
            })();
            match r {
                Ok(v) => {
                    if *json {
                        if let Err(e) = emit_json(&v) {
                            return fail(&e);
                        }
                    } else {
                        section("Quantization round-trip");
                        details(&v);
                    }
                    0
                }
                Err(e) => fail(&e),
            }
        }
    }
}
