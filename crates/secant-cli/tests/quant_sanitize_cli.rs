//! End-to-end tests of `secant sanitize` on quantized GGUF files (Milestone
//! S-4): Q8_0 and Q4_K scale headers (`d`, `dmin`) are scanned for NaN and Inf
//! by default, with subnormal and extreme checks available as opt-in
//! diagnostics. Non-finite repair zeroes the affected block, and every other
//! quantized type is still refused with exit 2.

#[path = "../../secant-core/tests/common/mod.rs"]
mod common;

use common::*;
use secant_core::builder::*;
use secant_core::container::scan_gguf_layout;
use secant_core::quant::*;
use serde_json::Value;
use std::process::Command;

const D_IN: usize = 512;
const D_OUT: usize = 4;

fn secant(args: &[&str]) -> (i32, Value, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_secant")).args(args).output().unwrap();
    let v = serde_json::from_slice(&o.stdout).unwrap_or(Value::Null);
    (o.status.code().unwrap_or(-1), v, String::from_utf8_lossy(&o.stderr).into_owned())
}

fn quantized(name: &str, fmt: QuantFormat, scale: f32) -> BuildTensor {
    let x: Vec<f32> = (0..D_OUT * D_IN).map(|i| ((i * 37 % 101) as f32 - 50.0) * scale).collect();
    let (bytes, _) = roundtrip_matrix(fmt, &x, D_OUT, D_IN).unwrap();
    BuildTensor { name: name.into(), dtype: fmt.name().into(), shape: vec![D_OUT, D_IN], data: bytes }
}

/// A GGUF holding one Q8_0, one Q4_K and one F32 norm tensor; returns the
/// bytes and the payload offsets of the two quantized tensors.
fn fixture() -> (Vec<u8>, i64, i64) {
    let norm = BuildTensor { name: "blk.0.attn_norm.weight".into(), dtype: "F32".into(), shape: vec![D_IN], data: f32_bytes(&[1.0; D_IN]) };
    let b = build_gguf(
        &[],
        &[quantized("blk.0.ffn_down.weight", QuantFormat::Q8_0, 0.01), quantized("blk.0.ssm_out.weight", QuantFormat::Q4K, 0.02), norm],
        32,
    )
    .unwrap();
    let l = scan_gguf_layout(&b).unwrap();
    (b, l.tensor("blk.0.ffn_down.weight").unwrap().offset, l.tensor("blk.0.ssm_out.weight").unwrap().offset)
}

fn set16(b: &mut [u8], off: i64, bits: u16) {
    b[off as usize..off as usize + 2].copy_from_slice(&bits.to_le_bytes());
}

fn get16(b: &[u8], off: i64) -> u16 {
    u16::from_le_bytes([b[off as usize], b[off as usize + 1]])
}

fn sanitize(dir: &Scratch, model: &str, tag: &str, extra: &[&str]) -> (i32, Value, String) {
    let receipt = dir.path(&format!("{}.receipt.json", tag));
    let mut args = vec!["sanitize", "--model", model, "--receipt", &receipt, "--attempt-id", "att-s4", "--json"];
    args.extend_from_slice(extra);
    let r = secant(&args);
    let on_disk: Value = serde_json::from_slice(&std::fs::read(&receipt).expect("receipt always written")).unwrap();
    assert_eq!(on_disk["status"], r.1["status"], "receipt file and --json agree");
    r
}

#[test]
fn genuine_q8_0_and_q4_k_files_scan_clean_instead_of_being_refused() {
    let dir = Scratch::new("s4-control");
    let (b, _, _) = fixture();
    let p = dir.path("control.gguf");
    write(&p, &b);
    let (code, rep, err) = sanitize(&dir, &p, "control", &[]);
    assert_eq!(code, 0, "{} {}", rep, err);
    assert_eq!(rep["status"], "clean");
    assert_eq!(rep["policy_version"], "zero-false-positive-b3-v2");
    let before = &rep["before"];
    assert_eq!(before["quant_tensors"], 2);
    assert_eq!(before["quant_blocks"], (D_OUT * D_IN / 32 + D_OUT * D_IN / 256) as i64);
    assert_eq!(before["float_elements"], D_IN as i64, "the F32 norm is still scanned element by element");
    assert_eq!(before["norm_rows"], 1);
    assert_eq!(before["total_findings"], 0);
    assert!(rep["input_unchanged"].as_bool().unwrap());
}

#[test]
fn crafted_scale_headers_follow_b3_2a_and_zero_nonfinite_blocks() {
    let dir = Scratch::new("s4-crafted");
    let (mut b, q8, q4) = fixture();
    let orig = b.clone();
    // Q8_0 block 3 d = NaN, block 5 d = 65504 (65504 * 127 > 1e4), Q4_K block 1
    // dmin = +Inf, Q4_K block 0 d = negative subnormal.
    let sites = [
        (q8 + 3 * 34, 0x7e00u16, "quant_scale_nan", "d", 3i64, "blk.0.ffn_down.weight"),
        (q8 + 5 * 34, 0x7bff, "quant_scale_extreme", "d", 5, "blk.0.ffn_down.weight"),
        (q4 + 144 + 2, 0x7c00, "quant_scale_inf", "dmin", 1, "blk.0.ssm_out.weight"),
        (q4, 0x8001, "quant_scale_subnormal", "d", 0, "blk.0.ssm_out.weight"),
    ];
    for s in &sites {
        set16(&mut b, s.0, s.1);
    }
    let p = dir.path("crafted.gguf");
    write(&p, &b);

    let (code, rep, _) = sanitize(&dir, &p, "scan", &[]);
    assert_eq!(code, 1, "findings remain: exit 1");
    assert_eq!(rep["status"], "findings");
    let before = &rep["before"];
    assert_eq!(before["total_findings"], 2);
    for s in [&sites[0], &sites[2]] {
        assert_eq!(before["counts"][s.2], 1, "{}", s.2);
        let ex = before["examples"].as_array().unwrap().iter().find(|e| e["kind"] == s.2).expect(s.2);
        assert_eq!(ex["field"], s.3);
        assert_eq!(ex["index"], s.4, "block index");
        assert_eq!(ex["tensor"], s.5);
        assert_eq!(ex["file_offset"], s.0, "file_offset points at the f16 field");
        assert_eq!(ex["bits"], format!("0x{:04x}", s.1));
    }
    assert!(before["counts"].get("quant_scale_extreme").is_none());
    assert!(before["counts"].get("quant_scale_subnormal").is_none());
    // row = block / blocks-per-row (512 / 32 = 16 Q8_0 blocks per row)
    assert_eq!(before["examples"].as_array().unwrap().iter().find(|e| e["kind"] == "quant_scale_nan").unwrap()["row"], 0);

    // B3-2A repair ignores the valid finite diagnostics and zeroes each block
    // whose scale header is NaN or Inf.
    let out1 = dir.path("partial.gguf");
    let (code, rep, _) = sanitize(&dir, &p, "partial", &["--clamp-nonfinite", "--out", &out1]);
    assert_eq!(code, 0);
    assert_eq!(rep["status"], "remediated");
    assert!(rep["output_published"].as_bool().unwrap());
    assert_eq!(rep["changes"]["quant_block_zeroed"], 2);
    assert_eq!(rep["after"]["total_findings"], 0);
    let partial = std::fs::read(&out1).unwrap();
    assert!(partial[sites[0].0 as usize..sites[0].0 as usize + Q8_0_BLOCK_BYTES].iter().all(|&x| x == 0));
    let q4_block = sites[2].0 as usize - 2;
    assert!(partial[q4_block..q4_block + Q4_K_BLOCK_BYTES].iter().all(|&x| x == 0));

    // Opting into all diagnostics reports and repairs all four planted cases.
    let out = dir.path("clean.gguf");
    let (code, rep, err) = sanitize(&dir, &p, "full", &["--clamp-nonfinite", "--clamp-extreme", "--flush-subnormal", "--out", &out]);
    assert_eq!(code, 0, "{} {}", rep, err);
    assert_eq!(rep["status"], "remediated");
    assert!(rep["output_published"].as_bool().unwrap());
    assert_eq!(rep["before"]["total_findings"], 4);
    for s in &sites {
        assert_eq!(rep["before"]["counts"][s.2], 1, "{}", s.2);
    }
    assert_eq!(rep["after"]["total_findings"], 0);
    assert_eq!(std::fs::read(&p).unwrap(), b, "input never modified");
    let c = std::fs::read(&out).unwrap();
    assert_eq!(c.len(), b.len());
    let changed: Vec<usize> = (0..c.len()).filter(|&i| c[i] != b[i]).collect();
    let mut allowed: Vec<usize> = (sites[0].0 as usize..sites[0].0 as usize + Q8_0_BLOCK_BYTES).collect();
    allowed.extend(q4_block..q4_block + Q4_K_BLOCK_BYTES);
    allowed.extend([sites[1].0 as usize, sites[1].0 as usize + 1, sites[3].0 as usize, sites[3].0 as usize + 1]);
    assert!(changed.iter().all(|i| allowed.contains(i)), "only affected blocks and diagnostic fields change: {:?}", changed);
    assert!(c[sites[0].0 as usize..sites[0].0 as usize + Q8_0_BLOCK_BYTES].iter().all(|&x| x == 0));
    assert!(c[q4_block..q4_block + Q4_K_BLOCK_BYTES].iter().all(|&x| x == 0));
    assert_eq!(get16(&c, sites[3].0), 0x8000, "negative subnormal -> -0");
    let v = get16(&c, sites[1].0);
    assert!(v & 0x8000 == 0 && v < 0x7c00 && v != 0, "{:#06x}", v);
    // the clamped scales are bounded replacements, not the genuine values
    assert_ne!(get16(&c, sites[1].0), get16(&orig, sites[1].0));
    // the published copy rescans clean through the CLI
    let (code, rep, _) = sanitize(&dir, &out, "rescan", &[]);
    assert_eq!(code, 0);
    assert_eq!(rep["status"], "clean");
}

#[test]
fn other_quantized_types_are_still_refused_with_exit_2() {
    let dir = Scratch::new("s4-iq2_xxs");
    // Build a 256-element Q4_K tensor (144 bytes) and retag as IQ2_XXS (type 16, 256 elements, 66 bytes/block)
    let t = quantized("w", QuantFormat::Q4K, 0.01);
    let mut b = build_gguf(&[], &[t], 32).unwrap();
    let pat: Vec<u8> = [1u64.to_le_bytes().to_vec(), vec![b'w'], 2u32.to_le_bytes().to_vec()].concat();
    let at = b.windows(pat.len()).position(|w| w == pat.as_slice()).unwrap() + pat.len() + 16;
    assert_eq!(u32::from_le_bytes(b[at..at + 4].try_into().unwrap()), 12);
    b[at..at + 4].copy_from_slice(&16u32.to_le_bytes());
    let p = dir.path("iq2_xxs.gguf");
    write(&p, &b);
    let (code, rep, _) = sanitize(&dir, &p, "iq2_xxs", &[]);
    assert_eq!(code, 2);
    assert_eq!(rep["status"], "unknown");
    assert!(rep["error"].as_str().unwrap().contains("no reviewed B3-2A block-header map for IQ2_XXS"), "{}", rep["error"]);
}

#[test]
fn all_ten_mixed_gguf_quantized_formats_and_dry_run_scan_clean_and_detect_nan() {
    use secant_core::quant_scale::QuantScaleFormat;
    let dir = Scratch::new("s4-mixed-10fmt");
    let formats = QuantScaleFormat::ALL;
    assert_eq!(formats.len(), 10);

    // Build a multi-tensor GGUF with 2 blocks per format and finite 1.0 (0x3c00) in every f16 header slot.
    let tensors: Vec<BuildTensor> = formats
        .iter()
        .enumerate()
        .map(|(idx, fmt)| {
            let mut data = vec![0u8; 2 * fmt.block_bytes()];
            for blk in 0..2 {
                for &(_, field_off) in fmt.fields() {
                    let pos = blk * fmt.block_bytes() + field_off;
                    data[pos..pos + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
                }
            }
            BuildTensor {
                name: format!("blk.{}.attn_q.weight", idx),
                dtype: fmt.name().into(),
                shape: vec![2, fmt.block_elems()],
                data,
            }
        })
        .collect();
    let b = build_gguf(&[], &tensors, 32).unwrap();
    let clean_path = dir.path("mixed_clean.gguf");
    write(&clean_path, &b);
    let (code, rep, err) = sanitize(&dir, &clean_path, "mixed_clean", &["--dry-run"]);
    assert_eq!(code, 0, "{} {}", rep, err);
    assert_eq!(rep["status"], "clean");
    assert_eq!(rep["before"]["quant_tensors"], 10);
    assert_eq!(rep["before"]["quant_blocks"], 20);

    // Plant NaN (0x7e00) in block 0 and +Inf (0x7c00) in block 1 across EVERY f16 header field (15 fields x 2 blocks = 30 findings).
    let mut poisoned = b.clone();
    let layout = scan_gguf_layout(&poisoned).unwrap();
    let mut total_fields = 0i64;
    for (idx, fmt) in formats.iter().enumerate() {
        let t = layout.tensor(&format!("blk.{}.attn_q.weight", idx)).unwrap();
        for &(_, field_off) in fmt.fields() {
            set16(&mut poisoned, t.offset + field_off as i64, 0x7e00);
            set16(&mut poisoned, t.offset + fmt.block_bytes() as i64 + field_off as i64, 0x7c00);
            total_fields += 1;
        }
    }
    assert_eq!(total_fields, 15);
    let poisoned_path = dir.path("mixed_poisoned.gguf");
    write(&poisoned_path, &poisoned);
    let (code, rep, _) = sanitize(&dir, &poisoned_path, "mixed_poisoned", &["--dry-run"]);
    assert_eq!(code, 1);
    assert_eq!(rep["status"], "findings");
    assert_eq!(rep["before"]["total_findings"], 30);
    assert_eq!(rep["before"]["counts"]["quant_scale_nan"], 15);
    assert_eq!(rep["before"]["counts"]["quant_scale_inf"], 15);

    let remediated_path = dir.path("mixed_remediated.gguf");
    let (code, rep, err) = sanitize(&dir, &poisoned_path, "mixed_remediated", &["--clamp-nonfinite", "--out", &remediated_path]);
    assert_eq!(code, 0, "{} {}", rep, err);
    assert_eq!(rep["status"], "remediated");
    assert_eq!(rep["after"]["total_findings"], 0);
    assert_eq!(rep["changes"]["quant_block_zeroed"], 20);
}

#[test]
fn a_quantized_norm_tensor_is_unknown_not_clean() {
    let dir = Scratch::new("s4-qnorm");
    let b = build_gguf(&[], &[quantized("blk.0.ffn_norm.weight", QuantFormat::Q8_0, 0.01)], 32).unwrap();
    let p = dir.path("qnorm.gguf");
    write(&p, &b);
    let (code, rep, _) = sanitize(&dir, &p, "qnorm", &[]);
    assert_eq!(code, 2);
    assert_eq!(rep["status"], "unknown");
    assert!(rep["error"].as_str().unwrap().contains("norm-row checks are implemented for unquantized floats only"));
}

#[test]
fn fp8_safetensors_cli_scans_clean_detects_findings_and_refuses_repair() {
    let dir = Scratch::new("s4-fp8");
    let clean = build_safetensors(
        &[
            BuildTensor {
                name: "model.layers.0.mlp.down_proj.weight".into(),
                dtype: "F8_E4M3".into(),
                shape: vec![2, 4],
                data: vec![0x00, 0x01, 0x38, 0x7E, 0x80, 0x87, 0xB8, 0xFE],
            },
            BuildTensor {
                name: "model.layers.0.mlp.down_proj.weight_scale_inv".into(),
                dtype: "F32".into(),
                shape: vec![1, 1],
                data: f32_bytes(&[0.0125]),
            },
            BuildTensor {
                name: "model.layers.0.self_attn.o_proj.weight".into(),
                dtype: "F8_E5M2".into(),
                shape: vec![2, 2],
                data: vec![0x3C, 0xBC, 0x01, 0x6F],
            },
        ],
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let clean_path = dir.path("fp8_clean.safetensors");
    write(&clean_path, &clean);
    let (code, rep, err) = sanitize(&dir, &clean_path, "fp8_clean", &[]);
    assert_eq!(code, 0, "{} {}", rep, err);
    assert_eq!(rep["status"], "clean");
    assert_eq!(rep["before"]["total_findings"], 0);
    assert_eq!(rep["before"]["float_elements"], 13);
    assert_eq!(rep["before"]["info"]["subnormal_tolerated"], 3);

    // inspect --json also succeeds on FP8 SafeTensors
    let (icode, irep, ierr) = secant(&["inspect", "--model", &clean_path, "--json"]);
    assert_eq!(icode, 0, "{} {}", irep, ierr);
    assert_eq!(irep["tensors"].as_array().unwrap().len(), 3);

    // Hostile FP8 SafeTensors -> exit 1 ("findings")
    let poisoned = build_safetensors(
        &[
            BuildTensor { name: "w_e4m3".into(), dtype: "F8_E4M3".into(), shape: vec![1, 3], data: vec![0x38, 0x7F, 0xFF] },
            BuildTensor { name: "w_e5m2".into(), dtype: "F8_E5M2".into(), shape: vec![1, 3], data: vec![0x7C, 0x7D, 0x7B] },
        ],
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let poisoned_path = dir.path("fp8_poisoned.safetensors");
    write(&poisoned_path, &poisoned);
    let (code, rep, _) = sanitize(&dir, &poisoned_path, "fp8_poisoned", &[]);
    assert_eq!(code, 1);
    assert_eq!(rep["status"], "findings");
    assert_eq!(rep["before"]["counts"]["nan"], 3);
    assert_eq!(rep["before"]["counts"]["inf"], 1);
    assert_eq!(rep["before"]["counts"]["extreme_finite"], 1);

    // Remediation on FP8 fails closed with exit 2 ("unknown")
    let out_path = dir.path("fp8_out.safetensors");
    let (code, rep, _) = sanitize(&dir, &poisoned_path, "fp8_repair", &["--clamp-nonfinite", "--out", &out_path]);
    assert_eq!(code, 2);
    assert_eq!(rep["status"], "unknown");
}

#[test]
fn u8_masks_batchnorm_counters_and_s0_zero_sentinels_cli() {
    let dir = Scratch::new("s4-u8-bn-s0");
    let sub = dir.path("ckpt");
    std::fs::create_dir_all(&sub).unwrap();
    let model = build_safetensors(
        &[
            BuildTensor {
                name: "gpt_neox.layers.0.attention.bias".into(),
                dtype: "U8".into(),
                shape: vec![1, 1, 2, 2],
                data: vec![1, 0, 1, 1],
            },
            BuildTensor {
                name: "encoder.layers.0.conv.norm.num_batches_tracked".into(),
                dtype: "I64".into(),
                shape: vec![],
                data: 12i64.to_le_bytes().to_vec(),
            },
            BuildTensor {
                name: "encoder.layers.0.conv.norm.weight".into(),
                dtype: "F32".into(),
                shape: vec![4],
                data: f32_bytes(&[1.0, 1.0, 1.0, 1.0]),
            },
            BuildTensor {
                name: "vision_model.patch_embedding.weight".into(),
                dtype: "F32".into(),
                shape: vec![2, 2],
                data: f32_bytes(&[0.5, -0.5, 0.25, 0.75]),
            },
        ],
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let model_path = format!("{}/model.safetensors", sub);
    write(&model_path, &model);
    write(
        &format!("{}/config.json", sub),
        br#"{"vocab_size":0,"num_local_experts":0,"num_experts_per_tok":0,"bos_token_id":null,"eos_token_id":null}"#,
    );

    let (code, rep, err) = sanitize(&dir, &model_path, "u8_bn_s0_clean", &[]);
    assert_eq!(code, 0, "{} {}", rep, err);
    assert_eq!(rep["status"], "clean");
    assert_eq!(rep["before"]["total_findings"], 0);
    assert_eq!(rep["before"]["ignored_integer_tensors"], 2);
    assert_eq!(rep["before"]["norm_rows"], 1);

    // Non-mask U8 weight tensor still fails closed with exit 2
    let packed_u8 = build_safetensors(
        &[BuildTensor { name: "model.layers.0.mlp.weight".into(), dtype: "U8".into(), shape: vec![2, 2], data: vec![1, 2, 3, 4] }],
        &std::collections::BTreeMap::new(),
    )
    .unwrap();
    let packed_path = dir.path("packed_u8.safetensors");
    write(&packed_path, &packed_u8);
    let (code, rep, _) = sanitize(&dir, &packed_path, "packed_u8", &[]);
    assert_eq!(code, 2);
    assert_eq!(rep["status"], "unknown");
}
