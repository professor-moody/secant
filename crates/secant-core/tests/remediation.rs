//! Regression controls for the independent R1-R8 cold review.
mod common;
use common::*;
use secant_core::builder::*;
use secant_core::container::*;
use secant_core::plan::{Edit, Plan};
use secant_core::quant::*;
use secant_core::subspace::*;
use std::collections::BTreeMap;

fn float_model() -> Vec<u8> {
    build_safetensors(&[BuildTensor { name: "w".into(), dtype: "F64".into(), shape: vec![2, 2], data: vec![0; 32] }], &BTreeMap::new())
        .unwrap()
}

fn float_edit() -> Edit {
    Edit {
        id: "valid".into(),
        tensor: "w".into(),
        op: "rank1-columns".into(),
        direction: vec![1., 0.],
        cols: vec![0],
        coeffs: vec![1.],
        ..Default::default()
    }
}

fn q8_model() -> Vec<u8> {
    build_gguf(&[], &[BuildTensor { name: "w".into(), dtype: "Q8_0".into(), shape: vec![2, 32], data: vec![0; 68] }], 32).unwrap()
}

fn patch(quant: bool, base: &str, out: &str, plan: &Plan, mode: PatchMode) -> bool {
    if quant {
        q8_0_patch(base, out, plan, mode).is_ok()
    } else {
        apply_plan(base, out, plan, mode).is_ok()
    }
}

#[test]
fn patch_transactions_preserve_files_on_late_failure_and_publish_noops() {
    for quant in [false, true] {
        let s = Scratch::new("transactions");
        let (base, out) = (s.path("base"), s.path("out"));
        let bytes = if quant { q8_model() } else { float_model() };
        write(&base, &bytes);
        write(&out, b"previous output");
        let mut invalid = float_edit();
        invalid.cols = vec![999];
        let plan = Plan { edits: vec![float_edit(), invalid], ..Default::default() };
        for mode in [PatchMode::CopyOnWrite, PatchMode::InPlace] {
            assert!(!patch(quant, &base, &out, &plan, mode));
            assert_eq!(std::fs::read(&base).unwrap(), bytes);
            assert_eq!(std::fs::read(&out).unwrap(), b"previous output");
        }
        let good = Plan { edits: vec![float_edit()], ..Default::default() };
        assert!(!patch(quant, &base, &base, &good, PatchMode::CopyOnWrite));
        let alias = s.path("alias");
        std::fs::hard_link(&base, &alias).unwrap();
        assert!(!patch(quant, &base, &alias, &good, PatchMode::CopyOnWrite));
        #[cfg(unix)]
        {
            let link = s.path("symlink");
            std::os::unix::fs::symlink(&base, &link).unwrap();
            assert!(!patch(quant, &base, &link, &good, PatchMode::CopyOnWrite));
        }
        assert!(patch(quant, &base, &out, &Plan::default(), PatchMode::CopyOnWrite));
        assert_eq!(std::fs::read(&out).unwrap(), bytes);
        let mut noop = float_edit();
        noop.coeffs = vec![0.];
        assert!(patch(quant, &base, &out, &Plan { edits: vec![noop], ..Default::default() }, PatchMode::CopyOnWrite));
        let directory = s.path("directory");
        std::fs::create_dir(&directory).unwrap();
        assert!(!patch(quant, &base, &directory, &good, PatchMode::CopyOnWrite));
        assert_eq!(std::fs::read(&base).unwrap(), bytes);
        assert!(!std::fs::read_dir(s.dir()).unwrap().any(|e| e.unwrap().file_name().to_string_lossy().starts_with(".secant-patch-")));
        assert!(patch(quant, &base, &out, &good, PatchMode::CopyOnWrite));
        assert_eq!(std::fs::read(&base).unwrap(), bytes);
        assert!(verify_patch(&base, &out, &["w".into()]).unwrap().ok);
        assert!(patch(quant, &base, &out, &good, PatchMode::InPlace));
    }
}

fn assert_family_rejects(bytes: &[u8]) {
    let s = Scratch::new("layout-reject");
    let (base, out) = (s.path("base"), s.path("out"));
    write(&base, bytes);
    assert!(layout_of_bytes(bytes, &base).is_err());
    assert!(Model::open(&base).is_err());
    assert!(apply_plan(&base, &out, &Plan::default(), PatchMode::CopyOnWrite).is_err());
    assert!(q8_0_patch(&base, &out, &Plan::default(), PatchMode::CopyOnWrite).is_err());
    assert!(!std::path::Path::new(&out).exists());
}

fn gguf_two() -> Vec<u8> {
    let ts: Vec<_> = ["tensor_a", "tensor_b"]
        .iter()
        .map(|n| BuildTensor { name: (*n).into(), dtype: "F32".into(), shape: vec![1, 1], data: vec![0; 4] })
        .collect();
    build_gguf(&[], &ts, 32).unwrap()
}

fn offset_field(bytes: &[u8], name: &str, dims: usize) -> usize {
    bytes.windows(name.len()).position(|s| s == name.as_bytes()).unwrap() + name.len() + 4 + dims * 8 + 4
}

#[test]
fn shared_layout_rejects_overlap_alignment_and_partial_quant_rows() {
    let body = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut st = (body.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(body);
    st.extend_from_slice(&[0; 4]);
    assert_family_rejects(&st);
    for off in [0u64, 1] {
        let mut gg = gguf_two();
        let field = offset_field(&gg, "tensor_b", 2);
        gg[field..field + 8].copy_from_slice(&off.to_le_bytes());
        assert_family_rejects(&gg);
    }
    for alignment in [0, 3, 1 << 21] {
        let mut gg = gguf_two();
        let value = 24 + 8 + "general.alignment".len() + 4;
        gg[value..value + 4].copy_from_slice(&(alignment as u32).to_le_bytes());
        assert_family_rejects(&gg);
    }
    let mut gg = q8_model();
    let dim = offset_field(&gg, "w", 2) - 4 - 16;
    gg[dim..dim + 8].copy_from_slice(&31u64.to_le_bytes());
    assert_family_rejects(&gg);
}

#[test]
fn legitimate_scalar_empty_padded_and_quantized_layouts_remain_valid() {
    let body = br#"{"a":{"dtype":"F64","shape":[],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[0],"data_offsets":[8,8]},"c":{"dtype":"F32","shape":[0,2],"data_offsets":[8,8]}}"#;
    let mut st = (body.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(body);
    st.extend_from_slice(&[0; 8]);
    for bytes in [st, gguf_two(), q8_model()] {
        let s = Scratch::new("valid-layout");
        let (base, out) = (s.path("base"), s.path("out"));
        write(&base, &bytes);
        assert!(Model::open(&base).is_ok());
        for q in [false, true] {
            assert!(patch(q, &base, &out, &Plan::default(), PatchMode::CopyOnWrite));
        }
    }
}

#[test]
fn verifier_covers_gap_trailer_and_total_length() {
    let s = Scratch::new("verify-gaps");
    let (base, out) = (s.path("base"), s.path("out"));
    let bytes = gguf_two();
    let l = layout_of_bytes(&bytes, &base).unwrap();
    write(&base, &bytes);
    for pos in [(l.tensors[0].offset + l.tensors[0].size) as usize, bytes.len() - 1] {
        let mut changed = bytes.clone();
        changed[pos] = 7;
        write(&out, &changed);
        assert!(!verify_patch(&base, &out, &[]).unwrap().ok);
    }
    let mut extended = bytes.clone();
    extended.push(0);
    write(&out, &extended);
    assert!(!verify_patch(&base, &out, &[]).unwrap().ok);
    write(&out, &bytes[..bytes.len() - 1]);
    assert!(!verify_patch(&base, &out, &[]).unwrap().ok);
}

#[test]
fn gram_matrix_checks_finite_rescalings_and_invalid_inputs() {
    for scale in [1e-300, 1.0, 1e300] {
        let named = BTreeMap::from([("a".into(), vec![scale, scale, 0.]), ("b".into(), vec![scale, 0., scale])]);
        let (basis, _) = gram_schmidt(&named, &["a".into(), "b".into()]).unwrap();
        for (i, a) in basis.values().enumerate() {
            for (j, b) in basis.values().enumerate() {
                assert!((dot(a, b) - if i == j { 1. } else { 0. }).abs() < 1e-12);
            }
        }
    }
    for v in [vec![f64::NAN, 1.], vec![f64::INFINITY, 0.], vec![f64::MAX, f64::MAX], vec![0., 0.]] {
        assert!(gram_schmidt(&BTreeMap::from([("a".into(), v)]), &["a".into()]).is_err());
    }
}

#[test]
fn quantizers_reject_unrepresentable_scales_before_block_writes() {
    let mut q8 = [77u8; 34];
    assert!(quantize_row_q8_0(&[f32::MAX; 32], &mut q8).is_err());
    assert_eq!(q8, [77; 34]);
    quantize_row_q8_0(&[65504.0 * 127.0; 32], &mut q8).unwrap();
    let mut dq = [0.; 32];
    dequantize_row_q8_0(&q8, &mut dq).unwrap();
    assert!(dq.iter().all(|x| x.is_finite()));
    let mut q4 = [77u8; 144];
    assert!(quantize_row_q4_k(&[f32::MAX; 256], &mut q4).is_err());
    assert_eq!(q4, [77; 144]);
    let x: Vec<_> = (0..256).map(|i| if i % 2 == 0 { -1e6 } else { 1e6 }).collect();
    quantize_row_q4_k(&x, &mut q4).unwrap();
    let mut dq4 = [0.; 256];
    dequantize_row_q4_k(&q4, &mut dq4).unwrap();
    assert!(dq4.iter().all(|x| x.is_finite()));
    let s = Scratch::new("quant-range");
    let (base, out) = (s.path("base"), s.path("out"));
    let bytes = q8_model();
    write(&base, &bytes);
    write(&out, b"keep");
    let mut edit = float_edit();
    edit.coeffs = vec![1e30];
    assert!(q8_0_patch(&base, &out, &Plan { edits: vec![edit], ..Default::default() }, PatchMode::CopyOnWrite).is_err());
    assert_eq!(std::fs::read(&out).unwrap(), b"keep");
    assert_eq!(std::fs::read(&base).unwrap(), bytes);
}

#[test]
fn compose_correlated_control_measures_combined_update_on_original_directions() {
    let s = Scratch::new("correlated-compose");
    let vectors = s.path("vectors");
    std::fs::create_dir(&vectors).unwrap();
    write(&format!("{vectors}/d_a_L0.json"), b"[1,0]");
    write(&format!("{vectors}/d_b_L0.json"), b"[1,1]");
    let profile = s.path("profile.json");
    // Same +1,+1 two-direction control as the cold review, followed by
    // differing activation/support and independent-tensor controls.
    for (ha, hb, separate, sign) in [("[1]", "[1]", false, 1.), ("[1,2]", "[2,1]", false, -1.), ("[1]", "[1]", true, 1.)] {
        write(&format!("{vectors}/h_a_L0.json"), ha.as_bytes());
        write(&format!("{vectors}/h_b_L0.json"), hb.as_bytes());
        let slots = serde_json::json!({"schema":"secant.wsp.profile.v1", "name":"correlated-directions-control", "default_target_shift":1.0,
            "slots":[{"id":"a","op":"rank1-columns","direction_token":"a","tensor_pattern":"w","tensor_st_pattern":"w","layer":0,"sign":1.,"top_k":1,"needs_direction":true},
                     {"id":"b","op":"rank1-columns","direction_token":"b","tensor_pattern":if separate {"other"} else {"w"},"tensor_st_pattern":if separate {"other"} else {"w"},"layer":0,"sign":sign,"top_k":1,"target_shift":2.,"needs_direction":true}]});
        // First iteration reproduces Astra's +1,+1 exactly.
        let mut slots = slots;
        if ha == "[1]" {
            slots["slots"][1]["target_shift"] = serde_json::json!(1.);
        }
        write(&profile, &serde_json::to_vec(&slots).unwrap());
        let flag_shift = if ha == "[1]" { 1. } else { 7. };
        let plan = compose(ComposeOptions {
            profile_path: profile.clone(),
            vectors_dir: vectors.clone(),
            target_shift: flag_shift,
            has_shift: true,
            ..Default::default()
        })
        .unwrap();
        let units = [vec![1., 0.], vec![std::f64::consts::FRAC_1_SQRT_2; 2]];
        for i in 0..2 {
            let mut delta_h = vec![0.; 2];
            for edit in &plan.edits {
                if edit.tensor != plan.edits[i].tensor {
                    continue;
                }
                for (col, coeff) in edit.cols.iter().zip(&edit.coeffs) {
                    for o in 0..2 {
                        delta_h[o] += edit.direction[o] * coeff * plan.edits[i].h_ref[*col as usize];
                    }
                }
            }
            assert!((dot(&units[i], &delta_h) - plan.edits[i].achieved_shift).abs() < 1e-12);
        }
        assert_eq!(plan.edits[0].achieved_shift, flag_shift);
        assert_eq!(plan.edits[1].achieved_shift, if ha == "[1]" { 1. } else { -2. });
    }
}

#[test]
fn verify_subspace_rejects_overflowed_norms_and_invalid_supplied_basis() {
    use secant_core::heal::*;
    let s = Scratch::new("verify-finite");
    let (base, healed, vectors) = (s.path("base"), s.path("healed"), s.path("vectors"));
    std::fs::create_dir(&vectors).unwrap();
    write(&format!("{vectors}/d_a.json"), b"[1,0]");
    write(&base, &float_model());
    let mut data = float_model();
    let t = layout_of_bytes(&data, &base).unwrap().tensor("w").unwrap().clone();
    data[t.offset as usize..t.offset as usize + 8].copy_from_slice(&1e308f64.to_le_bytes());
    write(&healed, &data);
    let mut opts = VerifySubspaceOptions { base, healed, vectors: vectors.clone(), tensors: vec!["w".into()], ..Default::default() };
    let report = verify_subspace(&opts).unwrap();
    assert_ne!(report.verdict, "pass");
    let mut protected = load_protected_directions(&vectors).unwrap();
    protected.basis = std::sync::Arc::new(vec![vec![0., 0.]]);
    opts.protected = Some(protected);
    assert!(verify_subspace(&opts).is_err());
}
