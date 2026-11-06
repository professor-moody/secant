//! Invariant tests for the RRR-critical primitives: logical orientation,
//! the Qwen3.5 ssm_out permutation, Gram-Schmidt, npy codec, patch/verify.

mod common;
use common::*;
use secant_core::builder::*;
use secant_core::container::*;
use secant_core::namespace::legacy_alias;
use secant_core::plan::{Plan, PLAN_SCHEMA, PROFILE_SCHEMA};
use secant_core::subspace::*;
use std::collections::BTreeMap;

fn lay(k: usize, v: usize, hv: usize) -> SsmOutLayout {
    SsmOutLayout { num_k_heads: k, num_v_heads: v, head_v_dim: hv, head_k_dim: 128, metadata_derived: true }
}

#[test]
fn documented_qwen35_geometry_matches_the_16_17_interleave() {
    let l = lay(16, 32, 128);
    let order = l.interleave_order().unwrap();
    // GGUF block j holds HF block [0,2,4,...,30,1,3,...,31][j]
    let expect: Vec<usize> = (0..16).map(|k| 2 * k).chain((0..16).map(|k| 2 * k + 1)).collect();
    assert_eq!(order, expect);
    // HF 0,1,2,3 -> GGUF 0,16,1,17
    let g: Vec<usize> = (0..4).map(|h| l.gguf_block_of_hf(h).unwrap()).collect();
    assert_eq!(g, vec![0, 16, 1, 17]);
    assert_eq!(l.in_width(), 4096);
    assert_eq!(l.tile(), 256); // r * d_v
    let p = SsmOutPermutation::new(4096, &l).unwrap();
    for hf in [0usize, 1, 127, 128, 255, 256, 4095] {
        let g = p.forward_in_col(hf).unwrap();
        assert_eq!(p.inverse_in_col(g).unwrap(), hf);
        // within a 128-wide block the offset is preserved
        assert_eq!(g % 128, hf % 128);
    }
    assert!(p.forward_in_col(4096).is_err());
}

#[test]
fn generalized_ratio_r4() {
    let l = lay(8, 32, 64);
    assert_eq!(l.ratio(), 4);
    let order = l.interleave_order().unwrap();
    assert_eq!(&order[..4], &[0, 4, 8, 12]);
    assert_eq!(order[8], 1);
    let mut s = order.clone();
    s.sort();
    assert_eq!(s, (0..32).collect::<Vec<_>>());
}

#[test]
fn ssm_metadata_fails_closed() {
    let mut m: BTreeMap<String, String> = BTreeMap::new();
    m.insert("general.architecture".into(), "qwen35".into());
    // no canonical keys => explicit legacy default, flagged
    let d = SsmOutLayout::from_meta(&m).unwrap();
    assert!(!d.metadata_derived);
    assert_eq!((d.num_k_heads, d.num_v_heads, d.head_v_dim), (16, 32, 128));
    // partial block => error, never mixed with defaults
    m.insert("qwen35.ssm.group_count".into(), "16".into());
    assert!(SsmOutLayout::from_meta(&m).is_err());
    m.insert("qwen35.ssm.time_step_rank".into(), "32".into());
    m.insert("qwen35.ssm.inner_size".into(), "4096".into());
    m.insert("qwen35.ssm.state_size".into(), "128".into());
    let ok = SsmOutLayout::from_meta(&m).unwrap();
    assert!(ok.metadata_derived && ok.head_v_dim == 128);
    // keys under a different architecture prefix are never consumed
    let mut other = BTreeMap::new();
    other.insert("general.architecture".into(), "llama".into());
    other.insert("qwen35.ssm.group_count".into(), "16".into());
    assert!(!SsmOutLayout::from_meta(&other).unwrap().metadata_derived);
    // not divisible / r == 1 identity / non-positive
    m.insert("qwen35.ssm.inner_size".into(), "4097".into());
    assert!(SsmOutLayout::from_meta(&m).is_err());
    m.insert("qwen35.ssm.inner_size".into(), "4096".into());
    m.insert("qwen35.ssm.group_count".into(), "32".into());
    assert_eq!(SsmOutLayout::from_meta(&m).unwrap().ratio(), 1);
    m.insert("qwen35.ssm.group_count".into(), "64".into());
    assert!(SsmOutLayout::from_meta(&m).is_err());
    m.insert("qwen35.ssm.group_count".into(), "0".into());
    assert!(SsmOutLayout::from_meta(&m).is_err());
}

#[test]
fn logical_orientation_agrees_across_containers() {
    // Same logical [d_out=3, d_in=5] matrix in both containers.
    let (d_out, d_in) = (3usize, 5usize);
    let vals: Vec<f64> = (0..d_out * d_in).map(|i| i as f64 + 1.0).collect();
    let t = BuildTensor { name: "w".into(), dtype: "F32".into(), shape: vec![d_out, d_in], data: f32_bytes(&vals) };
    let s = Scratch::new("orient");
    let st = s.path("a.safetensors");
    let gg = s.path("a.gguf");
    write(&st, &build_safetensors(&[t.clone()], &BTreeMap::new()).unwrap());
    write(&gg, &build_gguf(&[], &[t], 32).unwrap());
    let (ms, mg) = (Model::open(&st).unwrap(), Model::open(&gg).unwrap());
    let (ts, tg) = (ms.layout.tensor("w").unwrap(), mg.layout.tensor("w").unwrap());
    assert_eq!((ts.logical_out, ts.logical_in), (3, 5));
    assert_eq!((tg.logical_out, tg.logical_in), (3, 5));
    assert_eq!(tg.shape, vec![5, 3 + 0 * 5].into_iter().take(0).chain([5usize, 3]).collect::<Vec<_>>());
    for o in 0..d_out {
        for i in 0..d_in {
            let (a, b) = (ts.weight(o, i).unwrap(), tg.weight(o, i).unwrap());
            assert_eq!(a, b);
            let va = secant_core::dtype::le_f32(&ms.tensor_bytes(ts)[a as usize * 4..]);
            let vb = secant_core::dtype::le_f32(&mg.tensor_bytes(tg)[b as usize * 4..]);
            assert_eq!(va, vb);
            assert_eq!(va as f64, vals[o * d_in + i]);
        }
    }
    assert!(ts.weight(3, 0).is_err() && ts.weight(0, 5).is_err());
}

#[test]
fn gram_schmidt_meets_1e12_and_fails_closed_on_dependence() {
    let mut rng = Rng(3);
    let n = 64;
    let a = rng.vec(n);
    // b nearly parallel to a (1e-8 perturbation): single-pass MGS would leave residual coupling
    let b: Vec<f64> = a.iter().map(|x| x * 3.0 + 1e-8 * rng.unit()).collect();
    let c = rng.vec(n);
    let mut named = BTreeMap::new();
    named.insert("a".to_string(), a.clone());
    named.insert("b".to_string(), b);
    named.insert("c".to_string(), c);
    let order: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    let (basis, rep) = gram_schmidt(&named, &order).unwrap();
    assert!(rep.post_cosine_max_abs < 1e-12, "{}", rep.post_cosine_max_abs);
    for k in &order {
        assert!((l2_norm(&basis[k]) - 1.0).abs() < 1e-14);
    }
    // exactly dependent => error
    let mut dep = BTreeMap::new();
    dep.insert("a".to_string(), a.clone());
    dep.insert("b".to_string(), a.iter().map(|x| -2.0 * x).collect());
    assert!(gram_schmidt(&dep, &["a".to_string(), "b".to_string()]).is_err());
    // dimension mismatch => error
    let mut bad = BTreeMap::new();
    bad.insert("a".to_string(), vec![1.0, 2.0]);
    bad.insert("b".to_string(), vec![1.0, 2.0, 3.0]);
    assert!(gram_schmidt(&bad, &["a".to_string(), "b".to_string()]).is_err());
}

#[test]
fn npy_roundtrip_header_and_rejections() {
    let v = vec![1.5f64, -2.25, 3.0];
    let b = npy_bytes_f64(&v, &[3]).unwrap();
    assert_eq!(&b[..8], b"\x93NUMPY\x01\x00");
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    assert_eq!((10 + hl) % 64, 0);
    assert!(std::str::from_utf8(&b[10..10 + hl]).unwrap().starts_with("{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }"));
    assert_eq!(b[10 + hl - 1], b'\n');
    let back = load_numpy(&b).unwrap();
    assert_eq!(back.to_f64(), v);
    let m = npy_bytes_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let mb = load_numpy(&m).unwrap();
    assert_eq!(mb.shape, vec![2, 3]);
    assert_eq!(mb.to_f64(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut fortran = b.clone();
    let s = String::from_utf8_lossy(&fortran[10..10 + hl]).replace("False", "True ");
    fortran[10..10 + hl].copy_from_slice(s.as_bytes());
    assert!(load_numpy(&fortran).is_err());
    assert!(load_numpy(b"NOTNUMPY....").is_err());
    assert!(npy_bytes_f64(&v, &[4]).is_err());
}

fn small_model(s: &Scratch) -> (String, String) {
    let mut rng = Rng(1);
    let ts = vec![tensor("blk.0.ssm_out.weight", "F32", &[4, 16], &mut rng), tensor("other.weight", "F32", &[4, 4], &mut rng)];
    let p = s.path("m.gguf");
    write(&p, &build_gguf(&qwen_meta(2, 4, 4, 4), &ts, 32).unwrap());
    (p, "blk.0.ssm_out.weight".into())
}

#[test]
fn permute_hf_to_gguf_then_back_is_identity_and_verify_catches_tampering() {
    let s = Scratch::new("perm");
    let (base, name) = small_model(&s);
    let mk = |perm: &str| {
        serde_json::json!({"schema":PLAN_SCHEMA,"profile":"p","calibration":{"mode":"gamma"},
            "edits":[{"id":"P","role":"x","op":"permute-columns","tensor":name,"permutation":perm}]})
    };
    let (p1, p2) = (s.path("p1.json"), s.path("p2.json"));
    write(&p1, mk("qwen35-ssm-out-128-hf-to-gguf").to_string().as_bytes());
    write(&p2, mk("qwen35-ssm-out-128-gguf-to-hf").to_string().as_bytes());
    let (o1, o2) = (s.path("o1.gguf"), s.path("o2.gguf"));
    apply_plan(&base, &o1, &Plan::load(&p1).unwrap(), PatchMode::CopyOnWrite).unwrap();
    assert_ne!(sha256_file(&base), sha256_file(&o1));
    apply_plan(&o1, &o2, &Plan::load(&p2).unwrap(), PatchMode::CopyOnWrite).unwrap();
    assert_eq!(sha256_file(&base), sha256_file(&o2), "permutation then inverse must restore the exact bytes");
    // verify-patch: tamper the header and a non-target tensor
    let mut b = std::fs::read(&o1).unwrap();
    b[9] ^= 0xFF; // version/counts region of the header
    let bad = s.path("bad.gguf");
    write(&bad, &b);
    assert!(verify_patch(&base, &bad, &[name.clone()]).is_err() || !verify_patch(&base, &bad, &[name.clone()]).unwrap().ok);
    let m = Model::open(&o1).unwrap();
    let other = m.layout.tensor("other.weight").unwrap().offset as usize;
    let mut c = std::fs::read(&o1).unwrap();
    c[other] ^= 1;
    let bad2 = s.path("bad2.gguf");
    write(&bad2, &c);
    let r = verify_patch(&base, &bad2, &[name]).unwrap();
    assert!(!r.ok && r.unexpected_changed == vec!["other.weight".to_string()]);
    // in-place with SHA-256 snapshot verification
    let ip = s.path("ip.gguf");
    std::fs::copy(&base, &ip).unwrap();
    let rep = apply_plan(&ip, &ip, &Plan::load(&p1).unwrap(), PatchMode::InPlace).unwrap();
    assert!(rep.verify.unwrap().ok);
    assert_eq!(sha256_file(&ip), sha256_file(&o1));
}

#[test]
fn plan_and_profile_inputs_accept_canonical_and_legacy_schema_ids() {
    let s = Scratch::new("schema-alias");
    assert_eq!(PLAN_SCHEMA, "secant.wsp.plan.v1");
    assert_eq!(PROFILE_SCHEMA, "secant.wsp.profile.v1");
    let legacy_plan = legacy_alias(PLAN_SCHEMA).unwrap();
    let plan_with = |schema: &str| {
        serde_json::json!({"schema":schema,"profile":"p","calibration":{"mode":"gamma"},
            "edits":[{"id":"P","role":"x","op":"permute-columns","tensor":"w","permutation":"qwen35-ssm-out-128-hf-to-gguf"}]})
    };
    for (tag, schema, ok) in [
        ("canonical", PLAN_SCHEMA.to_string(), true),
        ("legacy", legacy_plan.clone(), true),
        ("profile-id", PROFILE_SCHEMA.to_string(), false),
        ("unknown", "other.wsp.plan.v1".to_string(), false),
    ] {
        let p = s.path(&format!("plan-{}.json", tag));
        write(&p, plan_with(&schema).to_string().as_bytes());
        match Plan::load(&p) {
            Ok(plan) => {
                assert!(ok, "{}: {:?} must be rejected", tag, schema);
                // whatever was read, the in-memory (and any re-written) plan carries the canonical id
                assert_eq!(plan.schema, PLAN_SCHEMA, "{}", tag);
            }
            Err(e) => assert!(!ok, "{}: {:?} rejected: {}", tag, schema, e),
        }
    }
    // a profile file read by compose: canonical and legacy ids both load, others do not
    let vdir = s.path("vectors");
    std::fs::create_dir_all(&vdir).unwrap();
    write(&format!("{}/d_loop_L0.npy", vdir), &npy_bytes_f64(&[1.0, 0.0, 0.0, 0.0], &[4]).unwrap());
    let legacy_profile = legacy_alias(PROFILE_SCHEMA).unwrap();
    for (tag, schema, ok) in [
        ("canonical", PROFILE_SCHEMA.to_string(), true),
        ("legacy", legacy_profile, true),
        ("plan-id", PLAN_SCHEMA.to_string(), false),
    ] {
        let prof = serde_json::json!({
            "schema": schema, "name": "alias-profile", "description": "schema alias test",
            "subspaces": [], "default_target_shift": 0.0, "default_gamma": 0.25,
            "layer_selection": {"strategy": "x", "tool": "x", "banked": "x"},
            "slots": [{"id":"P1","role":"x","direction_token":"loop","op":"projector-subtract","tensor_pattern":"blk.%d.w",
                       "tensor_st_pattern":"m.%d.w","layer":0,"sign":-1,"needs_direction":true,"gamma":0.5}]
        });
        let pp = s.path(&format!("profile-{}.json", tag));
        write(&pp, prof.to_string().as_bytes());
        let opts = ComposeOptions { profile_path: pp.clone(), vectors_dir: vdir.clone(), gamma: 0.25, has_gamma: true, ..Default::default() };
        match compose(opts) {
            Ok(plan) => {
                assert!(ok, "{}: profile {:?} must be rejected", tag, schema);
                assert_eq!(plan.schema, PLAN_SCHEMA, "{}: composed plan must carry the canonical id", tag);
            }
            Err(e) => {
                assert!(!ok, "{}: profile {:?} rejected: {}", tag, schema, e);
                assert!(e.message().contains("schema"), "{}: rejected for the wrong reason: {}", tag, e);
            }
        }
    }
}

#[test]
fn layout_rejects_truncated_and_garbage_files() {
    let s = Scratch::new("trunc");
    let (base, _) = small_model(&s);
    let b = std::fs::read(&base).unwrap();
    let t = s.path("t.gguf");
    write(&t, &b[..b.len() - 7]);
    assert!(load_layout(&t).is_err(), "payload beyond EOF must be rejected");
    let g = s.path("g.bin");
    write(&g, b"this is not a model file at all");
    assert!(load_layout(&g).is_err());
    assert!(load_layout(&s.path("missing.gguf")).is_err());
}

#[test]
fn rank1_on_quantized_float_edit_is_refused() {
    let s = Scratch::new("qrefuse");
    let mut rng = Rng(2);
    let x: Vec<f32> = rng.vec(4 * 32).iter().map(|v| *v as f32).collect();
    let (bytes, _) = secant_core::quant::roundtrip_matrix(secant_core::quant::QuantFormat::Q8_0, &x, 4, 32).unwrap();
    let t = BuildTensor { name: "w".into(), dtype: "Q8_0".into(), shape: vec![4, 32], data: bytes };
    let p = s.path("q.gguf");
    write(&p, &build_gguf(&[], &[t], 32).unwrap());
    let plan = serde_json::json!({"schema":PLAN_SCHEMA,"profile":"p","calibration":{"mode":"gamma"},
        "edits":[{"id":"E","role":"x","op":"projector-subtract","tensor":"w","direction":[1.0,0.0,0.0,0.0],"gamma":1.0}]});
    let pp = s.path("plan.json");
    write(&pp, plan.to_string().as_bytes());
    let err = apply_plan(&p, &s.path("o.gguf"), &Plan::load(&pp).unwrap(), PatchMode::CopyOnWrite).unwrap_err();
    assert!(err.message().contains("Q8_0"), "{}", err);
}

#[test]
fn hostile_offsets_are_errors_not_panics() {
    // GGUF with one tensor whose declared offset is u64::MAX, and a SafeTensors
    // header whose data_offsets sit at the i64 limit: both must be rejected, in
    // debug builds (overflow checks on) as in release.
    let _s = Scratch::new("hostile");
    let mut rng = Rng(4);
    let t = tensor("w", "F32", &[2, 2], &mut rng);
    let mut g = build_gguf(&[], &[t], 32).unwrap();
    // tensor info = name(8+1) ndims(4) dims(16) type(4) offset(8); the offset is the last 8 bytes before padding
    let off_at = 4 + 4 + 8 + 8 + (8 + "general.alignment".len() + 4 + 4) + (8 + 1 + 4 + 16 + 4);
    g[off_at..off_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    let e = scan_gguf_layout(&g).unwrap_err();
    assert!(e.message().contains("signed 64-bit"), "{}", e);
    let hdr = format!("{{\"w\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[{},{}]}}}}", i64::MAX - 4, i64::MAX);
    let mut st = (hdr.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(hdr.as_bytes());
    let e2 = layout_of_bytes(&st, "x.safetensors").unwrap_err();
    assert!(e2.message().contains("overflow") || e2.message().contains("exceeds"), "{}", e2);
}
