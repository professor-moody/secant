//! Builds `tests/c_smoke.c` against the staticlib and runs it, then checks the
//! same entry points through the Rust symbols (same bytes as `secant heal`).

use std::process::Command;

#[test]
fn c_program_links_libsecant_a_and_passes() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    // The staticlib is produced next to the test executable's profile dir.
    let exe = std::env::current_exe().unwrap();
    let profile_dir = exe.parent().unwrap().parent().unwrap().to_path_buf();
    let lib = profile_dir.join("libsecant.a");
    // `cargo test` only builds the rlib of this package; make sure the staticlib exists
    // for the profile under test (idempotent when it is already up to date).
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut build = Command::new(cargo);
    build.args(["build", "-p", "secant-ffi", "--manifest-path"]).arg(format!("{}/Cargo.toml", manifest));
    if !cfg!(debug_assertions) {
        build.arg("--release");
    }
    assert!(build.status().expect("run cargo build").success(), "building libsecant.a failed");
    assert!(lib.exists(), "libsecant.a not found at {}", lib.display());
    let out_dir = std::env::temp_dir().join(format!("secant-ffi-{}", std::process::id()));
    std::fs::create_dir_all(&out_dir).unwrap();
    let exe_c = out_dir.join("c_smoke");
    let cc = std::env::var("CC").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "cc".into());
    let mut cmd = Command::new(cc);
    cmd.args(["-O1", "-Wall", "-Werror", "-I"])
        .arg(format!("{}/include", manifest))
        .arg(format!("{}/tests/c_smoke.c", manifest))
        .arg(&lib)
        .arg("-lm");
    if cfg!(target_os = "macos") {
        cmd.args(["-framework", "CoreFoundation", "-lc++", "-lresolv"]);
    } else {
        cmd.args(["-lpthread", "-ldl"]);
    }
    let st = cmd.arg("-o").arg(&exe_c).status().expect("run C compiler");
    assert!(st.success(), "C compile/link failed");
    let run = Command::new(&exe_c).output().unwrap();
    assert!(run.status.success(), "c_smoke failed: {}", String::from_utf8_lossy(&run.stderr));
    println!("{}", String::from_utf8_lossy(&run.stdout));
    let _ = std::fs::remove_dir_all(&out_dir);
}

#[test]
fn rust_side_symbols_match_the_core_functions() {
    let dirs = [1f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut b: Vec<f32> = (0..12).map(|i| i as f32 * 0.1 - 0.5).collect();
    let mut b2 = b.clone();
    let mut leak = -1f64;
    let rc = unsafe { secant::secant_project_then_heal_f32_v1(dirs.as_ptr(), 2, 4, b.as_mut_ptr(), 3, &mut leak) };
    assert_eq!(rc, 0);
    let leak2 = secant_core::heal::project_then_heal_f32(&dirs, 2, 4, &mut b2, 3).unwrap();
    assert_eq!(b, b2);
    assert_eq!(leak.to_bits(), leak2.to_bits());
    // rows 0 and 1 of B are zeroed by projecting out e0, e1
    assert!(b[..6].iter().all(|x| x.abs() < 1e-7));
    let v = unsafe { std::ffi::CStr::from_ptr(secant::secant_version_v1()) }.to_str().unwrap();
    assert_eq!(v, env!("CARGO_PKG_VERSION"));
}
