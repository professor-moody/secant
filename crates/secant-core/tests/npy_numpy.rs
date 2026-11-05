//! `.npy` bytes written by Secant must equal `numpy.save` byte for byte (the
//! Go oracle's writer pads its header one byte too long). NOT_RUN when no
//! `python3` with numpy is available.

mod common;
use common::*;
use secant_core::subspace::*;
use std::process::Command;

fn numpy_save(scratch: &Scratch, dtype: &str, shape: &[usize], values: &[f64], name: &str) -> Option<Vec<u8>> {
    let out = scratch.path(name);
    let shape_py = format!("({},)", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","));
    let vals = values.iter().map(|v| format!("{:?}", v)).collect::<Vec<_>>().join(",");
    let code = format!("import numpy as np,sys; a=np.array([{}],dtype=np.{}).reshape({}); np.save(sys.argv[1], a)", vals, dtype, shape_py);
    let st = Command::new("python3").args(["-c", &code, &out]).output().ok()?;
    if !st.status.success() {
        return None;
    }
    std::fs::read(&out).ok()
}

#[test]
fn npy_bytes_equal_numpy_save() {
    let s = Scratch::new("npy");
    let mut rng = Rng(11);
    let mut checked = 0;
    // lengths chosen to hit different header-padding residues
    for (shape, f32mode) in [
        (vec![1usize], false),
        (vec![3], false),
        (vec![7], true),
        (vec![64], false),
        (vec![2, 3], true),
        (vec![5, 4], false),
        (vec![1000], true),
        (vec![12, 12, 3], false),
    ] {
        let n: usize = shape.iter().product();
        let v: Vec<f64> = if f32mode { rng.vec(n).iter().map(|x| (*x as f32) as f64).collect() } else { rng.vec(n) };
        let dtype = if f32mode { "float32" } else { "float64" };
        let Some(reference) = numpy_save(&s, dtype, &shape, &v, &format!("ref_{}.npy", checked)) else {
            println!("NOT_RUN npy_bytes_equal_numpy_save: python3 with numpy is not available");
            return;
        };
        let mine = if f32mode {
            npy_bytes_f32(&v.iter().map(|x| *x as f32).collect::<Vec<_>>(), &shape).unwrap()
        } else {
            npy_bytes_f64(&v, &shape).unwrap()
        };
        assert_eq!(mine, reference, "shape {:?} {}: bytes differ from numpy.save", shape, dtype);
        checked += 1;
    }
    println!("PASS {} arrays byte-identical to numpy.save", checked);
}
