//! C ABI for Secant. Built as `libsecant.a` (staticlib), `libsecant.{so,dylib}`
//! (cdylib) and an rlib. Every entry point is `_v1`-suffixed: the signatures
//! are frozen; a change ships as `_v2` beside it.
//!
//! Conventions: buffers are caller-owned, row-major, contiguous; the library
//! never retains a pointer past the call; no function allocates memory the
//! caller must free. Return codes: 0 ok; 1 null pointer or zero extent;
//! 2 size overflow; 3 the computation refused (NaN/Inf input, dependent or
//! misaligned input, geometry mismatch); 4 internal panic (caught at the
//! boundary). `secant_last_error_v1` returns the message for the most recent
//! non-zero code on the calling thread.

use secant_core::heal::project_then_heal_f32;
use secant_core::quant::{commutation_witness, QuantFormat};
use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

pub const SECANT_OK: i32 = 0;
pub const SECANT_ERR_ARG: i32 = 1;
pub const SECANT_ERR_OVERFLOW: i32 = 2;
pub const SECANT_ERR_REFUSED: i32 = 3;
pub const SECANT_ERR_PANIC: i32 = 4;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_err(msg: &str) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

fn code<F: FnOnce() -> Result<(), (i32, String)>>(f: F) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => SECANT_OK,
        Ok(Err((c, m))) => {
            set_err(&m);
            c
        }
        Err(_) => {
            set_err("internal panic caught at the C boundary");
            SECANT_ERR_PANIC
        }
    }
}

fn mul(a: usize, b: usize) -> Result<usize, (i32, String)> {
    a.checked_mul(b).ok_or((SECANT_ERR_OVERFLOW, format!("{} * {} overflows usize", a, b)))
}

/// Library version, a static NUL-terminated string (never free it).
#[no_mangle]
pub extern "C" fn secant_version_v1() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Message for the last non-zero return on this thread; valid until the next
/// failing call on the same thread. Empty string if none.
#[no_mangle]
pub extern "C" fn secant_last_error_v1() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Project-Then-Heal: replace `b` (row-major `d_out x r`) with
/// `B_locked = P(P B)`, `P = I - D^T D`, where `directions` is row-major
/// `k x d_out` (NOT assumed orthonormal; orthonormalized in float64 by
/// two-pass modified Gram-Schmidt). `*max_leak_out` receives
/// `max |D^T B_locked|` measured in float64 over the stored float32 values.
///
/// # Safety
/// `directions` must point to `k * d_out` floats, `b` to `d_out * r` writable
/// floats, `max_leak_out` to one writable double (may be NULL to skip).
#[no_mangle]
pub unsafe extern "C" fn secant_project_then_heal_f32_v1(
    directions: *const f32,
    k: usize,
    d_out: usize,
    b: *mut f32,
    r: usize,
    max_leak_out: *mut f64,
) -> i32 {
    code(|| {
        if directions.is_null() || b.is_null() || k == 0 || d_out == 0 || r == 0 {
            return Err((SECANT_ERR_ARG, "null pointer or zero extent".into()));
        }
        let (nd, nb) = (mul(k, d_out)?, mul(d_out, r)?);
        // SAFETY: the caller guarantees the extents (see function docs).
        let dirs = unsafe { slice::from_raw_parts(directions, nd) };
        let bm = unsafe { slice::from_raw_parts_mut(b, nb) };
        let leak = project_then_heal_f32(dirs, k, d_out, bm, r).map_err(|e| (SECANT_ERR_REFUSED, e.to_string()))?;
        if !max_leak_out.is_null() {
            unsafe { *max_leak_out = leak };
        }
        Ok(())
    })
}

/// Q8_0 commutation witness: `||D^T (Dequant(Quant(W)) - W)||_2` (Frobenius
/// over the `k x d_in` matrix) and its max-abs entry, for `w` row-major
/// `d_out x d_in` (`d_in` a multiple of 32) and `directions` row-major
/// `k x d_out` (orthonormalized internally).
///
/// # Safety
/// `directions` must point to `k * d_out` floats, `w` to `d_out * d_in`
/// floats; each out pointer to one writable double (either may be NULL).
#[no_mangle]
pub unsafe extern "C" fn secant_q8_0_commutation_witness_v1(
    directions: *const f32,
    k: usize,
    d_out: usize,
    w: *const f32,
    d_in: usize,
    out_l2: *mut f64,
    out_max_abs: *mut f64,
) -> i32 {
    code(|| {
        if directions.is_null() || w.is_null() || k == 0 || d_out == 0 || d_in == 0 {
            return Err((SECANT_ERR_ARG, "null pointer or zero extent".into()));
        }
        let (nd, nw) = (mul(k, d_out)?, mul(d_out, d_in)?);
        let dirs = unsafe { slice::from_raw_parts(directions, nd) };
        let wm = unsafe { slice::from_raw_parts(w, nw) };
        let rows: Vec<Vec<f64>> = (0..k).map(|i| dirs[i * d_out..(i + 1) * d_out].iter().map(|x| *x as f64).collect()).collect();
        if rows.iter().flatten().any(|x| !x.is_finite()) {
            return Err((SECANT_ERR_REFUSED, "directions hold NaN/Inf".into()));
        }
        let order: Vec<String> = (0..k).map(|i| format!("d{:06}", i)).collect();
        let named = order.iter().cloned().zip(rows).collect();
        let (basis, _) = secant_core::subspace::gram_schmidt(&named, &order).map_err(|e| (SECANT_ERR_REFUSED, e.to_string()))?;
        let ortho: Vec<Vec<f64>> = order.iter().map(|n| basis[n].clone()).collect();
        let wit = commutation_witness(QuantFormat::Q8_0, &ortho, wm, d_out, d_in).map_err(|e| (SECANT_ERR_REFUSED, e.to_string()))?;
        unsafe {
            if !out_l2.is_null() {
                *out_l2 = wit.witness_l2;
            }
            if !out_max_abs.is_null() {
                *out_max_abs = wit.witness_max_abs;
            }
        }
        Ok(())
    })
}
