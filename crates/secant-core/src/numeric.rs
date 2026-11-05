//! Arithmetic policy. The oracle (Go) does NOT forbid fused multiply-add in
//! most of its float64 code: on arm64 the Go compiler may compile `s += a*b`
//! to a single FMADD, on amd64 it does not (GOAMD64<v3). Only `read.go`
//! blocks fusion with explicit `float64()` conversions. Secant therefore
//! computes everything unfused (portable IEEE-754, identical on every host)
//! unless a caller opts into `go-arm64-fma` compatibility, which reproduces
//! the arm64 oracle's fused expressions so a differential run on an arm64
//! host can be bit-exact. The default is the portable result.

use std::sync::atomic::{AtomicBool, Ordering};

static GO_ARM64_FMA: AtomicBool = AtomicBool::new(false);

pub fn set_go_arm64_fma(on: bool) {
    GO_ARM64_FMA.store(on, Ordering::Relaxed);
}

pub fn go_arm64_fma() -> bool {
    GO_ARM64_FMA.load(Ordering::Relaxed)
}

/// `c + a*b`: fused only in go-arm64-fma compatibility mode.
#[inline]
pub fn fma_add(c: f64, a: f64, b: f64) -> f64 {
    if go_arm64_fma() {
        a.mul_add(b, c)
    } else {
        c + a * b
    }
}

/// `c - a*b`: fused (FMSUB) only in go-arm64-fma compatibility mode.
#[inline]
pub fn fma_sub(c: f64, a: f64, b: f64) -> f64 {
    if go_arm64_fma() {
        (-a).mul_add(b, c)
    } else {
        c - a * b
    }
}

/// Go `%g` / `%.Ng` float formatting (shortest or N significant digits,
/// exponent form when exp < -4 || exp >= precision (shortest: >= 21)).
pub fn go_fmt_g(v: f64, prec: Option<usize>) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "+Inf".into() } else { "-Inf".into() };
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    // digits and exponent via Rust's exact formatting
    let (digits, exp10) = match prec {
        Some(p) => {
            let p = p.max(1);
            let s = format!("{:.*e}", p - 1, v.abs());
            let (m, e) = s.split_once('e').unwrap();
            (m.replace('.', ""), e.parse::<i32>().unwrap())
        }
        None => {
            let s = format!("{:e}", v.abs());
            let (m, e) = s.split_once('e').unwrap();
            (m.replace('.', ""), e.parse::<i32>().unwrap())
        }
    };
    // %g with a precision strips trailing zeros (no '#' flag)
    let digits = digits.trim_end_matches('0').to_string();
    let digits = if digits.is_empty() { "0".to_string() } else { digits };
    let eprec = match prec {
        Some(p) => p.max(1) as i32,
        None => 21,
    };
    let eprec = if prec.is_none() && digits.len() as i32 > eprec { digits.len() as i32 } else { eprec };
    let sign = if v < 0.0 { "-" } else { "" };
    if exp10 < -4 || exp10 >= eprec {
        let mut m = digits[..1].to_string();
        if digits.len() > 1 {
            m.push('.');
            m.push_str(&digits[1..]);
        }
        let (es, ev) = if exp10 < 0 { ('-', -exp10) } else { ('+', exp10) };
        format!("{}{}e{}{:02}", sign, m, es, ev)
    } else if exp10 >= 0 {
        let int_len = (exp10 + 1) as usize;
        if digits.len() <= int_len {
            format!("{}{}{}", sign, digits, "0".repeat(int_len - digits.len()))
        } else {
            format!("{}{}.{}", sign, &digits[..int_len], &digits[int_len..])
        }
    } else {
        format!("{}0.{}{}", sign, "0".repeat((-exp10 - 1) as usize), digits)
    }
}

/// Go `%v` of a `[]string`: `[a b c]`.
pub fn go_fmt_slice(v: &[String]) -> String {
    format!("[{}]", v.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn g_format_matches_go() {
        assert_eq!(go_fmt_g(13.997742, Some(6)), "13.9977");
        assert_eq!(go_fmt_g(9.09e-9, Some(3)), "9.09e-09");
        assert_eq!(go_fmt_g(0.001, None), "0.001");
        assert_eq!(go_fmt_g(1e-12, None), "1e-12");
        assert_eq!(go_fmt_g(10.0, None), "10");
        assert_eq!(go_fmt_g(1.4826, None), "1.4826");
        assert_eq!(go_fmt_g(123456789.0, Some(6)), "1.23457e+08");
        assert_eq!(go_fmt_g(100000.0, Some(6)), "100000");
    }
}
