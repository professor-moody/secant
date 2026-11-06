//! Tensor-file security: the static weight-value sanitizer (`sanitize`) and the
//! boundary-value fixture generator (`probe`).
//!
//! Port of the Go oracle's `internal/wsp/sanitize.go` and `internal/wsp/probe.go`
//! (CLI semantics as in the oracle's `wsp sanitize` and `wsp probe` subcommands).
//!
//! Both receipts are written with the oracle's exact `encoding/json` layout
//! (struct field order, sorted map keys, `omitempty`, HTML escaping, Go float
//! formatting, 2-space indent) so that a receipt differs from the oracle's
//! only in the fields that are inherently run-specific: `started_utc`,
//! `finished_utc`, `receipt_sha256` (it seals the compact serialization that
//! contains both timestamps and every path), and the path-valued fields
//! (`input`, `options.output`, `options.out_dir`, `options.base`,
//! `fixtures[].path`).
//!
//! Policy scope (identical to the oracle, stated honestly): the sanitizer
//! DETECTS NaN, Inf, subnormal, extreme finite values, zero or tiny-L2 norm
//! rows and negative / out-of-range / duplicate routing ids. It REPAIRS only
//! three of those: NaN (to +0), +-Inf (to the signed largest value not above
//! `max_abs`) and duplicate routing ids (to the lowest unused valid expert).
//! Subnormal, extreme-finite, norm and negative/out-of-range findings are
//! never rewritten; a copy that still has one after the rescan is NOT
//! published and the status stays `findings`.

pub use probe::{
    generate_probe, write_probe_receipt, ProbeFixture, ProbeOptions, ProbeReport, ProbeTensor, ProbeUnavailable, PROBE_ARCHITECTURE,
    PROBE_CLASSES, PROBE_META_CLASS, PROBE_META_NUM_EXPERTS, PROBE_SCHEMA,
};
pub use sanitize::{
    sanitize_file, sanitize_file_report, write_sanitize_receipt, SanitizeFinding, SanitizeOptions, SanitizeReceipt, SanitizeScan,
    SANITIZE_SCHEMA,
};

// ---------------------------------------------------------------------------
// Go `encoding/json` compatible writer
// ---------------------------------------------------------------------------

pub(crate) mod gojson {
    use crate::builder::go_json_string;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// A JSON tree whose objects keep insertion order (Go struct field order).
    #[derive(Debug, Clone)]
    pub enum J {
        Null,
        Bool(bool),
        Int(i64),
        Float(f64),
        Str(String),
        Arr(Vec<J>),
        Obj(Vec<(String, J)>),
    }

    impl J {
        pub fn s(v: &str) -> J {
            J::Str(v.to_string())
        }
        pub fn strs(v: &[String]) -> J {
            J::Arr(v.iter().map(|x| J::Str(x.clone())).collect())
        }
        pub fn obj(v: Vec<(&str, J)>) -> J {
            J::Obj(v.into_iter().map(|(k, x)| (k.to_string(), x)).collect())
        }
        /// `json.Marshal`.
        pub fn compact(&self) -> String {
            let mut out = String::new();
            write(&mut out, self, false, 0);
            out
        }
        /// `json.MarshalIndent(v, "", "  ")`.
        pub fn pretty(&self) -> String {
            let mut out = String::new();
            write(&mut out, self, true, 0);
            out
        }
    }

    fn pad(out: &mut String, depth: usize) {
        for _ in 0..depth {
            out.push_str("  ");
        }
    }

    fn write(out: &mut String, v: &J, indent: bool, depth: usize) {
        match v {
            J::Null => out.push_str("null"),
            J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            J::Int(i) => out.push_str(&i.to_string()),
            J::Float(f) => out.push_str(&go_float(*f)),
            J::Str(s) => out.push_str(&go_json_string(s)),
            J::Arr(a) => {
                if a.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    if indent {
                        out.push('\n');
                        pad(out, depth + 1);
                    }
                    write(out, x, indent, depth + 1);
                }
                if indent {
                    out.push('\n');
                    pad(out, depth);
                }
                out.push(']');
            }
            J::Obj(o) => {
                if o.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (k, x)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    if indent {
                        out.push('\n');
                        pad(out, depth + 1);
                    }
                    out.push_str(&go_json_string(k));
                    out.push(':');
                    if indent {
                        out.push(' ');
                    }
                    write(out, x, indent, depth + 1);
                }
                if indent {
                    out.push('\n');
                    pad(out, depth);
                }
                out.push('}');
            }
        }
    }

    /// Shortest-round-trip decimal digits and decimal exponent of `|f|`
    /// (`d.ddd` x 10^exp), from Rust's `{:e}`.
    fn digits_exp(f: f64) -> (String, i32) {
        let s = format!("{:e}", f.abs());
        let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
        (m.replace('.', ""), e.parse::<i32>().unwrap_or(0))
    }

    fn fmt_e(neg: bool, digits: &str, exp: i32) -> String {
        let mut s = String::new();
        if neg {
            s.push('-');
        }
        s.push_str(&digits[..1]);
        if digits.len() > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        s.push('e');
        s.push(if exp < 0 { '-' } else { '+' });
        s.push_str(&format!("{:02}", exp.abs()));
        s
    }

    /// `encoding/json` float64 encoding: `f` format, switching to `e` format
    /// below 1e-6 or from 1e21, with `e-07` cleaned to `e-7`.
    pub fn go_float(f: f64) -> String {
        let a = f.abs();
        if a != 0.0 && (a < 1e-6 || a >= 1e21) {
            let (d, e) = digits_exp(f);
            let mut s = fmt_e(f.is_sign_negative(), &d, e);
            let n = s.len();
            let b = s.as_bytes();
            if n >= 4 && b[n - 4] == b'e' && b[n - 3] == b'-' && b[n - 2] == b'0' {
                let last = b[n - 1] as char;
                s.truncate(n - 2);
                s.push(last);
            }
            return s;
        }
        // Rust's Display never uses an exponent and is shortest-round-trip,
        // like strconv 'f' with precision -1.
        format!("{}", f)
    }

    /// `fmt` `%g` (shortest): exponent form when exp < -4 || exp >= 6.
    pub fn go_g(f: f64) -> String {
        if f.is_nan() {
            return "NaN".into();
        }
        if f.is_infinite() {
            return if f > 0.0 { "+Inf".into() } else { "-Inf".into() };
        }
        if f == 0.0 {
            return if f.is_sign_negative() { "-0".into() } else { "0".into() };
        }
        let (d, e) = digits_exp(f);
        if e < -4 || e >= 6 {
            return fmt_e(f < 0.0, &d, e);
        }
        format!("{}", f)
    }

    /// `time.Now().UTC().Format(time.RFC3339Nano)`.
    pub fn rfc3339_nano_now() -> String {
        rfc3339_nano(SystemTime::now())
    }

    pub fn rfc3339_nano(t: SystemTime) -> String {
        let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
        let secs = d.as_secs() as i64;
        let nanos = d.subsec_nanos();
        let days = secs.div_euclid(86400);
        let rem = secs.rem_euclid(86400);
        let (y, m, dd) = civil_from_days(days);
        let mut s = format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", y, m, dd, rem / 3600, (rem % 3600) / 60, rem % 60);
        if nanos > 0 {
            s.push('.');
            s.push_str(format!("{:09}", nanos).trim_end_matches('0'));
        }
        s.push('Z');
        s
    }

    // Howard Hinnant's days-to-civil algorithm.
    fn civil_from_days(z: i64) -> (i64, i64, i64) {
        let z = z + 719468;
        let era = z.div_euclid(146097);
        let doe = z.rem_euclid(146097);
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        (if m <= 2 { y + 1 } else { y }, m, d)
    }
}

// ---------------------------------------------------------------------------
// Go `path.Match` and `path.Clean`
// ---------------------------------------------------------------------------

pub(crate) mod gopath {
    fn decode_rune(s: &[u8]) -> (u32, usize) {
        if s.is_empty() {
            return (0xFFFD, 0);
        }
        let l = match s[0] {
            0x00..=0x7f => 1,
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            _ => return (0xFFFD, 1),
        };
        if s.len() < l {
            return (0xFFFD, 1);
        }
        match std::str::from_utf8(&s[..l]) {
            Ok(t) => (t.chars().next().map(|c| c as u32).unwrap_or(0xFFFD), l),
            Err(_) => (0xFFFD, 1),
        }
    }

    fn scan_chunk(mut pattern: &[u8]) -> (bool, &[u8], &[u8]) {
        let mut star = false;
        while !pattern.is_empty() && pattern[0] == b'*' {
            pattern = &pattern[1..];
            star = true;
        }
        let mut inrange = false;
        let mut i = 0;
        while i < pattern.len() {
            match pattern[i] {
                b'\\' => {
                    if i + 1 < pattern.len() {
                        i += 1;
                    }
                }
                b'[' => inrange = true,
                b']' => inrange = false,
                b'*' => {
                    if !inrange {
                        return (star, &pattern[..i], &pattern[i..]);
                    }
                }
                _ => {}
            }
            i += 1;
        }
        (star, pattern, &pattern[pattern.len()..])
    }

    fn get_esc(mut chunk: &[u8]) -> Result<(u32, &[u8]), ()> {
        if chunk.is_empty() || chunk[0] == b'-' || chunk[0] == b']' {
            return Err(());
        }
        if chunk[0] == b'\\' {
            chunk = &chunk[1..];
            if chunk.is_empty() {
                return Err(());
            }
        }
        let (r, n) = decode_rune(chunk);
        if r == 0xFFFD && n == 1 {
            return Err(());
        }
        let rest = &chunk[n..];
        if rest.is_empty() {
            return Err(());
        }
        Ok((r, rest))
    }

    fn match_chunk<'a>(mut chunk: &[u8], mut s: &'a [u8]) -> Result<(&'a [u8], bool), ()> {
        let mut failed = false;
        while !chunk.is_empty() {
            failed = failed || s.is_empty();
            let mut literal = false;
            match chunk[0] {
                b'[' => {
                    let mut r = 0u32;
                    if !failed {
                        let (rr, n) = decode_rune(s);
                        r = rr;
                        s = &s[n..];
                    }
                    chunk = &chunk[1..];
                    let mut negated = false;
                    if !chunk.is_empty() && chunk[0] == b'^' {
                        negated = true;
                        chunk = &chunk[1..];
                    }
                    let mut matched = false;
                    let mut nrange = 0;
                    loop {
                        if !chunk.is_empty() && chunk[0] == b']' && nrange > 0 {
                            chunk = &chunk[1..];
                            break;
                        }
                        let (lo, c2) = get_esc(chunk)?;
                        chunk = c2;
                        let mut hi = lo;
                        if chunk[0] == b'-' {
                            let (h, c3) = get_esc(&chunk[1..])?;
                            hi = h;
                            chunk = c3;
                        }
                        matched = matched || (lo <= r && r <= hi);
                        nrange += 1;
                    }
                    failed = failed || matched == negated;
                }
                b'?' => {
                    if !failed {
                        failed = s[0] == b'/';
                        let (_, n) = decode_rune(s);
                        s = &s[n..];
                    }
                    chunk = &chunk[1..];
                }
                b'\\' => {
                    chunk = &chunk[1..];
                    if chunk.is_empty() {
                        return Err(());
                    }
                    literal = true;
                }
                _ => literal = true,
            }
            if literal {
                if !failed {
                    failed = chunk[0] != s[0];
                    s = &s[1..];
                }
                chunk = &chunk[1..];
            }
        }
        if failed {
            return Ok((&s[s.len()..], false));
        }
        Ok((s, true))
    }

    /// Go `path.Match`; `Err(())` is `path.ErrBadPattern`.
    pub fn path_match(pattern: &str, name: &str) -> Result<bool, ()> {
        let mut pattern = pattern.as_bytes();
        let mut name = name.as_bytes();
        'pat: while !pattern.is_empty() {
            let (star, chunk, rest) = scan_chunk(pattern);
            pattern = rest;
            if star && chunk.is_empty() {
                return Ok(!name.contains(&b'/'));
            }
            match match_chunk(chunk, name) {
                Ok((t, true)) if t.is_empty() || !pattern.is_empty() => {
                    name = t;
                    continue;
                }
                Err(e) => return Err(e),
                _ => {}
            }
            if star {
                let mut i = 0;
                while i < name.len() && name[i] != b'/' {
                    match match_chunk(chunk, &name[i + 1..]) {
                        Ok((t, true)) => {
                            if pattern.is_empty() && !t.is_empty() {
                                i += 1;
                                continue;
                            }
                            name = t;
                            continue 'pat;
                        }
                        Err(e) => return Err(e),
                        _ => {}
                    }
                    i += 1;
                }
            }
            while !pattern.is_empty() {
                let (_, chunk, rest) = scan_chunk(pattern);
                pattern = rest;
                match_chunk(chunk, b"")?;
            }
            return Ok(false);
        }
        Ok(name.is_empty())
    }

    /// Lexical `filepath.Clean` (slash-separated).
    pub fn clean(path: &str) -> String {
        if path.is_empty() {
            return ".".into();
        }
        let rooted = path.starts_with('/');
        let mut out: Vec<&str> = Vec::new();
        for part in path.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    if let Some(last) = out.last() {
                        if *last != ".." {
                            out.pop();
                            continue;
                        }
                    }
                    if !rooted {
                        out.push("..");
                    }
                }
                p => out.push(p),
            }
        }
        let body = out.join("/");
        if rooted {
            format!("/{}", body)
        } else if body.is_empty() {
            ".".into()
        } else {
            body
        }
    }

    /// `filepath.Join(a, b)` for two elements.
    pub fn join(a: &str, b: &str) -> String {
        if a.is_empty() {
            return clean(b);
        }
        clean(&format!("{}/{}", a, b))
    }

    /// `filepath.Dir`.
    pub fn dir(path: &str) -> String {
        match path.rfind('/') {
            None => ".".into(),
            Some(i) => {
                let d = clean(&path[..=i]);
                d
            }
        }
    }

    /// `filepath.Abs`.
    pub fn abs(path: &str) -> std::io::Result<String> {
        if path.starts_with('/') {
            return Ok(clean(path));
        }
        let cwd = std::env::current_dir()?;
        Ok(clean(&format!("{}/{}", cwd.to_string_lossy(), path)))
    }
}

// ---------------------------------------------------------------------------
// Shared file helpers
// ---------------------------------------------------------------------------

pub(crate) mod fsx {
    use super::gopath;
    use crate::error::{Error, Result};
    use sha2::{Digest, Sha256};
    use std::fs::{File, Metadata, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};

    /// Render an I/O error the way Go's `syscall.Errno` renders it (lower-case
    /// strerror text, no `(os error N)` suffix).
    pub fn io_msg(e: &std::io::Error) -> String {
        let s = e.to_string();
        let s = match s.rfind(" (os error ") {
            Some(i) => s[..i].to_string(),
            None => s,
        };
        let mut c = s.chars();
        match c.next() {
            Some(f) => f.to_lowercase().collect::<String>() + c.as_str(),
            None => s,
        }
    }

    pub fn open_regular(path: &str) -> Result<File> {
        // The oracle wraps os.Open's own PathError a second time ("open P: open P: ...").
        let f = File::open(path).map_err(|e| Error::msg(format!("open {}: open {}: {}", path, path, io_msg(&e))))?;
        let md = f.metadata().map_err(|e| Error::msg(format!("stat {}: stat {}: {}", path, path, io_msg(&e))))?;
        if md.is_dir() {
            return Err(Error::msg(format!("{} is a directory", path)));
        }
        Ok(f)
    }

    #[cfg(unix)]
    pub fn read_exact_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        f.read_exact_at(buf, off)
    }
    #[cfg(unix)]
    pub fn write_all_at(f: &File, buf: &[u8], off: u64) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        f.write_all_at(buf, off)
    }
    #[cfg(windows)]
    pub fn read_exact_at(f: &File, mut buf: &mut [u8], mut off: u64) -> std::io::Result<()> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            let n = f.seek_read(buf, off)?;
            if n == 0 {
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
            }
            buf = &mut buf[n..];
            off += n as u64;
        }
        Ok(())
    }
    #[cfg(windows)]
    pub fn write_all_at(f: &File, mut buf: &[u8], mut off: u64) -> std::io::Result<()> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            let n = f.seek_write(buf, off)?;
            buf = &buf[n..];
            off += n as u64;
        }
        Ok(())
    }

    pub fn read_at(f: &File, buf: &mut [u8], off: i64) -> Result<()> {
        if off < 0 {
            return Err(Error::msg("negative file offset"));
        }
        read_exact_at(f, buf, off as u64).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                Error::msg("unexpected EOF")
            } else {
                Error::msg(io_msg(&e))
            }
        })
    }

    pub fn write_at(f: &File, buf: &[u8], off: i64) -> Result<()> {
        write_all_at(f, buf, off as u64).map_err(|e| Error::msg(io_msg(&e)))
    }

    /// SHA-256 of the whole open file (size taken from a fresh stat).
    pub fn hash_file(f: &File) -> Result<String> {
        let size = f.metadata().map_err(|e| Error::msg(io_msg(&e)))?.len();
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut off = 0u64;
        while off < size {
            let n = std::cmp::min(buf.len() as u64, size - off) as usize;
            read_exact_at(f, &mut buf[..n], off).map_err(|e| Error::msg(io_msg(&e)))?;
            h.update(&buf[..n]);
            off += n as u64;
        }
        Ok(crate::container::hex(&h.finalize()))
    }

    /// `os.SameFile`.
    pub fn same_file(a: &Metadata, b: &Metadata) -> bool {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            a.dev() == b.dev() && a.ino() == b.ino()
        }
        #[cfg(not(unix))]
        {
            a.len() == b.len() && a.modified().ok() == b.modified().ok()
        }
    }

    /// `os.Lstat` says the path does not exist (ENOENT only).
    pub fn lstat_not_exist(p: &str) -> bool {
        matches!(std::fs::symlink_metadata(p), Err(e) if e.kind() == std::io::ErrorKind::NotFound)
    }

    /// Removes the file when dropped.
    pub struct TempGuard(pub PathBuf);
    impl Drop for TempGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// `os.CreateTemp(dir, prefix+"*")`: a new 0600 read/write file.
    pub fn create_temp(dir: &str, prefix: &str) -> Result<(File, TempGuard)> {
        let mut last = None;
        for attempt in 0..64u32 {
            let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
            let name = format!("{}{}{}{}", prefix, std::process::id(), nanos, attempt);
            let p = Path::new(dir).join(name);
            let mut oo = OpenOptions::new();
            oo.read(true).write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                oo.mode(0o600);
            }
            match oo.open(&p) {
                Ok(f) => return Ok((f, TempGuard(p))),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some(e),
                Err(e) => return Err(Error::msg(format!("open {}: {}", p.display(), io_msg(&e)))),
            }
        }
        Err(Error::msg(format!("create temp in {}: {}", dir, last.map(|e| io_msg(&e)).unwrap_or_default())))
    }

    /// `writeJSONNew` / `WriteSanitizeReceipt`: temp file (0600) in the target
    /// directory, fsync, atomic hard link that refuses to replace anything.
    pub fn publish_new(dst: &str, data: &[u8]) -> Result<()> {
        let (mut f, guard) = create_temp(&gopath::dir(dst), ".wsp-json-")?;
        f.write_all(data).map_err(|e| Error::msg(io_msg(&e)))?;
        f.sync_all().map_err(|e| Error::msg(io_msg(&e)))?;
        drop(f);
        std::fs::hard_link(&guard.0, dst).map_err(|e| Error::msg(format!("link {} {}: {}", guard.0.display(), dst, io_msg(&e))))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// sanitize
// ---------------------------------------------------------------------------

pub mod sanitize {
    use super::fsx::{self, hash_file, io_msg, read_at, same_file, write_at};
    use super::gojson::{rfc3339_nano_now, J};
    use super::gopath;
    use crate::container::{scan_gguf_layout, scan_safetensors_layout, sha256_hex, Layout, TensorLocation};
    use crate::dtype::dtype_size;
    use crate::error::{Error, Result};
    use serde::{Deserialize, Serialize};
    use std::collections::{BTreeMap, HashSet};
    use std::fs::File;

    pub const SANITIZE_SCHEMA: &str = "secant.wsp.sanitize.v1";
    const MAX_HEADER: i64 = 128 << 20;

    fn ser_vec_null<S: serde::Serializer>(v: &Vec<String>, s: S) -> std::result::Result<S::Ok, S::Error> {
        if v.is_empty() {
            s.serialize_none()
        } else {
            v.serialize(s)
        }
    }
    pub(crate) fn de_vec_null<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Vec<String>, D::Error> {
        Ok(Option::<Vec<String>>::deserialize(d)?.unwrap_or_default())
    }

    /// Serialized into every receipt. Zero numerical limits take the documented
    /// defaults. Empty `routing_tensor` / `norm_tensor` serialize as JSON
    /// `null`, as a nil Go slice does.
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct SanitizeOptions {
        #[serde(default)]
        pub attempt_id: String,
        #[serde(default)]
        pub num_experts: i64,
        #[serde(default, serialize_with = "ser_vec_null", deserialize_with = "de_vec_null")]
        pub routing_tensor: Vec<String>,
        #[serde(default, serialize_with = "ser_vec_null", deserialize_with = "de_vec_null")]
        pub norm_tensor: Vec<String>,
        #[serde(default)]
        pub max_abs: f64,
        #[serde(default)]
        pub min_norm: f64,
        #[serde(default)]
        pub max_examples: i64,
        #[serde(default)]
        pub clamp_nonfinite: bool,
        #[serde(default)]
        pub dedup_routing_rows: bool,
        /// Secant extension (the oracle has no such option; opt-in, omitted from
        /// the receipt unless true): finite floats with |x| > `max_abs` become
        /// sign(x) * the largest dtype value not above `max_abs`.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        pub clamp_extreme: bool,
        /// Secant extension (opt-in, omitted from the receipt unless true):
        /// subnormal floats of the tensor's dtype become the same-signed zero.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        pub flush_subnormal: bool,
        #[serde(default)]
        pub output: String,
    }

    impl SanitizeOptions {
        pub fn to_j(&self) -> J {
            let list = |v: &Vec<String>| if v.is_empty() { J::Null } else { J::strs(v) };
            let mut v = vec![
                ("attempt_id", J::s(&self.attempt_id)),
                ("num_experts", J::Int(self.num_experts)),
                ("routing_tensor", list(&self.routing_tensor)),
                ("norm_tensor", list(&self.norm_tensor)),
                ("max_abs", J::Float(self.max_abs)),
                ("min_norm", J::Float(self.min_norm)),
                ("max_examples", J::Int(self.max_examples)),
                ("clamp_nonfinite", J::Bool(self.clamp_nonfinite)),
                ("dedup_routing_rows", J::Bool(self.dedup_routing_rows)),
            ];
            // omitempty-style: default runs serialize exactly as the oracle's.
            if self.clamp_extreme {
                v.push(("clamp_extreme", J::Bool(true)));
            }
            if self.flush_subnormal {
                v.push(("flush_subnormal", J::Bool(true)));
            }
            v.push(("output", J::s(&self.output)));
            J::obj(v)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct SanitizeFinding {
        pub kind: String,
        pub tensor: String,
        pub dtype: String,
        pub index: i64,
        pub row: i64,
        pub file_offset: i64,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        pub bits: String,
    }

    impl SanitizeFinding {
        fn to_j(&self) -> J {
            let mut v = vec![
                ("kind", J::s(&self.kind)),
                ("tensor", J::s(&self.tensor)),
                ("dtype", J::s(&self.dtype)),
                ("index", J::Int(self.index)),
                ("row", J::Int(self.row)),
                ("file_offset", J::Int(self.file_offset)),
            ];
            if !self.bits.is_empty() {
                v.push(("bits", J::s(&self.bits)));
            }
            J::obj(v)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct SanitizeScan {
        pub tensors: i64,
        pub float_elements: i64,
        pub routing_elements: i64,
        pub routing_rows: i64,
        pub norm_rows: i64,
        pub ignored_integer_tensors: i64,
        pub total_findings: i64,
        pub counts: BTreeMap<String, i64>,
        pub examples: Vec<SanitizeFinding>,
        pub examples_truncated: bool,
    }

    fn counts_j(m: &BTreeMap<String, i64>) -> J {
        J::Obj(m.iter().map(|(k, v)| (k.clone(), J::Int(*v))).collect())
    }

    impl SanitizeScan {
        pub fn to_j(&self) -> J {
            J::obj(vec![
                ("tensors", J::Int(self.tensors)),
                ("float_elements", J::Int(self.float_elements)),
                ("routing_elements", J::Int(self.routing_elements)),
                ("routing_rows", J::Int(self.routing_rows)),
                ("norm_rows", J::Int(self.norm_rows)),
                ("ignored_integer_tensors", J::Int(self.ignored_integer_tensors)),
                ("total_findings", J::Int(self.total_findings)),
                ("counts", counts_j(&self.counts)),
                ("examples", J::Arr(self.examples.iter().map(|e| e.to_j()).collect())),
                ("examples_truncated", J::Bool(self.examples_truncated)),
            ])
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct SanitizeReceipt {
        pub schema: String,
        pub attempt_id: String,
        pub policy_version: String,
        /// clean, remediated, findings, unknown
        pub status: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        pub error: String,
        pub started_utc: String,
        pub finished_utc: String,
        pub input: String,
        pub format: String,
        pub bytes: i64,
        pub pre_sha256: String,
        pub post_sha256: String,
        pub input_unchanged: bool,
        pub options: Option<SanitizeOptions>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub before: Option<SanitizeScan>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub after: Option<SanitizeScan>,
        pub changes: BTreeMap<String, i64>,
        pub output_published: bool,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        pub output_sha256: String,
        pub receipt_sha256: String,
    }

    impl SanitizeReceipt {
        /// The receipt as the oracle's `encoding/json` struct tree.
        pub fn to_j(&self) -> J {
            let mut v = vec![
                ("schema", J::s(&self.schema)),
                ("attempt_id", J::s(&self.attempt_id)),
                ("policy_version", J::s(&self.policy_version)),
                ("status", J::s(&self.status)),
            ];
            if !self.error.is_empty() {
                v.push(("error", J::s(&self.error)));
            }
            v.push(("started_utc", J::s(&self.started_utc)));
            v.push(("finished_utc", J::s(&self.finished_utc)));
            v.push(("input", J::s(&self.input)));
            v.push(("format", J::s(&self.format)));
            v.push(("bytes", J::Int(self.bytes)));
            v.push(("pre_sha256", J::s(&self.pre_sha256)));
            v.push(("post_sha256", J::s(&self.post_sha256)));
            v.push(("input_unchanged", J::Bool(self.input_unchanged)));
            v.push(("options", self.options.as_ref().map(|o| o.to_j()).unwrap_or(J::Null)));
            if let Some(b) = &self.before {
                v.push(("before", b.to_j()));
            }
            if let Some(a) = &self.after {
                v.push(("after", a.to_j()));
            }
            v.push(("changes", counts_j(&self.changes)));
            v.push(("output_published", J::Bool(self.output_published)));
            if !self.output_sha256.is_empty() {
                v.push(("output_sha256", J::s(&self.output_sha256)));
            }
            v.push(("receipt_sha256", J::s(&self.receipt_sha256)));
            J::obj(v)
        }
        /// `json.Marshal(rep)` (what `--json` prints, minus the newline).
        pub fn to_json_compact(&self) -> String {
            self.to_j().compact()
        }
        /// `json.MarshalIndent(rep, "", "  ")` (the receipt file, minus the newline).
        pub fn to_json_pretty(&self) -> String {
            self.to_j().pretty()
        }
        /// Recompute `receipt_sha256`: SHA-256 of the compact serialization with
        /// that field empty. A content digest, not a signature.
        pub fn compute_seal(&self) -> String {
            let mut c = self.clone();
            c.receipt_sha256.clear();
            sha256_hex(c.to_json_compact().as_bytes())
        }
    }

    /// Apply defaults and validate (oracle `sanitizeOptions`).
    pub fn normalize_options(o: &SanitizeOptions) -> Result<SanitizeOptions> {
        let mut o = o.clone();
        if o.max_abs == 0.0 {
            o.max_abs = 1e4;
        }
        if o.min_norm == 0.0 {
            o.min_norm = 1e-12;
        }
        if o.max_examples == 0 {
            o.max_examples = 128;
        }
        if o.attempt_id.is_empty() {
            return Err(Error::msg("attempt_id is required"));
        }
        if o.num_experts < 0
            || o.max_examples < 1
            || o.max_examples > 10000
            || o.max_abs <= 0.0
            || o.max_abs.is_nan()
            || o.max_abs.is_infinite()
            || o.min_norm <= 0.0
            || o.min_norm.is_nan()
            || o.min_norm.is_infinite()
        {
            return Err(Error::msg("invalid sanitizer policy limits"));
        }
        if (o.clamp_nonfinite || o.dedup_routing_rows || o.clamp_extreme || o.flush_subnormal) != (!o.output.is_empty()) {
            return Err(Error::msg("remediation flags and a separate --out must be supplied together"));
        }
        for p in o.routing_tensor.iter().chain(o.norm_tensor.iter()) {
            if p.is_empty() {
                return Err(Error::msg("empty tensor glob"));
            }
            if gopath::path_match(p, "").is_err() {
                return Err(Error::msg(format!("tensor glob {:?}: syntax error in pattern", p)));
            }
        }
        Ok(o)
    }

    /// Scan one input file; on success the receipt is returned, on any
    /// failure the error (UNKNOWN, never clean). Use [`sanitize_file_report`]
    /// to keep the receipt of a failed run.
    pub fn sanitize_file(input: &str, opts: &SanitizeOptions) -> Result<SanitizeReceipt> {
        let (rep, r) = sanitize_file_report(input, opts);
        r.map(|_| rep)
    }

    /// Oracle `SanitizeFile`: a receipt is ALWAYS returned, alongside the error
    /// (if any). `Status == "findings"` is NOT an error here; the CLI turns it
    /// into exit 1.
    pub fn sanitize_file_report(input: &str, opts: &SanitizeOptions) -> (SanitizeReceipt, Result<()>) {
        let mut rep = SanitizeReceipt {
            schema: SANITIZE_SCHEMA.into(),
            attempt_id: opts.attempt_id.clone(),
            policy_version: "unquantized-static-values-v1".into(),
            status: "unknown".into(),
            started_utc: rfc3339_nano_now(),
            input: input.into(),
            ..Default::default()
        };
        let r = run(&mut rep, input, opts);
        if let Err(e) = &r {
            rep.status = "unknown".into();
            rep.error = e.message().to_string();
        }
        rep.finished_utc = rfc3339_nano_now();
        rep.receipt_sha256 = String::new();
        rep.receipt_sha256 = rep.compute_seal();
        (rep, r)
    }

    fn run(rep: &mut SanitizeReceipt, input: &str, opts: &SanitizeOptions) -> Result<()> {
        let o = normalize_options(opts)?;
        rep.options = Some(o.clone());
        let f = fsx::open_regular(input)?;
        let st = f.metadata().map_err(|e| Error::msg(io_msg(&e)))?;
        rep.bytes = st.len() as i64;
        rep.pre_sha256 = hash_file(&f)?;
        // From here the post hash runs on every path (oracle's deferred hash).
        let mut res = body(rep, &f, &st, input, &o);
        let post = hash_file(&f);
        let current = std::fs::metadata(input);
        rep.post_sha256 = post.as_ref().cloned().unwrap_or_default();
        rep.input_unchanged =
            post.is_ok() && matches!(&current, Ok(c) if same_file(&st, c) && c.len() == st.len()) && rep.pre_sha256 == rep.post_sha256;
        if !rep.input_unchanged && res.is_ok() {
            res = Err(Error::msg("input changed during sanitization"));
        }
        res
    }

    fn body(rep: &mut SanitizeReceipt, f: &File, st: &std::fs::Metadata, input: &str, o: &SanitizeOptions) -> Result<()> {
        let size = st.len() as i64;
        let layout = sanitize_layout(f, size)?;
        rep.format = layout.format.clone();
        sanitize_selectors(&layout, o)?;
        let before = sanitize_scan(f, &layout, o, None, None)?;
        rep.status = if before.total_findings == 0 { "clean".into() } else { "findings".into() };
        rep.before = Some(before);
        if o.output.is_empty() {
            return Ok(());
        }
        if !fsx::lstat_not_exist(&o.output) {
            return Err(Error::msg(format!("output must not exist: {}", o.output)));
        }
        let (tmp, guard) = fsx::create_temp(&gopath::dir(&o.output), ".wsp-sanitize-")?;
        copy_prefix(f, &tmp, size)?;
        // Rewriting reads the original, so changes cannot affect classification.
        let mut changes = BTreeMap::new();
        sanitize_scan(f, &layout, o, Some(&tmp), Some(&mut changes))?;
        rep.changes = changes;
        let after = sanitize_scan(&tmp, &layout, o, None, None)?;
        let residual = after.total_findings != 0;
        rep.after = Some(after);
        if residual {
            rep.status = "findings".into();
            return Ok(());
        }
        tmp.sync_all().map_err(|e| Error::msg(io_msg(&e)))?;
        rep.output_sha256 = hash_file(&tmp)?;
        let post = hash_file(f)?;
        let current = std::fs::metadata(input).map_err(|e| Error::msg(io_msg(&e)))?;
        if post != rep.pre_sha256 || !same_file(st, &current) {
            return Err(Error::msg("input changed before publication"));
        }
        // link is atomic and refuses to replace an existing file, including symlinks.
        std::fs::hard_link(&guard.0, &o.output)
            .map_err(|e| Error::msg(format!("link {} {}: {}", guard.0.display(), o.output, io_msg(&e))))?;
        rep.output_published = true;
        rep.status = "remediated".into();
        Ok(())
    }

    fn copy_prefix(src: &File, dst: &File, size: i64) -> Result<()> {
        let mut buf = vec![0u8; 1 << 20];
        let mut off = 0i64;
        while off < size {
            let n = std::cmp::min(buf.len() as i64, size - off) as usize;
            read_at(src, &mut buf[..n], off)?;
            write_at(dst, &buf[..n], off)?;
            off += n as i64;
        }
        Ok(())
    }

    /// Write the receipt (indented JSON + newline, mode 0600) at a NEW path
    /// without clobbering anything.
    pub fn write_sanitize_receipt(dst: &str, rep: &SanitizeReceipt) -> Result<()> {
        if rep.attempt_id.is_empty() {
            return Err(Error::msg("receipt requires attempt_id"));
        }
        let mut b = rep.to_json_pretty();
        b.push('\n');
        fsx::publish_new(dst, b.as_bytes())
    }

    // ----- strict layout -----

    fn read_prefix(f: &File, n: i64) -> Result<Vec<u8>> {
        let mut b = vec![0u8; n as usize];
        read_at(f, &mut b, 0)?;
        Ok(b)
    }

    fn guarded<T>(what: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
            Ok(r) => r,
            Err(_) => Err(Error::msg(format!("{}: header arithmetic overflow", what))),
        }
    }

    /// Re-word the shared scanner's errors the way the oracle's scanner words
    /// them, so a receipt's `error` text agrees. `avail` is the number of bytes
    /// the scanner was given (the oracle's `io.ReadFull` reports a bare `EOF`
    /// when it hit the end having read nothing, `unexpected EOF` otherwise).
    fn go_scan_error(e: Error, avail: usize) -> Error {
        let m = e.message().to_string();
        if let Some(i) = m.find(" at offset ") {
            if let Some(rest) = m[i + 11..].strip_suffix(": unexpected EOF") {
                if rest.parse::<usize>().ok() == Some(avail) {
                    return Error::msg(format!("{}: EOF", &m[..i + 11 + rest.len()]));
                }
            }
        }
        if let Some(i) = m.find("unknown ggml type: ") {
            if let Some(name_end) = m[..i].strip_suffix(": ") {
                let id = &m[i + 19..];
                return Error::msg(format!(
                    "{}: unsupported ggml type {} (quantized): weight surgery is implemented for F32/F16/BF16/F64/I32/I64 payloads only",
                    name_end, id
                ));
            }
        }
        // the oracle's scanner accepts a negative dimension (sanitizeLayout rejects it later)
        if let Some(i) = m.find(" has a negative dimension in ") {
            if let Some(q) = m.find("tensor ") {
                return Error::msg(format!("{} shape overflow", &m[q..i]));
            }
        }
        // Go prints []int as [1 2], Rust as [1, 2]
        if m.starts_with("safetensors: tensor") {
            if let Some(a) = m.find(" but shape [") {
                if let Some(b) = m[a..].find(']') {
                    let fixed = m[a..a + b].replace(", ", " ");
                    return Error::msg(format!("{}{}{}", &m[..a], fixed, &m[a + b..]));
                }
            }
        }
        e
    }

    /// Validate the layout again before trusting it with value reads: arithmetic
    /// overflow, aliasing, duplicate names, payload extents, SafeTensors holes.
    /// Quantized tensors are rejected here with the oracle's message (the Go
    /// scanner fails the whole file; Secant's scanner lists them).
    pub fn sanitize_layout(f: &File, size: i64) -> Result<Layout> {
        let mut head = [0u8; 8];
        if size < 8 {
            return Err(Error::msg("EOF"));
        }
        read_at(f, &mut head, 0)?;
        let mut l: Layout;
        if &head[..4] == b"GGUF" {
            let v = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);
            if v != 2 && v != 3 {
                return Err(Error::msg("sanitizer supports little-endian GGUF v2/v3 only"));
            }
            let limit = std::cmp::min(size, MAX_HEADER);
            let mut want = std::cmp::min(limit, 1 << 20);
            l = loop {
                let buf = read_prefix(f, want)?;
                match guarded("gguf", || scan_gguf_layout(&buf)).map_err(|e| go_scan_error(e, buf.len())) {
                    Ok(l) => break l,
                    Err(e) if want < limit && e.message().contains("unexpected EOF") => {
                        want = std::cmp::min(limit, want * 4);
                    }
                    Err(e) => return Err(e),
                }
            };
            for t in &l.tensors {
                if t.is_quantized() {
                    return Err(Error::msg(format!(
                        "gguf: tensor {:?}: unsupported ggml type {} (quantized): weight surgery is implemented for F32/F16/BF16/F64/I32/I64 payloads only",
                        t.name,
                        t.ggml_type.map(|x| x.to_string()).unwrap_or_default()
                    )));
                }
            }
        } else {
            let n = u64::from_le_bytes(head);
            if n < 2 || n > (MAX_HEADER - 8) as u64 || n > (size - 8) as u64 {
                return Err(Error::msg("invalid SafeTensors header length"));
            }
            let mut header = vec![0u8; n as usize];
            read_at(f, &mut header, 8)?;
            check_json_header(&header)?;
            let buf = read_prefix(f, 8 + n as i64)?;
            l = guarded("safetensors", || scan_safetensors_layout(&buf)).map_err(|e| go_scan_error(e, buf.len()))?;
        }
        if l.data_offset < 8 || l.data_offset > size || l.tensors.is_empty() {
            return Err(Error::msg("empty or invalid tensor layout"));
        }
        let mut names: HashSet<&str> = HashSet::new();
        for t in &l.tensors {
            if !names.insert(t.name.as_str()) {
                return Err(Error::msg(format!("duplicate tensor name {:?}", t.name)));
            }
            let mut n: i64 = 1;
            for &d in &t.shape {
                let d = d as i64;
                if d < 0 || (d != 0 && n > i64::MAX / d) {
                    return Err(Error::msg(format!("tensor {:?} shape overflow", t.name)));
                }
                n *= d;
            }
            let w = dtype_size(&t.dtype);
            let bad = w == 0
                || n > i64::MAX / w
                || n * w != t.size
                || t.size < 0
                || t.offset < l.data_offset
                || t.offset > size
                || t.size > size - t.offset;
            if bad {
                return Err(Error::msg(format!("tensor {:?} has invalid extent or unsupported dtype", t.name)));
            }
        }
        let mut ranges: Vec<&TensorLocation> = l.tensors.iter().collect();
        ranges.sort_by(|a, b| a.offset.cmp(&b.offset).then(a.size.cmp(&b.size)));
        let mut end = l.data_offset;
        for t in ranges {
            if t.offset < end || (l.format == "safetensors" && t.offset != end) {
                return Err(Error::msg(format!("tensor {:?} overlaps or leaves a SafeTensors hole", t.name)));
            }
            end = t.offset + t.size;
        }
        if l.format == "safetensors" && end != size {
            return Err(Error::msg("unindexed SafeTensors payload bytes"));
        }
        l.file_size = size;
        Ok(l)
    }

    // ----- strict JSON header (duplicate keys, depth, required fields) -----

    enum Jv {
        Null,
        Other,
        Str,
        Arr,
        Obj(Vec<(String, Jv)>),
    }

    struct P<'a> {
        b: &'a [u8],
        i: usize,
    }

    impl<'a> P<'a> {
        fn ws(&mut self) {
            while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
                self.i += 1;
            }
        }
        fn err<T>(&self, what: &str) -> Result<T> {
            if self.i >= self.b.len() {
                return Err(Error::msg("unexpected end of JSON input"));
            }
            Err(Error::msg(format!("invalid JSON header at byte {}: {}", self.i, what)))
        }
        fn value(&mut self, depth: usize) -> Result<Jv> {
            if depth > 16 {
                return Err(Error::msg("header nesting exceeds limit"));
            }
            self.ws();
            if self.i >= self.b.len() {
                return Err(Error::msg("unexpected end of JSON input"));
            }
            match self.b[self.i] {
                b'{' => {
                    self.i += 1;
                    let mut items: Vec<(String, Jv)> = Vec::new();
                    let mut seen: HashSet<String> = HashSet::new();
                    self.ws();
                    if self.peek() == Some(b'}') {
                        self.i += 1;
                        return Ok(Jv::Obj(items));
                    }
                    loop {
                        self.ws();
                        if self.peek() != Some(b'"') {
                            return self.err("object key must be a string");
                        }
                        let k = self.string()?;
                        if !seen.insert(k.clone()) {
                            return Err(Error::msg(format!("duplicate or invalid JSON key {}", k)));
                        }
                        self.ws();
                        if self.peek() != Some(b':') {
                            return self.err("expected ':'");
                        }
                        self.i += 1;
                        let v = self.value(depth + 1)?;
                        items.push((k, v));
                        self.ws();
                        match self.peek() {
                            Some(b',') => self.i += 1,
                            Some(b'}') => {
                                self.i += 1;
                                return Ok(Jv::Obj(items));
                            }
                            _ => return self.err("expected ',' or '}'"),
                        }
                    }
                }
                b'[' => {
                    self.i += 1;
                    self.ws();
                    if self.peek() == Some(b']') {
                        self.i += 1;
                        return Ok(Jv::Arr);
                    }
                    loop {
                        self.value(depth + 1)?;
                        self.ws();
                        match self.peek() {
                            Some(b',') => self.i += 1,
                            Some(b']') => {
                                self.i += 1;
                                return Ok(Jv::Arr);
                            }
                            _ => return self.err("expected ',' or ']'"),
                        }
                    }
                }
                b'"' => {
                    self.string()?;
                    Ok(Jv::Str)
                }
                b't' => self.lit("true", Jv::Other),
                b'f' => self.lit("false", Jv::Other),
                b'n' => self.lit("null", Jv::Null),
                b'-' | b'0'..=b'9' => self.number(),
                _ => self.err("unexpected character"),
            }
        }
        fn peek(&self) -> Option<u8> {
            self.b.get(self.i).copied()
        }
        fn lit(&mut self, s: &str, v: Jv) -> Result<Jv> {
            if self.b[self.i..].starts_with(s.as_bytes()) {
                self.i += s.len();
                Ok(v)
            } else {
                self.err("bad literal")
            }
        }
        fn number(&mut self) -> Result<Jv> {
            if self.peek() == Some(b'-') {
                self.i += 1;
            }
            match self.peek() {
                Some(b'0') => self.i += 1,
                Some(b'1'..=b'9') => {
                    while matches!(self.peek(), Some(b'0'..=b'9')) {
                        self.i += 1;
                    }
                }
                _ => return self.err("bad number"),
            }
            if self.peek() == Some(b'.') {
                self.i += 1;
                if !matches!(self.peek(), Some(b'0'..=b'9')) {
                    return self.err("bad number");
                }
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.i += 1;
                }
            }
            if matches!(self.peek(), Some(b'e' | b'E')) {
                self.i += 1;
                if matches!(self.peek(), Some(b'+' | b'-')) {
                    self.i += 1;
                }
                if !matches!(self.peek(), Some(b'0'..=b'9')) {
                    return self.err("bad number");
                }
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.i += 1;
                }
            }
            Ok(Jv::Other)
        }
        fn hex4(&mut self) -> Result<u32> {
            if self.i + 4 > self.b.len() {
                return self.err("bad \\u escape");
            }
            let s = std::str::from_utf8(&self.b[self.i..self.i + 4]).map_err(|_| Error::msg("bad \\u escape"))?;
            let v = u32::from_str_radix(s, 16).map_err(|_| Error::msg("bad \\u escape"))?;
            self.i += 4;
            Ok(v)
        }
        fn string(&mut self) -> Result<String> {
            self.i += 1; // opening quote
            let mut out: Vec<u8> = Vec::new();
            loop {
                let c = match self.peek() {
                    Some(c) => c,
                    None => return Err(Error::msg("unexpected end of JSON input")),
                };
                self.i += 1;
                match c {
                    b'"' => break,
                    b'\\' => {
                        let e = match self.peek() {
                            Some(e) => e,
                            None => return Err(Error::msg("unexpected end of JSON input")),
                        };
                        self.i += 1;
                        match e {
                            b'"' => out.push(b'"'),
                            b'\\' => out.push(b'\\'),
                            b'/' => out.push(b'/'),
                            b'b' => out.push(8),
                            b'f' => out.push(12),
                            b'n' => out.push(b'\n'),
                            b'r' => out.push(b'\r'),
                            b't' => out.push(b'\t'),
                            b'u' => {
                                let mut cp = self.hex4()?;
                                if (0xD800..0xDC00).contains(&cp) {
                                    if self.b[self.i..].starts_with(b"\\u") {
                                        let save = self.i;
                                        self.i += 2;
                                        let lo = self.hex4()?;
                                        if (0xDC00..0xE000).contains(&lo) {
                                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                        } else {
                                            self.i = save;
                                            cp = 0xFFFD;
                                        }
                                    } else {
                                        cp = 0xFFFD;
                                    }
                                } else if (0xDC00..0xE000).contains(&cp) {
                                    cp = 0xFFFD;
                                }
                                let ch = char::from_u32(cp).unwrap_or('\u{FFFD}');
                                let mut tmp = [0u8; 4];
                                out.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                            }
                            _ => return self.err("bad escape"),
                        }
                    }
                    c if c < 0x20 => return self.err("control character in string"),
                    c => out.push(c),
                }
            }
            Ok(String::from_utf8_lossy(&out).into_owned())
        }
    }

    /// Oracle `sanitizeJSONHeader`.
    fn check_json_header(b: &[u8]) -> Result<()> {
        if b.is_empty() || b[0] != b'{' {
            return Err(Error::msg("SafeTensors header must start with an object"));
        }
        let mut p = P { b, i: 0 };
        let root = p.value(0)?;
        p.ws();
        if p.i != b.len() {
            return Err(Error::msg("trailing header JSON"));
        }
        let items = match root {
            Jv::Obj(items) => items,
            _ => return Err(Error::msg("SafeTensors header must start with an object")),
        };
        for (name, v) in items {
            if name == "__metadata__" {
                match v {
                    Jv::Null => {}
                    Jv::Obj(m) => {
                        for (_, x) in m {
                            if !matches!(x, Jv::Str) {
                                return Err(Error::msg("json: cannot unmarshal into Go value of type string"));
                            }
                        }
                    }
                    _ => return Err(Error::msg("json: cannot unmarshal into Go value of type map[string]string")),
                }
                continue;
            }
            let fields = match v {
                Jv::Null => Vec::new(),
                Jv::Obj(f) => f,
                _ => return Err(Error::msg("json: cannot unmarshal into Go value of type map[string]json.RawMessage")),
            };
            for key in ["dtype", "shape", "data_offsets"] {
                match fields.iter().find(|(k, _)| k == key) {
                    None | Some((_, Jv::Null)) => return Err(Error::msg(format!("tensor {:?} missing {}", name, key))),
                    _ => {}
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn options_defaults_and_validation() {
            let o = normalize_options(&SanitizeOptions { attempt_id: "a".into(), ..Default::default() }).unwrap();
            assert_eq!((o.max_abs, o.min_norm, o.max_examples), (1e4, 1e-12, 128));
            let base = SanitizeOptions { attempt_id: "a".into(), ..Default::default() };
            let bad = [
                SanitizeOptions { attempt_id: String::new(), ..base.clone() },
                SanitizeOptions { max_abs: -1.0, ..base.clone() },
                SanitizeOptions { max_abs: f64::NAN, ..base.clone() },
                SanitizeOptions { max_abs: f64::INFINITY, ..base.clone() },
                SanitizeOptions { min_norm: -1.0, ..base.clone() },
                SanitizeOptions { max_examples: -1, ..base.clone() },
                SanitizeOptions { max_examples: 10001, ..base.clone() },
                SanitizeOptions { num_experts: -1, ..base.clone() },
                SanitizeOptions { clamp_nonfinite: true, ..base.clone() },
                SanitizeOptions { output: "x".into(), ..base.clone() },
                SanitizeOptions { routing_tensor: vec!["[".into()], ..base.clone() },
                SanitizeOptions { norm_tensor: vec![String::new()], ..base.clone() },
            ];
            for b in bad {
                assert!(normalize_options(&b).is_err(), "{:?}", b);
            }
            assert!(normalize_options(&SanitizeOptions { dedup_routing_rows: true, output: "x".into(), ..base.clone() }).is_ok());
            // the opt-in repairs obey the same "flag and --out together" rule
            for flags in [(true, false), (false, true), (true, true)] {
                let f = SanitizeOptions { clamp_extreme: flags.0, flush_subnormal: flags.1, ..base.clone() };
                assert!(normalize_options(&f).is_err());
                assert!(normalize_options(&SanitizeOptions { output: "x".into(), ..f }).is_ok());
            }
            // default runs serialize without the new keys; used flags appear in order before "output"
            let d = base.to_j().compact();
            assert!(!d.contains("clamp_extreme") && !d.contains("flush_subnormal"));
            let u = SanitizeOptions { clamp_extreme: true, flush_subnormal: true, output: "x".into(), ..base }.to_j().compact();
            assert!(u.contains("\"dedup_routing_rows\":false,\"clamp_extreme\":true,\"flush_subnormal\":true,\"output\":\"x\""), "{}", u);
        }

        #[test]
        fn json_header_strictness() {
            let ok = br#"{"__metadata__":{"a":"b"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
            assert!(check_json_header(ok).is_ok());
            for bad in [
                &br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{}}"#[..],
                br#"{"x":{"dtype":"F32","dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                br#"{"x":{"dtype":"F32","data_offsets":[0,4]}}"#,
                br#"{"x":{"dtype":"F32","shape":null,"data_offsets":[0,4]}}"#,
                br#"{"__metadata__":{"a":1}}"#,
                br#"{"x":5}"#,
                br#"[1]"#,
                b"null",
                br#"{} x"#,
                br#"{"a":{"b":{"c":{"d":{"e":{"f":{"g":{"h":{"i":{"j":{"k":{"l":{"m":{"n":{"o":{"p":{"q":1}}}}}}}}}}}}}}}}}"#,
            ] {
                assert!(check_json_header(bad).is_err(), "{}", String::from_utf8_lossy(bad));
            }
            // a key written with an escape collides with its plain spelling
            assert!(check_json_header(br#"{"a":{},"a":{}}"#).is_err());
        }

        #[test]
        fn quantize_nan_and_inf_replacement_bits() {
            // +Inf -> signed max below the bound; the sign bit is preserved for -Inf
            let pos = finite_bound("F32", 1.0001);
            assert_eq!(f32::from_bits(pos as u32), 1.0001f32.min(f32::from_bits(pos as u32)));
            assert!(f32::from_bits(pos as u32) as f64 <= 1.0001);
        }
    }

    // ----- selectors -----

    pub(crate) fn sanitize_match(name: &str, patterns: &[String], automatic: &str) -> bool {
        let lower = name.to_lowercase();
        if automatic == "routing" && (lower.contains("tid2eid") || lower.contains("expert_id")) {
            return true;
        }
        if automatic == "norm" && lower.contains("norm") {
            return true;
        }
        patterns.iter().any(|p| matches!(gopath::path_match(p, name), Ok(true)))
    }

    /// Contiguous slots per row: GGUF ne[0], SafeTensors shape[-1].
    pub(crate) fn row_width(t: &TensorLocation, format: &str) -> i64 {
        if t.shape.is_empty() {
            return 1;
        }
        if format == "gguf" {
            return t.shape[0] as i64;
        }
        t.shape[t.shape.len() - 1] as i64
    }

    fn sanitize_selectors(l: &Layout, o: &SanitizeOptions) -> Result<()> {
        for patterns in [&o.routing_tensor, &o.norm_tensor] {
            for p in patterns.iter() {
                let matched = l.tensors.iter().any(|t| matches!(gopath::path_match(p, &t.name), Ok(true)));
                if !matched {
                    return Err(Error::msg(format!("tensor glob {:?} matched nothing", p)));
                }
            }
        }
        for t in &l.tensors {
            let routing = sanitize_match(&t.name, &o.routing_tensor, "routing");
            let norm = sanitize_match(&t.name, &o.norm_tensor, "norm");
            let integer = t.dtype == "I32" || t.dtype == "I64";
            if routing {
                if !integer || t.shape.is_empty() || t.size == 0 || o.num_experts <= 0 {
                    return Err(Error::msg(format!("routing tensor {:?} requires I32/I64, nonempty rows, and --num-experts > 0", t.name)));
                }
                if row_width(t, &l.format) > 1 << 20 {
                    return Err(Error::msg("routing row exceeds 1048576 elements"));
                }
            }
            if norm && (integer || t.size == 0) {
                return Err(Error::msg(format!("norm tensor {:?} requires nonempty floating point vectors", t.name)));
            }
        }
        Ok(())
    }

    // ----- float classification -----

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Kind {
        Normal,
        Nan,
        Inf,
        Subnormal,
    }

    impl Kind {
        fn name(self) -> &'static str {
            match self {
                Kind::Normal => "",
                Kind::Nan => "nan",
                Kind::Inf => "inf",
                Kind::Subnormal => "subnormal",
            }
        }
    }

    fn pow2(k: i32) -> f64 {
        // exact power of two (k in the normal double range)
        f64::from_bits(((1023 + k) as u64) << 52)
    }

    /// Oracle `sanitizeFloat`: value (as float64) and IEEE class of the
    /// dtype-width bit pattern `b`.
    pub fn classify(b: u64, dtype: &str) -> (f64, Kind) {
        let (exponent, mantissa): (u64, u64);
        let mut v = 0.0f64;
        match dtype {
            "F16" => {
                exponent = 0x7c00;
                mantissa = 0x03ff;
                let e = ((b >> 10) & 31) as i32;
                if e == 0 {
                    v = ((b & mantissa) as f64) * pow2(-24);
                } else {
                    v = ((1024 + (b & mantissa)) as f64) * pow2(e - 25);
                }
                if b & 0x8000 != 0 {
                    v = -v;
                }
            }
            "BF16" => {
                exponent = 0x7f80;
                mantissa = 0x007f;
                v = f32::from_bits((b as u32) << 16) as f64;
            }
            "F32" => {
                exponent = 0x7f800000;
                mantissa = 0x007fffff;
                v = f32::from_bits(b as u32) as f64;
            }
            _ => {
                exponent = 0x7ff0000000000000;
                mantissa = 0x000fffffffffffff;
                if dtype == "F64" {
                    v = f64::from_bits(b);
                }
            }
        }
        if b & exponent == exponent {
            if b & mantissa != 0 {
                return (f64::NAN, Kind::Nan);
            }
            if b >> (dtype_size(dtype) as u32 * 8 - 1) != 0 {
                return (f64::NEG_INFINITY, Kind::Inf);
            }
            return (f64::INFINITY, Kind::Inf);
        }
        if b & exponent == 0 && b & mantissa != 0 {
            return (v, Kind::Subnormal);
        }
        (v, Kind::Normal)
    }

    /// Largest positive finite encoding not above `bound` (binary search over
    /// the ordered positive IEEE encodings).
    pub fn finite_bound(dtype: &str, bound: f64) -> u64 {
        let mut hi: u64 = match dtype {
            "F16" => 0x7bff,
            "BF16" => 0x7f7f,
            "F32" => 0x7f7fffff,
            "F64" => 0x7fefffffffffffff,
            _ => 0,
        };
        let mut lo = 0u64;
        while lo < hi {
            let mid = lo + (hi - lo + 1) / 2;
            let (v, _) = classify(mid, dtype);
            if v <= bound {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }

    fn get_bits(b: &[u8], w: i64) -> u64 {
        match w {
            2 => u16::from_le_bytes([b[0], b[1]]) as u64,
            4 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64,
            _ => u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        }
    }

    fn put_bits(b: &mut [u8], w: i64, v: u64) {
        match w {
            2 => b[..2].copy_from_slice(&(v as u16).to_le_bytes()),
            4 => b[..4].copy_from_slice(&(v as u32).to_le_bytes()),
            _ => b[..8].copy_from_slice(&v.to_le_bytes()),
        }
    }

    fn get_int(b: &[u8], dtype: &str) -> i64 {
        if dtype == "I32" {
            (u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i32) as i64
        } else {
            u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as i64
        }
    }

    /// `math.Hypot`, the pure-Go algorithm (what arm64 runs). `1+q*q` is an
    /// ordinary expression in Go, hence `fma_add`.
    pub fn go_hypot(p: f64, q: f64) -> f64 {
        let (mut p, mut q) = (p.abs(), q.abs());
        if p == f64::INFINITY || q == f64::INFINITY {
            return f64::INFINITY;
        }
        if p.is_nan() || q.is_nan() {
            return f64::NAN;
        }
        if p < q {
            std::mem::swap(&mut p, &mut q);
        }
        if p == 0.0 {
            return 0.0;
        }
        q /= p;
        p * crate::numeric::fma_add(1.0, q, q).sqrt()
    }

    fn add(s: &mut SanitizeScan, o: &SanitizeOptions, kind: &str, t: &TensorLocation, i: i64, row: i64, bits: u64) {
        s.total_findings += 1;
        *s.counts.entry(kind.to_string()).or_insert(0) += 1;
        if (s.examples.len() as i64) < o.max_examples {
            let w = dtype_size(&t.dtype);
            let encoded = if !kind.starts_with("norm_") && kind != "routing_width_exceeds_experts" {
                format!("0x{:0width$x}", bits, width = (w * 2) as usize)
            } else {
                String::new()
            };
            s.examples.push(SanitizeFinding {
                kind: kind.into(),
                tensor: t.name.clone(),
                dtype: t.dtype.clone(),
                index: i,
                row,
                file_offset: t.offset + i * w,
                bits: encoded,
            });
        } else {
            s.examples_truncated = true;
        }
    }

    /// Oracle `sanitizeScan`. With `dst` it rewrites the repairable findings in
    /// the copy (always classifying from the original `f`).
    pub(crate) fn sanitize_scan(
        f: &File,
        l: &Layout,
        o: &SanitizeOptions,
        dst: Option<&File>,
        mut changes: Option<&mut BTreeMap<String, i64>>,
    ) -> Result<SanitizeScan> {
        let mut s = SanitizeScan { tensors: l.tensors.len() as i64, examples: Vec::new(), ..Default::default() };
        for t in &l.tensors {
            let w = dtype_size(&t.dtype);
            let n = t.size / w;
            let width = row_width(t, &l.format);
            if t.dtype == "I32" || t.dtype == "I64" {
                if !sanitize_match(&t.name, &o.routing_tensor, "routing") {
                    s.ignored_integer_tensors += 1;
                    continue;
                }
                let mut row_bytes = vec![0u8; (width * w) as usize];
                let mut base = 0i64;
                while base < n {
                    read_at(f, &mut row_bytes, t.offset + base * w)?;
                    let row = base / width;
                    s.routing_rows += 1;
                    s.routing_elements += width;
                    if width > o.num_experts {
                        add(&mut s, o, "routing_width_exceeds_experts", t, base, row, 0);
                    }
                    let mut seen: HashSet<i64> = HashSet::new();
                    let mut used: HashSet<i64> = HashSet::new();
                    let mut valid = true;
                    for j in 0..width {
                        let slot = &row_bytes[(j * w) as usize..];
                        let v = get_int(slot, &t.dtype);
                        if v < 0 {
                            add(&mut s, o, "routing_negative", t, base + j, row, get_bits(slot, w));
                            valid = false;
                        } else if v >= o.num_experts {
                            add(&mut s, o, "routing_oob", t, base + j, row, get_bits(slot, w));
                            valid = false;
                        }
                        if seen.contains(&v) {
                            add(&mut s, o, "routing_duplicate", t, base + j, row, get_bits(slot, w));
                        }
                        seen.insert(v);
                        used.insert(v);
                    }
                    if let (Some(d), true) = (dst, o.dedup_routing_rows && valid && width <= o.num_experts) {
                        seen = HashSet::new();
                        let mut next = 0i64;
                        for j in 0..width {
                            let v = get_int(&row_bytes[(j * w) as usize..], &t.dtype);
                            if seen.contains(&v) {
                                while used.contains(&next) {
                                    next += 1;
                                }
                                if next >= o.num_experts || (t.dtype == "I32" && next > i32::MAX as i64) {
                                    return Err(Error::msg("no representable unused expert"));
                                }
                                put_bits(&mut row_bytes[(j * w) as usize..], w, next as u64);
                                used.insert(next);
                                if let Some(c) = changes.as_deref_mut() {
                                    *c.entry("routing_duplicate".to_string()).or_insert(0) += 1;
                                }
                            }
                            seen.insert(v);
                        }
                        write_at(d, &row_bytes, t.offset + base * w)?;
                    }
                    base += width;
                }
                continue;
            }
            let norm = sanitize_match(&t.name, &o.norm_tensor, "norm");
            let mut buf = vec![0u8; 64 << 10];
            let (mut all_zero, mut finite) = (true, true);
            let mut row_norm = 0.0f64;
            let mut clamp_bound: Option<u64> = None;
            let rewrite = o.clamp_nonfinite || o.clamp_extreme || o.flush_subnormal;
            let mut bound_of = || *clamp_bound.get_or_insert_with(|| finite_bound(&t.dtype, o.max_abs));
            let sign_bit = 1u64 << (w as u32 * 8 - 1);
            let mut base = 0i64;
            while base < n {
                let count = std::cmp::min(n - base, buf.len() as i64 / w);
                let chunk = &mut buf[..(count * w) as usize];
                read_at(f, chunk, t.offset + base * w)?;
                for j in 0..count {
                    let i = base + j;
                    let slot_off = (j * w) as usize;
                    let b = get_bits(&chunk[slot_off..], w);
                    let (v, kind) = classify(b, &t.dtype);
                    let row = if width > 0 { i / width } else { 0 };
                    s.float_elements += 1;
                    if kind != Kind::Normal {
                        add(&mut s, o, kind.name(), t, i, row, b);
                    }
                    if kind != Kind::Nan && kind != Kind::Inf && v.abs() > o.max_abs {
                        add(&mut s, o, "extreme_finite", t, i, row, b);
                    }
                    if norm {
                        all_zero = all_zero && v == 0.0;
                        finite = finite && kind != Kind::Nan && kind != Kind::Inf;
                        row_norm = go_hypot(row_norm, v);
                        if (i + 1) % width == 0 {
                            s.norm_rows += 1;
                            if all_zero {
                                add(&mut s, o, "norm_all_zero", t, i - width + 1, row, 0);
                            } else if finite && row_norm < o.min_norm {
                                add(&mut s, o, "norm_below_min", t, i - width + 1, row, 0);
                            }
                            all_zero = true;
                            finite = true;
                            row_norm = 0.0;
                        }
                    }
                    if dst.is_some() {
                        let sign = if b & sign_bit != 0 { sign_bit } else { 0 };
                        let repl: Option<(u64, &str)> = if o.clamp_nonfinite && kind == Kind::Nan {
                            Some((0, "nan"))
                        } else if o.clamp_nonfinite && kind == Kind::Inf {
                            Some((bound_of() | sign, "inf"))
                        } else if o.flush_subnormal && kind == Kind::Subnormal {
                            Some((sign, "subnormal"))
                        } else if o.clamp_extreme && kind != Kind::Nan && kind != Kind::Inf && v.abs() > o.max_abs {
                            Some((bound_of() | sign, "extreme_finite"))
                        } else {
                            None
                        };
                        if let Some((replacement, name)) = repl {
                            put_bits(&mut chunk[slot_off..], w, replacement);
                            if let Some(c) = changes.as_deref_mut() {
                                *c.entry(name.to_string()).or_insert(0) += 1;
                            }
                        }
                    }
                }
                if let (Some(d), true) = (dst, rewrite) {
                    write_at(d, chunk, t.offset + base * w)?;
                }
                base += count;
            }
        }
        Ok(s)
    }
}

// ---------------------------------------------------------------------------
// probe
// ---------------------------------------------------------------------------

pub mod probe {
    use super::fsx::{self, io_msg, read_at, write_at};
    use super::gojson::{go_g, rfc3339_nano_now, J};
    use super::gopath;
    use super::sanitize::{row_width, sanitize_file_report, SanitizeOptions};
    use crate::builder::{build_gguf, build_safetensors, sha256_bytes_hex, write_file_exclusive, BuildTensor, MetaValue};
    use crate::container::{detect_format_bytes, file_sha256, inspect_model, load_layout, verify_patch, Layout, TensorLocation};
    use crate::dtype::{dtype_size, is_float_dtype, TensorBuffer};
    use crate::error::{Error, Result};
    use serde::{Deserialize, Serialize};
    use std::collections::{BTreeMap, BTreeSet};
    use std::io::Read;

    pub const PROBE_SCHEMA: &str = "secant.wsp.probe.v1";
    /// Fixture metadata: GGUF `general.architecture` (and `general.name` prefix).
    pub const PROBE_ARCHITECTURE: &str = "secant-probe";
    /// Fixture metadata keys (GGUF KV and SafeTensors `__metadata__`).
    pub const PROBE_META_CLASS: &str = "secant.probe.class";
    pub const PROBE_META_NUM_EXPERTS: &str = "secant.probe.num_experts";
    pub const PROBE_CLASSES: [&str; 4] = ["class1", "class2", "class3", "class4"];
    const PROBE_D_MODEL: i64 = 16;

    fn ser_vec_null<S: serde::Serializer>(v: &Vec<String>, s: S) -> std::result::Result<S::Ok, S::Error> {
        if v.is_empty() {
            s.serialize_none()
        } else {
            v.serialize(s)
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct ProbeOptions {
        /// class1..class4 | all
        #[serde(default)]
        pub class: String,
        /// safetensors | gguf | both
        #[serde(default)]
        pub format: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        pub base: String,
        #[serde(default)]
        pub num_experts: i64,
        #[serde(default)]
        pub out_dir: String,
        #[serde(default)]
        pub attempt_id: String,
    }

    impl ProbeOptions {
        pub fn to_j(&self) -> J {
            let mut v = vec![("class", J::s(&self.class)), ("format", J::s(&self.format))];
            if !self.base.is_empty() {
                v.push(("base", J::s(&self.base)));
            }
            v.push(("num_experts", J::Int(self.num_experts)));
            v.push(("out_dir", J::s(&self.out_dir)));
            v.push(("attempt_id", J::s(&self.attempt_id)));
            J::obj(v)
        }
        fn classes(&self) -> Vec<&'static str> {
            if self.class == "all" {
                PROBE_CLASSES.to_vec()
            } else {
                PROBE_CLASSES.iter().copied().filter(|c| *c == self.class).collect()
            }
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct ProbeTensor {
        pub name: String,
        pub dtype: String,
        pub shape: Vec<usize>,
        pub role: String,
        pub cells_written: i64,
        pub cells_skipped: i64,
        pub expected_kinds: Vec<String>,
        /// Empty serializes as `null` (a nil Go slice).
        #[serde(default, serialize_with = "ser_vec_null", deserialize_with = "super::sanitize::de_vec_null")]
        pub observed_kinds: Vec<String>,
    }

    impl ProbeTensor {
        fn to_j(&self) -> J {
            J::obj(vec![
                ("name", J::s(&self.name)),
                ("dtype", J::s(&self.dtype)),
                ("shape", J::Arr(self.shape.iter().map(|d| J::Int(*d as i64)).collect())),
                ("role", J::s(&self.role)),
                ("cells_written", J::Int(self.cells_written)),
                ("cells_skipped", J::Int(self.cells_skipped)),
                ("expected_kinds", J::strs(&self.expected_kinds)),
                ("observed_kinds", if self.observed_kinds.is_empty() { J::Null } else { J::strs(&self.observed_kinds) }),
            ])
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct ProbeFixture {
        pub class: String,
        /// synthetic | base-derived
        pub mode: String,
        pub format: String,
        pub file: String,
        pub path: String,
        pub bytes: i64,
        pub sha256: String,
        pub tensors: Vec<ProbeTensor>,
        pub inspect_ok: bool,
        pub sanitize_status: String,
        #[serde(rename = "sanitize_total_findings")]
        pub sanitize_total: i64,
        pub sanitize_counts: BTreeMap<String, i64>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub notes: Vec<String>,
    }

    impl ProbeFixture {
        fn to_j(&self) -> J {
            let mut v = vec![
                ("class", J::s(&self.class)),
                ("mode", J::s(&self.mode)),
                ("format", J::s(&self.format)),
                ("file", J::s(&self.file)),
                ("path", J::s(&self.path)),
                ("bytes", J::Int(self.bytes)),
                ("sha256", J::s(&self.sha256)),
                ("tensors", J::Arr(self.tensors.iter().map(|t| t.to_j()).collect())),
                ("inspect_ok", J::Bool(self.inspect_ok)),
                ("sanitize_status", J::s(&self.sanitize_status)),
                ("sanitize_total_findings", J::Int(self.sanitize_total)),
                ("sanitize_counts", J::Obj(self.sanitize_counts.iter().map(|(k, x)| (k.clone(), J::Int(*x))).collect())),
            ];
            if !self.notes.is_empty() {
                v.push(("notes", J::strs(&self.notes)));
            }
            J::obj(v)
        }
    }

    /// A requested class that could not be built from the supplied base.
    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct ProbeUnavailable {
        pub class: String,
        pub reason: String,
    }

    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    pub struct ProbeReport {
        pub schema: String,
        pub attempt_id: String,
        /// generated | unknown
        pub status: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        pub error: String,
        pub started_utc: String,
        pub finished_utc: String,
        pub options: Option<ProbeOptions>,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        pub base_pre_sha256: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        pub base_post_sha256: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub base_unchanged: Option<bool>,
        pub fixtures: Vec<ProbeFixture>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub unavailable: Vec<ProbeUnavailable>,
        pub receipt_sha256: String,
    }

    impl ProbeReport {
        pub fn to_j(&self) -> J {
            let mut v = vec![("schema", J::s(&self.schema)), ("attempt_id", J::s(&self.attempt_id)), ("status", J::s(&self.status))];
            if !self.error.is_empty() {
                v.push(("error", J::s(&self.error)));
            }
            v.push(("started_utc", J::s(&self.started_utc)));
            v.push(("finished_utc", J::s(&self.finished_utc)));
            v.push(("options", self.options.as_ref().map(|o| o.to_j()).unwrap_or(J::Null)));
            if !self.base_pre_sha256.is_empty() {
                v.push(("base_pre_sha256", J::s(&self.base_pre_sha256)));
            }
            if !self.base_post_sha256.is_empty() {
                v.push(("base_post_sha256", J::s(&self.base_post_sha256)));
            }
            if let Some(b) = self.base_unchanged {
                v.push(("base_unchanged", J::Bool(b)));
            }
            v.push(("fixtures", J::Arr(self.fixtures.iter().map(|f| f.to_j()).collect())));
            if !self.unavailable.is_empty() {
                v.push((
                    "unavailable",
                    J::Arr(self.unavailable.iter().map(|u| J::obj(vec![("class", J::s(&u.class)), ("reason", J::s(&u.reason))])).collect()),
                ));
            }
            v.push(("receipt_sha256", J::s(&self.receipt_sha256)));
            J::obj(v)
        }
        pub fn to_json_compact(&self) -> String {
            self.to_j().compact()
        }
        pub fn to_json_pretty(&self) -> String {
            self.to_j().pretty()
        }
        pub fn compute_seal(&self) -> String {
            let mut c = self.clone();
            c.receipt_sha256.clear();
            crate::container::sha256_hex(c.to_json_compact().as_bytes())
        }
    }

    // ---------- dtype bit helpers ----------

    struct FloatFmt {
        total: u32,
        exp: u32,
        mant: u32,
    }

    fn float_format_of(dtype: &str) -> Option<FloatFmt> {
        match dtype {
            "F16" => Some(FloatFmt { total: 16, exp: 5, mant: 10 }),
            "BF16" => Some(FloatFmt { total: 16, exp: 8, mant: 7 }),
            "F32" => Some(FloatFmt { total: 32, exp: 8, mant: 23 }),
            "F64" => Some(FloatFmt { total: 64, exp: 11, mant: 52 }),
            _ => None,
        }
    }

    impl FloatFmt {
        fn exp_mask(&self) -> u64 {
            ((1u64 << self.exp) - 1) << self.mant
        }
        fn quiet(&self) -> u64 {
            1u64 << (self.mant - 1)
        }
        fn sign(&self) -> u64 {
            1u64 << (self.total - 1)
        }
    }

    /// Exact dtype-width pattern of a named non-finite value.
    fn spec_bits(dtype: &str, spec: &str) -> Result<u64> {
        let f = float_format_of(dtype).ok_or_else(|| Error::msg(format!("non-finite specs need a float dtype, got {}", dtype)))?;
        Ok(match spec {
            "nan" => f.exp_mask() | f.quiet(),
            "nan-neg" => f.sign() | f.exp_mask() | f.quiet(),
            "nan-payload" => f.exp_mask() | f.quiet() | 1,
            "nan-signal" => f.exp_mask() | 1,
            "+inf" => f.exp_mask(),
            "-inf" => f.sign() | f.exp_mask(),
            _ => return Err(Error::msg(format!("unknown non-finite spec {:?}", spec))),
        })
    }

    fn put_bits(b: &mut TensorBuffer, i: i64, bits: u64) {
        let i = i as usize;
        match dtype_size(&b.dtype) {
            2 => b.data[i * 2..i * 2 + 2].copy_from_slice(&(bits as u16).to_le_bytes()),
            4 => b.data[i * 4..i * 4 + 4].copy_from_slice(&(bits as u32).to_le_bytes()),
            _ => b.data[i * 8..i * 8 + 8].copy_from_slice(&bits.to_le_bytes()),
        }
    }

    // ---------- pattern grid ----------

    /// A tensor addressed as rows x width in the sanitizer's row convention.
    struct Grid {
        buf: TensorBuffer,
        rows: i64,
        width: i64,
        written: i64,
        skipped: i64,
        kinds: BTreeSet<String>,
    }

    impl Grid {
        fn new(buf: TensorBuffer, rows: i64, width: i64) -> Grid {
            Grid { buf, rows, width, written: 0, skipped: 0, kinds: BTreeSet::new() }
        }
        fn idx(&self, r: i64, c: i64) -> Option<i64> {
            if r < 0 || r >= self.rows || c < 0 || c >= self.width {
                return None;
            }
            Some(r * self.width + c)
        }
        fn put_spec(&mut self, r: i64, c: i64, spec: &str) -> Result<()> {
            let i = match self.idx(r, c) {
                Some(i) => i,
                None => {
                    self.skipped += 1;
                    return Ok(());
                }
            };
            let bits = spec_bits(&self.buf.dtype, spec)?;
            put_bits(&mut self.buf, i, bits);
            self.written += 1;
            if spec.ends_with("inf") {
                self.kinds.insert("inf".into());
            } else {
                self.kinds.insert("nan".into());
            }
            Ok(())
        }
        /// Writes a finite value with dtype rounding; refuses a value the dtype
        /// cannot hold (an F16 cell asked for 1e38 would silently become +Inf).
        fn put_finite(&mut self, r: i64, c: i64, v: f64) -> Result<()> {
            let i = match self.idx(r, c) {
                Some(i) => i,
                None => {
                    self.skipped += 1;
                    return Ok(());
                }
            };
            self.buf.set_float(i, v);
            let got = self.buf.get(i);
            if got.is_nan() || got.is_infinite() || (v != 0.0 && got.abs() < v.abs() / 2.0) {
                return Err(Error::msg(format!("dtype {} cannot represent {} (stored {})", self.buf.dtype, go_g(v), go_g(got))));
            }
            self.written += 1;
            if got.abs() > 1e4 {
                self.kinds.insert("extreme_finite".into());
            }
            Ok(())
        }
        fn put_int(&mut self, r: i64, c: i64, v: i64) {
            let i = match self.idx(r, c) {
                Some(i) => i,
                None => {
                    self.skipped += 1;
                    return;
                }
            };
            if self.buf.set_int(i, v).is_err() {
                self.skipped += 1; // value outside the dtype range (e.g. a 2^32 entry in an I32 table)
                return;
            }
            self.written += 1;
        }
        fn kind_list(&self) -> Vec<String> {
            self.kinds.iter().cloned().collect()
        }
    }

    // ---------- roles ----------

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Role {
        Router,
        Scale,
        Bias,
        NanSort,
        Routing,
        Norm,
        NormSub,
    }

    impl Role {
        fn name(self) -> &'static str {
            match self {
                Role::Router => "router",
                Role::Scale => "scale",
                Role::Bias => "bias",
                Role::NanSort => "nansort",
                Role::Routing => "routing",
                Role::Norm => "norm",
                Role::NormSub => "norm-sub",
            }
        }
        fn from_name(s: &str) -> Role {
            match s {
                "router" => Role::Router,
                "scale" => Role::Scale,
                "bias" => Role::Bias,
                "nansort" => Role::NanSort,
                "routing" => Role::Routing,
                "norm" => Role::Norm,
                _ => Role::NormSub,
            }
        }
        /// Fill the benign baseline (synthetic tensors only).
        fn init(self, g: &mut Grid, n: i64) {
            let total = g.rows * g.width;
            match self {
                Role::Router | Role::Scale | Role::Bias | Role::NanSort => {
                    let seed: i64 = match self {
                        Role::Router => 1,
                        Role::Scale => 2,
                        Role::Bias => 3,
                        _ => 4,
                    };
                    for i in 0..total {
                        g.buf.set_float(i, (((i * 37 + seed) % 2001) - 1000) as f64 / 1001.0);
                    }
                }
                Role::Routing => {
                    for r in 0..g.rows {
                        for c in 0..g.width {
                            let _ = g.buf.set_int(r * g.width + c, (r + c) % n);
                        }
                    }
                }
                Role::Norm | Role::NormSub => {
                    for i in 0..total {
                        g.buf.set_float(i, 1.0);
                    }
                }
            }
        }
        fn apply(self, g: &mut Grid, n: i64) -> Result<()> {
            match self {
                Role::Router => apply_router(g),
                Role::Scale | Role::Bias => apply_scale(g),
                Role::NanSort => apply_nansort(g),
                Role::Routing => {
                    apply_routing(g, n);
                    Ok(())
                }
                Role::Norm => {
                    apply_norm_zero(g);
                    if g.rows >= 2 {
                        write_subnormal_row(g, 1);
                    }
                    Ok(())
                }
                Role::NormSub => {
                    write_subnormal_row(g, 0);
                    Ok(())
                }
            }
        }
    }

    fn apply_router(g: &mut Grid) -> Result<()> {
        // NaN and -Inf router/gate logits: row 0 mixes them, row 1 is all -Inf
        // (a "masked" expert row), row 2 is all NaN.
        g.put_spec(0, 0, "nan")?;
        g.put_spec(0, 1, "-inf")?;
        for c in 0..g.width {
            g.put_spec(1, c, "-inf")?;
            g.put_spec(2, c, "nan")?;
        }
        Ok(())
    }

    fn apply_scale(g: &mut Grid) -> Result<()> {
        g.put_spec(0, 0, "nan")?;
        g.put_finite(0, 1, 1e38)?;
        g.put_finite(0, 2, -1e38)
    }

    fn apply_nansort(g: &mut Grid) -> Result<()> {
        let cycle = ["nan", "nan-neg", "nan-payload", "nan-signal"];
        for c in 0..g.width {
            g.put_spec(0, c, "nan")?;
            g.put_spec(1, c, cycle[(c % 4) as usize])?;
            if c == 0 {
                g.put_spec(2, c, "nan")?;
            } else {
                g.put_finite(2, c, 0.25 * c as f64)?;
            }
            if c == g.width - 1 {
                g.put_spec(3, c, "nan")?;
            } else {
                g.put_finite(3, c, 0.5)?;
            }
            if c % 2 == 0 {
                g.put_spec(4, c, "nan")?;
            } else {
                g.put_finite(4, c, 0.5)?;
            }
            g.put_spec(5, c, ["+inf", "-inf", "nan"][(c % 3) as usize])?;
        }
        Ok(())
    }

    fn apply_routing(g: &mut Grid, n: i64) {
        let (min_i, max_i) = if g.buf.dtype == "I64" { (i64::MIN, i64::MAX) } else { (i32::MIN as i64, i32::MAX as i64) };
        // (row, col, value, kind, i64-only)
        let specs: [(i64, i64, i64, &str, bool); 7] = [
            (1, 0, -1, "routing_negative", false),
            (2, 0, min_i, "routing_negative", false),
            (3, 0, n, "routing_oob", false),
            (4, 0, max_i, "routing_oob", false),
            (6, 0, 1i64 << 32, "routing_oob", true), // truncates to expert 0 in an int32 reader
            (7, 0, i32::MIN as i64, "routing_negative", true),
            (8, 0, i32::MAX as i64, "routing_oob", true),
        ];
        for (row, col, val, kind, i64only) in specs {
            if i64only && g.buf.dtype != "I64" {
                continue;
            }
            let before = g.written;
            g.put_int(row, col, val);
            if g.written > before {
                g.kinds.insert(kind.into());
            }
        }
        // row 5: duplicate expert ids in one row
        let before = g.written;
        g.put_int(5, 0, 0);
        g.put_int(5, 1, 0);
        if g.written == before + 2 {
            g.kinds.insert("routing_duplicate".into());
        }
    }

    fn row_norm_below(g: &Grid, row: i64, min: f64) -> bool {
        let mut sq = 0.0f64;
        for c in 0..g.width {
            let i = g.idx(row, c).unwrap_or(0);
            sq = super::sanitize::go_hypot(sq, g.buf.get(i));
        }
        sq < min
    }

    fn apply_norm_zero(g: &mut Grid) {
        for c in 0..g.width {
            match g.idx(0, c) {
                Some(i) => {
                    put_bits(&mut g.buf, i, 0);
                    g.written += 1;
                }
                None => g.skipped += 1,
            }
        }
        if g.written > 0 {
            g.kinds.insert("norm_all_zero".into());
        }
    }

    fn write_subnormal_row(g: &mut Grid, row: i64) {
        let mut n = 0;
        for c in 0..g.width {
            match g.idx(row, c) {
                Some(i) => {
                    put_bits(&mut g.buf, i, (c % 3) as u64 + 1); // smallest three subnormal magnitudes
                    g.written += 1;
                    n += 1;
                }
                None => g.skipped += 1,
            }
        }
        if n > 0 {
            g.kinds.insert("subnormal".into());
            if row_norm_below(g, row, 1e-12) {
                g.kinds.insert("norm_below_min".into());
            }
        }
    }

    // ---------- synthetic specs ----------

    struct Synth {
        name: &'static str,
        role: &'static str,
        dtype: &'static str,
        shape: Vec<usize>,
    }

    fn top_k(n: i64) -> usize {
        if n < 4 {
            n as usize
        } else {
            4
        }
    }

    fn synth_specs(class: &str, n: i64) -> Vec<Synth> {
        let d = PROBE_D_MODEL as usize;
        let s = |name, role, dtype, shape: Vec<usize>| Synth { name, role, dtype, shape };
        match class {
            "class1" => vec![
                s("blk.0.ffn_gate_inp.weight", "router", "F32", vec![n as usize, d]),
                s("blk.1.ffn_gate_inp.weight", "router", "BF16", vec![n as usize, d]),
                s("blk.2.ffn_gate_inp.weight", "router", "F16", vec![n as usize, d]),
            ],
            "class2" => vec![
                s("blk.0.attn_output.scale", "scale", "F32", vec![d]),
                s("blk.0.attn_output.bias", "bias", "F32", vec![d]),
                s("blk.1.attn_output.scale", "scale", "BF16", vec![d]),
                s("blk.1.attn_output.bias", "bias", "BF16", vec![d]),
                s("blk.2.attn_output.scale", "scale", "F64", vec![d]),
            ],
            "class3" => vec![
                s("blk.0.router_scores.weight", "nansort", "F32", vec![6, 8]),
                s("blk.1.router_scores.weight", "nansort", "BF16", vec![6, 8]),
                s("blk.2.router_scores.weight", "nansort", "F16", vec![6, 8]),
            ],
            "class4" => {
                let k = top_k(n);
                vec![
                    s("blk.0.tid2eid", "routing", "I32", vec![6, k]),
                    s("blk.0.expert_id", "routing", "I64", vec![9, k]),
                    s("blk.0.attn_norm.weight", "norm", "F32", vec![3, d]),
                    s("blk.1.attn_norm.weight", "norm", "BF16", vec![3, d]),
                    s("blk.2.attn_norm.weight", "norm", "F16", vec![3, d]),
                ]
            }
            _ => Vec::new(),
        }
    }

    fn class_roles(class: &str) -> &'static [&'static str] {
        match class {
            "class1" => &["router"],
            "class2" => &["scale", "bias"],
            "class3" => &["nansort"],
            "class4" => &["routing", "norm"],
            _ => &[],
        }
    }

    fn slug(class: &str) -> &'static str {
        match class {
            "class1" => "router-nan-neginf",
            "class2" => "scale-bias-nan-extreme",
            "class3" => "nan-sort-arrays",
            _ => "routing-and-norm",
        }
    }

    /// SafeTensors-order shape (last axis contiguous) as rows x width.
    fn grid_of(shape: &[usize]) -> (i64, i64) {
        let width = shape[shape.len() - 1] as i64;
        let rows = shape[..shape.len() - 1].iter().fold(1i64, |a, d| a * *d as i64);
        (rows, width)
    }

    /// Same for a tensor scanned out of a container (raw storage order).
    fn layout_grid(t: &TensorLocation, format: &str) -> (i64, i64) {
        let width = row_width(t, format);
        let w = dtype_size(&t.dtype);
        if width <= 0 || w == 0 {
            return (0, 0);
        }
        ((t.size / w) / width, width)
    }

    // ---------- generation ----------

    fn normalize(o: &ProbeOptions) -> Result<ProbeOptions> {
        let mut o = o.clone();
        if o.class.is_empty() {
            o.class = "all".into();
        }
        if o.format.is_empty() {
            o.format = "both".into();
        }
        if o.num_experts == 0 {
            o.num_experts = 8;
        }
        if o.attempt_id.is_empty() {
            return Err(Error::msg("attempt_id is required"));
        }
        if o.out_dir.is_empty() {
            return Err(Error::msg("out_dir is required"));
        }
        if !(o.class == "all" || PROBE_CLASSES.contains(&o.class.as_str())) {
            return Err(Error::msg(format!("class {:?}: want class1|class2|class3|class4|all", o.class)));
        }
        if o.format != "safetensors" && o.format != "gguf" && o.format != "both" {
            return Err(Error::msg(format!("format {:?}: want safetensors|gguf|both", o.format)));
        }
        if o.num_experts < 2 || o.num_experts > 1 << 20 {
            return Err(Error::msg(format!("num_experts {} outside [2, {}]", o.num_experts, 1 << 20)));
        }
        Ok(o)
    }

    /// Oracle `GenerateProbe`: writes the requested fixtures into a NEW
    /// directory. A report is ALWAYS returned (status `unknown` plus the error
    /// on failure, and on error nothing is left in `out_dir`).
    pub fn generate_probe(o: &ProbeOptions) -> (ProbeReport, Result<()>) {
        let mut rep = ProbeReport {
            schema: PROBE_SCHEMA.into(),
            attempt_id: o.attempt_id.clone(),
            status: "unknown".into(),
            started_utc: rfc3339_nano_now(),
            fixtures: Vec::new(),
            ..Default::default()
        };
        let mut created: Vec<String> = Vec::new();
        let mut dir_made = false;
        let r = generate(&mut rep, o, &mut created, &mut dir_made);
        if let Err(e) = &r {
            rep.status = "unknown".into();
            rep.error = e.message().to_string();
            rep.fixtures = Vec::new();
            for p in &created {
                let _ = std::fs::remove_file(p);
            }
            if dir_made {
                let _ = std::fs::remove_dir(&rep.options.as_ref().map(|x| x.out_dir.clone()).unwrap_or_else(|| o.out_dir.clone()));
            }
        }
        rep.finished_utc = rfc3339_nano_now();
        rep.receipt_sha256 = String::new();
        rep.receipt_sha256 = rep.compute_seal();
        (rep, r)
    }

    fn generate(rep: &mut ProbeReport, o: &ProbeOptions, created: &mut Vec<String>, dir_made: &mut bool) -> Result<()> {
        let o = normalize(o)?;
        rep.options = Some(o.clone());
        let mut base_format = String::new();
        if !o.base.is_empty() {
            base_format = detect_format(&o.base)?;
            if o.format != "both" && o.format != base_format {
                return Err(Error::msg(format!(
                    "--format {} conflicts with the {} base; use both or {}",
                    o.format, base_format, base_format
                )));
            }
            rep.base_pre_sha256 = file_sha256(&o.base)?.0;
        }
        let parent = gopath::dir(&o.out_dir);
        std::fs::create_dir_all(&parent).map_err(|e| Error::msg(format!("mkdir {}: {}", parent, io_msg(&e))))?;
        // atomic reservation: fails if it exists
        make_dir(&o.out_dir).map_err(|e| Error::msg(format!("out-dir must be a new directory: mkdir {}: {}", o.out_dir, io_msg(&e))))?;
        *dir_made = true;

        for class in o.classes() {
            if !o.base.is_empty() {
                match from_base(&o, class, &base_format, created)? {
                    (Some(fx), _) => rep.fixtures.push(fx),
                    (None, reason) => rep.unavailable.push(ProbeUnavailable { class: class.into(), reason }),
                }
                continue;
            }
            for format in ["safetensors", "gguf"] {
                if o.format != "both" && o.format != format {
                    continue;
                }
                let fx = synthetic(&o, class, format, created).map_err(|e| Error::msg(format!("{} {}: {}", class, format, e.message())))?;
                rep.fixtures.push(fx);
            }
        }
        if rep.fixtures.is_empty() {
            let un: Vec<String> = rep.unavailable.iter().map(|u| format!("{{{} {}}}", u.class, u.reason)).collect();
            return Err(Error::msg(format!("no fixture could be generated from base {}: [{}]", o.base, un.join(" "))));
        }
        if !o.base.is_empty() {
            let post = file_sha256(&o.base)?.0;
            rep.base_post_sha256 = post.clone();
            let same = post == rep.base_pre_sha256;
            rep.base_unchanged = Some(same);
            if !same {
                return Err(Error::msg("base changed during generation"));
            }
        }
        rep.status = "generated".into();
        Ok(())
    }

    fn make_dir(p: &str) -> std::io::Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new().mode(0o755).create(p)
        }
        #[cfg(not(unix))]
        {
            std::fs::create_dir(p)
        }
    }

    /// Oracle `DetectFormat`.
    fn detect_format(path: &str) -> Result<String> {
        let mut f = fsx::open_regular(path)?;
        let mut head = Vec::with_capacity(9);
        f.by_ref().take(9).read_to_end(&mut head).map_err(|e| Error::msg(format!("read {}: {}", path, io_msg(&e))))?;
        let base = path.rsplit('/').next().unwrap_or(path);
        Ok(detect_format_bytes(&head, base)?.to_string())
    }

    fn synthetic(o: &ProbeOptions, class: &str, format: &str, created: &mut Vec<String>) -> Result<ProbeFixture> {
        let n = o.num_experts;
        let specs = synth_specs(class, n);
        let mut tensors: Vec<BuildTensor> = Vec::new();
        let mut meta: Vec<ProbeTensor> = Vec::new();
        for s in &specs {
            let role = Role::from_name(s.role);
            let (rows, width) = grid_of(&s.shape);
            let buf = TensorBuffer::new(s.dtype, vec![0u8; (rows * width * dtype_size(s.dtype)) as usize]);
            let mut g = Grid::new(buf, rows, width);
            role.init(&mut g, n);
            role.apply(&mut g, n).map_err(|e| Error::msg(format!("tensor {}: {}", s.name, e.message())))?;
            if g.written == 0 {
                return Err(Error::msg(format!("tensor {}: pattern wrote no cells", s.name)));
            }
            meta.push(ProbeTensor {
                name: s.name.into(),
                dtype: s.dtype.into(),
                shape: s.shape.clone(),
                role: s.role.into(),
                cells_written: g.written,
                cells_skipped: g.skipped,
                expected_kinds: g.kind_list(),
                observed_kinds: Vec::new(),
            });
            tensors.push(BuildTensor { name: s.name.into(), dtype: s.dtype.into(), shape: s.shape.clone(), data: g.buf.data });
        }
        let data = if format == "safetensors" {
            let mut m = BTreeMap::new();
            m.insert("format".to_string(), "pt".to_string());
            m.insert(PROBE_META_CLASS.to_string(), class.to_string());
            m.insert(PROBE_META_NUM_EXPERTS.to_string(), o.num_experts.to_string());
            build_safetensors(&tensors, &m)?
        } else {
            let kv = vec![
                ("general.architecture".to_string(), MetaValue::Str(PROBE_ARCHITECTURE.into())),
                ("general.name".to_string(), MetaValue::Str(format!("{}-{}", PROBE_ARCHITECTURE, class))),
                (PROBE_META_CLASS.to_string(), MetaValue::Str(class.into())),
                (PROBE_META_NUM_EXPERTS.to_string(), MetaValue::U32(o.num_experts as u32)),
                ("general.alignment".to_string(), MetaValue::U32(32)),
            ];
            build_gguf(&kv, &tensors, 32)?
        };
        let file = format!("{}-{}.{}", class, slug(class), format);
        let path = gopath::join(&o.out_dir, &file);
        write_file_exclusive(&path, &data)?;
        created.push(path.clone());
        let mut fx = ProbeFixture {
            class: class.into(),
            mode: "synthetic".into(),
            format: format.into(),
            file,
            path,
            bytes: data.len() as i64,
            sha256: sha256_bytes_hex(&data),
            tensors: meta,
            ..Default::default()
        };
        self_check(&mut fx, o, true)?;
        Ok(fx)
    }

    /// Run the container inspection and the sanitizer on a written fixture and
    /// compare the observed finding kinds with the writer's intent, per tensor.
    /// `exact` additionally forbids any finding the writer did not intend.
    fn self_check(fx: &mut ProbeFixture, o: &ProbeOptions, exact: bool) -> Result<()> {
        inspect_model(&fx.path).map_err(|e| Error::msg(format!("fixture {} fails container inspection: {}", fx.file, e.message())))?;
        fx.inspect_ok = true;
        let opts =
            SanitizeOptions { attempt_id: o.attempt_id.clone(), num_experts: o.num_experts, max_examples: 10000, ..Default::default() };
        let (rec, r) = sanitize_file_report(&fx.path, &opts);
        if let Err(e) = r {
            return Err(Error::msg(format!("fixture {}: sanitizer could not scan: {}", fx.file, e.message())));
        }
        let before = rec.before.clone().unwrap_or_default();
        fx.sanitize_status = rec.status.clone();
        fx.sanitize_total = before.total_findings;
        fx.sanitize_counts = before.counts.clone();
        let mut observed: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for e in &before.examples {
            observed.entry(e.tensor.clone()).or_default().insert(e.kind.clone());
        }
        let mut union: BTreeSet<String> = BTreeSet::new();
        for t in fx.tensors.iter_mut() {
            if let Some(ks) = observed.get(&t.name) {
                for k in ks {
                    t.observed_kinds.push(k.clone());
                }
            }
            t.observed_kinds.sort();
            for k in &t.expected_kinds {
                union.insert(k.clone());
            }
        }
        if before.examples_truncated {
            // Per-tensor attribution is unavailable; check the aggregate counts.
            for k in &union {
                if before.counts.get(k).copied().unwrap_or(0) == 0 {
                    return Err(Error::msg(format!("fixture {}: expected finding kind {} absent from sanitizer counts", fx.file, k)));
                }
            }
            fx.notes.push("sanitizer examples truncated; per-tensor attribution checked at aggregate granularity only".into());
            return Ok(());
        }
        let empty = BTreeSet::new();
        for t in &fx.tensors {
            let obs = observed.get(&t.name).unwrap_or(&empty);
            for k in &t.expected_kinds {
                if !obs.contains(k) {
                    return Err(Error::msg(format!(
                        "fixture {} tensor {}: expected finding {} not reported by the sanitizer (observed {})",
                        fx.file,
                        t.name,
                        k,
                        go_slice(&t.observed_kinds)
                    )));
                }
            }
            if exact {
                let want: BTreeSet<&String> = t.expected_kinds.iter().collect();
                for k in obs {
                    if !want.contains(k) {
                        return Err(Error::msg(format!(
                            "fixture {} tensor {}: unintended finding {} (expected {})",
                            fx.file,
                            t.name,
                            k,
                            go_slice(&t.expected_kinds)
                        )));
                    }
                }
            }
        }
        if exact {
            let known: BTreeSet<&String> = fx.tensors.iter().map(|t| &t.name).collect();
            for name in observed.keys() {
                if !known.contains(name) {
                    return Err(Error::msg(format!("fixture {}: findings on unlisted tensor {}", fx.file, name)));
                }
            }
        }
        Ok(())
    }

    /// Go `%v` of a []string.
    fn go_slice(v: &[String]) -> String {
        format!("[{}]", v.join(" "))
    }

    // ---------- base-derived ----------

    struct Choice {
        role: Role,
        tensor: usize,
        why: &'static str,
    }

    fn base_select(class: &str, l: &Layout) -> Vec<Choice> {
        let mut out: Vec<Choice> = Vec::new();
        let mut used: BTreeSet<usize> = BTreeSet::new();
        let lower = |t: &TensorLocation| t.name.to_lowercase();
        let is_router = |t: &TensorLocation| {
            let n = lower(t);
            is_float_dtype(&t.dtype) && (n.contains("gate_inp") || n.contains("router") || n.contains("mlp.gate.weight"))
        };
        let is_norm = |t: &TensorLocation| is_float_dtype(&t.dtype) && lower(t).contains("norm");
        let first =
            |out: &mut Vec<Choice>, used: &mut BTreeSet<usize>, role: Role, why: &'static str, m: &dyn Fn(&TensorLocation) -> bool| {
                for (i, t) in l.tensors.iter().enumerate() {
                    if used.contains(&i) || !m(t) {
                        continue;
                    }
                    out.push(Choice { role, tensor: i, why });
                    used.insert(i);
                    return;
                }
            };
        match class {
            "class1" => first(&mut out, &mut used, Role::Router, "router/gate tensor by name", &is_router),
            "class2" => {
                first(&mut out, &mut used, Role::Scale, "name ends in scale", &|t| {
                    let n = lower(t);
                    is_float_dtype(&t.dtype) && (n.ends_with(".scale") || n.ends_with("_scale") || n.ends_with(".scales"))
                });
                first(&mut out, &mut used, Role::Bias, "name ends in bias", &|t| is_float_dtype(&t.dtype) && lower(t).ends_with(".bias"));
            }
            "class3" => {
                first(&mut out, &mut used, Role::NanSort, "router/gate tensor by name", &is_router);
                if out.is_empty() {
                    first(&mut out, &mut used, Role::NanSort, "no router tensor; first float tensor with >=6 rows and >=4 columns", &|t| {
                        let (r, w) = layout_grid(t, &l.format);
                        is_float_dtype(&t.dtype) && t.shape.len() >= 2 && r >= 6 && w >= 4
                    });
                }
            }
            "class4" => {
                first(&mut out, &mut used, Role::Routing, "name contains tid2eid or expert_id", &|t| {
                    let n = lower(t);
                    (t.dtype == "I32" || t.dtype == "I64") && (n.contains("tid2eid") || n.contains("expert_id"))
                });
                first(&mut out, &mut used, Role::Norm, "first norm tensor", &is_norm);
                if let Some(last) = out.last() {
                    if last.role == Role::Norm {
                        let (r, _) = layout_grid(&l.tensors[last.tensor], &l.format);
                        if r < 2 {
                            first(
                                &mut out,
                                &mut used,
                                Role::NormSub,
                                "norm tensor is a single vector; subnormal pattern goes into a second norm tensor",
                                &is_norm,
                            );
                        }
                    }
                }
            }
            _ => {}
        }
        out
    }

    /// Oracle `LoadLayout`: the Go scanner rejects a GGUF with any quantized
    /// tensor, Secant's lists them, so reject them here with the Go message.
    fn go_load_layout(path: &str) -> Result<Layout> {
        let l = load_layout(path)?;
        for t in &l.tensors {
            if t.is_quantized() {
                return Err(Error::msg(format!(
                    "gguf: tensor {:?}: unsupported ggml type {} (quantized): weight surgery is implemented for F32/F16/BF16/F64/I32/I64 payloads only",
                    t.name,
                    t.ggml_type.map(|x| x.to_string()).unwrap_or_default()
                )));
            }
        }
        Ok(l)
    }

    /// Returns `(Some(fixture), "")` or `(None, reason)` when the class cannot
    /// be built from this base.
    fn from_base(o: &ProbeOptions, class: &str, format: &str, created: &mut Vec<String>) -> Result<(Option<ProbeFixture>, String)> {
        let layout = go_load_layout(&o.base).map_err(|e| Error::msg(format!("base layout: {}", e.message())))?;
        let choices = base_select(class, &layout);
        if choices.is_empty() {
            return Ok((None, format!("base has no tensor suitable for {} by the selection rules", class)));
        }
        let mut not_covered: Vec<String> = Vec::new();
        for role in class_roles(class) {
            if !choices.iter().any(|c| c.role.name() == *role) {
                not_covered.push(format!("role {}: no suitable tensor in the base; NOT covered by this fixture", role));
            }
        }
        let file = format!("{}-{}.base-derived.{}", class, slug(class), format);
        let path = gopath::join(&o.out_dir, &file);
        crate::container::copy_file(&o.base, &path)?;
        created.push(path.clone());
        let out = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| Error::msg(format!("open {}: {}", path, io_msg(&e))))?;
        let n = o.num_experts;
        let mut fx = ProbeFixture {
            class: class.into(),
            mode: "base-derived".into(),
            format: format.into(),
            file,
            path: path.clone(),
            notes: not_covered,
            ..Default::default()
        };
        let mut changed: Vec<String> = Vec::new();
        for c in &choices {
            let t = &layout.tensors[c.tensor];
            let mut raw = vec![0u8; t.size as usize];
            read_at(&out, &mut raw, t.offset)
                .map_err(|e| Error::msg(format!("read tensor {} [{},{}): {}", t.name, t.offset, t.offset + t.size, e.message())))?;
            let (rows, width) = layout_grid(t, format);
            let mut g = Grid::new(TensorBuffer::new(&t.dtype, raw), rows, width);
            c.role.apply(&mut g, n).map_err(|e| Error::msg(format!("tensor {}: {}", t.name, e.message())))?;
            if g.written == 0 {
                // NOTE (oracle behaviour, replicated): the copy of the base made
                // above stays in out-dir and in `created`, so it is only removed
                // if a LATER error fails the run.
                return Ok((None, format!("tensor {} is too small for the {} pattern", t.name, c.role.name())));
            }
            write_at(&out, &g.buf.data, t.offset)?;
            changed.push(t.name.clone());
            fx.tensors.push(ProbeTensor {
                name: t.name.clone(),
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
                role: c.role.name().into(),
                cells_written: g.written,
                cells_skipped: g.skipped,
                expected_kinds: g.kind_list(),
                observed_kinds: Vec::new(),
            });
            fx.notes.push(format!("{} -> {} ({})", c.role.name(), t.name, c.why));
        }
        out.sync_all().map_err(|e| Error::msg(io_msg(&e)))?;
        drop(out);
        let v = verify_patch(&o.base, &path, &changed).map_err(|e| Error::msg(format!("verify-patch: {}", e.message())))?;
        if !v.ok {
            return Err(Error::msg(format!("header or untouched-tensor identity broken: {}", v.failure_summary())));
        }
        let (sum, size) = file_sha256(&path)?;
        fx.sha256 = sum;
        fx.bytes = size;
        self_check(&mut fx, o, false)?;
        fx.notes.push("header region and all untouched tensors byte-identical to the base (VerifyPatch ok)".into());
        Ok((Some(fx), String::new()))
    }

    /// Publish the receipt (indented JSON + newline, mode 0600) at a NEW path.
    pub fn write_probe_receipt(dst: &str, rep: &ProbeReport) -> Result<()> {
        if rep.attempt_id.is_empty() {
            return Err(Error::msg("receipt requires attempt_id"));
        }
        let mut b = rep.to_json_pretty();
        b.push('\n');
        fsx::publish_new(dst, b.as_bytes())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn grid(dtype: &str, rows: i64, width: i64) -> Grid {
            Grid::new(TensorBuffer::new(dtype, vec![0u8; (rows * width * dtype_size(dtype)) as usize]), rows, width)
        }

        #[test]
        fn spec_bits_are_exact_per_dtype() {
            assert_eq!(spec_bits("F32", "nan").unwrap(), 0x7fc00000);
            assert_eq!(spec_bits("F32", "nan-neg").unwrap(), 0xffc00000);
            assert_eq!(spec_bits("F32", "nan-payload").unwrap(), 0x7fc00001);
            assert_eq!(spec_bits("F32", "nan-signal").unwrap(), 0x7f800001);
            assert_eq!(spec_bits("F32", "+inf").unwrap(), 0x7f800000);
            assert_eq!(spec_bits("F32", "-inf").unwrap(), 0xff800000);
            assert_eq!(spec_bits("F16", "nan").unwrap(), 0x7e00);
            assert_eq!(spec_bits("BF16", "-inf").unwrap(), 0xff80);
            assert_eq!(spec_bits("F64", "nan-signal").unwrap(), 0x7ff0000000000001);
            assert!(spec_bits("I32", "nan").is_err());
            assert!(spec_bits("F32", "bogus").is_err());
        }

        #[test]
        fn f16_cannot_hold_extreme_scale() {
            let mut g = grid("F16", 1, 16);
            let e = apply_scale(&mut g).unwrap_err();
            assert!(e.message().contains("cannot represent 1e+38 (stored +Inf)"), "{}", e.message());
            let mut g = grid("BF16", 1, 16);
            apply_scale(&mut g).unwrap();
            assert!(g.kinds.contains("nan") && g.kinds.contains("extreme_finite"));
        }

        #[test]
        fn routing_boundary_values_i32_and_i64() {
            let mut g32 = grid("I32", 9, 4);
            apply_routing(&mut g32, 8);
            // 2^32 does not fit an I32 table and is skipped, not clamped
            assert_eq!(g32.skipped, 0 + 0); // the i64-only rows are not attempted for I32
            assert_eq!(g32.buf.get(3 * 4), 8.0);
            assert_eq!(g32.buf.get(2 * 4), i32::MIN as f64);
            assert_eq!(g32.kind_list(), vec!["routing_duplicate", "routing_negative", "routing_oob"]);
            let mut g64 = grid("I64", 9, 4);
            apply_routing(&mut g64, 8);
            let d = &g64.buf.data;
            assert_eq!(i64::from_le_bytes(d[6 * 32..6 * 32 + 8].try_into().unwrap()), 1i64 << 32);
            assert_eq!(i64::from_le_bytes(d[2 * 32..2 * 32 + 8].try_into().unwrap()), i64::MIN);
            assert_eq!(i64::from_le_bytes(d[4 * 32..4 * 32 + 8].try_into().unwrap()), i64::MAX);
        }

        #[test]
        fn norm_roles_report_the_intended_kinds() {
            let mut g = grid("F32", 3, 16);
            Role::Norm.init(&mut g, 8);
            Role::Norm.apply(&mut g, 8).unwrap();
            assert_eq!(g.kind_list(), vec!["norm_all_zero", "norm_below_min", "subnormal"]);
            // F16: the smallest subnormal norm (~2e-7) is above the 1e-12 floor
            let mut h = grid("F16", 3, 16);
            Role::Norm.init(&mut h, 8);
            Role::Norm.apply(&mut h, 8).unwrap();
            assert_eq!(h.kind_list(), vec!["norm_all_zero", "subnormal"]);
        }

        #[test]
        fn nansort_variants_and_grid_addressing() {
            let mut g = grid("F32", 6, 8);
            Role::NanSort.init(&mut g, 8);
            Role::NanSort.apply(&mut g, 8).unwrap();
            let bits = |r: i64, c: i64| u32::from_le_bytes(g.buf.data[((r * 8 + c) * 4) as usize..][..4].try_into().unwrap());
            assert_eq!(bits(1, 0), 0x7fc00000);
            assert_eq!(bits(1, 1), 0xffc00000);
            assert_eq!(bits(1, 2), 0x7fc00001);
            assert_eq!(bits(1, 3), 0x7f800001);
            assert_eq!(bits(5, 0), 0x7f800000);
            assert_eq!(bits(5, 1), 0xff800000);
            assert_eq!(grid_of(&[6, 8]), (6, 8));
            assert_eq!(grid_of(&[16]), (1, 16));
            assert_eq!(grid_of(&[2, 3, 4]), (6, 4));
        }

        #[test]
        fn option_normalization() {
            let ok = ProbeOptions { attempt_id: "a".into(), out_dir: "d".into(), ..Default::default() };
            let n = normalize(&ok).unwrap();
            assert_eq!((n.class.as_str(), n.format.as_str(), n.num_experts), ("all", "both", 8));
            for bad in [
                ProbeOptions { class: "class9".into(), ..ok.clone() },
                ProbeOptions { format: "onnx".into(), ..ok.clone() },
                ProbeOptions { num_experts: 1, ..ok.clone() },
                ProbeOptions { num_experts: (1 << 20) + 1, ..ok.clone() },
                ProbeOptions { attempt_id: String::new(), ..ok.clone() },
                ProbeOptions { out_dir: String::new(), ..ok.clone() },
            ] {
                assert!(normalize(&bad).is_err());
            }
        }

        #[test]
        fn base_selection_rules() {
            use crate::container::Layout;
            let t = |name: &str, dtype: &str, shape: Vec<usize>| {
                let w = dtype_size(dtype);
                let n: usize = shape.iter().product();
                TensorLocation {
                    name: name.into(),
                    dtype: dtype.into(),
                    shape: shape.clone(),
                    rows: shape[0],
                    cols: *shape.get(1).unwrap_or(&1),
                    logical_out: 0,
                    logical_in: 0,
                    offset: 0,
                    size: n as i64 * w,
                    gguf_column_major: false,
                    ggml_type: None,
                }
            };
            let l = Layout {
                format: "safetensors".into(),
                path: String::new(),
                file_size: 0,
                data_offset: 0,
                metadata: Default::default(),
                tensors: vec![
                    t("a.mlp.gate.weight", "I32", vec![2, 2]),
                    t("a.norm", "F32", vec![16]),
                    t("b.norm", "F32", vec![16]),
                    t("c.tid2eid", "I64", vec![2, 2]),
                ],
            };
            let c = base_select("class4", &l);
            let got: Vec<(&str, usize)> = c.iter().map(|x| (x.role.name(), x.tensor)).collect();
            assert_eq!(got, vec![("routing", 3), ("norm", 1), ("norm-sub", 2)]);
            assert!(base_select("class1", &l).is_empty()); // the gate is an integer tensor: not a router
        }
    }
}

// Re-exports used by the CLI.
pub use gopath::{abs as go_abs, clean as go_clean};

#[doc(hidden)]
pub mod internals {
    pub use super::fsx::lstat_not_exist;
    pub use super::gojson::{go_float, go_g, rfc3339_nano};
    pub use super::gopath::{dir as go_dir, path_match};
    pub use super::sanitize::{classify, finite_bound, go_hypot, Kind};
}

#[cfg(test)]
mod tests {
    use super::internals::*;

    #[test]
    fn go_float_matches_encoding_json() {
        assert_eq!(go_float(1e4), "10000");
        assert_eq!(go_float(1e-12), "1e-12");
        assert_eq!(go_float(1.5e-7), "1.5e-7");
        assert_eq!(go_float(1e-6), "0.000001");
        assert_eq!(go_float(1e21), "1e+21");
        assert_eq!(go_float(1.0001), "1.0001");
        assert_eq!(go_float(0.0), "0");
        assert_eq!(go_float(f64::MAX), "1.7976931348623157e+308");
        assert_eq!(go_float(-2.5e-9), "-2.5e-9");
    }

    #[test]
    fn go_g_matches_fmt() {
        assert_eq!(go_g(1e38), "1e+38");
        assert_eq!(go_g(100000.0), "100000");
        assert_eq!(go_g(1e6), "1e+06");
        assert_eq!(go_g(0.0001), "0.0001");
        assert_eq!(go_g(0.00001), "1e-05");
        assert_eq!(go_g(f64::INFINITY), "+Inf");
        assert_eq!(go_g(f64::NAN), "NaN");
        assert_eq!(go_g(-1e38), "-1e+38");
    }

    #[test]
    fn rfc3339_formatting() {
        use std::time::{Duration, UNIX_EPOCH};
        assert_eq!(rfc3339_nano(UNIX_EPOCH), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_nano(UNIX_EPOCH + Duration::new(1_700_000_000, 120_000_000)), "2023-11-14T22:13:20.12Z");
        assert_eq!(rfc3339_nano(UNIX_EPOCH + Duration::new(951_782_400, 1)), "2000-02-29T00:00:00.000000001Z");
    }

    #[test]
    fn path_match_cases() {
        assert_eq!(path_match("custom_*", "custom_route"), Ok(true));
        assert_eq!(path_match("*.weight", "blk.0.weight"), Ok(true));
        assert_eq!(path_match("a?c", "abc"), Ok(true));
        assert_eq!(path_match("a?c", "a/c"), Ok(false));
        assert_eq!(path_match("*", "a/b"), Ok(false));
        assert_eq!(path_match("[a-c]x", "bx"), Ok(true));
        assert_eq!(path_match("[^a-c]x", "bx"), Ok(false));
        assert_eq!(path_match("\\*", "*"), Ok(true));
        assert_eq!(path_match("[", ""), Err(()));
        assert_eq!(path_match("a[", ""), Err(()));
        assert_eq!(path_match("[]a]", "]"), Err(()));
        assert_eq!(path_match("x\\", ""), Err(()));
        assert_eq!(path_match("b*[", "abc"), Err(()));
        assert_eq!(path_match("missing*", "blk.0"), Ok(false));
    }

    #[test]
    fn classify_ieee_classes() {
        for b in 1u64..1024 {
            let (v, k) = classify(b, "F16");
            assert_eq!(v, b as f64 * 2f64.powi(-24));
            assert_eq!(k, Kind::Subnormal);
        }
        assert_eq!(classify(0x7e01, "F16").1, Kind::Nan);
        assert_eq!(classify(0x7c00, "F16").1, Kind::Inf);
        assert_eq!(classify(0xfc00, "F16").0, f64::NEG_INFINITY);
        assert_eq!(classify(0x8000, "BF16").1, Kind::Normal);
        assert_eq!(classify(0x7f7fffff, "F32").0, f32::MAX as f64);
        assert_eq!(classify(0x7ff8000000000001, "F64").1, Kind::Nan);
        assert_eq!(classify(1, "F64").1, Kind::Subnormal);
    }

    #[test]
    fn finite_bound_never_exceeds_limit() {
        for dtype in ["F16", "BF16", "F32", "F64"] {
            for limit in [0.1, 1.0001, 10000.0, 70000.0, f64::MAX] {
                let b = finite_bound(dtype, limit);
                let (v, k) = classify(b, dtype);
                assert!(k != Kind::Nan && k != Kind::Inf && v <= limit, "{} {} -> {}", dtype, limit, v);
            }
        }
        // F16 max finite is 65504
        assert_eq!(finite_bound("F16", 70000.0), 0x7bff);
    }

    #[test]
    fn hypot_special_cases() {
        assert_eq!(go_hypot(0.0, 0.0), 0.0);
        assert_eq!(go_hypot(3.0, 4.0), 5.0);
        assert_eq!(go_hypot(f64::INFINITY, f64::NAN), f64::INFINITY);
        assert!(go_hypot(f64::NAN, 1.0).is_nan());
    }
}
