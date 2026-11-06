//! Static weight-value sink scanner (`secant scan-sinks`), a port of the
//! reference oracle's `wsp scan-sinks` (`internal/wsp/sinks.go`).
//!
//! A weight file that is perfectly format-valid still carries VALUES, and four
//! families of consumer code turn a hostile value into a memory-safety, DoS or
//! abort event: a float cast to an index (`FLOAT_TO_INDEX_CAST`), an unchecked
//! routing/index tensor (`UNVALIDATED_ROUTING_OR_INDEX_TENSOR`), a division by
//! a quantisation scale (`QUANT_SCALE_DIV_OR_OVERFLOW`), and a release
//! assertion over loaded values (`WEIGHT_VALUE_RELEASE_ASSERT`). This scanner
//! is a line-oriented HYPOTHESIS generator: it names a site and says whether a
//! guard-shaped expression is visible earlier in the same function. It does
//! not prove reachability or exploitability.
//!
//! Deterministic: files are visited in sorted order, findings are sorted by
//! (score desc, file, line, class), and the receipt carries a digest over every
//! scanned `(path, sha256)` pair, so the same tree gives the same receipt.
//!
//! Parity with the oracle: same file selection, rules, scores, ordering and
//! `encoding/json` layout. Recorded divergences:
//!   * `schema` is `secant.wsp-sink-scan.v1` (the oracle writes the legacy
//!     namespace), and the third `limitations` entry is worded without the
//!     oracle's internal rule references.
//!   * A line longer than 1 MiB: the oracle's `bufio.Scanner` stops at it and
//!     silently drops the rest of that file; Secant scans every line.
//!   * Regex classes are ASCII here (`regex::bytes`, Unicode off). Go's `(?i)`
//!     also folds U+212A (Kelvin) and U+017F (long s), and Go's `\s` excludes
//!     `\v`; neither occurs in real source identifiers.
//!   * `risk_score` rounding uses `numeric::fma_add`, so `--go-arm64-fma`
//!     reproduces the arm64 oracle's fused `f*10+0.5` exactly.

use crate::error::{Error, Result};
use crate::security::go_abs;
use crate::security::gojson::J;
use regex::bytes::{Regex, RegexBuilder};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::OnceLock;

pub const SINK_SCAN_SCHEMA: &str = "secant.wsp-sink-scan.v1";

pub const SINK_FLOAT_TO_INDEX_CAST: &str = "FLOAT_TO_INDEX_CAST";
pub const SINK_UNVALIDATED_INDEX: &str = "UNVALIDATED_ROUTING_OR_INDEX_TENSOR";
pub const SINK_QUANT_SCALE_DIV: &str = "QUANT_SCALE_DIV_OR_OVERFLOW";
pub const SINK_RELEASE_ASSERT: &str = "WEIGHT_VALUE_RELEASE_ASSERT";

/// Sink classes in report order.
pub const SINK_CLASSES: [&str; 4] = [SINK_FLOAT_TO_INDEX_CAST, SINK_UNVALIDATED_INDEX, SINK_QUANT_SCALE_DIV, SINK_RELEASE_ASSERT];

const MAX_FILE_BYTES: usize = 4 << 20;
const MAX_SCOPE_BACK: usize = 250;

pub const LIMITATIONS: [&str; 3] = [
    "line-oriented heuristic: macros, templates and cross-function guards are not resolved",
    "guard_present means a guard-shaped expression appears earlier in the same function scope; it is not a proof that the guard dominates the sink",
    "a hit is a hypothesis that needs a replay and a read of the source at the site; reachability from a weight file and product impact are separate claims",
];

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SinkScanOptions {
    pub source: String,
    /// Maximum ranked findings kept in `findings`; `<= 0` takes 100.
    pub top: i64,
    pub attempt_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SinkFinding {
    pub file: String,
    pub line: i64,
    pub sink_class: String,
    pub snippet: String,
    pub guard_present: bool,
    pub guard_kind: String,
    pub risk_score: f64,
    pub function: String,
    pub language: String,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SinkScanReport {
    pub schema: String,
    pub attempt_id: String,
    pub source: String,
    pub started_utc: String,
    pub finished_utc: String,
    pub files_scanned: i64,
    pub files_skipped: i64,
    pub lines_scanned: i64,
    pub tree_digest: String,
    pub total_findings: i64,
    pub class_counts: BTreeMap<String, i64>,
    pub unguarded_counts: BTreeMap<String, i64>,
    pub top: i64,
    pub truncated: bool,
    pub findings: Vec<SinkFinding>,
    pub limitations: Vec<String>,
    pub options: SinkScanOptions,
}

fn s_or_skip(v: &mut Vec<(&'static str, J)>, k: &'static str, s: &str) {
    if !s.is_empty() {
        v.push((k, J::s(s)));
    }
}

fn counts(m: &BTreeMap<String, i64>) -> J {
    J::Obj(m.iter().map(|(k, v)| (k.clone(), J::Int(*v))).collect())
}

impl SinkFinding {
    fn to_j(&self) -> J {
        let mut v = vec![
            ("file", J::s(&self.file)),
            ("line", J::Int(self.line)),
            ("sink_class", J::s(&self.sink_class)),
            ("snippet", J::s(&self.snippet)),
            ("guard_present", J::Bool(self.guard_present)),
        ];
        s_or_skip(&mut v, "guard_kind", &self.guard_kind);
        v.push(("risk_score", J::Float(self.risk_score)));
        s_or_skip(&mut v, "function", &self.function);
        v.push(("language", J::s(&self.language)));
        s_or_skip(&mut v, "notes", &self.notes);
        J::obj(v)
    }
}

impl SinkScanReport {
    /// The receipt as the oracle's `encoding/json` struct tree.
    pub fn to_j(&self) -> J {
        let mut v = vec![("schema", J::s(&self.schema))];
        s_or_skip(&mut v, "attempt_id", &self.attempt_id);
        v.extend([
            ("source", J::s(&self.source)),
            ("started_utc", J::s(&self.started_utc)),
            ("finished_utc", J::s(&self.finished_utc)),
            ("files_scanned", J::Int(self.files_scanned)),
            ("files_skipped", J::Int(self.files_skipped)),
            ("lines_scanned", J::Int(self.lines_scanned)),
            ("tree_digest_sha256", J::s(&self.tree_digest)),
            ("total_findings", J::Int(self.total_findings)),
            ("class_counts", counts(&self.class_counts)),
            ("unguarded_counts", counts(&self.unguarded_counts)),
            ("top", J::Int(self.top)),
            ("truncated", J::Bool(self.truncated)),
            ("findings", J::Arr(self.findings.iter().map(|f| f.to_j()).collect())),
            ("limitations", J::strs(&self.limitations)),
        ]);
        let mut o = vec![("source", J::s(&self.options.source)), ("top", J::Int(self.options.top))];
        s_or_skip(&mut o, "attempt_id", &self.options.attempt_id);
        v.push(("options", J::obj(o)));
        J::obj(v)
    }
    /// `json.MarshalIndent(rep, "", "  ")` (without the trailing newline).
    pub fn to_json_pretty(&self) -> String {
        self.to_j().pretty()
    }
}

// ------------------------------------------------------------------ patterns

struct Res {
    c_cast: Regex,
    static_cast: Regex,
    fn_cast: Regex,
    rust_cast: Regex,
    floaty: Regex,
    test_path: Regex,
    kernel_path: Regex,
    load_path: Regex,
    nan_guard: Regex,
    self_eq: Regex,
    quant_guard: Regex,
    check_macro: Regex,
    fn_start_c: Regex,
    fn_start_rust: Regex,
    idx_sub: Regex,
    idx_ptr: Regex,
    div_scale1: Regex,
    div_scale2: Regex,
    fp16_scale: Regex,
    loaded_value: Regex,
    cmp: Regex,
    clamp_call: Regex,
}

/// RE2-compatible byte regex: ASCII classes, as Go's `regexp` on source text.
fn re(p: &str) -> Regex {
    RegexBuilder::new(p).unicode(false).build().unwrap_or_else(|e| panic!("sink pattern {:?}: {}", p, e))
}

fn res() -> &'static Res {
    static R: OnceLock<Res> = OnceLock::new();
    R.get_or_init(|| {
        let cast_type = r"(?:u?int(?:8|16|32|64)_t|size_t|ssize_t|ptrdiff_t|intptr_t|int|unsigned|long(?:\s+long)?|short|index_t|idx_t)";
        let idx_names = r"tid2eid|expert_ids?|expert_idx|expert_indices|selected_experts|topk_ids|sorted_token_ids|token_idx|token_ids?|indices|row_splits|perm(?:utation)?";
        Res {
            c_cast: re(&format!(r"\(\s*{}\s*\)\s*[\w(*&\[~-]", cast_type)),
            static_cast: re(&format!(r"(?:static|reinterpret)_cast\s*<\s*(?:const\s+)?{}\s*>\s*\(", cast_type)),
            fn_cast: re(r"\b(?:lrintf?|llrintf?|lroundf?|llroundf?|nearbyintf?|rintf?)\s*\("),
            rust_cast: re(r"\bas\s+(?:i|u)(?:8|16|32|64|size)\b"),
            floaty: re(r"(?i)\b(?:logits?|probs?|probabilit\w*|scores?|softmax|topk|top_k|gate|router|routing_w\w*|weights?|scales?|temperature|float|half|bfloat16?|bf16|fp16|fp32|f32|f16|floorf?|ceilf?|roundf?|truncf?|expf?|logf?|sqrtf?|tanhf?|powf?|fabsf?|fmaf?|rsqrt\w*|GGML_FP16_TO_FP32|__half2float|__bfloat162float|static_cast\s*<\s*float|\d+\.\d*f\b|\d+e[+-]?\d+f?\b)"),
            test_path: re(r"(?i)(?:^|/)(?:tests?|testing|testdata|bench(?:mark)?s?|examples?|samples?|docs?|tools/(?:perf|bench)\w*)/|_test\.(?:cc|cpp|c|rs)$|(?:^|/)test_\w+\.(?:cc|cpp|c)$"),
            kernel_path: re(r"(?i)(?:kernel|cuda|\.cu$|\.cuh$|moe|expert|router|routing|ssm|delta|quant|dequant|mamba|attention|sampl|topk|gate|csrc|ops?/)"),
            load_path: re(r"(?i)(?:load|loader|read|checkpoint|bundle|restore|safetensor|gguf|serializ|deserializ|import|model_|weights?)"),
            nan_guard: re(r"(?i)\b(?:std::)?(?:isnan|isfinite|isinf|is_nan|is_finite|is_infinite)\s*\(|\.is_nan\(\)|\.is_finite\(\)|nan_to_num"),
            self_eq: re(r"!\s*\(\s*([A-Za-z_]\w*)\s*==\s*([A-Za-z_]\w*)\s*\)|\b([A-Za-z_]\w*)\s*!=\s*([A-Za-z_]\w*)\b"),
            quant_guard: re(r"(?i)(?:\b(?:d|dmin|scale\w*|amax|id|iscale)\s*(?:==|!=|<=|>=|<|>)\s*[-+]?(?:0|0\.0f?|eps\w*|EPS\w*|FLT_\w+|1e-\d+f?)|\bfabsf?\s*\([^)]*\)\s*(?:<|>|<=|>=)|isfinite|isnan|\bepsilon\b|\b(?:d|scale\w*|amax)\s*\?|\?\s*1\.0f?\s*/)"),
            check_macro: re(r"\b(?:CHECK(?:_EQ|_NE|_LT|_LE|_GT|_GE)?|GGML_ASSERT|TORCH_CHECK\w*|TORCH_INTERNAL_ASSERT|ORT_ENFORCE|ORT_THROW|TF_CHECK_OK|TF_QCHECK_OK|LOG\s*\(\s*FATAL\s*\)|OP_REQUIRES\w*|RET_CHECK\w*|abort|assert|DCHECK\w*)\b"),
            fn_start_c: re(r"^[A-Za-z_~][\w:<>,\s\*&~]*\b([A-Za-z_~][\w:~]*)\s*\([^;]*$"),
            fn_start_rust: re(r#"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:unsafe\s+)?(?:extern\s+"C"\s+)?fn\s+(\w+)"#),
            idx_sub: re(&format!(r"\[\s*[^\]\[\n]*\b({})\w*(?:\s*\[[^\]\n]*\])?[^\]\n]*\]", idx_names)),
            idx_ptr: re(&format!(r"\*\s*\(\s*\w+\s*\+\s*[^)\n]*\b({})\b[^)\n]*\)", idx_names)),
            div_scale1: re(r"\b1(?:\.0*)?f?\s*/\s*\(?\s*(?:[A-Za-z_]\w*(?:->|\.))?(d|dmin|dm|amax|absmax|iscale|inv_scale|id|\w*scale\w*)\s*\)?"),
            div_scale2: re(r"[\w)\]]\s*/\s*\(?\s*(?:[A-Za-z_]\w*(?:->|\.))?(scale\w*|\w*_scale\w*|dmin|amax|absmax|\w+\.d|\w+->d|\w+\.dm|\w+->dm)\b"),
            fp16_scale: re(r"(?:GGML_(?:COMPUTE_)?FP16_TO_FP32|GGML_CPU_FP16_TO_FP32|ggml_fp16_to_fp32|__half2float|half_to_float|fp16_to_fp32|F16C|half2float|f16_to_f32|f16::to_f32|half::f16)\s*\(?\s*[^;)]*\b(d|dmin|dm|scale\w*|\w*scale\w*)\b"),
            loaded_value: re(r"(?i)\b(?:shape|dims?|ne\[|nb\[|numel|n_experts?|num_experts|dtype|type|offset|n_dims|n_tensors|idx|index|expert|size|count|rank|ggml_nbytes|data)\b"),
            cmp: re(r"(?:<|>|<=|>=|==|!=)"),
            clamp_call: re(r"(?i)\b(?:clamp|min|max|MIN|MAX)\s*\("),
        }
    })
}

/// Source extensions and their language labels (oracle `sinkExt`).
pub fn language_of_ext(ext: &str) -> Option<&'static str> {
    Some(match ext {
        ".c" | ".h" => "c",
        ".cc" | ".cpp" | ".cxx" | ".hpp" | ".hh" | ".hxx" | ".inc" | ".inl" | ".mm" | ".metal" => "cpp",
        ".cu" | ".cuh" => "cuda",
        ".rs" => "rust",
        _ => return None,
    })
}

/// Go `filepath.Ext`: the suffix from the final `.` of the final element.
fn go_ext(path: &str) -> &str {
    let b = path.as_bytes();
    let mut i = b.len();
    while i > 0 && b[i - 1] != b'/' {
        if b[i - 1] == b'.' {
            return &path[i - 1..];
        }
        i -= 1;
    }
    ""
}

// ------------------------------------------------------------------ byte helpers

/// Go `strings.TrimSpace` over bytes (ASCII and Unicode White_Space; invalid
/// UTF-8 is never space).
fn trim_space(b: &[u8]) -> &[u8] {
    /// Width of the whitespace character starting at `i`, if it is one.
    fn space_at(b: &[u8], i: usize) -> Option<usize> {
        let c = *b.get(i)?;
        if matches!(c, b'\t' | b'\n' | 0x0B | 0x0C | b'\r' | b' ') {
            return Some(1);
        }
        let w = match c {
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => return None,
        };
        let ch = std::str::from_utf8(b.get(i..i + w)?).ok()?.chars().next()?;
        if ch.is_whitespace() {
            Some(w)
        } else {
            None
        }
    }
    let mut lo = 0;
    while let Some(w) = space_at(b, lo) {
        lo += w;
    }
    let mut hi = b.len();
    'outer: while hi > lo {
        for w in 1..=4 {
            if hi >= lo + w && space_at(b, hi - w) == Some(w) {
                hi -= w;
                continue 'outer;
            }
        }
        break;
    }
    &b[lo..hi]
}

fn contains(h: &[u8], n: &[u8]) -> bool {
    h.windows(n.len()).any(|w| w == n)
}

/// Go `encoding/json` string semantics for raw bytes: every byte of an
/// invalid UTF-8 sequence becomes U+FFFD.
fn go_lossy(b: &[u8]) -> String {
    let mut out = String::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match std::str::from_utf8(&b[i..]) {
            Ok(s) => {
                out.push_str(s);
                break;
            }
            Err(e) => {
                let ok = e.valid_up_to();
                out.push_str(std::str::from_utf8(&b[i..i + ok]).unwrap());
                out.push('\u{FFFD}');
                i += ok + 1;
            }
        }
    }
    out
}

fn snippet_of(text: &[u8]) -> String {
    let replaced: Vec<u8> = text.iter().map(|&c| if c == b'\t' { b' ' } else { c }).collect();
    let t = trim_space(&replaced);
    if t.len() > 200 {
        let mut s = go_lossy(&t[..200]);
        s.push_str("...");
        s
    } else {
        go_lossy(t)
    }
}

fn round1(f: f64) -> f64 {
    crate::numeric::fma_add(0.5, f, 10.0) as i64 as f64 / 10.0
}

// ------------------------------------------------------------------ per line

struct Line {
    n: i64,
    text: Vec<u8>,
    code: Vec<u8>,
}

/// Blank comment bodies and string-literal contents, keeping columns and
/// carrying block-comment state across lines (oracle `stripComments`).
fn strip_comments(line: &[u8], in_block: &mut bool) -> Vec<u8> {
    let mut b = line.to_vec();
    let mut i = 0;
    while i < b.len() {
        if *in_block {
            if i + 1 < b.len() && b[i] == b'*' && b[i + 1] == b'/' {
                b[i] = b' ';
                b[i + 1] = b' ';
                i += 2;
                *in_block = false;
                continue;
            }
            b[i] = b' ';
            i += 1;
            continue;
        }
        if i + 1 < b.len() && b[i] == b'/' && b[i + 1] == b'*' {
            b[i] = b' ';
            b[i + 1] = b' ';
            i += 2;
            *in_block = true;
        } else if i + 1 < b.len() && b[i] == b'/' && b[i + 1] == b'/' {
            for c in b.iter_mut().skip(i) {
                *c = b' ';
            }
            i = b.len();
        } else if b[i] == b'"' {
            let mut j = i + 1;
            while j < b.len() && b[j] != b'"' {
                if b[j] == b'\\' {
                    j += 1;
                }
                j += 1;
            }
            let mut k = i + 1;
            while k < j && k < b.len() {
                b[k] = b' ';
                k += 1;
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    b
}

/// `bufio.ScanLines`: split on `\n`, drop one trailing `\r`, no empty final
/// token. Unlike the oracle there is no 1 MiB line cap (see module docs).
fn split_lines(data: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, &c) in data.iter().enumerate() {
        if c == b'\n' {
            out.push(drop_cr(&data[start..i]));
            start = i + 1;
        }
    }
    if start < data.len() {
        out.push(drop_cr(&data[start..]));
    }
    out
}

fn drop_cr(b: &[u8]) -> &[u8] {
    if b.last() == Some(&b'\r') {
        &b[..b.len() - 1]
    } else {
        b
    }
}

fn find_str<'a>(r: &Regex, s: &'a [u8]) -> &'a [u8] {
    r.find(s).map(|m| m.as_bytes()).unwrap_or(b"")
}

fn scope_has(scope: &[Line], code: &[u8], r: &Regex, nan: bool) -> (bool, String) {
    let res = res();
    let check = |s: &[u8]| -> Option<String> {
        let m = find_str(r, s);
        if !m.is_empty() {
            return Some(go_lossy(trim_space(m)));
        }
        if nan {
            for c in res.self_eq.captures_iter(s) {
                let g = |i: usize| c.get(i).map(|m| m.as_bytes()).unwrap_or(b"");
                if (!g(1).is_empty() && g(1) == g(2)) || (!g(3).is_empty() && g(3) == g(4)) {
                    return Some(go_lossy(trim_space(g(0))));
                }
            }
        }
        None
    };
    if let Some(m) = check(code) {
        return (true, m);
    }
    for l in scope.iter().rev() {
        if let Some(m) = check(&l.code) {
            return (true, m);
        }
    }
    (false, String::new())
}

fn float_to_index(code: &[u8], lang: &str) -> bool {
    let r = res();
    if lang == "rust" {
        return r.rust_cast.is_match(code) && r.floaty.is_match(code);
    }
    let loc = r.c_cast.find(code).or_else(|| r.static_cast.find(code)).or_else(|| r.fn_cast.find(code));
    match loc {
        Some(m) => r.floaty.is_match(&code[m.start()..]),
        None => false,
    }
}

fn idx_subscript(code: &[u8]) -> Option<String> {
    let r = res();
    r.idx_sub.captures(code).or_else(|| r.idx_ptr.captures(code)).and_then(|c| c.get(1)).map(|m| go_lossy(m.as_bytes()))
}

fn index_guard(scope: &[Line], code: &[u8], name: &str) -> (bool, String) {
    let r = res();
    let id = re(&format!(r"\b\w*{}\w*\b", regex::escape(name)));
    if contains(code, b"?") && r.cmp.is_match(code) {
        return (true, "same-line-conditional".into());
    }
    for l in scope.iter().rev() {
        let c = &l.code;
        if !id.is_match(c) {
            continue;
        }
        if r.check_macro.is_match(c) && r.cmp.is_match(c) {
            return (true, "check-macro".into());
        }
        if contains(c, b"if") && r.cmp.is_match(c) {
            return (true, "if-compare".into());
        }
        if r.clamp_call.is_match(c) {
            return (true, "clamp".into());
        }
    }
    (false, String::new())
}

fn quant_scale(code: &[u8]) -> Option<(&'static str, &'static str)> {
    let r = res();
    if r.div_scale1.is_match(code) {
        return Some(("reciprocal of quant scale", "div"));
    }
    if r.div_scale2.is_match(code) {
        return Some(("division by scale", "div"));
    }
    if r.fp16_scale.is_match(code) {
        return Some(("fp16 scale conversion", "conv"));
    }
    None
}

/// Oracle `releaseAssert`, bug-compatible: `assert`, `DCHECK*`, `OP_REQUIRES*`
/// and `RET_CHECK*` are not release aborts; a bare `abort` IS reported (the
/// oracle's comment says otherwise, its code reports it).
fn release_assert(code: &[u8]) -> String {
    let m = go_lossy(find_str(&res().check_macro, code));
    if m.is_empty()
        || m == "assert"
        || m == "OP_REQUIRES"
        || m.starts_with("DCHECK")
        || m.starts_with("OP_REQUIRES")
        || m.starts_with("RET_CHECK")
    {
        return String::new();
    }
    m
}

fn enclosing_function(lines: &[Line], i: usize, lang: &str) -> (String, usize) {
    let r = res();
    let lo = i.saturating_sub(MAX_SCOPE_BACK);
    let mut k = i as i64;
    while k >= lo as i64 {
        let ku = k as usize;
        k -= 1;
        let c = &lines[ku].code;
        if lang == "rust" {
            if let Some(m) = r.fn_start_rust.captures(c) {
                return (go_lossy(&m[1]), ku);
            }
            continue;
        }
        if c.is_empty() || matches!(c[0], b' ' | b'\t' | b'#' | b'}' | b'{') {
            continue;
        }
        let t = trim_space(c);
        let skip = t.ends_with(b";")
            || t.starts_with(b"//")
            || t.starts_with(b"using ")
            || t.starts_with(b"typedef")
            || t.starts_with(b"namespace")
            || t.starts_with(b"struct ")
            || t.starts_with(b"class ")
            || t.starts_with(b"enum ")
            || (t.starts_with(b"template") && !contains(t, b"("));
        if skip {
            continue;
        }
        if let Some(m) = r.fn_start_c.captures(c) {
            let open = (ku..=ku + 3).take_while(|&j| j < lines.len()).any(|j| contains(&lines[j].code, b"{"));
            if open {
                return (go_lossy(&m[1]), ku);
            }
        }
    }
    (String::new(), lo)
}

/// Scan one file's bytes; returns its findings and the line count.
pub fn scan_sink_file(rel: &str, lang: &str, data: &[u8]) -> (Vec<SinkFinding>, i64) {
    let r = res();
    let mut in_block = false;
    let lines: Vec<Line> = split_lines(data)
        .into_iter()
        .enumerate()
        .map(|(i, t)| Line { n: i as i64 + 1, text: t.to_vec(), code: strip_comments(t, &mut in_block) })
        .collect();
    let mut out = Vec::new();
    let test_path = r.test_path.is_match(rel.as_bytes());
    let kernel_path = r.kernel_path.is_match(rel.as_bytes());
    let load_path = r.load_path.is_match(rel.as_bytes());
    for (i, ln) in lines.iter().enumerate() {
        let code = &ln.code[..];
        let tc = trim_space(code);
        if tc.is_empty() || tc.starts_with(b"#include") {
            continue;
        }
        let (func, fn_start) = enclosing_function(&lines, i, lang);
        let mut add = |class: &str, base: f64, guard: bool, kind: String, notes: String| {
            let mut score = base;
            let mut notes = notes;
            if kernel_path {
                score += 1.5;
            }
            if load_path {
                score += 0.5;
            }
            if guard {
                score *= 0.25;
            }
            if test_path {
                score *= 0.2;
            }
            if lang == "rust" && class == SINK_FLOAT_TO_INDEX_CAST {
                score *= 0.4; // `as` on floats saturates and maps NaN to 0 in Rust: no UB
                notes = String::from_utf8(
                    trim_space(format!("{} rust float-to-int `as` saturates (NaN->0); logic error, not UB", notes).as_bytes()).to_vec(),
                )
                .unwrap();
            }
            if score > 10.0 {
                score = 10.0;
            }
            out.push(SinkFinding {
                file: rel.to_string(),
                line: ln.n,
                sink_class: class.to_string(),
                snippet: snippet_of(&ln.text),
                guard_present: guard,
                guard_kind: kind,
                risk_score: round1(score),
                function: func.clone(),
                language: lang.to_string(),
                notes,
            });
        };
        let scope = &lines[fn_start.min(i)..i];
        if float_to_index(code, lang) {
            let (guard, gk) = scope_has(scope, code, &r.nan_guard, true);
            add(SINK_FLOAT_TO_INDEX_CAST, 6.0, guard, if guard { gk } else { String::new() }, String::new());
        }
        if let Some(m) = idx_subscript(code) {
            let (guard, kind) = index_guard(scope, code, &m);
            add(SINK_UNVALIDATED_INDEX, 6.5, guard, kind, format!("index name: {}", m));
        }
        if let Some((m, divisor)) = quant_scale(code) {
            let (guard, kind) = scope_has(scope, code, &r.quant_guard, false);
            add(SINK_QUANT_SCALE_DIV, if divisor == "conv" { 3.0 } else { 7.0 }, guard, kind, m.to_string());
        }
        let mac = release_assert(code);
        let loaded = r.loaded_value.is_match(code);
        if !mac.is_empty() && (load_path || kernel_path || loaded) {
            add(SINK_RELEASE_ASSERT, if loaded { 5.5 } else { 4.0 }, false, String::new(), format!("macro: {}", mac));
        }
    }
    (out, lines.len() as i64)
}

fn rfc3339_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    crate::security::internals::rfc3339_nano(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs))
}

fn io_msg(e: &std::io::Error) -> String {
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

/// Walk `dir` like `filepath.WalkDir`: entries in name order, hidden and VCS
/// directories pruned, only regular files with a known extension kept. An
/// unreadable directory counts as one skipped file.
fn walk(dir: &std::path::Path, files: &mut Vec<String>, skipped: &mut i64) {
    let rd = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => {
            *skipped += 1;
            return;
        }
    };
    let mut entries: Vec<(Vec<u8>, std::fs::DirEntry)> = Vec::new();
    for e in rd {
        match e {
            Ok(e) => entries.push((std::os::unix::ffi::OsStrExt::as_bytes(e.file_name().as_os_str()).to_vec(), e)),
            Err(_) => {
                *skipped += 1;
                return;
            }
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, e) in entries {
        let ft = match e.file_type() {
            Ok(t) => t,
            Err(_) => {
                *skipped += 1;
                continue;
            }
        };
        let path = e.path();
        if ft.is_dir() {
            if matches!(name.as_slice(), b".git" | b"node_modules" | b".hg" | b".svn") || name.starts_with(b".") {
                continue;
            }
            walk(&path, files, skipped);
            continue;
        }
        let p = path.to_string_lossy().into_owned();
        if ft.is_file() && language_of_ext(&go_ext(&p).to_lowercase()).is_some() {
            files.push(p);
        }
    }
}

/// Walk a source tree and return the ranked candidate sinks (oracle `SinkScan`).
pub fn sink_scan(o: &SinkScanOptions) -> Result<SinkScanReport> {
    if o.source.is_empty() {
        return Err(Error::msg("--source is required"));
    }
    let st = std::fs::metadata(&o.source).map_err(|e| Error::msg(format!("source {}: stat {}: {}", o.source, o.source, io_msg(&e))))?;
    if !st.is_dir() {
        return Err(Error::msg(format!("source {} is not a directory", o.source)));
    }
    let mut o = o.clone();
    if o.top <= 0 {
        o.top = 100;
    }
    let root = go_abs(&o.source).map_err(|e| Error::msg(io_msg(&e)))?;
    let mut rep = SinkScanReport {
        schema: SINK_SCAN_SCHEMA.into(),
        attempt_id: o.attempt_id.clone(),
        source: root.clone(),
        started_utc: rfc3339_now(),
        finished_utc: String::new(),
        files_scanned: 0,
        files_skipped: 0,
        lines_scanned: 0,
        tree_digest: String::new(),
        total_findings: 0,
        class_counts: SINK_CLASSES.iter().map(|c| (c.to_string(), 0)).collect(),
        unguarded_counts: SINK_CLASSES.iter().map(|c| (c.to_string(), 0)).collect(),
        top: o.top,
        truncated: false,
        findings: vec![],
        limitations: LIMITATIONS.iter().map(|s| s.to_string()).collect(),
        options: o.clone(),
    };
    let mut files = Vec::new();
    walk(std::path::Path::new(&root), &mut files, &mut rep.files_skipped);
    files.sort();
    let mut digest = Sha256::new();
    let mut all: Vec<SinkFinding> = Vec::new();
    let prefix = if root.ends_with('/') { root.clone() } else { format!("{}/", root) };
    for p in &files {
        let rel = p.strip_prefix(&prefix).unwrap_or(p).to_string();
        let data = match std::fs::read(p) {
            Ok(d) if d.len() <= MAX_FILE_BYTES => d,
            _ => {
                rep.files_skipped += 1;
                continue;
            }
        };
        digest.update(format!("{} {}\n", rel, crate::container::sha256_hex(&data)).as_bytes());
        rep.files_scanned += 1;
        let lang = language_of_ext(&go_ext(p).to_lowercase()).unwrap();
        let (fs, n) = scan_sink_file(&rel, lang, &data);
        rep.lines_scanned += n;
        all.extend(fs);
    }
    rep.tree_digest = crate::container::hex(&digest.finalize());
    all.sort_by(|a, b| {
        b.risk_score
            .partial_cmp(&a.risk_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.file.as_bytes().cmp(b.file.as_bytes()))
            .then(a.line.cmp(&b.line))
            .then_with(|| a.sink_class.cmp(&b.sink_class))
    });
    for f in &all {
        *rep.class_counts.entry(f.sink_class.clone()).or_insert(0) += 1;
        if !f.guard_present {
            *rep.unguarded_counts.entry(f.sink_class.clone()).or_insert(0) += 1;
        }
    }
    rep.total_findings = all.len() as i64;
    if all.len() as i64 > o.top {
        rep.truncated = true;
        all.truncate(o.top as usize);
    }
    rep.findings = all;
    rep.finished_utc = rfc3339_now();
    Ok(rep)
}

/// Write the receipt (indented JSON + newline) to a NEW file, creating
/// parent directories (oracle `WriteSinkScan`).
pub fn write_sink_scan(dst: &str, rep: &SinkScanReport) -> Result<()> {
    use std::io::Write;
    let p = std::path::Path::new(dst);
    if let Some(parent) = p.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| Error::msg(format!("mkdir {}: {}", parent.display(), io_msg(&e))))?;
        }
    }
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(p)
        .map_err(|e| Error::msg(format!("open {}: {}", dst, io_msg(&e))))?;
    let mut b = rep.to_json_pretty();
    b.push('\n');
    f.write_all(b.as_bytes()).map_err(|e| Error::msg(format!("write {}: {}", dst, io_msg(&e))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classes(src: &str, rel: &str, lang: &str) -> Vec<(i64, String, bool)> {
        scan_sink_file(rel, lang, src.as_bytes()).0.into_iter().map(|f| (f.line, f.sink_class, f.guard_present)).collect()
    }

    #[test]
    fn each_sink_class_fires_on_its_shape_and_honours_guards() {
        let src = "static int route(const float * logits, int n) {\n    int e = (int) logits[0];\n    return e;\n}\nstatic int routed(const float * logits, int n) {\n    if (std::isnan(logits[0])) return 0;\n    int e = (int) logits[0];\n    return e;\n}\nvoid moe(const int * expert_ids, float * out, const float * w) {\n    out[0] = w[expert_ids[0]];\n}\nvoid dq(const block_q8_0 * x, float * y) {\n    const float id = 1.0f / x->d;\n    y[0] = id;\n}\nvoid load(int n_dims) {\n    GGML_ASSERT(n_dims <= 4);\n}\n";
        let got = classes(src, "src/ggml.c", "c");
        assert!(got.contains(&(2, SINK_FLOAT_TO_INDEX_CAST.into(), false)), "{:?}", got);
        assert!(got.contains(&(7, SINK_FLOAT_TO_INDEX_CAST.into(), true)), "isnan guard in scope: {:?}", got);
        assert!(got.contains(&(11, SINK_UNVALIDATED_INDEX.into(), false)), "{:?}", got);
        assert!(got.contains(&(14, SINK_QUANT_SCALE_DIV.into(), false)), "{:?}", got);
        assert!(got.contains(&(18, SINK_RELEASE_ASSERT.into(), false)), "{:?}", got);
        // comments and strings are blanked before matching
        assert!(classes("void f(){\n  // int e = (int) logits[0];\n  puts(\"(int) logits\");\n}\n", "a.c", "c").is_empty());
    }

    #[test]
    fn rust_casts_are_downweighted_and_scores_follow_the_path_rules() {
        let (f, n) = scan_sink_file("src/router.rs", "rust", b"fn pick(logits: &[f32]) -> usize {\n    logits[0] as usize\n}\n");
        assert_eq!(n, 3);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].function, "pick");
        // base 6 + kernel path 1.5 (router), * 0.4 for Rust = 3.0
        assert_eq!(f[0].risk_score, 3.0);
        assert!(f[0].notes.contains("saturates"));
        // the same C line under tests/ is scaled by 0.2: 6 * 0.2 = 1.2
        let (f, _) = scan_sink_file("tests/a.c", "c", b"int f(float *logits) {\n  return (int) logits[0];\n}\n");
        assert_eq!(f[0].risk_score, 1.2);
    }

    #[test]
    fn a_line_over_one_mib_does_not_truncate_the_file() {
        // Recorded divergence: the oracle's bufio.Scanner stops at a >1 MiB line
        // and drops the rest of the file; every line is scanned here.
        let mut src = b"int f(float *logits) {\n".to_vec();
        src.extend(std::iter::repeat_n(b'x', (1 << 20) + 7));
        src.extend_from_slice(b"\n  return (int) logits[0];\n}\n");
        let (f, n) = scan_sink_file("a.c", "c", &src);
        assert_eq!(n, 4);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].line, 3);
    }

    #[test]
    fn go_helpers_match_the_oracle_conventions() {
        assert_eq!(go_ext("a/b.CU"), ".CU");
        assert_eq!(go_ext("a.b/c"), "");
        assert_eq!(go_ext("a/.c"), ".c");
        assert_eq!(trim_space(b" \t x y \r\n"), b"x y");
        assert_eq!(trim_space("\u{a0}x\u{3000}".as_bytes()), b"x");
        assert_eq!(go_lossy(b"a\xe2\x82b"), "a\u{fffd}\u{fffd}b");
        assert_eq!(split_lines(b"a\r\nb\n\nc"), vec![&b"a"[..], b"b", b"", b"c"]);
        assert_eq!(split_lines(b"a\n"), vec![&b"a"[..]]);
        assert_eq!(round1(0.35), 0.4);
        assert_eq!(snippet_of(&[b'x'; 300]).len(), 203);
        let mut blk = false;
        assert_eq!(strip_comments(b"a /* b", &mut blk), b"a     ");
        assert!(blk);
        assert_eq!(strip_comments(b"c */ d \"e\\\"f\" g", &mut blk), b"     d \"    \" g");
    }
}
