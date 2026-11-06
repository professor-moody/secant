//! CLI helpers shared by every subcommand module.

use secant_core::Error;
use serde::Serialize;

/// Indented JSON (two spaces, trailing newline), as the oracle's `emitJSON`.
pub fn emit_json<T: Serialize>(v: &T) -> Result<(), Error> {
    let s = json_pretty(v)?;
    print!("{}", s);
    Ok(())
}

pub fn json_pretty<T: Serialize>(v: &T) -> Result<String, Error> {
    let mut s = serde_json::to_string_pretty(v).map_err(|e| Error::msg(e.to_string()))?;
    s.push('\n');
    Ok(s)
}

/// Print `error: <msg>` to stderr and return the oracle-compatible exit code
/// (1 = gate failed, 2 = usage / I-O / undeterminable).
pub fn fail(e: &Error) -> i32 {
    eprintln!("Error: {}", e);
    e.exit_code()
}

pub fn usage(msg: &str) -> i32 {
    eprintln!("Error: {}", msg);
    eprintln!("Hint: run `secant <command> --help` for options and examples.");
    2
}

pub fn section(title: &str) {
    println!("\n{}\n{}", title, "=".repeat(title.len()));
}

pub fn field(label: &str, value: impl std::fmt::Display) {
    println!("  {:<24} {}", label, value);
}

pub fn badge(pass: bool) -> &'static str {
    if pass {
        "[PASS]"
    } else {
        "[FAIL]"
    }
}

pub fn bytes(n: u64) -> String {
    let mut size = n as f64;
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut unit = 0;
    while size >= 1024.0 && unit + 1 < units.len() {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} B", n)
    } else {
        format!("{:.2} {} ({} bytes)", size, units[unit], n)
    }
}

pub fn table(headers: &[&str], rows: &[Vec<String>]) {
    let widths: Vec<usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| rows.iter().filter_map(|r| r.get(i)).map(|v| v.chars().count()).max().unwrap_or(0).max(h.len()))
        .collect();
    let print_row = |row: &[String]| {
        for (i, cell) in row.iter().enumerate() {
            print!("  {:width$}", cell, width = widths[i]);
        }
        println!();
    };
    print_row(&headers.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    print_row(&widths.iter().map(|n| "-".repeat(*n)).collect::<Vec<_>>());
    for row in rows {
        print_row(row);
    }
}

/// Render nested diagnostic values as labeled fields, never as raw JSON.
pub fn details(value: &serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, value) in map {
                let label = key.replace('_', " ");
                match value {
                    serde_json::Value::Object(_) => {
                        section(&label);
                        details(value);
                    }
                    serde_json::Value::Array(values) => {
                        section(&label);
                        for (i, v) in values.iter().enumerate() {
                            field("Entry", i + 1);
                            details(v);
                        }
                    }
                    // A boolean can describe a mode or observation, not a gate.
                    // Only command-specific gate renderers should use PASS/FAIL.
                    serde_json::Value::Bool(b) => field(&label, b),
                    serde_json::Value::Number(n) if key == "bytes" => field(&label, bytes(n.as_u64().unwrap_or(0))),
                    serde_json::Value::Number(n) if n.is_f64() => field(&label, format!("{:.6e}", n.as_f64().unwrap())),
                    serde_json::Value::String(s) => field(&label, s),
                    serde_json::Value::Null => field(&label, "UNKNOWN"),
                    _ => field(&label, value),
                }
            }
        }
        serde_json::Value::String(s) => field("Value", s),
        _ => field("Value", value),
    }
}

/// Split a comma-separated list, dropping empty parts.
pub fn split_csv(s: &str) -> Vec<String> {
    s.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()
}
