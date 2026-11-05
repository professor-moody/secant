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
    2
}

/// Split a comma-separated list, dropping empty parts.
pub fn split_csv(s: &str) -> Vec<String> {
    s.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()
}
