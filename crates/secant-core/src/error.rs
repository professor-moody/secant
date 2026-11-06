//! Error type with the same three-way exit contract as the Go oracle
//! (its `wsp` subcommands): 0 = the check ran and passed; 1 = the check ran and its
//! gate failed; 2 = usage, I/O, unsupported input, or a result that could not
//! be determined.

use std::fmt;

#[derive(Debug, Clone)]
pub enum Error {
    /// Usage, I/O, unsupported input, or undeterminable result (exit 2).
    Msg(String),
    /// A gate that ran and failed (leakage over tolerance, verification
    /// failure after edits). Exit 1.
    Gate(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn msg(s: impl Into<String>) -> Self {
        Error::Msg(s.into())
    }
    pub fn gate(s: impl Into<String>) -> Self {
        Error::Gate(s.into())
    }
    pub fn exit_code(&self) -> i32 {
        match self {
            Error::Msg(_) => 2,
            Error::Gate(_) => 1,
        }
    }
    pub fn message(&self) -> &str {
        match self {
            Error::Msg(s) | Error::Gate(s) => s,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Msg(e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Msg(e.to_string())
    }
}

/// `bail!("fmt", args)` returns `Err(Error::Msg(..))`.
#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => { return Err($crate::error::Error::Msg(format!($($arg)*))) };
}

/// `fail!` returns a gate failure (exit 1).
#[macro_export]
macro_rules! gate_fail {
    ($($arg:tt)*) => { return Err($crate::error::Error::Gate(format!($($arg)*))) };
}
