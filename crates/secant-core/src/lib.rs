//! Secant core. Self-contained: no dependency on any Crucible-internal crate.
//!
//! Modules mirror the Go reference implementation (`internal/wsp`) that serves
//! as the byte-exact differential conformance oracle.

pub mod builder;
pub mod container;
pub mod dtype;
pub mod error;
pub mod heal;
pub mod numeric;
pub mod plan;
pub mod quant;
pub mod read;
pub mod security;
pub mod subspace;

pub use error::{Error, Result};
