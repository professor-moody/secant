//! Secant core. Self-contained: no dependency on any external internal crate.
//!
//! Modules mirror the Go reference implementation (`internal/wsp`) that serves
//! as the byte-exact differential conformance oracle. The `mp` module (`secant
//! audit-trojan` Marchenko-Pastur spectral outlier scanner) is a Rust-native
//! extension with no Go counterpart.

pub mod analyze;
pub mod builder;
pub mod container;
pub mod dtype;
pub mod error;
pub mod heal;
pub mod mp;
pub mod namespace;
pub mod numeric;
pub mod plan;
pub mod quant;
pub mod quant_scale;
pub mod read;
pub mod security;
pub mod sinks;
pub mod subspace;

pub use error::{Error, Result};
pub mod rotation;
