//! # Cross-Rust Version Compatibility

pub mod ranges;
pub mod strings;

#[cfg(feature = "std")]
pub mod threads;
