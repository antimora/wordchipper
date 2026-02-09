//! # Vocab Support Tooling

mod pattern_tools;
mod resource_tools;
mod specials_tools;
pub mod testing;
pub mod validators;

#[doc(inline)]
pub use resource_tools::ConstUrlResource;
#[doc(inline)]
pub use specials_tools::{format_carrot, format_reserved_carrot};
