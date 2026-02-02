//! # Thread Regex Pool
#![allow(unused)]

use crate::concurrency::pool_toy;
use crate::concurrency::pool_toy::PoolToy;
use crate::regex::regex_supplier::RegexSupplier;
use crate::regex::regex_wrapper::RegexWrapper;
use core::fmt::Debug;
use core::num::NonZero;
use std::sync::atomic::AtomicUsize;

/// Interior-Mutable Thread-Local Regex Pool
///
/// In HPC applications, under some loads, interior buffers in compiled regex
/// can block. This pool exists to mitigate that, by cloning regex-per-thread.
#[derive(Debug, Clone)]
pub struct RegexWrapperPool {
    pool: PoolToy<RegexWrapper>,
}

impl From<RegexWrapper> for RegexWrapperPool {
    fn from(regex: RegexWrapper) -> Self {
        Self::new(regex)
    }
}

impl RegexWrapperPool {
    /// Create a new `RegexPool`
    ///
    /// ## Arguments
    /// * `regex` - The regex to pool.
    ///
    /// ## Returns
    /// A new `RegexWrapperPool` instance.
    pub fn new(regex: RegexWrapper) -> Self {
        let pool = PoolToy::init(regex, None);

        Self { pool }
    }

    /// Returns the number of regex instances in the pool.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.pool.len()
    }
}

impl RegexSupplier for RegexWrapperPool {
    #[inline(always)]
    fn get_regex(&self) -> &RegexWrapper {
        self.pool.get()
    }

    fn get_pattern(&self) -> String {
        self.get_regex().as_str().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::string::ToString;
    use crate::regex::regex_wrapper::RegexWrapperPattern;
    #[test]
    fn test_regex_pool() {
        let pattern: RegexWrapperPattern = r"foo".into();
        let regex: RegexWrapper = pattern.compile().unwrap();

        let pool: RegexWrapperPool = RegexWrapperPool::new(regex.clone());

        assert_eq!(pool.get_pattern(), r"foo");
        assert!(format!("{:?}", pool).contains(&format!("{:?}", regex).to_string()));
    }
}
