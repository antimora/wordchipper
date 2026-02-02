//! # Thread Regex Pool
#![allow(unused)]

use crate::alloc::sync::Arc;
use crate::regex::regex_supplier::RegexSupplier;
use crate::regex::regex_wrapper::RegexWrapper;
use crate::types::CommonHashMap;
use core::fmt::Debug;
use core::num::NonZero;
use core::sync::atomic::Ordering;
use parking_lot::RwLock;
use std::cell::RefCell;
use std::num::NonZeroU64;
use std::sync::atomic::AtomicUsize;
use std::thread;
use std::thread::ThreadId;

fn unsafe_threadid_to_u64(thread_id: &ThreadId) -> u64 {
    unsafe { std::mem::transmute(thread_id) }
}

/// Stub
struct FakeThreadId(NonZeroU64);

fn hash_current_thread() -> usize {
    // It's easier to use unsafe than to use nightly. Rust has this nice u64 thread id counter
    // that works great for our use case of avoiding collisions in our array. Unfortunately,
    // it's private. However, there are only so many ways you can layout a u64, so just transmute
    // https://github.com/rust-lang/rust/issues/67939
    const _: [u8; 8] = [0; std::mem::size_of::<std::thread::ThreadId>()];
    const _: [u8; 8] = [0; std::mem::size_of::<FakeThreadId>()];
    let x = unsafe {
        std::mem::transmute::<std::thread::ThreadId, FakeThreadId>(thread::current().id()).0
    };
    u64::from(x) as usize
}

/// Interior-Mutable Thread-Local Regex Pool
///
/// In HPC applications, under some loads, interior buffers in compiled regex
/// can block. This pool exists to mitigate that, by cloning regex-per-thread.
pub struct RegexWrapperPool {
    pool: Vec<RegexWrapper>,

    counter: AtomicUsize,
}

impl Clone for RegexWrapperPool {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            counter: AtomicUsize::new(0),
        }
    }
}

impl Debug for RegexWrapperPool {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("RegexPool")
            .field("regex", &self.get_regex())
            .finish()
    }
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
        let max_pool = std::thread::available_parallelism()
            .unwrap_or(NonZero::new(128).unwrap())
            .get() as u64;

        let pool = (0..max_pool).map(|_| regex.clone()).collect::<Vec<_>>();

        Self {
            pool,
            counter: AtomicUsize::new(0),
        }
    }

    /// Returns the number of regex instances in the pool.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.pool.len()
    }
}

impl RegexSupplier for RegexWrapperPool {
    fn get_regex(&self) -> &RegexWrapper {
        // let tid = hash_current_thread();
        let id = self
            .counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| {
                Some((x + 1) % self.pool.len())
            })
            .unwrap();
        &self.pool[id % self.pool.len()]
    }

    fn get_pattern(&self) -> String {
        self.pool[0].as_str().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::string::ToString;
    use crate::regex::regex_wrapper::RegexWrapperPattern;
    use fancy_regex::internal::compile;

    #[test]
    fn test_regex_pool() {
        let pattern: RegexWrapperPattern = r"foo".into();
        let regex: RegexWrapper = pattern.compile().unwrap();

        let pool: RegexWrapperPool = RegexWrapperPool::new(regex.clone());

        assert_eq!(pool.get_pattern(), r"foo");
        assert!(format!("{:?}", pool).contains(&format!("{:?}", regex).to_string()));
    }
}
