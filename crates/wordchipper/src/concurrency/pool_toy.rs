//! # Thread Pool Toy

use core::fmt::Debug;
use std::num::{NonZero, NonZeroU64, NonZeroUsize};
use std::thread;

struct FakeThreadId(NonZeroU64);

pub fn hash_current_thread() -> usize {
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

/// Current Thread -> T Pool.
pub struct PoolToy<T>
where
    T: Clone + Send,
{
    pool: Vec<T>,
}

impl<T> PoolToy<T>
where
    T: Clone + Send,
{
    /// Create a new thread-local pool with the given vector of items.
    pub fn new(pool: Vec<T>) -> Self {
        assert!(!pool.is_empty());
        Self { pool }
    }

    /// Initialize a new thread-local pool with the given item and maximum pool size.
    pub fn init(
        item: T,
        max_pool: Option<NonZeroUsize>,
    ) -> Self {
        let sys_max: usize = std::thread::available_parallelism()
            .unwrap_or(NonZero::new(128).unwrap())
            .get();

        let max_pool = max_pool.map(|x| x.get()).unwrap_or(sys_max);

        let max_pool = core::cmp::min(max_pool, sys_max);

        Self::new(vec![item; max_pool])
    }

    /// Get a reference to the item for the current thread.
    pub fn get(&self) -> &T {
        let tid = hash_current_thread();
        &self.pool[tid % self.pool.len()]
    }

    /// Get the length of the pool.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.pool.len()
    }
}

impl<T> Clone for PoolToy<T>
where
    T: Clone + Send,
{
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}

impl<T> Debug for PoolToy<T>
where
    T: Clone + Send + Debug,
{
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("PoolToy")
            .field("item", &self.pool[0])
            .field("len", &self.pool.len())
            .finish()
    }
}
