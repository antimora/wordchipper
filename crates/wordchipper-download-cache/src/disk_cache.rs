//! # Wordchipper Disk Cache

use std::path::PathBuf;
use anyhow::Context;
use crate::WORDCHIPPER_CACHE_CONFIG;

/// Options for [`DiskDownloadCache`].
#[derive(Clone, Default, Debug)]
pub struct DiskDownloadCacheOptions {
    /// Optional path to the cache directory.
    ///
    pub cache_dir: Option<PathBuf>,

    /// Optional path to the data directory.
    pub data_dir: Option<PathBuf>,
}

/// Disk cache for downloaded files.
#[derive(Clone, Debug)]
pub struct DiskDownloadCache {
    /// Cache directory.
    pub cache_dir: PathBuf,

    /// Data directory.
    pub data_dir: PathBuf,
}

impl DiskDownloadCache {
    /// Construct a new [`DiskDownloadCache`].
    pub fn init(options: DiskDownloadCacheOptions) -> anyhow::Result<Self> {
        let cache_dir = WORDCHIPPER_CACHE_CONFIG
            .resolve_cache_dir(options.cache_dir)
            .context("failed to resolve cache directory")?;

        let data_dir = WORDCHIPPER_CACHE_CONFIG
            .resolve_data_dir(options.data_dir)
            .context("failed to resolve data directory")?;

        Ok(Self {
            cache_dir,
            data_dir,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;
    use serial_test::serial;
    use crate::disk_cache::{DiskDownloadCache, DiskDownloadCacheOptions};
    use crate::{WORDCHIPPER_CACHE_CONFIG, WORDCHIPPER_CACHE_DIR, WORDCHIPPER_DATA_DIR};

    #[test]
    #[serial]
    fn test_resolve_dirs() {
        let orig_cache_dir = env::var(WORDCHIPPER_CACHE_DIR);
        let orig_data_dir = env::var(WORDCHIPPER_CACHE_DIR);

        let pds = WORDCHIPPER_CACHE_CONFIG
            .project_dirs()
            .expect("failed to get project dirs");

        let user_cache_dir = PathBuf::from("/tmp/wordchipper/cache");
        let user_data_dir = PathBuf::from("/tmp/wordchipper/data");

        let env_cache_dir = PathBuf::from("/tmp/wordchipper/env_cache");
        let env_data_dir = PathBuf::from("/tmp/wordchipper/env_data");

        // No env vars
        unsafe {
            env::remove_var(WORDCHIPPER_CACHE_DIR);
            env::remove_var(WORDCHIPPER_DATA_DIR);
        }

        let cache = DiskDownloadCache::init(DiskDownloadCacheOptions {
            cache_dir: Some(user_cache_dir.clone()),
            data_dir: Some(user_data_dir.clone()),
        })
        .unwrap();
        assert_eq!(&cache.cache_dir, &user_cache_dir);
        assert_eq!(&cache.data_dir, &user_data_dir);

        let cache = DiskDownloadCache::init(DiskDownloadCacheOptions::default()).unwrap();
        assert_eq!(&cache.cache_dir, &pds.cache_dir().to_path_buf());
        assert_eq!(&cache.data_dir, &pds.data_dir().to_path_buf());

        // With env var.
        unsafe {
            env::set_var(WORDCHIPPER_CACHE_DIR, env_cache_dir.to_str().unwrap());
            env::set_var(WORDCHIPPER_DATA_DIR, env_data_dir.to_str().unwrap());
        }

        let cache = DiskDownloadCache::init(DiskDownloadCacheOptions {
            cache_dir: Some(user_cache_dir.clone()),
            data_dir: Some(user_data_dir.clone()),
        })
        .unwrap();
        assert_eq!(&cache.cache_dir, &user_cache_dir);
        assert_eq!(&cache.data_dir, &user_data_dir);

        let cache = DiskDownloadCache::init(DiskDownloadCacheOptions::default()).unwrap();
        assert_eq!(&cache.cache_dir, &env_cache_dir);
        assert_eq!(&cache.data_dir, &env_data_dir);

        // restore original env var.
        match orig_cache_dir {
            Ok(original) => unsafe { env::set_var(WORDCHIPPER_CACHE_DIR, original) },
            Err(_) => unsafe { env::remove_var(WORDCHIPPER_CACHE_DIR) },
        }
        match orig_data_dir {
            Ok(original) => unsafe { env::set_var(WORDCHIPPER_DATA_DIR, original) },
            Err(_) => unsafe { env::remove_var(WORDCHIPPER_DATA_DIR) },
        }
    }
}