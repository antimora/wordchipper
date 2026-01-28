//! # wordchipper-download-cache

use directories_next::ProjectDirs;
use std::env;
use std::path::{Path, PathBuf};

/// Attempt to build a System/$USER [`ProjectDirs`] for wordchipper.
///
/// Used to determine the default cache and data directories.
pub fn wordchipper_project_dirs() -> Option<ProjectDirs> {
    ProjectDirs::from("io", "crates", "wordchipper")
}

/// Environment variable key to override the default cache directory.
pub const WORDCHIPPER_CACHE_DIR: &str = "WORDCHIPPER_CACHE_DIR";

/// Get the cache directory for wordchipper.
///
/// The resolution order is:
/// 1. `path`, if present.
/// 2. [`WORDCHIPPER_CACHE_DIR`] env var.
/// 3. `project_dirs().cache_dir()`
/// 4. None
pub fn resolve_cache_dir<P: AsRef<Path>>(path: Option<P>) -> Option<PathBuf> {
    if let Some(path) = path {
        path.as_ref().to_path_buf().into()
    } else if let Ok(path) = env::var(WORDCHIPPER_CACHE_DIR) {
        PathBuf::from(path).into()
    } else if let Some(pds) = wordchipper_project_dirs() {
        pds.cache_dir().to_path_buf().into()
    } else {
        None
    }
}

/// Environment variable key to override the default data directory.
pub const WORDCHIPPER_DATA_DIR: &str = "WORDCHIPPER_DATA_DIR";

/// Get the data directory for wordchipper.
///
/// The resolution order is:
/// 1. `path`, if present.
/// 2. [`WORDCHIPPER_DATA_DIR`] env var.
/// 3. `project_dirs().data_dir()`
/// 4. `None`
pub fn resolve_data_dir<P: AsRef<Path>>(path: Option<P>) -> Option<PathBuf> {
    if let Some(path) = path {
        path.as_ref().to_path_buf().into()
    } else if let Ok(path) = env::var(WORDCHIPPER_DATA_DIR) {
        PathBuf::from(path).into()
    } else if let Some(pds) = wordchipper_project_dirs() {
        pds.data_dir().to_path_buf().into()
    } else {
        None
    }
}

/// Options for [`DiskDownloadCache`].
#[derive(Clone, Default, Debug)]
pub struct DiskDownloadCacheOptions {
    /// Optional path to the cache directory.
    ///
    pub cache_dir: Option<PathBuf>,

    /// Optional path to the data directory.
    pub data_dir: Option<PathBuf>,
}

impl DiskDownloadCacheOptions {
    /// Resolve the options into a [`DiskDownloadCache`].
    pub fn resolve(mut self) -> anyhow::Result<Self> {
        self.cache_dir = resolve_cache_dir(self.cache_dir);
        if self.cache_dir.is_none() {
            anyhow::bail!("Unable to resolve cache dir.");
        }

        self.data_dir = resolve_data_dir(self.data_dir);
        if self.data_dir.is_none() {
            anyhow::bail!("Unable to resolve data dir.");
        }

        Ok(self)
    }
}

/// Disk cache for downloaded files.
#[derive(Clone, Debug)]
pub struct DiskDownloadCache {
    options: DiskDownloadCacheOptions,
}

impl DiskDownloadCache {
    /// Construct a new [`DiskDownloadCache`].
    pub fn init(options: DiskDownloadCacheOptions) -> anyhow::Result<Self> {
        let options = options.resolve()?;

        Ok(Self { options })
    }

    /// Get the resolved options for this cache.
    pub fn options(&self) -> &DiskDownloadCacheOptions {
        &self.options
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_resolve_cache_dir() {
        let original = env::var(WORDCHIPPER_CACHE_DIR);

        let no_path: Option<PathBuf> = None;
        let path = Some(PathBuf::from("/tmp/wordchipper"));
        let pds = wordchipper_project_dirs();
        let var_path = PathBuf::from("/tmp/wordchipper-cache");

        // No env var.
        unsafe { env::remove_var(WORDCHIPPER_CACHE_DIR) }

        assert_eq!(resolve_cache_dir(path.clone()), path.clone());

        assert_eq!(
            resolve_cache_dir(no_path.clone()),
            pds.map(|pds| pds.cache_dir().to_path_buf())
        );

        // With env var.
        unsafe { env::set_var(WORDCHIPPER_CACHE_DIR, var_path.to_str().unwrap()) }

        assert_eq!(resolve_cache_dir(path.clone()), path.clone());

        assert_eq!(resolve_cache_dir(no_path), Some(var_path));

        // restore original env var.
        match original {
            Ok(original) => unsafe { env::set_var(WORDCHIPPER_CACHE_DIR, original) },
            Err(_) => unsafe { env::remove_var(WORDCHIPPER_CACHE_DIR) },
        }
    }

    #[test]
    #[serial]
    fn test_resolve_data_dir() {
        let original = env::var(WORDCHIPPER_DATA_DIR);

        let no_path: Option<PathBuf> = None;
        let path = Some(PathBuf::from("/tmp/wordchipper"));
        let pds = wordchipper_project_dirs();
        let var_path = PathBuf::from("/tmp/wordchipper-data");

        // No env var.
        unsafe { env::remove_var(WORDCHIPPER_DATA_DIR) }

        assert_eq!(resolve_data_dir(path.clone()), path.clone());

        assert_eq!(
            resolve_data_dir(no_path.clone()),
            pds.map(|pds| pds.data_dir().to_path_buf())
        );

        // With env var.
        unsafe { env::set_var(WORDCHIPPER_DATA_DIR, var_path.to_str().unwrap()) }

        assert_eq!(resolve_data_dir(path.clone()), path.clone());

        assert_eq!(resolve_data_dir(no_path), Some(var_path));

        // restore original env var.
        match original {
            Ok(original) => unsafe { env::set_var(WORDCHIPPER_DATA_DIR, original) },
            Err(_) => unsafe { env::remove_var(WORDCHIPPER_DATA_DIR) },
        }
    }
}
