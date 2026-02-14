//! # Resource Loader

use std::path::PathBuf;

use crate::resources::KeyedResource;

/// A trait for loading resources.
pub trait ResourceLoader {
    /// Load a resource.
    #[cfg(feature = "std")]
    fn load_resource_path<R: Into<KeyedResource>>(
        &mut self,
        resource: R,
    ) -> anyhow::Result<PathBuf>;
}

#[cfg(feature = "download")]
impl ResourceLoader for crate::disk_cache::WordchipperDiskCache {
    fn load_resource_path<R: Into<KeyedResource>>(
        &mut self,
        resource: R,
    ) -> anyhow::Result<PathBuf> {
        let resource = resource.into();
        self.load_cached_path(&resource.key, &resource.resource.urls, true)
    }
}
