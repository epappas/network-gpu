use lru::LruCache;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

// use crate::error::ClientError;

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_tensor_cache_size: usize,
    pub max_device_cache_size: usize,
    pub tensor_ttl: Duration,
    pub device_info_ttl: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_tensor_cache_size: 1000,
            max_device_cache_size: 100,
            tensor_ttl: Duration::from_secs(3600), // 1 hour
            device_info_ttl: Duration::from_secs(300), // 5 minutes
        }
    }
}

#[derive(Debug, Clone)]
pub struct CachedItem<T> {
    pub data: T,
    pub cached_at: Instant,
    pub ttl: Duration,
}

impl<T> CachedItem<T> {
    pub fn new(data: T, ttl: Duration) -> Self {
        Self {
            data,
            cached_at: Instant::now(),
            ttl,
        }
    }
    
    pub fn is_expired(&self) -> bool {
        self.cached_at.elapsed() > self.ttl
    }
}

#[derive(Debug)]
pub struct ClientCache {
    tensor_cache: RwLock<LruCache<String, CachedItem<Arc<networkgpu_proto::TensorDescriptor>>>>,
    device_cache: RwLock<LruCache<i32, CachedItem<networkgpu_proto::DeviceInfo>>>,
    memory_allocation_cache: RwLock<HashMap<String, CachedItem<networkgpu_proto::MemoryPointer>>>,
    config: CacheConfig,
}

impl ClientCache {
    pub fn new(config: CacheConfig) -> Self {
        let tensor_cache_size = NonZeroUsize::new(config.max_tensor_cache_size)
            .unwrap_or(NonZeroUsize::new(1000).unwrap());
        let device_cache_size = NonZeroUsize::new(config.max_device_cache_size)
            .unwrap_or(NonZeroUsize::new(100).unwrap());
        
        Self {
            tensor_cache: RwLock::new(LruCache::new(tensor_cache_size)),
            device_cache: RwLock::new(LruCache::new(device_cache_size)),
            memory_allocation_cache: RwLock::new(HashMap::new()),
            config,
        }
    }
    
    // Tensor cache operations
    pub fn get_tensor(&self, tensor_id: &str) -> Option<Arc<networkgpu_proto::TensorDescriptor>> {
        let mut cache = self.tensor_cache.write();
        if let Some(cached_item) = cache.get(tensor_id) {
            if !cached_item.is_expired() {
                return Some(cached_item.data.clone());
            } else {
                cache.pop(tensor_id);
            }
        }
        None
    }
    
    pub fn put_tensor(&self, tensor_id: String, tensor: networkgpu_proto::TensorDescriptor) {
        let cached_item = CachedItem::new(Arc::new(tensor), self.config.tensor_ttl);
        self.tensor_cache.write().put(tensor_id, cached_item);
    }
    
    pub fn remove_tensor(&self, tensor_id: &str) {
        self.tensor_cache.write().pop(tensor_id);
    }
    
    // Device cache operations
    pub fn get_device_info(&self, device_id: i32) -> Option<networkgpu_proto::DeviceInfo> {
        let mut cache = self.device_cache.write();
        if let Some(cached_item) = cache.get(&device_id) {
            if !cached_item.is_expired() {
                return Some(cached_item.data.clone());
            } else {
                cache.pop(&device_id);
            }
        }
        None
    }
    
    pub fn put_device_info(&self, device_id: i32, device_info: networkgpu_proto::DeviceInfo) {
        let cached_item = CachedItem::new(device_info, self.config.device_info_ttl);
        self.device_cache.write().put(device_id, cached_item);
    }
    
    // Memory allocation cache operations
    pub fn get_memory_allocation(&self, allocation_id: &str) -> Option<networkgpu_proto::MemoryPointer> {
        let mut cache = self.memory_allocation_cache.write();
        if let Some(cached_item) = cache.get(allocation_id) {
            if !cached_item.is_expired() {
                return Some(cached_item.data.clone());
            } else {
                cache.remove(allocation_id);
            }
        }
        None
    }
    
    pub fn put_memory_allocation(
        &self,
        allocation_id: String,
        memory_pointer: networkgpu_proto::MemoryPointer,
    ) {
        let cached_item = CachedItem::new(memory_pointer, Duration::from_secs(3600));
        self.memory_allocation_cache.write().insert(allocation_id, cached_item);
    }
    
    pub fn remove_memory_allocation(&self, allocation_id: &str) {
        self.memory_allocation_cache.write().remove(allocation_id);
    }
    
    // Cache maintenance
    pub fn cleanup_expired(&self) {
        // Clean up tensor cache
        {
            let mut tensor_cache = self.tensor_cache.write();
            let expired_keys: Vec<_> = tensor_cache
                .iter()
                .filter(|(_, item)| item.is_expired())
                .map(|(key, _)| key.clone())
                .collect();
            
            for key in expired_keys {
                tensor_cache.pop(&key);
            }
        }
        
        // Clean up device cache
        {
            let mut device_cache = self.device_cache.write();
            let expired_keys: Vec<_> = device_cache
                .iter()
                .filter(|(_, item)| item.is_expired())
                .map(|(key, _)| *key)
                .collect();
            
            for key in expired_keys {
                device_cache.pop(&key);
            }
        }
        
        // Clean up memory allocation cache
        {
            let mut mem_cache = self.memory_allocation_cache.write();
            let expired_keys: Vec<_> = mem_cache
                .iter()
                .filter(|(_, item)| item.is_expired())
                .map(|(key, _)| key.clone())
                .collect();
            
            for key in expired_keys {
                mem_cache.remove(&key);
            }
        }
    }
    
    pub fn clear(&self) {
        self.tensor_cache.write().clear();
        self.device_cache.write().clear();
        self.memory_allocation_cache.write().clear();
    }
    
    pub fn cache_stats(&self) -> CacheStats {
        let tensor_cache = self.tensor_cache.read();
        let device_cache = self.device_cache.read();
        let mem_cache = self.memory_allocation_cache.read();
        
        CacheStats {
            tensor_cache_size: tensor_cache.len(),
            tensor_cache_capacity: tensor_cache.cap().get(),
            device_cache_size: device_cache.len(),
            device_cache_capacity: device_cache.cap().get(),
            memory_allocation_cache_size: mem_cache.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub tensor_cache_size: usize,
    pub tensor_cache_capacity: usize,
    pub device_cache_size: usize,
    pub device_cache_capacity: usize,
    pub memory_allocation_cache_size: usize,
}