use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub server: ServerSettings,
    pub gpu: GPUSettings,
    pub memory: MemorySettings,
    pub security: SecuritySettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
    pub max_connections: usize,
    pub request_timeout_seconds: u64,
    pub keepalive_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUSettings {
    pub device_ids: Vec<i32>,
    pub max_concurrent_kernels: usize,
    pub max_streams_per_device: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySettings {
    pub pool_size_mb: u64,
    pub max_allocation_mb: u64,
    pub enable_memory_pool: bool,
    pub alignment_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySettings {
    pub enable_tls: bool,
    pub cert_file: Option<String>,
    pub key_file: Option<String>,
    pub ca_file: Option<String>,
    pub require_client_auth: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server: ServerSettings {
                max_connections: 100,
                request_timeout_seconds: 300,
                keepalive_seconds: 60,
            },
            gpu: GPUSettings {
                device_ids: vec![], // Auto-detect
                max_concurrent_kernels: 32,
                max_streams_per_device: 16,
            },
            memory: MemorySettings {
                pool_size_mb: 1024,
                max_allocation_mb: 512,
                enable_memory_pool: true,
                alignment_bytes: 256,
            },
            security: SecuritySettings {
                enable_tls: false,
                cert_file: None,
                key_file: None,
                ca_file: None,
                require_client_auth: false,
            },
        }
    }
}

impl ServerConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        
        // Try parsing as TOML first
        if let Ok(config) = toml::from_str::<ServerConfig>(&content) {
            return Ok(config);
        }
        
        // Try parsing as JSON
        if let Ok(config) = serde_json::from_str::<ServerConfig>(&content) {
            return Ok(config);
        }
        
        // Try parsing as YAML
        if let Ok(config) = serde_yaml::from_str::<ServerConfig>(&content) {
            return Ok(config);
        }
        
        Err(anyhow::anyhow!("Failed to parse config file as TOML, JSON, or YAML"))
    }
    
    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }
}