use anyhow::Result;
use dashmap::DashMap;
use log::{debug, info, warn};
use networkgpu_proto::{DeviceInfo, DeviceStats};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

use crate::config::ServerConfig;
use crate::error::GPUError;
use crate::memory::MemoryManager;
use crate::stream::StreamManager;

pub type DeviceId = i32;
pub type AllocationToken = String;

#[derive(Debug)]
pub struct GPUResourceManager {
    devices: RwLock<HashMap<DeviceId, Arc<GPUDevice>>>,
    device_allocations: DashMap<AllocationToken, DeviceAllocation>,
    config: ServerConfig,
}

#[derive(Debug)]
pub struct GPUDevice {
    pub id: DeviceId,
    // pub device: Arc<CudaDevice>, // Commented out for now
    pub info: DeviceInfo,
    pub memory_manager: Arc<MemoryManager>,
    pub stream_manager: Arc<StreamManager>,
    pub stats: DeviceStats,
    pub allocated_memory: AtomicU64,
    pub kernel_launches: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct DeviceAllocation {
    pub token: AllocationToken,
    pub device_id: DeviceId,
    pub client_id: String,
    pub memory_limit: u64,
    pub allocated_at: std::time::SystemTime,
}

impl GPUResourceManager {
    pub async fn new(config: ServerConfig) -> Result<Self, GPUError> {
        let mut devices = HashMap::new();
        
        // Auto-detect devices if none specified
        let device_ids = if config.gpu.device_ids.is_empty() {
            // For cudarc 0.16+, use num_devices()
            let num_devices = match std::process::Command::new("nvidia-smi")
                .arg("-L")
                .output() 
            {
                Ok(output) => {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    output_str.lines().count() as u32
                }
                Err(_) => 1, // Default to 1 device if nvidia-smi fails
            };
            (0..num_devices as i32).collect::<Vec<_>>()
        } else {
            config.gpu.device_ids.clone()
        };
        
        info!("Initializing {} GPU devices", device_ids.len());
        
        for device_id in device_ids {
            match Self::initialize_device(device_id, &config).await {
                Ok(device) => {
                    info!("Initialized GPU device {}: {}", device_id, device.info.name);
                    devices.insert(device_id, Arc::new(device));
                }
                Err(e) => {
                    warn!("Failed to initialize GPU device {}: {}", device_id, e);
                }
            }
        }
        
        if devices.is_empty() {
            return Err(GPUError::InternalError("No GPU devices available".to_string()));
        }
        
        Ok(Self {
            devices: RwLock::new(devices),
            device_allocations: DashMap::new(),
            config,
        })
    }
    
    async fn initialize_device(device_id: DeviceId, config: &ServerConfig) -> Result<GPUDevice, GPUError> {
        // let device = CudaDevice::new(device_id as usize)
        //     .map_err(|e| GPUError::CudaError(e.to_string()))?;
        
        let name = format!("GPU Device {}", device_id);
        let total_memory = 8 * 1024 * 1024 * 1024u64; // Default to 8GB, will be updated when cudarc API is available
        let free_memory = total_memory;
        
        let device_info = DeviceInfo {
            device_id,
            name: name.clone(),
            total_memory,
            free_memory,
            compute_capability_major: 0, // TODO: Get from device
            compute_capability_minor: 0,
            multiprocessor_count: 0,
            max_threads_per_block: 1024,
            max_block_dim_x: 1024,
            max_block_dim_y: 1024,
            max_block_dim_z: 64,
            max_grid_dim_x: 65535,
            max_grid_dim_y: 65535,
            max_grid_dim_z: 65535,
            warp_size: 32,
            max_pitch: 2147483647,
            max_threads_per_multiprocessor: 2048,
            clock_rate: 1000000,
            texture_alignment: 512,
            concurrent_kernels: true,
            ecc_enabled: false,
            pci_bus_id: 0,
            pci_device_id: 0,
            uuid: Uuid::new_v4().to_string(),
        };
        
        let device_stats = DeviceStats {
            device_id,
            memory_used: 0,
            memory_free: free_memory,
            utilization_percent: 0.0,
            active_kernels: 0,
            total_kernel_launches: 0,
        };
        
        let memory_manager = MemoryManager::new(
            device_id,
            config.memory.pool_size_mb * 1024 * 1024,
            config.memory.alignment_bytes,
        )?;
        
        let stream_manager = StreamManager::new(
            device_id,
            config.gpu.max_streams_per_device,
        )?;
        
        Ok(GPUDevice {
            id: device_id,
            // device,
            info: device_info,
            memory_manager: Arc::new(memory_manager),
            stream_manager: Arc::new(stream_manager),
            stats: device_stats,
            allocated_memory: AtomicU64::new(0),
            kernel_launches: AtomicU64::new(0),
        })
    }
    
    pub fn get_device_info(&self, device_id: DeviceId) -> Result<DeviceInfo, GPUError> {
        let devices = self.devices.read();
        let device = devices.get(&device_id)
            .ok_or(GPUError::InvalidDevice(device_id))?;
        
        let mut info = device.info.clone();
        // For now, simulate memory usage based on allocations
        let allocated = device.allocated_memory.load(Ordering::Relaxed);
        info.free_memory = info.total_memory.saturating_sub(allocated);
        
        Ok(info)
    }
    
    pub fn list_devices(&self) -> Vec<DeviceInfo> {
        let devices = self.devices.read();
        devices.values()
            .map(|device| {
                let mut info = device.info.clone();
                let allocated = device.allocated_memory.load(Ordering::Relaxed);
                info.free_memory = info.total_memory.saturating_sub(allocated);
                info
            })
            .collect()
    }
    
    pub fn allocate_device(
        &self,
        device_id: DeviceId,
        client_id: String,
        memory_limit: u64,
    ) -> Result<AllocationToken, GPUError> {
        let devices = self.devices.read();
        let _device = devices.get(&device_id)
            .ok_or(GPUError::InvalidDevice(device_id))?;
        
        let token = Uuid::new_v4().to_string();
        let allocation = DeviceAllocation {
            token: token.clone(),
            device_id,
            client_id,
            memory_limit,
            allocated_at: std::time::SystemTime::now(),
        };
        
        self.device_allocations.insert(token.clone(), allocation);
        debug!("Allocated device {} to client with token {}", device_id, token);
        
        Ok(token)
    }
    
    pub fn release_device(&self, token: &str) -> Result<(), GPUError> {
        if let Some((_, allocation)) = self.device_allocations.remove(token) {
            debug!("Released device {} for token {}", allocation.device_id, token);
            Ok(())
        } else {
            Err(GPUError::InternalError(format!("Invalid allocation token: {}", token)))
        }
    }
    
    pub fn get_device(&self, device_id: DeviceId) -> Result<Arc<GPUDevice>, GPUError> {
        let devices = self.devices.read();
        devices.get(&device_id)
            .cloned()
            .ok_or(GPUError::InvalidDevice(device_id))
    }
    
    pub fn get_device_stats(&self, device_id: DeviceId) -> Result<DeviceStats, GPUError> {
        let device = self.get_device(device_id)?;
        
        let used_memory = device.allocated_memory.load(Ordering::Relaxed);
        let free_memory = device.info.total_memory.saturating_sub(used_memory);
        
        Ok(DeviceStats {
            device_id,
            memory_used: used_memory,
            memory_free: free_memory,
            utilization_percent: 0.0, // TODO: Implement GPU utilization monitoring
            active_kernels: 0, // TODO: Track active kernels
            total_kernel_launches: device.kernel_launches.load(Ordering::Relaxed),
        })
    }
    
    pub fn device_count(&self) -> usize {
        self.devices.read().len()
    }
    
    pub fn synchronize_device(&self, device_id: DeviceId) -> Result<(), GPUError> {
        let _device = self.get_device(device_id)?;
        // For now, simulate synchronization - actual CUDA sync will be implemented later
        debug!("Simulating device synchronization for device {}", device_id);
        std::thread::sleep(std::time::Duration::from_millis(1));
        Ok(())
    }
}