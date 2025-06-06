use anyhow::Result;
use log::{debug, info};
use networkgpu_proto::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::cache::{CacheConfig, ClientCache};
use crate::connection::{ConnectionConfig, ConnectionPool};
use crate::error::ClientError;
use crate::tensor::{RemoteTensor, TensorHandle};

#[derive(Debug, Clone)]
#[derive(Default)]
pub struct ClientConfig {
    pub connection: ConnectionConfig,
    pub cache: CacheConfig,
    pub enable_compression: bool,
    pub default_device_id: i32,
}


#[derive(Debug)]
pub struct NetworkGPUClient {
    connection_pool: Arc<ConnectionPool>,
    cache: Arc<ClientCache>,
    config: ClientConfig,
    device_allocations: RwLock<HashMap<i32, String>>, // device_id -> allocation_token
    available_devices: RwLock<Vec<DeviceInfo>>,
}

impl Clone for NetworkGPUClient {
    fn clone(&self) -> Self {
        Self {
            connection_pool: self.connection_pool.clone(),
            cache: self.cache.clone(),
            config: self.config.clone(),
            device_allocations: RwLock::new(HashMap::new()),
            available_devices: RwLock::new(Vec::new()),
        }
    }
}

impl NetworkGPUClient {
    pub async fn new(
        server_urls: Vec<String>,
        config: ClientConfig,
    ) -> Result<Self, ClientError> {
        info!("Connecting to NetworkGPU servers: {:?}", server_urls);
        
        let connection_pool = Arc::new(
            ConnectionPool::new(server_urls, config.connection.clone()).await?
        );
        
        let cache = Arc::new(ClientCache::new(config.cache.clone()));
        
        let client = Self {
            connection_pool,
            cache,
            config,
            device_allocations: RwLock::new(HashMap::new()),
            available_devices: RwLock::new(Vec::new()),
        };
        
        client.refresh_device_list().await?;
        
        info!("NetworkGPU client initialized successfully");
        Ok(client)
    }
    
    pub async fn get_client(&self) -> Result<network_gpu_service_client::NetworkGpuServiceClient<tonic::transport::Channel>, ClientError> {
        self.connection_pool.get_client().await
    }
    
    pub async fn list_devices(&self) -> Result<Vec<DeviceInfo>, ClientError> {
        // Try cache first
        let devices = self.available_devices.read().await;
        if !devices.is_empty() {
            return Ok(devices.clone());
        }
        drop(devices);
        
        // Refresh from server
        self.refresh_device_list().await
    }
    
    async fn refresh_device_list(&self) -> Result<Vec<DeviceInfo>, ClientError> {
        let mut client = self.get_client().await?;
        
        let request = ListDevicesRequest {};
        let response = client.list_devices(request).await?;
        let result = response.into_inner();
        
        if !result.success {
            return Err(ClientError::DeviceOperationFailed(result.error_message));
        }
        
        // Update cache and available devices
        let mut devices = self.available_devices.write().await;
        *devices = result.devices.clone();
        
        for device in &result.devices {
            self.cache.put_device_info(device.device_id, device.clone());
        }
        
        debug!("Refreshed device list: {} devices available", result.devices.len());
        Ok(result.devices)
    }
    
    pub async fn get_device_info(&self, device_id: i32) -> Result<DeviceInfo, ClientError> {
        // Try cache first
        if let Some(device_info) = self.cache.get_device_info(device_id) {
            return Ok(device_info);
        }
        
        // Fetch from server
        let mut client = self.get_client().await?;
        
        let request = GetDeviceInfoRequest { device_id };
        let response = client.get_device_info(request).await?;
        let result = response.into_inner();
        
        if !result.success {
            return Err(ClientError::DeviceOperationFailed(result.error_message));
        }
        
        let device_info = result.device_info
            .ok_or_else(|| ClientError::DeviceOperationFailed("No device info returned".to_string()))?;
        
        // Cache the result
        self.cache.put_device_info(device_id, device_info.clone());
        
        Ok(device_info)
    }
    
    pub async fn allocate_device(
        &self,
        device_id: i32,
        client_id: String,
        memory_limit: u64,
    ) -> Result<String, ClientError> {
        let mut client = self.get_client().await?;
        
        let request = AllocateDeviceRequest {
            device_id,
            client_id,
            memory_limit,
        };
        
        let response = client.allocate_device(request).await?;
        let result = response.into_inner();
        
        if !result.success {
            return Err(ClientError::DeviceOperationFailed(result.error_message));
        }
        
        // Store allocation token
        let mut allocations = self.device_allocations.write().await;
        allocations.insert(device_id, result.allocation_token.clone());
        
        debug!("Allocated device {} with token {}", device_id, result.allocation_token);
        Ok(result.allocation_token)
    }
    
    pub async fn release_device(&self, device_id: i32) -> Result<(), ClientError> {
        let allocation_token = {
            let mut allocations = self.device_allocations.write().await;
            allocations.remove(&device_id)
        };
        
        if let Some(token) = allocation_token {
            let mut client = self.get_client().await?;
            
            let request = ReleaseDeviceRequest {
                allocation_token: token,
            };
            
            let response = client.release_device(request).await?;
            let result = response.into_inner();
            
            if !result.success {
                return Err(ClientError::DeviceOperationFailed(result.error_message));
            }
            
            debug!("Released device {}", device_id);
        }
        
        Ok(())
    }
    
    pub async fn create_tensor(
        &self,
        shape: &[i64],
        dtype: &str,
        device_id: i32,
        requires_grad: bool,
        initial_data: Option<&[u8]>,
    ) -> Result<TensorHandle, ClientError> {
        let mut client = self.get_client().await?;
        
        let request = CreateTensorRequest {
            shape: shape.to_vec(),
            dtype: dtype.to_string(),
            device_id,
            requires_grad,
            initial_data: initial_data.unwrap_or(&[]).to_vec(),
            pin_memory: false,
        };
        
        let response = client.create_tensor(request).await?;
        let result = response.into_inner();
        
        if !result.success {
            return Err(ClientError::TensorOperationFailed(result.error_message));
        }
        
        let tensor_descriptor = result.tensor
            .ok_or_else(|| ClientError::TensorOperationFailed("No tensor descriptor returned".to_string()))?;
        
        let tensor_id = tensor_descriptor.tensor_id.clone();
        
        // Cache the tensor
        self.cache.put_tensor(tensor_id.clone(), tensor_descriptor.clone());
        
        let handle = TensorHandle::new(tensor_id, tensor_descriptor, Arc::new(self.clone()));
        
        debug!("Created tensor {} with shape {:?}", handle.id, shape);
        Ok(handle)
    }
    
    pub async fn destroy_tensor(&self, tensor_id: &str) -> Result<(), ClientError> {
        let mut client = self.get_client().await?;
        
        let request = DestroyTensorRequest {
            tensor_id: tensor_id.to_string(),
        };
        
        let response = client.destroy_tensor(request).await?;
        let result = response.into_inner();
        
        if !result.success {
            return Err(ClientError::TensorOperationFailed(result.error_message));
        }
        
        // Remove from cache
        self.cache.remove_tensor(tensor_id);
        
        debug!("Destroyed tensor {}", tensor_id);
        Ok(())
    }
    
    pub async fn get_tensor(&self, tensor_id: &str) -> Result<TensorHandle, ClientError> {
        // Try cache first
        if let Some(descriptor) = self.cache.get_tensor(tensor_id) {
            return Ok(TensorHandle::new(
                tensor_id.to_string(),
                (*descriptor).clone(),
                Arc::new(self.clone()),
            ));
        }
        
        Err(ClientError::InvalidTensorId(tensor_id.to_string()))
    }
    
    pub async fn create_tensor_from_data(
        &self,
        data: &[f32],
        shape: &[i64],
        device_id: i32,
    ) -> Result<RemoteTensor, ClientError> {
        // Convert f32 data to bytes
        let byte_data: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                std::mem::size_of_val(data),
            )
        };
        
        let handle = self.create_tensor(
            shape,
            "float32",
            device_id,
            false,
            Some(byte_data),
        ).await?;
        
        RemoteTensor::new(handle)
    }
    
    pub async fn randn(
        &self,
        shape: &[i64],
        device_id: i32,
    ) -> Result<RemoteTensor, ClientError> {
        // Generate random data
        let element_count: usize = shape.iter().map(|&x| x as usize).product();
        let mut data = Vec::with_capacity(element_count);
        
        for _ in 0..element_count {
            data.push(fastrand::f32() * 2.0 - 1.0); // Random values between -1 and 1
        }
        
        self.create_tensor_from_data(&data, shape, device_id).await
    }
    
    pub async fn zeros(
        &self,
        shape: &[i64],
        device_id: i32,
    ) -> Result<RemoteTensor, ClientError> {
        let element_count: usize = shape.iter().map(|&x| x as usize).product();
        let data = vec![0.0f32; element_count];
        
        self.create_tensor_from_data(&data, shape, device_id).await
    }
    
    pub async fn ones(
        &self,
        shape: &[i64],
        device_id: i32,
    ) -> Result<RemoteTensor, ClientError> {
        let element_count: usize = shape.iter().map(|&x| x as usize).product();
        let data = vec![1.0f32; element_count];
        
        self.create_tensor_from_data(&data, shape, device_id).await
    }
    
    pub async fn synchronize_device(&self, device_id: i32) -> Result<(), ClientError> {
        let mut client = self.get_client().await?;
        
        let request = SynchronizeDeviceRequest { device_id };
        let response = client.synchronize_device(request).await?;
        let result = response.into_inner();
        
        if !result.success {
            return Err(ClientError::DeviceOperationFailed(result.error_message));
        }
        
        Ok(())
    }
    
    pub async fn health_check(&self) -> Result<HealthCheckResponse, ClientError> {
        let mut client = self.get_client().await?;
        
        let request = HealthCheckRequest {};
        let response = client.health_check(request).await?;
        
        Ok(response.into_inner())
    }
    
    pub fn cleanup_cache(&self) {
        self.cache.cleanup_expired();
    }
    
    pub fn cache_stats(&self) -> crate::cache::CacheStats {
        self.cache.cache_stats()
    }
}