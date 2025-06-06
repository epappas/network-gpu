use anyhow::Result;
use futures::Stream;
use log::info;
use networkgpu_proto::*;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use crate::config::ServerConfig;
use crate::error::GPUError;
use crate::gpu::GPUResourceManager;
use crate::tensor::TensorManager;

type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

pub struct NetworkGPUServiceImpl {
    gpu_manager: Arc<GPUResourceManager>,
    tensor_managers: Arc<dashmap::DashMap<i32, Arc<TensorManager>>>,
    config: ServerConfig,
    server_start_time: std::time::SystemTime,
    request_count: Arc<std::sync::atomic::AtomicU64>,
}

impl NetworkGPUServiceImpl {
    pub async fn new(config: ServerConfig) -> Result<Self, GPUError> {
        let gpu_manager = Arc::new(GPUResourceManager::new(config.clone()).await?);
        let tensor_managers = Arc::new(dashmap::DashMap::new());
        
        // Initialize tensor managers for each device
        let devices = gpu_manager.list_devices();
        for device_info in devices {
            let device = gpu_manager.get_device(device_info.device_id)?;
            let tensor_manager = Arc::new(TensorManager::new(
                device_info.device_id,
                device.memory_manager.clone(),
            ));
            tensor_managers.insert(device_info.device_id, tensor_manager);
        }
        
        info!("NetworkGPU service initialized with {} devices", tensor_managers.len());
        
        Ok(Self {
            gpu_manager,
            tensor_managers,
            config,
            server_start_time: std::time::SystemTime::now(),
            request_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }
    
    pub fn device_count(&self) -> usize {
        self.gpu_manager.device_count()
    }
    
    fn increment_request_count(&self) {
        self.request_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn get_tensor_manager(&self, device_id: i32) -> Result<Arc<TensorManager>, GPUError> {
        self.tensor_managers
            .get(&device_id)
            .map(|entry| entry.value().clone())
            .ok_or(GPUError::InvalidDevice(device_id))
    }
}

#[tonic::async_trait]
impl network_gpu_service_server::NetworkGpuService for NetworkGPUServiceImpl {
    async fn get_device_info(
        &self,
        request: Request<GetDeviceInfoRequest>,
    ) -> Result<Response<GetDeviceInfoResponse>, Status> {
        self.increment_request_count();
        let req = request.into_inner();
        
        match self.gpu_manager.get_device_info(req.device_id) {
            Ok(device_info) => {
                let response = GetDeviceInfoResponse {
                    device_info: Some(device_info),
                    success: true,
                    error_message: String::new(),
                };
                Ok(Response::new(response))
            }
            Err(e) => {
                let response = GetDeviceInfoResponse {
                    device_info: None,
                    success: false,
                    error_message: e.to_string(),
                };
                Ok(Response::new(response))
            }
        }
    }
    
    async fn list_devices(
        &self,
        _request: Request<ListDevicesRequest>,
    ) -> Result<Response<ListDevicesResponse>, Status> {
        self.increment_request_count();
        
        let devices = self.gpu_manager.list_devices();
        let response = ListDevicesResponse {
            devices,
            success: true,
            error_message: String::new(),
        };
        Ok(Response::new(response))
    }
    
    async fn allocate_device(
        &self,
        request: Request<AllocateDeviceRequest>,
    ) -> Result<Response<AllocateDeviceResponse>, Status> {
        self.increment_request_count();
        let req = request.into_inner();
        
        match self.gpu_manager.allocate_device(
            req.device_id,
            req.client_id,
            req.memory_limit,
        ) {
            Ok(token) => {
                let response = AllocateDeviceResponse {
                    allocation_token: token,
                    success: true,
                    error_message: String::new(),
                };
                Ok(Response::new(response))
            }
            Err(e) => Err(e.into()),
        }
    }
    
    async fn release_device(
        &self,
        request: Request<ReleaseDeviceRequest>,
    ) -> Result<Response<ReleaseDeviceResponse>, Status> {
        self.increment_request_count();
        let req = request.into_inner();
        
        match self.gpu_manager.release_device(&req.allocation_token) {
            Ok(()) => {
                let response = ReleaseDeviceResponse {
                    success: true,
                    error_message: String::new(),
                };
                Ok(Response::new(response))
            }
            Err(e) => Err(e.into()),
        }
    }
    
    async fn allocate_memory(
        &self,
        request: Request<AllocateMemoryRequest>,
    ) -> Result<Response<AllocateMemoryResponse>, Status> {
        self.increment_request_count();
        let req = request.into_inner();
        
        let device = self.gpu_manager.get_device(req.device_id)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        
        match device.memory_manager.allocate(req.size, req.device_id) {
            Ok(allocation) => {
                let memory_pointer = MemoryPointer {
                    ptr: allocation.ptr,
                    size: allocation.size,
                    device_id: req.device_id,
                    allocation_id: allocation.id,
                    alignment: req.alignment,
                };
                
                let response = AllocateMemoryResponse {
                    memory_pointer: Some(memory_pointer),
                    success: true,
                    error_message: String::new(),
                };
                Ok(Response::new(response))
            }
            Err(e) => Err(e.into()),
        }
    }
    
    async fn deallocate_memory(
        &self,
        request: Request<DeallocateMemoryRequest>,
    ) -> Result<Response<DeallocateMemoryResponse>, Status> {
        self.increment_request_count();
        let req = request.into_inner();
        
        // Find which device this allocation belongs to
        let devices = self.gpu_manager.list_devices();
        let mut result = Err(GPUError::InternalError("Allocation not found".to_string()));
        
        for device_info in devices {
            if let Ok(device) = self.gpu_manager.get_device(device_info.device_id) {
                if device.memory_manager.get_allocation(&req.allocation_id).is_some() {
                    result = device.memory_manager.deallocate(&req.allocation_id);
                    break;
                }
            }
        }
        
        match result {
            Ok(()) => {
                let response = DeallocateMemoryResponse {
                    success: true,
                    error_message: String::new(),
                };
                Ok(Response::new(response))
            }
            Err(e) => Err(e.into()),
        }
    }
    
    type MemoryCopyStream = ResponseStream<MemoryCopyResponse>;
    
    async fn memory_copy(
        &self,
        request: Request<Streaming<MemoryCopyRequest>>,
    ) -> Result<Response<Self::MemoryCopyStream>, Status> {
        self.increment_request_count();
        let mut stream = request.into_inner();
        let (tx, rx) = mpsc::channel(32);
        
        let gpu_manager = self.gpu_manager.clone();
        tokio::spawn(async move {
            while let Some(req_result) = stream.next().await {
                match req_result {
                    Ok(copy_req) => {
                        let response = match Self::handle_memory_copy_static(&gpu_manager, copy_req).await {
                            Ok(bytes_copied) => MemoryCopyResponse {
                                success: true,
                                bytes_copied,
                                error_message: String::new(),
                                host_data: vec![],
                            },
                            Err(e) => MemoryCopyResponse {
                                success: false,
                                bytes_copied: 0,
                                error_message: e.to_string(),
                                host_data: vec![],
                            },
                        };
                        
                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(Status::internal(e.to_string()))).await;
                        break;
                    }
                }
            }
        });
        
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
    
    async fn memory_set(
        &self,
        request: Request<MemorySetRequest>,
    ) -> Result<Response<MemorySetResponse>, Status> {
        self.increment_request_count();
        let _req = request.into_inner();
        
        // TODO: Implement memory set operation
        let response = MemorySetResponse {
            success: true,
            error_message: String::new(),
        };
        Ok(Response::new(response))
    }
    
    async fn launch_kernel(
        &self,
        request: Request<LaunchKernelRequest>,
    ) -> Result<Response<LaunchKernelResponse>, Status> {
        self.increment_request_count();
        let _req = request.into_inner();
        
        // TODO: Implement kernel launch
        let response = LaunchKernelResponse {
            success: true,
            error_message: String::new(),
            kernel_id: uuid::Uuid::new_v4().to_string(),
        };
        Ok(Response::new(response))
    }
    
    async fn synchronize_device(
        &self,
        request: Request<SynchronizeDeviceRequest>,
    ) -> Result<Response<SynchronizeDeviceResponse>, Status> {
        self.increment_request_count();
        let req = request.into_inner();
        
        let _device = self.gpu_manager.get_device(req.device_id)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        
        match self.gpu_manager.synchronize_device(req.device_id) {
            Ok(()) => {
                let response = SynchronizeDeviceResponse {
                    success: true,
                    error_message: String::new(),
                };
                Ok(Response::new(response))
            }
            Err(e) => {
                let response = SynchronizeDeviceResponse {
                    success: false,
                    error_message: e.to_string(),
                };
                Ok(Response::new(response))
            }
        }
    }
    
    async fn create_stream(
        &self,
        request: Request<CreateStreamRequest>,
    ) -> Result<Response<CreateStreamResponse>, Status> {
        self.increment_request_count();
        let req = request.into_inner();
        
        let device = self.gpu_manager.get_device(req.device_id)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        
        match device.stream_manager.create_stream(req.device_id, req.flags, req.priority) {
            Ok(stream_id) => {
                let stream_info = StreamInfo {
                    stream_id,
                    device_id: req.device_id,
                    flags: req.flags,
                    priority: req.priority,
                };
                
                let response = CreateStreamResponse {
                    stream_info: Some(stream_info),
                    success: true,
                    error_message: String::new(),
                };
                Ok(Response::new(response))
            }
            Err(e) => Err(e.into()),
        }
    }
    
    async fn destroy_stream(
        &self,
        request: Request<DestroyStreamRequest>,
    ) -> Result<Response<DestroyStreamResponse>, Status> {
        self.increment_request_count();
        let req = request.into_inner();
        
        // Find which device this stream belongs to
        let devices = self.gpu_manager.list_devices();
        let mut result = Err(GPUError::InvalidStreamId(req.stream_id.clone()));
        
        for device_info in devices {
            if let Ok(device) = self.gpu_manager.get_device(device_info.device_id) {
                if device.stream_manager.destroy_stream(&req.stream_id).is_ok() {
                    result = Ok(());
                    break;
                }
            }
        }
        
        match result {
            Ok(()) => {
                let response = DestroyStreamResponse {
                    success: true,
                    error_message: String::new(),
                };
                Ok(Response::new(response))
            }
            Err(e) => Err(e.into()),
        }
    }
    
    async fn create_tensor(
        &self,
        request: Request<CreateTensorRequest>,
    ) -> Result<Response<CreateTensorResponse>, Status> {
        self.increment_request_count();
        let req = request.into_inner();
        
        let tensor_manager = self.get_tensor_manager(req.device_id)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        
        let initial_data = if req.initial_data.is_empty() {
            None
        } else {
            Some(req.initial_data.as_slice())
        };
        
        match tensor_manager.create_tensor(
            &req.shape,
            &req.dtype,
            req.device_id,
            req.requires_grad,
            initial_data,
        ) {
            Ok(tensor_id) => {
                let tensor = tensor_manager.get_tensor(&tensor_id)
                    .map_err(|e| Status::internal(e.to_string()))?;
                
                let response = CreateTensorResponse {
                    tensor: Some(tensor.descriptor),
                    success: true,
                    error_message: String::new(),
                };
                Ok(Response::new(response))
            }
            Err(e) => Err(e.into()),
        }
    }
    
    async fn destroy_tensor(
        &self,
        request: Request<DestroyTensorRequest>,
    ) -> Result<Response<DestroyTensorResponse>, Status> {
        self.increment_request_count();
        let req = request.into_inner();
        
        // Find which device this tensor belongs to
        let devices = self.gpu_manager.list_devices();
        let mut result = Err(GPUError::InvalidTensorId(req.tensor_id.clone()));
        
        for device_info in devices {
            if let Ok(tensor_manager) = self.get_tensor_manager(device_info.device_id) {
                if tensor_manager.destroy_tensor(&req.tensor_id).is_ok() {
                    result = Ok(());
                    break;
                }
            }
        }
        
        match result {
            Ok(()) => {
                let response = DestroyTensorResponse {
                    success: true,
                    error_message: String::new(),
                };
                Ok(Response::new(response))
            }
            Err(e) => Err(e.into()),
        }
    }
    
    async fn tensor_operation(
        &self,
        request: Request<TensorOperationRequest>,
    ) -> Result<Response<TensorOperationResponse>, Status> {
        self.increment_request_count();
        let req = request.into_inner();
        
        // Extract device_id from the first input tensor
        let first_tensor_id = req.input_tensors.first()
            .ok_or_else(|| Status::invalid_argument("No input tensors provided"))?;
        
        // Find which device this tensor belongs to
        let mut device_id = None;
        let devices = self.gpu_manager.list_devices();
        
        for device_info in devices {
            if let Ok(tensor_manager) = self.get_tensor_manager(device_info.device_id) {
                if tensor_manager.get_tensor(first_tensor_id).is_ok() {
                    device_id = Some(device_info.device_id);
                    break;
                }
            }
        }
        
        let device_id = device_id.ok_or_else(|| 
            Status::not_found("Input tensor not found on any device"))?;
        
        let tensor_manager = self.get_tensor_manager(device_id)
            .map_err(|e| Status::internal(e.to_string()))?;
        
        // Convert parameters
        let mut params = HashMap::new();
        for (key, param) in req.parameters {
            // Convert TensorOpParam to string for simplicity
            let value = match param.value {
                Some(networkgpu_proto::tensor_op_param::Value::IntValue(v)) => v.to_string(),
                Some(networkgpu_proto::tensor_op_param::Value::FloatValue(v)) => v.to_string(),
                Some(networkgpu_proto::tensor_op_param::Value::StringValue(v)) => v,
                Some(networkgpu_proto::tensor_op_param::Value::BoolValue(v)) => v.to_string(),
                _ => String::new(),
            };
            params.insert(key, value);
        }
        
        match tensor_manager.tensor_operation(
            TensorOpType::try_from(req.operation as i32).unwrap_or(TensorOpType::TensorOpUnknown),
            &req.input_tensors,
            &req.output_tensor,
            &params,
        ) {
            Ok(()) => {
                let result_tensor = tensor_manager.get_tensor(&req.output_tensor)
                    .map_err(|e| Status::internal(e.to_string()))?;
                
                let response = TensorOperationResponse {
                    success: true,
                    error_message: String::new(),
                    result_tensor: Some(result_tensor.descriptor),
                };
                Ok(Response::new(response))
            }
            Err(e) => Err(e.into()),
        }
    }
    
    type GetTensorDataStream = ResponseStream<TensorDataChunk>;
    
    async fn get_tensor_data(
        &self,
        request: Request<GetTensorDataRequest>,
    ) -> Result<Response<Self::GetTensorDataStream>, Status> {
        self.increment_request_count();
        let req = request.into_inner();
        
        let (tx, rx) = mpsc::channel(32);
        let gpu_manager = self.gpu_manager.clone();
        
        tokio::spawn(async move {
            // Find which device this tensor belongs to
            let devices = gpu_manager.list_devices();
            let mut tensor_manager: Option<Arc<crate::tensor::TensorManager>> = None;
            
            let _devices = devices;
            
            if let Some(tensor_manager) = tensor_manager {
                match tensor_manager.get_tensor_data(&req.tensor_id, req.offset, req.size) {
                    Ok(data) => {
                        let chunk = TensorDataChunk {
                            tensor_id: req.tensor_id,
                            offset: req.offset,
                            data,
                            total_size: req.size,
                            is_last_chunk: true,
                        };
                        let _ = tx.send(Ok(chunk)).await;
                    }
                    Err(e) => {
                        let _ = tx.send(Err(Status::internal(e.to_string()))).await;
                    }
                }
            } else {
                let _ = tx.send(Err(Status::not_found("Tensor not found"))).await;
            }
        });
        
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
    
    async fn set_tensor_data(
        &self,
        request: Request<Streaming<TensorDataChunk>>,
    ) -> Result<Response<SetTensorDataResponse>, Status> {
        self.increment_request_count();
        let mut stream = request.into_inner();
        
        let mut total_bytes = 0u64;
        let mut tensor_id = String::new();
        
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    if tensor_id.is_empty() {
                        tensor_id = chunk.tensor_id.clone();
                    }
                    
                    // Find which device this tensor belongs to
                    let devices = self.gpu_manager.list_devices();
                    let mut found = false;
                    
                    for device_info in devices {
                        if let Ok(tensor_manager) = self.get_tensor_manager(device_info.device_id) {
                            if let Ok(()) = tensor_manager.set_tensor_data(
                                &chunk.tensor_id,
                                &chunk.data,
                                chunk.offset,
                            ) {
                                total_bytes += chunk.data.len() as u64;
                                found = true;
                                break;
                            }
                        }
                    }
                    
                    if !found {
                        return Err(Status::not_found("Tensor not found"));
                    }
                }
                Err(e) => {
                    return Err(Status::internal(e.to_string()));
                }
            }
        }
        
        let response = SetTensorDataResponse {
            success: true,
            error_message: String::new(),
            bytes_written: total_bytes,
        };
        Ok(Response::new(response))
    }
    
    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let devices = self.gpu_manager.list_devices();
        let uptime = self.server_start_time.elapsed().unwrap_or_default().as_secs();
        
        let response = HealthCheckResponse {
            status: health_check_response::Status::Serving as i32,
            available_devices: devices,
            server_uptime_seconds: uptime,
            server_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        Ok(Response::new(response))
    }
    
    async fn get_server_stats(
        &self,
        _request: Request<GetServerStatsRequest>,
    ) -> Result<Response<GetServerStatsResponse>, Status> {
        let total_requests = self.request_count.load(std::sync::atomic::Ordering::Relaxed);
        let devices = self.gpu_manager.list_devices();
        
        let mut device_stats = Vec::new();
        for device_info in devices {
            if let Ok(stats) = self.gpu_manager.get_device_stats(device_info.device_id) {
                device_stats.push(stats);
            }
        }
        
        let response = GetServerStatsResponse {
            total_requests,
            active_connections: 0, // TODO: Track active connections
            total_memory_allocated: 0, // TODO: Sum across all devices
            total_tensors_created: 0, // TODO: Sum across all tensor managers
            device_stats,
            cpu_usage_percent: 0.0, // TODO: Implement CPU monitoring
            memory_usage_bytes: 0, // TODO: Implement memory monitoring
        };
        Ok(Response::new(response))
    }
}

impl NetworkGPUServiceImpl {
    async fn handle_memory_copy(&self, _req: MemoryCopyRequest) -> Result<u64, GPUError> {
        Ok(0)
    }

    async fn handle_memory_copy_static(_gpu_manager: &Arc<crate::gpu::GPUResourceManager>, _req: MemoryCopyRequest) -> Result<u64, GPUError> {
        Ok(0)
    }
}