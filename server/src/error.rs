use thiserror::Error;

#[derive(Error, Debug)]
pub enum GPUError {
    #[error("Invalid device ID: {0}")]
    InvalidDevice(i32),
    
    #[error("Device allocation failed")]
    AllocationFailed,
    
    #[error("Memory allocation failed: {0}")]
    MemoryAllocationFailed(String),
    
    #[error("Memory deallocation failed: {0}")]
    MemoryDeallocationFailed(String),
    
    #[error("CUDA error: {0}")]
    CudaError(String),
    
    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),
    
    #[error("Tensor operation failed: {0}")]
    TensorOperationFailed(String),
    
    #[error("Stream operation failed: {0}")]
    StreamOperationFailed(String),
    
    #[error("Device synchronization failed")]
    SynchronizationFailed,
    
    #[error("Invalid tensor ID: {0}")]
    InvalidTensorId(String),
    
    #[error("Invalid stream ID: {0}")]
    InvalidStreamId(String),
    
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl From<cudarc::driver::DriverError> for GPUError {
    fn from(err: cudarc::driver::DriverError) -> Self {
        GPUError::CudaError(err.to_string())
    }
}

impl From<GPUError> for tonic::Status {
    fn from(err: GPUError) -> Self {
        match err {
            GPUError::InvalidDevice(_) => tonic::Status::invalid_argument(err.to_string()),
            GPUError::AllocationFailed => tonic::Status::resource_exhausted(err.to_string()),
            GPUError::MemoryAllocationFailed(_) => tonic::Status::resource_exhausted(err.to_string()),
            GPUError::ResourceLimitExceeded(_) => tonic::Status::resource_exhausted(err.to_string()),
            GPUError::InvalidTensorId(_) => tonic::Status::not_found(err.to_string()),
            GPUError::InvalidStreamId(_) => tonic::Status::not_found(err.to_string()),
            _ => tonic::Status::internal(err.to_string()),
        }
    }
}