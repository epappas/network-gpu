use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClientError {
    #[error("Connection error: {0}")]
    ConnectionError(String),
    
    #[error("gRPC error: {0}")]
    GrpcError(#[from] tonic::Status),
    
    #[error("Transport error: {0}")]
    TransportError(#[from] tonic::transport::Error),
    
    #[error("Invalid server URL: {0}")]
    InvalidUrl(String),
    
    #[error("Device operation failed: {0}")]
    DeviceOperationFailed(String),
    
    #[error("Memory operation failed: {0}")]
    MemoryOperationFailed(String),
    
    #[error("Tensor operation failed: {0}")]
    TensorOperationFailed(String),
    
    #[error("Invalid tensor ID: {0}")]
    InvalidTensorId(String),
    
    #[error("Invalid device ID: {0}")]
    InvalidDeviceId(i32),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Timeout error: {0}")]
    TimeoutError(String),
    
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl From<std::io::Error> for ClientError {
    fn from(err: std::io::Error) -> Self {
        ClientError::ConnectionError(err.to_string())
    }
}

// impl From<serde_json::Error> for ClientError {
//     fn from(err: serde_json::Error) -> Self {
//         ClientError::SerializationError(err.to_string())
//     }
// }