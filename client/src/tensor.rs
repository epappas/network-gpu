use anyhow::Result;
// use bytes::Bytes;
use futures::StreamExt;
use log::debug;
use networkgpu_proto::*;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::client::NetworkGPUClient;
use crate::error::ClientError;

#[derive(Debug, Clone)]
pub struct TensorHandle {
    pub id: String,
    pub descriptor: Arc<RwLock<TensorDescriptor>>,
    pub client: Arc<NetworkGPUClient>,
}

#[derive(Debug, Clone)]
pub struct RemoteTensor {
    pub handle: TensorHandle,
    pub shape: Vec<i64>,
    pub dtype: String,
    pub device_id: i32,
    pub requires_grad: bool,
}

impl TensorHandle {
    pub fn new(
        id: String,
        descriptor: TensorDescriptor,
        client: Arc<NetworkGPUClient>,
    ) -> Self {
        Self {
            id,
            descriptor: Arc::new(RwLock::new(descriptor)),
            client,
        }
    }
    
    pub async fn get_data(&self) -> Result<Vec<u8>, ClientError> {
        let descriptor = self.descriptor.read().await;
        let size = descriptor.size_bytes;
        drop(descriptor);
        
        self.get_data_range(0, size).await
    }
    
    pub async fn get_data_range(&self, offset: u64, size: u64) -> Result<Vec<u8>, ClientError> {
        let mut client = self.client.get_client().await?;
        
        let request = GetTensorDataRequest {
            tensor_id: self.id.clone(),
            offset,
            size,
        };
        
        let mut stream = client.get_tensor_data(request).await?.into_inner();
        let mut data = Vec::new();
        
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            data.extend_from_slice(&chunk.data);
        }
        
        debug!("Retrieved {} bytes for tensor {}", data.len(), self.id);
        Ok(data)
    }
    
    pub async fn set_data(&self, data: &[u8]) -> Result<(), ClientError> {
        self.set_data_range(data, 0).await
    }
    
    pub async fn set_data_range(&self, data: &[u8], offset: u64) -> Result<(), ClientError> {
        let mut client = self.client.get_client().await?;
        
        let (tx, rx) = tokio::sync::mpsc::channel(32);
        
        // Send data in chunks
        let tensor_id = self.id.clone();
        let data = data.to_vec();
        tokio::spawn(async move {
            const CHUNK_SIZE: usize = 64 * 1024; // 64KB chunks
            
            for (i, chunk) in data.chunks(CHUNK_SIZE).enumerate() {
                let chunk_offset = offset + (i * CHUNK_SIZE) as u64;
                let is_last = (i + 1) * CHUNK_SIZE >= data.len();
                
                let tensor_chunk = TensorDataChunk {
                    tensor_id: tensor_id.clone(),
                    offset: chunk_offset,
                    data: chunk.to_vec(),
                    total_size: data.len() as u64,
                    is_last_chunk: is_last,
                };
                
                if tx.send(tensor_chunk).await.is_err() {
                    break;
                }
            }
        });
        
        let request_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let response = client.set_tensor_data(request_stream).await?;
        
        let result = response.into_inner();
        if !result.success {
            return Err(ClientError::TensorOperationFailed(result.error_message));
        }
        
        debug!("Set {} bytes for tensor {}", result.bytes_written, self.id);
        Ok(())
    }
    
    pub async fn copy_to(&self, other: &TensorHandle) -> Result<(), ClientError> {
        let data = self.get_data().await?;
        other.set_data(&data).await
    }
    
    pub async fn shape(&self) -> Vec<i64> {
        self.descriptor.read().await.shape.clone()
    }
    
    pub async fn dtype(&self) -> String {
        self.descriptor.read().await.dtype.clone()
    }
    
    pub async fn device_id(&self) -> i32 {
        self.descriptor.read().await.device_id
    }
    
    pub async fn size_bytes(&self) -> u64 {
        self.descriptor.read().await.size_bytes
    }
    
    pub async fn requires_grad(&self) -> bool {
        self.descriptor.read().await.requires_grad
    }
}

impl RemoteTensor {
    pub fn new(handle: TensorHandle) -> Result<Self, ClientError> {
        let descriptor = match handle.descriptor.try_read() {
            Ok(desc) => desc.clone(),
            Err(_) => return Err(ClientError::InternalError("Cannot access tensor descriptor".to_string())),
        };
        
        Ok(Self {
            shape: descriptor.shape.clone(),
            dtype: descriptor.dtype.clone(),
            device_id: descriptor.device_id,
            requires_grad: descriptor.requires_grad,
            handle,
        })
    }
    
    pub async fn add(&self, other: &RemoteTensor) -> Result<RemoteTensor, ClientError> {
        self.binary_operation(other, TensorOpType::TensorOpAdd).await
    }
    
    pub async fn subtract(&self, other: &RemoteTensor) -> Result<RemoteTensor, ClientError> {
        self.binary_operation(other, TensorOpType::TensorOpSub).await
    }
    
    pub async fn multiply(&self, other: &RemoteTensor) -> Result<RemoteTensor, ClientError> {
        self.binary_operation(other, TensorOpType::TensorOpMul).await
    }
    
    pub async fn divide(&self, other: &RemoteTensor) -> Result<RemoteTensor, ClientError> {
        self.binary_operation(other, TensorOpType::TensorOpDiv).await
    }
    
    pub async fn matmul(&self, other: &RemoteTensor) -> Result<RemoteTensor, ClientError> {
        self.binary_operation(other, TensorOpType::TensorOpMatmul).await
    }
    
    async fn binary_operation(
        &self,
        other: &RemoteTensor,
        op_type: TensorOpType,
    ) -> Result<RemoteTensor, ClientError> {
        // Calculate output shape based on operation
        let output_shape = match op_type {
            TensorOpType::TensorOpMatmul => self.calculate_matmul_shape(&other.shape)?,
            _ => self.shape.clone(), // Element-wise operations preserve shape
        };
        
        let output_tensor = self.handle.client.create_tensor(
            &output_shape,
            &self.dtype,
            self.device_id,
            false, // No gradients for now
            None,
        ).await?;
        
        // Execute operation
        let mut client = self.handle.client.get_client().await?;
        
        let request = TensorOperationRequest {
            operation: op_type as i32,
            input_tensors: vec![self.handle.id.clone(), other.handle.id.clone()],
            output_tensor: output_tensor.id.clone(),
            parameters: std::collections::HashMap::new(),
            stream_id: String::new(),
        };
        
        let response = client.tensor_operation(request).await?;
        let result = response.into_inner();
        
        if !result.success {
            return Err(ClientError::TensorOperationFailed(result.error_message));
        }
        
        RemoteTensor::new(output_tensor)
    }
    
    pub async fn transpose(&self, dim0: i64, dim1: i64) -> Result<RemoteTensor, ClientError> {
        let mut output_shape = self.shape.clone();
        output_shape.swap(dim0 as usize, dim1 as usize);
        
        let output_tensor = self.handle.client.create_tensor(
            &output_shape,
            &self.dtype,
            self.device_id,
            false,
            None,
        ).await?;
        
        let mut client = self.handle.client.get_client().await?;
        
        let mut parameters = std::collections::HashMap::new();
        parameters.insert(
            "dim0".to_string(),
            networkgpu_proto::TensorOpParam {
                value: Some(networkgpu_proto::tensor_op_param::Value::IntValue(dim0)),
            },
        );
        parameters.insert(
            "dim1".to_string(),
            networkgpu_proto::TensorOpParam {
                value: Some(networkgpu_proto::tensor_op_param::Value::IntValue(dim1)),
            },
        );
        
        let request = TensorOperationRequest {
            operation: TensorOpType::TensorOpTranspose as i32,
            input_tensors: vec![self.handle.id.clone()],
            output_tensor: output_tensor.id.clone(),
            parameters,
            stream_id: String::new(),
        };
        
        let response = client.tensor_operation(request).await?;
        let result = response.into_inner();
        
        if !result.success {
            return Err(ClientError::TensorOperationFailed(result.error_message));
        }
        
        RemoteTensor::new(output_tensor)
    }
    
    pub async fn reshape(&self, new_shape: &[i64]) -> Result<RemoteTensor, ClientError> {
        // Validate that the total number of elements is preserved
        let current_elements: i64 = self.shape.iter().product();
        let new_elements: i64 = new_shape.iter().product();
        
        if current_elements != new_elements {
            return Err(ClientError::TensorOperationFailed(
                format!("Cannot reshape tensor with {} elements to shape with {} elements",
                        current_elements, new_elements)
            ));
        }
        
        let output_tensor = self.handle.client.create_tensor(
            new_shape,
            &self.dtype,
            self.device_id,
            false,
            None,
        ).await?;
        
        let mut client = self.handle.client.get_client().await?;
        
        let mut parameters = std::collections::HashMap::new();
        parameters.insert(
            "shape".to_string(),
            networkgpu_proto::TensorOpParam {
                value: Some(networkgpu_proto::tensor_op_param::Value::IntArray(
                    networkgpu_proto::IntArray {
                        values: new_shape.to_vec(),
                    }
                )),
            },
        );
        
        let request = TensorOperationRequest {
            operation: TensorOpType::TensorOpReshape as i32,
            input_tensors: vec![self.handle.id.clone()],
            output_tensor: output_tensor.id.clone(),
            parameters,
            stream_id: String::new(),
        };
        
        let response = client.tensor_operation(request).await?;
        let result = response.into_inner();
        
        if !result.success {
            return Err(ClientError::TensorOperationFailed(result.error_message));
        }
        
        RemoteTensor::new(output_tensor)
    }
    
    pub async fn sum(&self, dim: Option<i64>) -> Result<RemoteTensor, ClientError> {
        let output_shape = if let Some(dim) = dim {
            let mut shape = self.shape.clone();
            shape.remove(dim as usize);
            shape
        } else {
            vec![1] // Scalar result
        };
        
        let output_tensor = self.handle.client.create_tensor(
            &output_shape,
            &self.dtype,
            self.device_id,
            false,
            None,
        ).await?;
        
        let mut client = self.handle.client.get_client().await?;
        
        let mut parameters = std::collections::HashMap::new();
        if let Some(dim) = dim {
            parameters.insert(
                "dim".to_string(),
                networkgpu_proto::TensorOpParam {
                    value: Some(networkgpu_proto::tensor_op_param::Value::IntValue(dim)),
                },
            );
        }
        
        let request = TensorOperationRequest {
            operation: TensorOpType::TensorOpReduceSum as i32,
            input_tensors: vec![self.handle.id.clone()],
            output_tensor: output_tensor.id.clone(),
            parameters,
            stream_id: String::new(),
        };
        
        let response = client.tensor_operation(request).await?;
        let result = response.into_inner();
        
        if !result.success {
            return Err(ClientError::TensorOperationFailed(result.error_message));
        }
        
        RemoteTensor::new(output_tensor)
    }
    
    pub async fn sigmoid(&self) -> Result<RemoteTensor, ClientError> {
        self.unary_operation(TensorOpType::TensorOpSigmoid).await
    }
    
    pub async fn relu(&self) -> Result<RemoteTensor, ClientError> {
        self.unary_operation(TensorOpType::TensorOpRelu).await
    }
    
    async fn unary_operation(&self, op_type: TensorOpType) -> Result<RemoteTensor, ClientError> {
        let output_tensor = self.handle.client.create_tensor(
            &self.shape,
            &self.dtype,
            self.device_id,
            false,
            None,
        ).await?;
        
        let mut client = self.handle.client.get_client().await?;
        
        let request = TensorOperationRequest {
            operation: op_type as i32,
            input_tensors: vec![self.handle.id.clone()],
            output_tensor: output_tensor.id.clone(),
            parameters: std::collections::HashMap::new(),
            stream_id: String::new(),
        };
        
        let response = client.tensor_operation(request).await?;
        let result = response.into_inner();
        
        if !result.success {
            return Err(ClientError::TensorOperationFailed(result.error_message));
        }
        
        RemoteTensor::new(output_tensor)
    }
    
    pub async fn to_cpu(&self) -> Result<Vec<u8>, ClientError> {
        self.handle.get_data().await
    }
    
    fn calculate_matmul_shape(&self, other_shape: &[i64]) -> Result<Vec<i64>, ClientError> {
        if self.shape.len() < 2 || other_shape.len() < 2 {
            return Err(ClientError::TensorOperationFailed(
                "Matrix multiplication requires at least 2D tensors".to_string()
            ));
        }
        
        let m = self.shape[self.shape.len() - 2];
        let k1 = self.shape[self.shape.len() - 1];
        let k2 = other_shape[other_shape.len() - 2];
        let n = other_shape[other_shape.len() - 1];
        
        if k1 != k2 {
            return Err(ClientError::TensorOperationFailed(
                format!("Matrix dimensions don't match for multiplication: {} vs {}", k1, k2)
            ));
        }
        
        // For simplicity, assume 2D matmul for now
        Ok(vec![m, n])
    }
}