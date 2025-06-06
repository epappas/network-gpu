use anyhow::Result;
use dashmap::DashMap;
use log::debug;
use uuid::Uuid;

use crate::error::GPUError;

pub type StreamId = String;

#[derive(Debug)]
pub struct StreamManager {
    device_id: i32,
    streams: DashMap<StreamId, StreamInfo>,
    max_streams: usize,
}

#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub id: StreamId,
    pub device_id: i32,
    pub flags: u32,
    pub priority: i32,
    pub created_at: std::time::SystemTime,
}

impl StreamManager {
    pub fn new(device_id: i32, max_streams: usize) -> Result<Self, GPUError> {
        Ok(Self {
            device_id,
            streams: DashMap::new(),
            max_streams,
        })
    }
    
    pub fn create_stream(
        &self,
        device_id: i32,
        flags: u32,
        priority: i32,
    ) -> Result<StreamId, GPUError> {
        if self.streams.len() >= self.max_streams {
            return Err(GPUError::ResourceLimitExceeded(
                "Maximum number of streams reached".to_string()
            ));
        }
        
        let stream_id = Uuid::new_v4().to_string();
        let stream_info = StreamInfo {
            id: stream_id.clone(),
            device_id,
            flags,
            priority,
            created_at: std::time::SystemTime::now(),
        };
        
        self.streams.insert(stream_id.clone(), stream_info);
        debug!("Created stream {} for device {}", stream_id, device_id);
        
        Ok(stream_id)
    }
    
    pub fn destroy_stream(&self, stream_id: &str) -> Result<(), GPUError> {
        if let Some((_, stream_info)) = self.streams.remove(stream_id) {
            // Stream will be automatically destroyed when dropped
            debug!("Destroyed stream {} for device {}", stream_id, stream_info.device_id);
            Ok(())
        } else {
            Err(GPUError::InvalidStreamId(stream_id.to_string()))
        }
    }
    
    pub fn get_stream(&self, stream_id: &str) -> Result<StreamInfo, GPUError> {
        self.streams.get(stream_id)
            .map(|entry| (*entry.value()).clone())
            .ok_or_else(|| GPUError::InvalidStreamId(stream_id.to_string()))
    }
    
    pub fn synchronize_stream(&self, stream_id: &str) -> Result<(), GPUError> {
        let _stream = self.get_stream(stream_id)?;
        debug!("Simulating stream synchronization for {}", stream_id);
        Ok(())
    }
    
    pub fn list_streams(&self) -> Vec<String> {
        self.streams.iter().map(|entry| entry.key().clone()).collect()
    }
    
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }
}