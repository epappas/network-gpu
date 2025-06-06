use anyhow::Result;
use dashmap::DashMap;
use log::debug;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

use crate::error::GPUError;

pub type AllocationId = String;

#[derive(Debug)]
pub struct MemoryManager {
    device_id: i32,
    allocations: DashMap<AllocationId, MemoryAllocation>,
    memory_pool: Mutex<MemoryPool>,
    total_allocated: AtomicU64,
    pool_size: u64,
    alignment: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub id: AllocationId,
    pub ptr: u64, // Use u64 to store pointer addresses for now
    pub size: u64,
    pub device_id: i32,
    pub allocated_at: std::time::SystemTime,
    pub from_pool: bool,
}

#[derive(Debug)]
struct MemoryPool {
    free_blocks: VecDeque<MemoryBlock>,
    total_size: u64,
    used_size: u64,
}

#[derive(Debug, Clone)]
struct MemoryBlock {
    ptr: u64,
    size: u64,
}

impl MemoryManager {
    pub fn new(
        device_id: i32,
        pool_size: u64,
        alignment: usize,
    ) -> Result<Self, GPUError> {
        // For now, create an empty memory pool - actual CUDA allocation will be implemented later
        let memory_pool = MemoryPool {
            free_blocks: VecDeque::new(),
            total_size: pool_size,
            used_size: 0,
        };
        
        Ok(Self {
            device_id,
            allocations: DashMap::new(),
            memory_pool: Mutex::new(memory_pool),
            total_allocated: AtomicU64::new(0),
            pool_size,
            alignment,
        })
    }
    
    pub fn allocate(&self, size: u64, device_id: i32) -> Result<MemoryAllocation, GPUError> {
        let aligned_size = self.align_size(size);
        
        // Try to allocate from pool first
        if let Some(allocation) = self.allocate_from_pool(aligned_size, device_id)? {
            return Ok(allocation);
        }
        
        // Fall back to direct allocation
        self.allocate_direct(aligned_size, device_id)
    }
    
    fn allocate_from_pool(
        &self,
        size: u64,
        device_id: i32,
    ) -> Result<Option<MemoryAllocation>, GPUError> {
        let mut pool = self.memory_pool.lock();
        
        // Find a suitable block
        for (index, block) in pool.free_blocks.iter().enumerate() {
            if block.size >= size {
                let block = pool.free_blocks.remove(index).unwrap();
                
                // Split block if necessary
                if block.size > size {
                    let remaining_block = MemoryBlock {
                        ptr: block.ptr + size,
                        size: block.size - size,
                    };
                    pool.free_blocks.push_back(remaining_block);
                }
                
                pool.used_size += size;
                
                let allocation_id = Uuid::new_v4().to_string();
                let allocation = MemoryAllocation {
                    id: allocation_id.clone(),
                    ptr: block.ptr,
                    size,
                    device_id,
                    allocated_at: std::time::SystemTime::now(),
                    from_pool: true,
                };
                
                self.allocations.insert(allocation_id, allocation.clone());
                self.total_allocated.fetch_add(size, Ordering::Relaxed);
                
                debug!("Allocated {} bytes from pool for device {}", size, device_id);
                return Ok(Some(allocation));
            }
        }
        
        Ok(None)
    }
    
    fn allocate_direct(&self, size: u64, device_id: i32) -> Result<MemoryAllocation, GPUError> {
        // For now, simulate allocation with a dummy pointer - actual CUDA allocation will be implemented later
        let ptr = 0x1000u64 + (fastrand::u64(..0x1000000) << 12); // Simulate unique pointer
        
        let allocation_id = Uuid::new_v4().to_string();
        let allocation = MemoryAllocation {
            id: allocation_id.clone(),
            ptr,
            size,
            device_id,
            allocated_at: std::time::SystemTime::now(),
            from_pool: false,
        };
        
        self.allocations.insert(allocation_id, allocation.clone());
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
        
        debug!("Direct allocated {} bytes for device {}", size, device_id);
        Ok(allocation)
    }
    
    pub fn deallocate(&self, allocation_id: &str) -> Result<(), GPUError> {
        let (_, allocation) = self.allocations.remove(allocation_id)
            .ok_or_else(|| GPUError::MemoryDeallocationFailed(
                format!("Invalid allocation ID: {}", allocation_id)
            ))?;
        
        self.total_allocated.fetch_sub(allocation.size, Ordering::Relaxed);
        
        if allocation.from_pool {
            // Return to pool
            let mut pool = self.memory_pool.lock();
            pool.free_blocks.push_back(MemoryBlock {
                ptr: allocation.ptr,
                size: allocation.size,
            });
            pool.used_size -= allocation.size;
            
            // Coalesce adjacent blocks
            self.coalesce_blocks(&mut pool);
            
            debug!("Returned {} bytes to pool", allocation.size);
        } else {
            // Direct deallocation - CUDA handles this automatically when DevicePtr is dropped
            debug!("Direct deallocated {} bytes", allocation.size);
        }
        
        Ok(())
    }
    
    pub fn copy_host_to_device(
        &self,
        allocation_id: &str,
        host_data: &[u8],
        offset: u64,
    ) -> Result<(), GPUError> {
        let allocation = self.allocations.get(allocation_id)
            .ok_or_else(|| GPUError::MemoryDeallocationFailed(
                format!("Invalid allocation ID: {}", allocation_id)
            ))?;
        
        if offset + host_data.len() as u64 > allocation.size {
            return Err(GPUError::MemoryAllocationFailed(
                "Data size exceeds allocation size".to_string()
            ));
        }
        
        let _dest_ptr = allocation.ptr + offset;
        
        // For now, simulate memory copy - actual CUDA copy will be implemented later
        debug!("Simulating host to device copy of {} bytes", host_data.len());
        
        Ok(())
    }
    
    pub fn copy_device_to_host(
        &self,
        allocation_id: &str,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, GPUError> {
        let allocation = self.allocations.get(allocation_id)
            .ok_or_else(|| GPUError::MemoryDeallocationFailed(
                format!("Invalid allocation ID: {}", allocation_id)
            ))?;
        
        if offset + size > allocation.size {
            return Err(GPUError::MemoryAllocationFailed(
                "Requested size exceeds allocation size".to_string()
            ));
        }
        
        let _src_ptr = allocation.ptr + offset;
        
        // For now, simulate memory copy - actual CUDA copy will be implemented later
        let host_data = vec![0u8; size as usize];
        debug!("Simulating device to host copy of {} bytes", size);
        
        Ok(host_data)
    }
    
    pub fn get_allocation(&self, allocation_id: &str) -> Option<MemoryAllocation> {
        self.allocations.get(allocation_id).map(|entry| entry.value().clone())
    }
    
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated.load(Ordering::Relaxed)
    }
    
    fn align_size(&self, size: u64) -> u64 {
        let alignment = self.alignment as u64;
        (size + alignment - 1) & !(alignment - 1)
    }
    
    fn coalesce_blocks(&self, pool: &mut MemoryPool) {
        // Sort blocks by address
        pool.free_blocks.make_contiguous().sort_by_key(|block| block.ptr);
        
        let mut i = 0;
        while i < pool.free_blocks.len().saturating_sub(1) {
            let current = &pool.free_blocks[i];
            let next = &pool.free_blocks[i + 1];
            
            // Check if blocks are adjacent
            if current.ptr + current.size == next.ptr {
                // Merge blocks
                let merged_block = MemoryBlock {
                    ptr: current.ptr,
                    size: current.size + next.size,
                };
                
                pool.free_blocks.remove(i + 1);
                pool.free_blocks[i] = merged_block;
            } else {
                i += 1;
            }
        }
    }
}