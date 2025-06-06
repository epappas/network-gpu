use anyhow::Result;
use log::{debug, warn};
use networkgpu_proto::network_gpu_service_client::NetworkGpuServiceClient;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tonic::transport::{Channel, Endpoint};

use crate::error::ClientError;

#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub keepalive_time: Duration,
    pub keepalive_timeout: Duration,
    pub max_idle_connections: usize,
    pub max_connections_per_endpoint: usize,
    pub retry_attempts: usize,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(300),
            keepalive_time: Duration::from_secs(60),
            keepalive_timeout: Duration::from_secs(5),
            max_idle_connections: 8,
            max_connections_per_endpoint: 4,
            retry_attempts: 3,
        }
    }
}

#[derive(Debug)]
pub struct ConnectionPool {
    endpoints: Vec<String>,
    connections: RwLock<VecDeque<PooledConnection>>,
    round_robin_index: AtomicUsize,
    config: ConnectionConfig,
    total_connections: AtomicUsize,
}

#[derive(Debug)]
struct PooledConnection {
    client: NetworkGpuServiceClient<Channel>,
    endpoint: String,
    created_at: Instant,
    last_used: Instant,
    use_count: u64,
}

impl ConnectionPool {
    pub async fn new(
        endpoints: Vec<String>,
        config: ConnectionConfig,
    ) -> Result<Self, ClientError> {
        if endpoints.is_empty() {
            return Err(ClientError::ConfigError("No endpoints provided".to_string()));
        }
        
        let pool = Self {
            endpoints: endpoints.clone(),
            connections: RwLock::new(VecDeque::new()),
            round_robin_index: AtomicUsize::new(0),
            config,
            total_connections: AtomicUsize::new(0),
        };
        
        // Pre-create some connections
        for endpoint in &endpoints {
            for _ in 0..2 {
                if let Ok(client) = pool.create_connection(endpoint).await {
                    let connection = PooledConnection {
                        client,
                        endpoint: endpoint.clone(),
                        created_at: Instant::now(),
                        last_used: Instant::now(),
                        use_count: 0,
                    };
                    pool.connections.write().push_back(connection);
                    pool.total_connections.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        
        debug!("Created connection pool with {} pre-created connections", 
               pool.connections.read().len());
        
        Ok(pool)
    }
    
    pub async fn get_client(&self) -> Result<NetworkGpuServiceClient<Channel>, ClientError> {
        // Try to get an existing connection first
        if let Some(mut connection) = self.get_pooled_connection() {
            connection.last_used = Instant::now();
            connection.use_count += 1;
            return Ok(connection.client);
        }
        
        let endpoint = self.select_endpoint();
        let client = self.create_connection(&endpoint).await?;
        
        debug!("Created new connection to {}", endpoint);
        Ok(client)
    }
    
    pub async fn return_client(&self, client: NetworkGpuServiceClient<Channel>, endpoint: String) {
        let mut connections = self.connections.write();
        
        if connections.len() < self.config.max_idle_connections {
            let connection = PooledConnection {
                client,
                endpoint,
                created_at: Instant::now(),
                last_used: Instant::now(),
                use_count: 0,
            };
            connections.push_back(connection);
        }
    }
    
    async fn create_connection(&self, endpoint: &str) -> Result<NetworkGpuServiceClient<Channel>, ClientError> {
        let endpoint = Endpoint::from_shared(endpoint.to_string())
            .map_err(|e| ClientError::InvalidUrl(e.to_string()))?
            .connect_timeout(self.config.connect_timeout)
            .timeout(self.config.request_timeout)
            .keep_alive_while_idle(true);
        
        let channel = timeout(self.config.connect_timeout, endpoint.connect())
            .await
            .map_err(|_| ClientError::TimeoutError("Connection timeout".to_string()))?
            .map_err(ClientError::TransportError)?;
        
        Ok(NetworkGpuServiceClient::new(channel))
    }
    
    fn get_pooled_connection(&self) -> Option<PooledConnection> {
        let mut connections = self.connections.write();
        
        // Remove stale connections
        let now = Instant::now();
        while let Some(connection) = connections.front() {
            if now.duration_since(connection.last_used) > self.config.keepalive_time {
                connections.pop_front();
                self.total_connections.fetch_sub(1, Ordering::Relaxed);
            } else {
                break;
            }
        }
        
        connections.pop_front()
    }
    
    fn select_endpoint(&self) -> String {
        let index = self.round_robin_index.fetch_add(1, Ordering::Relaxed);
        self.endpoints[index % self.endpoints.len()].clone()
    }
    
    pub fn connection_count(&self) -> usize {
        self.connections.read().len()
    }
    
    pub fn total_connections(&self) -> usize {
        self.total_connections.load(Ordering::Relaxed)
    }
    
    pub async fn health_check(&self) -> Result<(), ClientError> {
        for endpoint in &self.endpoints {
            match self.create_connection(endpoint).await {
                Ok(mut client) => {
                    match client.health_check(networkgpu_proto::HealthCheckRequest {}).await {
                        Ok(response) => {
                            let health = response.into_inner();
                            if health.status != networkgpu_proto::health_check_response::Status::Serving as i32 {
                                warn!("Endpoint {} is not serving", endpoint);
                            } else {
                                debug!("Endpoint {} is healthy", endpoint);
                            }
                        }
                        Err(e) => {
                            warn!("Health check failed for {}: {}", endpoint, e);
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to connect to {}: {}", endpoint, e);
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct LoadBalancer {
    pools: Vec<Arc<ConnectionPool>>,
    strategy: LoadBalanceStrategy,
    current_index: AtomicUsize,
}

#[derive(Debug, Clone)]
pub enum LoadBalanceStrategy {
    RoundRobin,
    LeastConnections,
    Random,
}

impl LoadBalancer {
    pub fn new(pools: Vec<Arc<ConnectionPool>>, strategy: LoadBalanceStrategy) -> Self {
        Self {
            pools,
            strategy,
            current_index: AtomicUsize::new(0),
        }
    }
    
    pub async fn get_client(&self) -> Result<NetworkGpuServiceClient<Channel>, ClientError> {
        let pool = self.select_pool();
        pool.get_client().await
    }
    
    fn select_pool(&self) -> &Arc<ConnectionPool> {
        match self.strategy {
            LoadBalanceStrategy::RoundRobin => {
                let index = self.current_index.fetch_add(1, Ordering::Relaxed);
                &self.pools[index % self.pools.len()]
            }
            LoadBalanceStrategy::LeastConnections => {
                self.pools
                    .iter()
                    .min_by_key(|pool| pool.total_connections())
                    .unwrap_or(&self.pools[0])
            }
            LoadBalanceStrategy::Random => {
                let index = fastrand::usize(..self.pools.len());
                &self.pools[index]
            }
        }
    }
}