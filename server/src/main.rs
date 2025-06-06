use anyhow::Result;
use clap::Parser;
use env_logger;
use log::info;
use std::net::SocketAddr;
use tonic::transport::Server;

mod gpu;
mod memory;
mod service;
mod tensor;
mod stream;
mod error;
mod config;

use config::ServerConfig;
use service::NetworkGPUServiceImpl;

#[derive(Parser, Debug)]
#[command(name = "networkgpu-server")]
#[command(about = "NetworkGPU Server - Remote GPU access via gRPC")]
struct Args {
    #[arg(short, long, default_value = "0.0.0.0:50051")]
    address: SocketAddr,
    
    #[arg(short, long)]
    config: Option<String>,
    
    #[arg(short, long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(&args.log_level)
    ).init();
    
    info!("Starting NetworkGPU Server on {}", args.address);
    
    // Load configuration
    let config = if let Some(config_path) = args.config {
        ServerConfig::from_file(&config_path)?
    } else {
        ServerConfig::default()
    };
    
    // Initialize GPU service
    let gpu_service = NetworkGPUServiceImpl::new(config).await?;
    
    info!("GPU service initialized successfully");
    info!("Available devices: {}", gpu_service.device_count());
    
    // Start gRPC server
    Server::builder()
        .add_service(networkgpu_proto::network_gpu_service_server::NetworkGpuServiceServer::new(gpu_service))
        .serve(args.address)
        .await?;
    
    Ok(())
}