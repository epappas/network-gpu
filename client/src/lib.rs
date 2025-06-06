pub mod connection;
pub mod client;
pub mod error;
pub mod tensor;
pub mod cache;

pub use client::NetworkGPUClient;
pub use connection::ConnectionPool;
pub use error::ClientError;
pub use tensor::{RemoteTensor, TensorHandle};

// use networkgpu_proto::*;