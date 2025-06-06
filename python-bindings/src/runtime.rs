use once_cell::sync::Lazy;
use tokio::runtime::Runtime;

/// Global Tokio runtime for async operations
pub static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("Failed to create Tokio runtime")
});