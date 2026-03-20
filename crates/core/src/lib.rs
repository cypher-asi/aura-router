pub mod errors;

pub use errors::AppError;

pub type Result<T> = std::result::Result<T, AppError>;
