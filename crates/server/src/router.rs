use axum::routing::{get, post};
use axum::Router;

use crate::handlers;
use crate::state::AppState;

pub fn create_router() -> Router<AppState> {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/messages", post(handlers::proxy::messages))
        .route("/v1/generate-image", post(handlers::image_gen::generate_image))
        .route(
            "/v1/generate-image/stream",
            post(handlers::image_gen::generate_image_stream),
        )
        .route(
            "/v1/generate-image/config",
            get(handlers::image_gen::generate_image_config),
        )
}
