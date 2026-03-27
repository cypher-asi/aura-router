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
        .route("/v1/generate-3d", post(handlers::generate_3d::generate_3d))
        .route(
            "/v1/generate-3d/stream",
            post(handlers::generate_3d::generate_3d_stream),
        )
        .route(
            "/v1/generate-3d/:taskId",
            get(handlers::generate_3d::get_3d_status),
        )
}
