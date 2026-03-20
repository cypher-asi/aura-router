use axum::routing::get;
use axum::Router;

use crate::handlers;
use crate::state::AppState;

pub fn create_router() -> Router<AppState> {
    Router::new().route("/health", get(handlers::health))
}
