mod handlers;
mod router;
mod state;

use tower_http::cors::{Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use aura_router_auth::{InternalToken, TokenValidator};

use crate::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,aura_router=debug,tower_http=debug".into()),
        )
        .init();

    let auth0_domain = std::env::var("AUTH0_DOMAIN").expect("AUTH0_DOMAIN must be set");
    let auth0_audience = std::env::var("AUTH0_AUDIENCE").expect("AUTH0_AUDIENCE must be set");
    let cookie_secret =
        std::env::var("AUTH_COOKIE_SECRET").expect("AUTH_COOKIE_SECRET must be set");
    let internal_token =
        std::env::var("INTERNAL_SERVICE_TOKEN").expect("INTERNAL_SERVICE_TOKEN must be set");
    let anthropic_api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let z_billing_url = std::env::var("Z_BILLING_URL").expect("Z_BILLING_URL must be set");
    let z_billing_api_key =
        std::env::var("Z_BILLING_API_KEY").expect("Z_BILLING_API_KEY must be set");

    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "3000".to_string())
        .parse()
        .expect("PORT must be a valid number");

    let validator = TokenValidator::new(auth0_domain, auth0_audience, cookie_secret);

    let cors = match std::env::var("CORS_ORIGINS") {
        Ok(origins) => {
            let origins: Vec<_> = origins
                .split(',')
                .filter_map(|o| o.trim().parse().ok())
                .collect();
            CorsLayer::new()
                .allow_origin(origins)
                .allow_methods(Any)
                .allow_headers(Any)
        }
        Err(_) => CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any),
    };

    let rate_limit_rpm: u32 = std::env::var("RATE_LIMIT_RPM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    let state = AppState {
        validator,
        internal_token: InternalToken(internal_token),
        http_client: reqwest::Client::new(),
        rate_limiter: std::sync::Arc::new(
            aura_router_proxy::rate_limit::RateLimiter::new(rate_limit_rpm, 60),
        ),
        anthropic_api_key,
        openai_api_key: std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|s| !s.is_empty()),
        z_billing_url,
        z_billing_api_key,
        aura_network_url: std::env::var("AURA_NETWORK_URL")
            .ok()
            .filter(|s| !s.is_empty()),
        aura_network_token: std::env::var("AURA_NETWORK_TOKEN")
            .ok()
            .filter(|s| !s.is_empty()),
        aura_storage_url: std::env::var("AURA_STORAGE_URL")
            .ok()
            .filter(|s| !s.is_empty()),
        aura_storage_token: std::env::var("AURA_STORAGE_TOKEN")
            .ok()
            .filter(|s| !s.is_empty()),
    };

    let app = router::create_router()
        .with_state(state)
        .layer(cors)
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024)) // 10MB
        .layer(tower::limit::ConcurrencyLimitLayer::new(512))
        .layer(TraceLayer::new_for_http());

    let bind_addr = format!("0.0.0.0:{port}");
    tracing::info!(address = %bind_addr, "aura-router starting");

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    tracing::info!(address = %bind_addr, "aura-router listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let ctrl_c = tokio::signal::ctrl_c();
            #[cfg(unix)]
            {
                let mut sigterm =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("Failed to install SIGTERM handler");
                tokio::select! {
                    _ = ctrl_c => {},
                    _ = sigterm.recv() => {},
                }
            }
            #[cfg(not(unix))]
            {
                ctrl_c.await.ok();
            }
            tracing::info!("Shutdown signal received");
        })
        .await?;

    Ok(())
}
