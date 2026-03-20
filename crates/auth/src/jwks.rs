use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use jsonwebtoken::DecodingKey;
use serde::Deserialize;

const CACHE_TTL: Duration = Duration::from_secs(300); // 5 minutes

#[derive(Debug, Deserialize)]
struct JwksResponse {
    keys: Vec<JwkKey>,
}

#[derive(Debug, Deserialize)]
struct JwkKey {
    kid: String,
    n: String,
    e: String,
}

struct CacheState {
    keys: HashMap<String, DecodingKey>,
    fetched_at: Instant,
}

#[derive(Clone)]
pub struct JwksClient {
    jwks_url: String,
    http: reqwest::Client,
    cache: Arc<RwLock<Option<CacheState>>>,
}

impl JwksClient {
    pub fn new(auth0_domain: &str) -> Self {
        Self {
            jwks_url: format!("https://{auth0_domain}/.well-known/jwks.json"),
            http: reqwest::Client::new(),
            cache: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn get_key(&self, kid: &str) -> Result<DecodingKey, String> {
        // Check cache — return if found and not expired
        {
            let cache = self.cache.read().await;
            if let Some(ref state) = *cache {
                if state.fetched_at.elapsed() < CACHE_TTL {
                    if let Some(key) = state.keys.get(kid) {
                        return Ok(key.clone());
                    }
                }
            }
        }

        // Cache miss or expired — fetch fresh JWKS
        self.refresh_and_get(kid).await
    }

    async fn refresh_and_get(&self, kid: &str) -> Result<DecodingKey, String> {
        let resp = self
            .http
            .get(&self.jwks_url)
            .send()
            .await
            .map_err(|e| format!("JWKS fetch failed: {e}"))?;

        let jwks: JwksResponse = resp
            .json()
            .await
            .map_err(|e| format!("JWKS parse failed: {e}"))?;

        let mut keys = HashMap::new();
        for key in &jwks.keys {
            if let Ok(decoding_key) = DecodingKey::from_rsa_components(&key.n, &key.e) {
                keys.insert(key.kid.clone(), decoding_key);
            }
        }

        let result = keys
            .get(kid)
            .cloned()
            .ok_or_else(|| format!("Key ID '{kid}' not found in JWKS"));

        // Update cache
        let mut cache = self.cache.write().await;
        *cache = Some(CacheState {
            keys,
            fetched_at: Instant::now(),
        });

        result
    }
}
