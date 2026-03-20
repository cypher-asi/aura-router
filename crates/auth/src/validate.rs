use jsonwebtoken::{decode, Algorithm, DecodingKey, Validation};
use serde::Deserialize;

use crate::jwks::JwksClient;

const SELF_SIGNED_KID: &str = "jFNXMnFjGrSoDafnLQBohoCNalWcFcTjnKEbkRzWFBHyYJFikdLMHP";

#[derive(Debug, Clone, Deserialize)]
pub struct TokenClaims {
    pub id: Option<String>,
    pub sub: Option<String>,
}

impl TokenClaims {
    pub fn user_id(&self) -> Option<&str> {
        self.id.as_deref().or(self.sub.as_deref())
    }
}

#[derive(Clone)]
pub struct TokenValidator {
    jwks: JwksClient,
    cookie_secret: String,
    auth0_domain: String,
    auth0_audience: String,
}

impl TokenValidator {
    pub fn new(auth0_domain: String, auth0_audience: String, cookie_secret: String) -> Self {
        Self {
            jwks: JwksClient::new(&auth0_domain),
            cookie_secret,
            auth0_domain,
            auth0_audience,
        }
    }

    pub async fn validate(&self, token: &str) -> Result<TokenClaims, String> {
        let header =
            jsonwebtoken::decode_header(token).map_err(|e| format!("Invalid token header: {e}"))?;

        let kid = header.kid.as_deref().unwrap_or("");

        if kid == SELF_SIGNED_KID {
            self.validate_hs256(token)
        } else {
            self.validate_rs256(token, kid).await
        }
    }

    fn validate_hs256(&self, token: &str) -> Result<TokenClaims, String> {
        let key = DecodingKey::from_secret(self.cookie_secret.as_bytes());
        let mut validation = Validation::new(Algorithm::HS256);
        validation.validate_exp = false;
        validation.validate_aud = false;
        validation.required_spec_claims.clear();

        decode::<TokenClaims>(token, &key, &validation)
            .map(|data| data.claims)
            .map_err(|e| format!("HS256 validation failed: {e}"))
    }

    async fn validate_rs256(&self, token: &str, kid: &str) -> Result<TokenClaims, String> {
        let key = self.jwks.get_key(kid).await?;

        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_audience(&[&self.auth0_audience]);
        validation.set_issuer(&[format!("https://{}/", self.auth0_domain)]);

        decode::<TokenClaims>(token, &key, &validation)
            .map(|data| data.claims)
            .map_err(|e| format!("RS256 validation failed: {e}"))
    }
}
