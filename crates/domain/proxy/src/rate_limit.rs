//! Per-user rate limiting.
//!
//! Simple in-memory sliding window rate limiter. Tracks request
//! counts per user within a configurable time window.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

/// Per-user rate limiter.
pub struct RateLimiter {
    /// Max requests per window per user.
    max_requests: u32,
    /// Window duration in seconds.
    window_secs: u64,
    /// User request tracking: user_id → (count, window_start).
    state: Mutex<HashMap<String, (u32, Instant)>>,
}

impl RateLimiter {
    /// Create a new rate limiter.
    ///
    /// `max_requests`: maximum requests per user per window.
    /// `window_secs`: window duration in seconds.
    pub fn new(max_requests: u32, window_secs: u64) -> Self {
        Self {
            max_requests,
            window_secs,
            state: Mutex::new(HashMap::new()),
        }
    }

    /// Check if a request is allowed for the given user.
    ///
    /// Returns `Ok(())` if allowed, `Err(retry_after_secs)` if rate limited.
    pub fn check(&self, user_id: &str) -> Result<(), u64> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();

        let entry = state.entry(user_id.to_string()).or_insert((0, now));

        // Reset window if expired
        if now.duration_since(entry.1).as_secs() >= self.window_secs {
            entry.0 = 0;
            entry.1 = now;
        }

        if entry.0 >= self.max_requests {
            let retry_after = self.window_secs - now.duration_since(entry.1).as_secs();
            return Err(retry_after);
        }

        entry.0 += 1;
        Ok(())
    }
}
