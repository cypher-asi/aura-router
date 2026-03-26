<h1 align="center">aura-router</h1>

<p align="center">
  <b>LLM proxy and billing router for the AURA platform.</b>
</p>

## Overview

aura-router is the LLM proxy layer for AURA. All LLM requests from clients (desktop, web, mobile) route through this service. It authenticates users, checks credit balance, forwards requests to the LLM provider with the platform API key, and records usage for billing and stats.

The platform API key never reaches the client — it lives only on the server.

---

## Quick Start

### Prerequisites

- Rust toolchain
- z-billing service running (for credit checks)
- Anthropic API key (and optionally OpenAI)

### Setup

```
cp .env.example .env
# Edit .env with your API keys and service URLs

cargo run
```

The server starts on `http://0.0.0.0:3000` by default.

### Health Check

```
curl http://localhost:3000/health
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `PORT` | No | Server port (default: 3000, Render uses 10000) |
| `AUTH0_DOMAIN` | Yes | Auth0 domain for JWKS |
| `AUTH0_AUDIENCE` | Yes | Auth0 audience identifier |
| `AUTH_COOKIE_SECRET` | Yes | Shared secret for HS256 token validation (same as aura-network) |
| `INTERNAL_SERVICE_TOKEN` | Yes | Token for service-to-service auth |
| `ANTHROPIC_API_KEY` | Yes | Platform Anthropic API key |
| `OPENAI_API_KEY` | No | Platform OpenAI API key (required for GPT models) |
| `Z_BILLING_URL` | Yes | z-billing service URL |
| `Z_BILLING_API_KEY` | Yes | z-billing service API key |
| `AURA_NETWORK_URL` | No | aura-network URL for usage recording |
| `AURA_NETWORK_TOKEN` | No | aura-network internal service token |
| `AURA_STORAGE_URL` | No | aura-storage URL for message storage |
| `AURA_STORAGE_TOKEN` | No | aura-storage internal service token |
| `CORS_ORIGINS` | No | Comma-separated allowed origins. Omit for permissive (dev mode) |
| `RATE_LIMIT_RPM` | No | Max requests per minute per user (default: 60) |

---

## Authentication

All proxy endpoints require a JWT in the `Authorization: Bearer <token>` header. Tokens are obtained by logging in via zOS API (`POST https://zosapi.zero.tech/api/v2/accounts/login`).

Both RS256 (Auth0 JWKS) and HS256 (shared secret) tokens are accepted — same token format as aura-network and aura-storage.

---

## API Reference

See [docs/api.md](docs/api.md) for the full API reference.

---

## Architecture

```
Client (aura-code / mobile / web)
    |
    | JWT + Anthropic-format request
    v
aura-router
    |
    |-- 1. Validate JWT
    |-- 2. Check credits (z-billing)
    |-- 3. [Enrichment hook - future]
    |-- 4. Forward to provider (Anthropic / OpenAI)
    |-- 5. Stream response back to client
    |-- 6. Debit credits (z-billing)
    |-- 7. Record usage (aura-network)
    |-- 8. Store messages (aura-storage)
    |
    v
LLM Provider (api.anthropic.com / api.openai.com)
```

---

## License

MIT
