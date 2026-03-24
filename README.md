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

### Health

| Method | Path | Description | Auth |
|---|---|---|---|
| GET | `/health` | Liveness check | None |

### LLM Proxy

| Method | Path | Description | Auth |
|---|---|---|---|
| POST | `/v1/messages` | Proxy LLM request | JWT |

The `/v1/messages` endpoint accepts Anthropic-compatible request bodies. The router resolves the provider from the `model` field and forwards accordingly.

**Supported models:**
- Anthropic: `claude-*` (e.g., `claude-sonnet-4-6`, `claude-opus-4-6`)
- OpenAI: `gpt-*`, `o1-*`, `o3-*`, `o4-*`, `codex-*` (requires `OPENAI_API_KEY`)

**Request flow:**
1. Authenticate via JWT (`Authorization: Bearer <token>`)
2. Check credits via z-billing (pre-flight balance check)
3. Forward to LLM provider with platform API key
4. Stream or return response to client
5. Debit credits via z-billing (post-completion)
6. Record usage to aura-network (stats)
7. Store messages to aura-storage (if session context headers present)

**Streaming:** Set `"stream": true` in the request body. Response streams as `text/event-stream` (SSE). Token counts are captured from the stream for billing.

**Session context headers (optional, for message storage):**
- `X-Aura-Session-Id` — Session UUID
- `X-Aura-Agent-Id` — Project agent UUID
- `X-Aura-Project-Id` — Project UUID
- `X-Aura-Org-Id` — Organization UUID

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

### Error Handling

| Failure | Response |
|---|---|
| Invalid/missing JWT | 401 Unauthorized |
| Insufficient credits | 402 Payment Required |
| Rate limited | 429 Too Many Requests (with Retry-After header) |
| Unsupported model | 400 Bad Request |
| z-billing unreachable | 503 Service Unavailable |
| Provider unreachable | 502 Bad Gateway |
| Provider error (429, 500, etc.) | Passthrough (same status) |
| aura-network/storage unreachable | Logged, response not affected |

---

## Cross-Service Integration

| Service | How aura-router calls it |
|---|---|
| z-billing | `POST /v1/usage/check` (pre-check), `POST /v1/usage` (debit) via `X-API-Key` |
| aura-network | `POST /internal/usage` via `X-Internal-Token` |
| aura-storage | `POST /internal/events` via `X-Internal-Token` |

---

## License

MIT
