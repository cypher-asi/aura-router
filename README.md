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
- Anthropic API key (and optionally OpenAI/xAI)

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
| `XAI_API_KEY` | No | Platform xAI API key (required for Grok models and xAI Remote MCP tools) |
| `MOONSHOT_API_KEY` | No | Platform Moonshot API key (required for Kimi K3) |
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
    |-- 4. Forward to provider (Anthropic / OpenAI / xAI / Moonshot)
    |-- 5. Stream response back to client
    |-- 6. Debit credits (z-billing)
    |-- 7. Record usage (aura-network)
    |-- 8. Store messages (aura-storage)
    |
    v
LLM Provider (api.anthropic.com / api.openai.com / api.x.ai / api.moonshot.ai)
```

Grok requests with tools, `xai_tools`, `server_tools`, or `xai_mcp_servers`
route through xAI's OpenAI-compatible Responses API so Remote MCP and
server-side xAI tools can run upstream.
Grok model routing uses Aura's platform `XAI_API_KEY`; user X account access
belongs behind explicit MCP server/tool integrations.
Aura prompt-cache keys are forwarded to xAI Chat Completions as
`x-grok-conv-id` and to xAI Responses as `prompt_cache_key`.
Kimi K3 routes directly to Moonshot's OpenAI-compatible Chat Completions API
using Aura's platform `MOONSHOT_API_KEY`; stable cache keys are forwarded as
`prompt_cache_key`.

---

## License

MIT
