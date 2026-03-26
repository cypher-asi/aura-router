# aura-router API Reference

LLM proxy service for the Aura platform. Authenticates users, enforces rate limits and credit billing, then forwards requests to the appropriate LLM provider using platform-managed API keys.

Base URL: `https://<deployment>/`

---

## Authentication

All authenticated endpoints require a JWT in the `Authorization` header:

```
Authorization: Bearer <token>
```

Two signing algorithms are accepted:

| Algorithm | Source |
|-----------|--------|
| RS256 | Auth0 JWKS (same tokens issued by aura-network) |
| HS256 | Shared secret (`AUTH_COOKIE_SECRET`) |

---

## Endpoints

### GET /health

Health check. No authentication required.

**Response** `200 OK`

```json
{
  "status": "ok",
  "timestamp": "2026-03-24T12:00:00.000Z"
}
```

---

### POST /v1/messages

Anthropic-compatible LLM proxy. Authenticates the caller, verifies credit balance, forwards the request to the resolved LLM provider, returns the response, and records usage in the background.

**Authentication:** JWT (required)

**Content-Type:** `application/json`

**Body size limit:** 25 MB (supports image content blocks)

#### Request Body

Follows the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) format. All fields not listed below are passed through to the provider untouched.

##### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model identifier. Determines which provider receives the request (see [Provider Routing](#provider-routing)). |
| `messages` | array | Conversation history. Each element is an object with `role` (`"user"` or `"assistant"`) and `content`. |

##### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stream` | boolean | `false` | Enable Server-Sent Events streaming. |
| `max_tokens` | integer | — | Maximum number of tokens to generate. |
| `temperature` | float | — | Sampling temperature. |
| `system` | string | — | System prompt prepended to the conversation. |

Any additional Anthropic-compatible fields (e.g. `top_p`, `top_k`, `stop_sequences`, `metadata`, `tools`, `tool_choice`) are forwarded as-is.

##### Optional Headers

These headers attach the request to an Aura session for event recording in aura-storage. All are optional; if omitted, the request is still proxied but no session events are stored.

| Header | Type | Description |
|--------|------|-------------|
| `X-Aura-Session-Id` | UUID | Session identifier |
| `X-Aura-Agent-Id` | UUID | Project agent identifier |
| `X-Aura-Project-Id` | UUID | Project identifier |
| `X-Aura-Org-Id` | UUID | Organization identifier (optional even when other session headers are present) |

#### Provider Routing

The `model` field determines which upstream provider handles the request.

| Model prefix | Provider | Upstream endpoint |
|-------------|----------|-------------------|
| `claude-*` | Anthropic | `https://api.anthropic.com/v1/messages` |
| `gpt-*`, `o1-*`, `o3-*`, `o4-*`, `codex-*` | OpenAI | `https://api.openai.com/v1/chat/completions` |

Unsupported model prefixes return `400 Bad Request`.

OpenAI routing requires the `OPENAI_API_KEY` environment variable to be configured; if it is absent, requests for OpenAI models return `400 Bad Request`.

#### Non-Streaming Response

When `stream` is `false` (or omitted), the provider's full JSON response is returned as-is.

**Response headers:**

```
Content-Type: application/json
X-Context-Usage: 0.4532
X-Model-Max-Tokens: 200000
```

| Header | Description |
|--------|-------------|
| `X-Context-Usage` | Float (0-1) representing how much of the model's context window has been consumed. Calculated from `input_tokens / max_context_tokens`. |
| `X-Model-Max-Tokens` | Maximum context window size for the model (in tokens). |

#### Streaming Response

When `stream` is `true`, the provider's SSE stream is forwarded to the client with context usage appended.

**Response headers:**

```
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
X-Model-Max-Tokens: 200000
```

Each event follows the standard SSE format (`data: {...}\n\n`). The final provider event is `data: [DONE]`.

After the provider stream ends, the router appends a custom `x_context_usage` event:

```
event: x_context_usage
data: {"contextUsage":0.4532,"inputTokens":90640,"outputTokens":1500,"maxTokens":200000}
```

| Field | Description |
|-------|-------------|
| `contextUsage` | Float (0-1) representing context window consumption. |
| `inputTokens` | Total input tokens for this request. |
| `outputTokens` | Total output tokens generated. |
| `maxTokens` | Maximum context window for the model. |

#### Request Flow

```
Client                    aura-router              z-billing         Provider        Background
  |                            |                       |                 |                |
  |-- POST /v1/messages ------>|                       |                 |                |
  |                            |-- 1. Validate JWT     |                 |                |
  |                            |-- 2. Rate limit check |                 |                |
  |                            |-- 3. Parse model      |                 |                |
  |                            |-- 4. Resolve provider |                 |                |
  |                            |-- 5. Pre-check ------>|                 |                |
  |                            |   (min 1 credit)      |                 |                |
  |                            |<-- credits ok --------|                 |                |
  |                            |-- 6. Forward request ------------------>|                |
  |<-- 7. Return response -----|<------------------------------------- --|                |
  |                            |-- 8. Debit actual cost --------------->z-billing         |
  |                            |-- 9. Record usage ------------------->aura-network       |
  |                            |-- 10. Store events ------------------>aura-storage       |
```

1. **Validate JWT** — Verify the bearer token (RS256 via JWKS or HS256 via shared secret). Reject with `401` on failure.
2. **Rate limit check** — Enforce per-user sliding window. Reject with `429` if exceeded.
3. **Parse request** — Extract `model` and `stream` from the request body. Reject with `400` if the body is invalid or `model` is missing.
4. **Resolve provider** — Map the model prefix to a provider. Reject with `400` if the model is unsupported or the provider is not configured.
5. **Pre-check credits** — Call z-billing to confirm the user has at least 1 credit. Reject with `402` if insufficient, `503` if z-billing is unreachable.
6. **Forward to provider** — Send the request to the upstream provider using the platform API key. Return `502` if the provider is unreachable.
7. **Return response** — Stream or return the provider response to the client.
8. **Debit credits** _(background)_ — Post actual token usage cost to z-billing. Fire-and-forget.
9. **Record usage** _(background)_ — Post usage stats to aura-network. Fire-and-forget.
10. **Store events** _(background)_ — If session headers were present, post conversation events to aura-storage. Fire-and-forget.

#### Rate Limiting

Requests are rate-limited per user using a sliding window algorithm.

| Parameter | Value |
|-----------|-------|
| Window | 1 minute (sliding) |
| Default limit | 60 requests per minute |
| Configurable via | `RATE_LIMIT_RPM` environment variable |

When the limit is exceeded, the response includes a `Retry-After` header indicating how many seconds the client should wait before retrying.

#### Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description"
  }
}
```

##### Error Codes

| HTTP Status | Code | Description |
|-------------|------|-------------|
| 400 | `BAD_REQUEST` | Invalid JSON, missing `model` field, unsupported model prefix, or OpenAI provider not configured |
| 401 | `UNAUTHORIZED` | Missing or invalid JWT |
| 402 | `INSUFFICIENT_CREDITS` | User does not have enough credits. Balance and required amount included in the message string. |
| 429 | `RATE_LIMITED` | Per-user rate limit exceeded. Response includes `Retry-After` header. |
| 502 | `PROVIDER_ERROR` | Upstream LLM provider is unreachable or returned an unexpected error |
| 503 | `BILLING_UNAVAILABLE` | z-billing service is unreachable |

##### 402 Example

```json
{
  "error": {
    "code": "INSUFFICIENT_CREDITS",
    "message": "Insufficient credits: balance=0, required=1"
  }
}
```

##### 429 Example

```
HTTP/1.1 429 Too Many Requests
Retry-After: 12
Content-Type: application/json

{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Too many requests. Retry after 12 seconds."
  }
}
```

---

## Cross-Service Integration

aura-router communicates with three backend services. The pre-check call is synchronous and blocks the request; all other calls are fire-and-forget in the background after the client receives its response.

### z-billing

| Operation | Method | Endpoint | Timing |
|-----------|--------|----------|--------|
| Credit pre-check | POST | `/v1/usage/check` | Synchronous (blocks request if insufficient) |
| Debit actual cost | POST | `/v1/usage` | Background (fire-and-forget) |

### aura-network

| Operation | Method | Endpoint | Timing |
|-----------|--------|----------|--------|
| Record usage stats | POST | `/internal/usage` | Background (fire-and-forget). Sends `orgId`, `projectId`, `zeroUserId`, `durationMs`. |

### aura-storage

| Operation | Method | Endpoint | Timing |
|-----------|--------|----------|--------|
| Store conversation events | POST | `/internal/events` | Background (fire-and-forget) |

---

## Image Generation

### POST /v1/generate-image

Generate an image using OpenAI or Gemini. Returns S3 URLs for watermarked and original images.

**Authentication:** JWT (required)

**Request body:**

```json
{
  "prompt": "string (required)",
  "size": "1024x1024 | 1536x1024 | 1024x1536 | 256x256 | 512x512 | auto (default: 1024x1024)",
  "model": "gpt-image-1 | dall-e-3 | dall-e-2 | gemini-nano-banana (default: gpt-image-1)",
  "images": ["url or base64 data URL"] (optional, reference images)
}
```

**Response:** `200 OK`

```json
{
  "success": true,
  "imageUrl": "https://aura-images.s3...watermarked.png",
  "originalUrl": "https://aura-images.s3...original.png",
  "meta": {
    "model": "gpt-image-1",
    "size": "1024x1024",
    "prompt": "original prompt",
    "provider": "openai",
    "created": 1711234567
  }
}
```

A style lock prompt is automatically appended to all generation requests for consistent product render output.

**Billing:** Flat per-generation cost (500 credits for GPT-Image-1, 400 for DALL-E 3, 200 for DALL-E 2, 300 for Gemini).

---

### POST /v1/generate-image/stream

Same as above but returns SSE stream with progress and partial image events.

**Authentication:** JWT (required)

**Request body:** Same as `POST /v1/generate-image`

**Response:** SSE stream (`text/event-stream`)

Events:

```
data: {"type":"start","ts":"2026-03-26T10:00:00Z"}

data: {"type":"progress","percent":10,"message":"Generating image..."}

data: {"type":"partial-image","data":"data:image/png;base64,..."}

data: {"type":"progress","percent":50,"message":"Refining..."}

data: {"type":"completed","imageUrl":"https://...","originalUrl":"https://...","meta":{...}}
```

Error event:
```
data: {"type":"error","code":"GENERATION_FAILED","message":"..."}
```

---

### GET /v1/generate-image/config

Returns available image generation models and estimated generation times.

**Authentication:** JWT (required)

**Response:** `200 OK`

```json
{
  "defaultModel": "gpt-image-1",
  "models": [
    {
      "id": "gpt-image-1",
      "name": "GPT Image 1",
      "provider": "openai",
      "etaMs": 20000,
      "supportsReferences": true
    },
    {
      "id": "dall-e-3",
      "name": "DALL-E 3",
      "provider": "openai",
      "etaMs": 15000,
      "supportsReferences": false
    },
    {
      "id": "gemini-nano-banana",
      "name": "Gemini Flash Image",
      "provider": "google",
      "etaMs": 25000,
      "supportsReferences": true
    }
  ]
}
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | `3000` | Server listen port |
| `AUTH0_DOMAIN` | Yes | — | Auth0 tenant domain for JWKS endpoint |
| `AUTH0_AUDIENCE` | Yes | — | Auth0 audience identifier for token validation |
| `AUTH_COOKIE_SECRET` | Yes | — | Shared secret for HS256 token validation |
| `INTERNAL_SERVICE_TOKEN` | Yes | — | Token for service-to-service authentication |
| `ANTHROPIC_API_KEY` | Yes | — | Platform Anthropic API key (used for all `claude-*` requests) |
| `OPENAI_API_KEY` | No | — | Platform OpenAI API key (required for `gpt-*`/`o1-*`/`o3-*`/`o4-*`/`codex-*` models) |
| `Z_BILLING_URL` | Yes | — | z-billing service base URL |
| `Z_BILLING_API_KEY` | Yes | — | API key for z-billing requests |
| `AURA_NETWORK_URL` | No | — | aura-network base URL for usage recording |
| `AURA_NETWORK_TOKEN` | No | — | Internal service token for aura-network |
| `AURA_STORAGE_URL` | No | — | aura-storage base URL for event recording |
| `AURA_STORAGE_TOKEN` | No | — | Internal service token for aura-storage |
| `GOOGLE_API_KEY` | No | — | Google API key (required for Gemini image generation) |
| `S3_BUCKET_NAME` | No | — | S3 bucket for image uploads (required for image generation) |
| `AWS_REGION` | No | `us-east-1` | AWS region for S3 |
| `AWS_ACCESS_KEY_ID` | No | — | AWS credentials for S3 (required for image generation) |
| `AWS_SECRET_ACCESS_KEY` | No | — | AWS credentials for S3 (required for image generation) |
| `WATERMARK_PATH` | No | `assets/watermark.png` | Path to watermark image file |
| `CORS_ORIGINS` | No | — | Comma-separated list of allowed CORS origins |
| `RATE_LIMIT_RPM` | No | `60` | Maximum requests per minute per user |
