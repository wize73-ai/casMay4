# CasaLingua API Reference

This section provides detailed documentation for all API endpoints available in the CasaLingua application.

## Authentication

Most API endpoints require authentication. CasaLingua supports the following authentication methods:

1. **API Key Authentication**: Pass your API key in the `Authorization` header:
   ```
   Authorization: Bearer cslg_8f4b2d1e7a3c5b9e2f1d8a7c4b2e5f9d
   ```

2. **JWT Token Authentication**: For user-based authentication, obtain a JWT token and pass it in the `Authorization` header:
   ```
   Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

## Base URL

All API endpoints are relative to the base URL of your CasaLingua instance:

```
https://your-casalingua-instance.com/
```

For local development:

```
http://localhost:8000/
```

## Response Format

All API endpoints return responses in a consistent format:

```json
{
  "status": "success",  // or "error"
  "message": "Operation completed successfully", // human-readable message
  "data": {
    // endpoint-specific response data
  },
  "metadata": {
    "request_id": "uuid-here",
    "timestamp": 1620160000,
    "version": "1.0.0",
    "process_time": 0.123
  }
}
```

## Available Endpoints

| Category | Endpoint | Description | Documentation |
|----------|----------|-------------|---------------|
| **Pipeline** | `/pipeline/translate` | Translates text between languages | [Translation](./translation.md) |
| **Pipeline** | `/pipeline/simplify` | Simplifies complex text | [Simplification](./simplification.md) |
| **Pipeline** | `/pipeline/detect` | Detects the language of text | [Language Detection](./language-detection.md) |
| **Pipeline** | `/pipeline/analyze` | Analyzes text for various attributes | [Text Analysis](./text-analysis.md) |
| **Pipeline** | `/pipeline/summarize` | Generates a summary of text | [Summarization](./summarization.md) |
| **Pipeline** | `/pipeline/anonymize` | Anonymizes personally identifiable information | [Anonymization](./anonymization.md) |
| **Streaming** | `/streaming/translate` | Streams translation results | [Streaming](./streaming.md) |
| **Streaming** | `/streaming/analyze` | Streams analysis results | [Streaming](./streaming.md) |
| **RAG** | `/rag/translate` | Knowledge-enhanced translation | [RAG Translation](./rag.md#translation) |
| **RAG** | `/rag/query` | Queries knowledge base | [RAG Query](./rag.md#query) |
| **RAG** | `/rag/chat` | Knowledge-enhanced chat | [RAG Chat](./rag.md#chat) |
| **Admin** | `/admin/system/info` | System information | [Admin API](./admin.md) |
| **Admin** | `/admin/models` | Model management | [Admin API](./admin.md#models) |
| **Health** | `/health` | Health checks | [Health Checks](./health.md) |

## Error Handling

When an error occurs, the API responds with an appropriate HTTP status code and a JSON response with details:

```json
{
  "status": "error",
  "message": "Error message describing what went wrong",
  "error": {
    "code": "ERROR_CODE",
    "details": "Additional error details"
  },
  "metadata": {
    "request_id": "uuid-here",
    "timestamp": 1620160000
  }
}
```

Common HTTP status codes:
- `400 Bad Request`: Invalid input parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Input validation failed
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server-side error

## Rate Limiting

API endpoints are subject to rate limiting. The current limits are:
- 100 requests per minute for authenticated users
- 10 requests per minute for unauthenticated users

Rate limit information is included in the response headers:
- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Remaining requests in the current window
- `X-RateLimit-Reset`: Time (Unix timestamp) when the rate limit resets