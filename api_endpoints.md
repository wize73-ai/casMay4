# API Endpoints

## Admin Routes (`/app/api/routes/admin.py`)

| Method | Path                           | Description                                       |
|--------|--------------------------------|---------------------------------------------------|
| GET    | /system/info                   | Get system information                            |
| GET    | /system/config                 | Get system configuration                          |
| POST   | /system/config                 | Update system configuration                       |
| GET    | /models                        | List available models                             |
| GET    | /models/{model_id}             | Get model details                                 |
| POST   | /models/{model_id}/load        | Load model into memory                            |
| POST   | /models/{model_id}/unload      | Unload model from memory                          |
| GET    | /languages                     | List supported languages                          |
| GET    | /metrics                       | Get system metrics                                |
| GET    | /metrics/time-series/{series_name} | Get time series metrics                        |
| GET    | /logs                          | Search audit logs                                 |
| GET    | /logs/export                   | Export audit logs                                 |
| POST   | /api-keys                      | Create API key                                    |
| GET    | /api-keys                      | List API keys                                     |
| DELETE | /api-keys/{key_id}             | Revoke API key                                    |

## Pipeline Routes (`/app/api/routes/pipeline.py`)

| Method | Path                           | Description                                       |
|--------|--------------------------------|---------------------------------------------------|
| POST   | /translate                     | Translate text                                    |
| POST   | /translate/batch               | Batch translate texts                             |
| POST   | /translate/document            | Upload and translate a document                   |
| GET    | /translate/document/{task_id}  | Get document translation status                   |
| POST   | /detect                        | Detect language                                   |
| POST   | /detect-language               | Detect language (alias route)                     |
| POST   | /analyze                       | Analyze text                                      |
| POST   | /summarize                     | Summarize text                                    |
| POST   | /verify                        | Verify translation quality                        |
| GET    | /tasks/{task_id}               | Get task status                                   |

## RAG Routes (`/app/api/routes/rag.py`)

| Method | Path                           | Description                                       |
|--------|--------------------------------|---------------------------------------------------|
| POST   | /rag/translate                 | Translate text with RAG enhancement               |
| POST   | /rag/query                     | Query knowledge base                              |
| POST   | /rag/chat                      | Chat with RAG enhancement                         |
| POST   | /rag/documents/upload          | Upload document to knowledge base                 |
| GET    | /rag/documents/{document_id}   | Get document by ID                                |
| DELETE | /rag/documents/{document_id}   | Delete document from knowledge base               |
| GET    | /rag/conversations/{conversation_id} | Get conversation history                    |
| POST   | /rag/documents/ingest-from-config | Ingest documents from GitHub config            |

## Bloom Housing Routes (`/app/api/routes/bloom_housing.py`)

| Method | Path                           | Description                                       |
|--------|--------------------------------|---------------------------------------------------|
| POST   | /bloom-housing/translate       | Translate text (Bloom Housing format)             |
| POST   | /bloom-housing/detect-language | Detect language (Bloom Housing format)            |
| POST   | /bloom-housing/analyze         | Analyze text (Bloom Housing format)               |
| POST   | /bloom-housing/translate-document | Upload and translate document (Bloom Housing format) |
| GET    | /bloom-housing/document-status/{document_id} | Get document translation status (Bloom Housing format) |

## Streaming Routes (`/app/api/routes/streaming.py`)

| Method | Path                           | Description                                       |
|--------|--------------------------------|---------------------------------------------------|
| POST   | /streaming/translate           | Stream translation of text                        |
| POST   | /streaming/analyze             | Stream text analysis                              |

## Health Routes (`/app/api/routes/health.py`)

| Method | Path                           | Description                                       |
|--------|--------------------------------|---------------------------------------------------|
| GET    | /health                        | Basic health check                                |
| GET    | /health/detailed               | Detailed health check                             |
| GET    | /health/models                 | Model health check                                |
| GET    | /health/database               | Database health check                             |
| GET    | /readiness                     | Readiness probe for Kubernetes                    |
| GET    | /liveness                      | Liveness probe for Kubernetes                     |

## Routes Worth Testing Separately

1. **Translation API endpoints**:
   - `/translate` - Core translation functionality
   - `/translate/batch` - Performance with multiple texts
   - `/translate/document` - Document handling capabilities

2. **RAG-enhanced endpoints**:
   - `/rag/translate` - Knowledge-enhanced translations
   - `/rag/chat` - Conversational capabilities with knowledge retrieval

3. **Streaming endpoints**:
   - `/streaming/translate` - Real-time translation performance
   - `/streaming/analyze` - Real-time analysis capabilities

4. **Health check endpoints**:
   - `/health/detailed` - System diagnostics
   - `/health/models` - Model availability and performance

5. **Admin endpoints**:
   - `/models/{model_id}/load` - Model loading performance
   - `/system/config` - Configuration management

6. **Language detection**:
   - `/detect` - Language identification accuracy

7. **Document processing**:
   - `/translate/document` - Document translation quality and performance
   - `/rag/documents/upload` - Document indexing for RAG

8. **Bloom Housing compatibility layer**:
   - Test at least one Bloom Housing endpoint to verify compatibility