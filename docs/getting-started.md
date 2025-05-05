# Getting Started with CasaLingua

This guide will help you set up and start using CasaLingua for translation and text simplification, with a special focus on housing legal documents.

## Prerequisites

- Python 3.8+ installed
- Git (for cloning the repository)
- 8GB+ RAM for running basic models
- 16GB+ RAM recommended for full functionality
- NVIDIA GPU (optional but recommended for better performance)

## Installation

### Option 1: Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/your-organization/casalingua.git
   cd casalingua
   ```

2. Start the application using Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. Verify the installation:
   ```bash
   curl http://localhost:8000/health
   ```

### Option 2: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-organization/casalingua.git
   cd casalingua
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required models:
   ```bash
   ./scripts/download_models.py
   ```

5. Start the application:
   ```bash
   ./scripts/start.sh  # On Windows: scripts\start.bat
   ```

6. Verify the installation:
   ```bash
   curl http://localhost:8000/health
   ```

## Configuration

### Basic Configuration

The default configuration is suitable for most use cases. If you need to customize the application, modify the appropriate config file:

- Development: `config/development.json`
- Production: `config/production.json`

For detailed configuration documentation, see the [Configuration Guide](./configuration/README.md).

### Database Configuration

CasaLingua supports both SQLite (default) and PostgreSQL for data persistence. The default SQLite configuration works out of the box, but for production deployments, consider using PostgreSQL:

```bash
# Switch to PostgreSQL (replace with your actual credentials)
python scripts/toggle_db_config.py --type postgres --host db.example.com --username dbuser --password dbpass

# Or use SQLite for development
python scripts/toggle_db_config.py --type sqlite
```

See the [Database Configuration Guide](./configuration/database.md) for detailed instructions.

### Environment Variables

Key environment variables you may want to set:

- `CASALINGUA_ENV`: Set to `development` or `production`
- `CASALINGUA_PORT`: Port to run the API server (default: 8000)
- `CASALINGUA_HOST`: Host to bind the API server (default: 0.0.0.0)
- `CASALINGUA_MODELS_DIR`: Directory to store models (default: models/)
- `CASALINGUA_LOG_LEVEL`: Logging level (default: INFO)
- `CASALINGUA_USE_GPU`: Whether to use GPU if available (default: true)
- `CASALINGUA_DATABASE_URL`: Database connection URL (default: sqlite:///data/casalingua.db)

### Model Configuration

Model configuration is specified in `config/model_registry.json`. You can modify this file to:

- Add new models
- Configure model parameters
- Disable specific models
- Set up model fallbacks

## Quick Start

### Translate Text

```bash
curl -X POST http://localhost:8000/pipeline/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The lease agreement requires tenants to maintain the property in good condition.",
    "source_language": "en",
    "target_language": "es"
  }'
```

### Simplify Text

```bash
curl -X POST http://localhost:8000/pipeline/simplify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The tenant shall indemnify and hold harmless the landlord from and against any and all claims, actions, suits, judgments and demands brought or recovered against the landlord by reason of any negligent or willful act or omission of the tenant.",
    "language": "en",
    "target_level": "simple",
    "domain": "legal-housing"
  }'
```

### Detect Language

```bash
curl -X POST http://localhost:8000/pipeline/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you doing today?"
  }'
```

## Authentication

### API Key Authentication

For production use, you should authenticate with an API key:

```bash
curl -X POST http://localhost:8000/pipeline/translate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "Hello world",
    "target_language": "es"
  }'
```

### Creating an API Key

```bash
curl -X POST http://localhost:8000/admin/api-keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ADMIN_KEY" \
  -d '{
    "name": "My API Key",
    "scopes": ["translation:read", "translation:write"]
  }'
```

## Common Operations

### Enabling Verification

To verify the quality of translations or simplifications:

```bash
curl -X POST http://localhost:8000/pipeline/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "target_language": "es",
    "verify": true
  }'
```

### Housing Legal Document Simplification

For simplifying legal housing documents with domain-specific handling:

```bash
curl -X POST http://localhost:8000/pipeline/simplify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Lessor reserves the right to access the premises for inspection purposes with 24 hours advance notice provided to the lessee, except in cases of emergency wherein immediate access may be required.",
    "language": "en",
    "target_level": "simple",
    "domain": "legal-housing",
    "verify_output": true,
    "auto_fix": true
  }'
```

## Monitoring and Management

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed

# Model health check
curl http://localhost:8000/health/models
```

### System Information

```bash
curl http://localhost:8000/admin/system/info \
  -H "Authorization: Bearer ADMIN_KEY"
```

### Model Management

```bash
# List all models
curl http://localhost:8000/admin/models \
  -H "Authorization: Bearer ADMIN_KEY"

# Get information about a specific model
curl http://localhost:8000/admin/models/mbart_translation \
  -H "Authorization: Bearer ADMIN_KEY"
```

## Troubleshooting

### Common Issues

1. **Models not loading**: Check models directory permissions and available disk space
2. **Out of memory errors**: Reduce concurrent requests or use smaller models
3. **API timeout**: Adjust timeout settings for large text inputs
4. **Unsupported language**: Check supported languages and make sure language codes are correct

### Viewing Logs

```bash
# Docker logs
docker-compose logs -f app

# Local installation logs
tail -f logs/app/app.log
```

### Checking Metrics

```bash
curl http://localhost:8000/admin/metrics \
  -H "Authorization: Bearer ADMIN_KEY"
```

## Next Steps

- [API Reference](./api/README.md): Complete API documentation
- [Models Overview](./models/README.md): Learn about the available models
- [Quality Assurance](./quality/README.md): Learn about veracity auditing and quality metrics
- [Architecture](./architecture/README.md): Understanding CasaLingua's architecture