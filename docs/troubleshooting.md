# Troubleshooting Guide

This document helps you diagnose and resolve common issues with CasaLingua.

## Common Issues and Solutions

### Installation Issues

#### Docker Container Fails to Start

**Symptoms:**
- Container exits immediately after starting
- Docker logs show memory-related errors

**Solutions:**
1. Increase Docker memory allocation (recommended: 8GB+)
2. Check Docker settings → Resources → Memory
3. Verify available disk space (need at least 10GB free)

```bash
# Check Docker logs
docker-compose logs app
```

#### Python Dependencies Installation Errors

**Symptoms:**
- `pip install -r requirements.txt` fails
- Missing compiler errors for some packages

**Solutions:**
1. Install required system packages:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-dev build-essential
   
   # macOS
   brew install python3
   
   # Windows
   # Install Visual C++ Build Tools
   ```
2. Try creating a fresh virtual environment:
   ```bash
   python -m venv new_venv
   source new_venv/bin/activate  # Windows: new_venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Model Loading Issues

#### Out of Memory Errors When Loading Models

**Symptoms:**
- Server crashes during startup
- "CUDA out of memory" or similar errors

**Solutions:**
1. Modify `config/default.json` to use lower precision:
   ```json
   "models": {
     "precision": "float16"
   }
   ```
2. Disable GPU for some models:
   ```json
   "mbart_translation": {
     "device": "cpu"
   }
   ```
3. Reduce batch size:
   ```json
   "models": {
     "batch_size": 4
   }
   ```

#### Model Files Not Found

**Symptoms:**
- "Model not found" errors
- Server fails to load models

**Solutions:**
1. Run the model download script:
   ```bash
   python scripts/download_models.py
   ```
2. Check model paths in config:
   ```bash
   cat config/model_registry.json
   ```
3. Ensure permissions are correct on model directory:
   ```bash
   chmod -R 755 models/
   ```

### API Issues

#### Authentication Errors

**Symptoms:**
- API returns 401 Unauthorized
- "Invalid API key" errors

**Solutions:**
1. Verify the API key is correct
2. Check API key format in request:
   ```
   Authorization: Bearer YOUR_API_KEY
   ```
3. Generate a new API key:
   ```bash
   curl -X POST http://localhost:8000/admin/api-keys \
     -H "Authorization: Bearer ADMIN_KEY" \
     -d '{"name": "New API Key", "scopes": ["translation:read", "translation:write"]}'
   ```

#### Request Timeout

**Symptoms:**
- Request times out for large text
- Server takes too long to respond

**Solutions:**
1. Increase timeout setting in client:
   ```python
   response = requests.post(url, headers=headers, data=json.dumps(data), timeout=120)
   ```
2. Break large text into smaller chunks
3. Adjust server timeout in config:
   ```json
   "server": {
     "timeout": 120
   }
   ```

### Translation Issues

#### Poor Translation Quality

**Symptoms:**
- Translation doesn't maintain meaning
- Incorrect language detection

**Solutions:**
1. Explicitly specify source language instead of auto-detection
2. Enable verification:
   ```json
   "verify": true
   ```
3. Try a different model:
   ```json
   "model_name": "mt5"
   ```
4. Check if language pair is well-supported:
   ```bash
   curl http://localhost:8000/admin/languages
   ```

#### Translation Missing Parts of Text

**Symptoms:**
- Parts of input text missing from translation
- Truncated output

**Solutions:**
1. Check if text exceeds model max length:
   ```json
   "max_length": 512
   ```
2. Break text into smaller segments
3. Try batch translation for long texts

### Simplification Issues

#### Housing Legal Text Not Properly Simplified

**Symptoms:**
- Legal jargon remains in output
- Key terms aren't consistently preserved

**Solutions:**
1. Specify domain explicitly:
   ```json
   "domain": "legal-housing"
   ```
2. Enable verification and auto-fixing:
   ```json
   "verify_output": true,
   "auto_fix": true
   ```
3. Update to latest model registry
4. Review legal text dictionaries in processor

#### Simplification Changes Meaning

**Symptoms:**
- Critical information missing in simplified text
- Meaning altered significantly

**Solutions:**
1. Enable verification to detect meaning changes:
   ```json
   "verify_output": true
   ```
2. Set a more conservative simplification level:
   ```json
   "target_level": "moderate"  // instead of "simple"
   ```
3. Manually review simplifications for legal documents

### Veracity Auditing Issues

#### Verification Always Fails

**Symptoms:**
- Verification results always return "verified": false
- Many issues reported in verification results

**Solutions:**
1. Check threshold settings in config:
   ```json
   "veracity": {
     "threshold": 0.7,  // Try lowering this value
     "min_confidence": 0.6  // Try lowering this value
   }
   ```
2. Check if reference embeddings are loaded correctly
3. Ensure model_manager is properly initialized in veracity auditor

#### Verification Takes Too Long

**Symptoms:**
- API requests with verification enabled take much longer
- Timeouts during verification

**Solutions:**
1. Adjust max sample size for verification:
   ```json
   "veracity": {
     "max_sample_size": 500  // Lower this value
   }
   ```
2. Implement simpler verification for bulk operations
3. Process verification in background if possible

## Diagnostic Commands

### System Status

```bash
# Check overall health
curl http://localhost:8000/health

# Detailed component status
curl http://localhost:8000/health/detailed

# Check models status
curl http://localhost:8000/health/models
```

### Logs

```bash
# View application logs
docker-compose logs -f app

# View specific log file
tail -f logs/app/app.log

# Check error logs
grep ERROR logs/app/app.log

# Check audit logs
tail -f logs/audit/audit_*.jsonl
```

### Performance Metrics

```bash
# View system metrics
curl http://localhost:8000/admin/metrics \
  -H "Authorization: Bearer ADMIN_KEY"

# Check time series metrics
curl "http://localhost:8000/admin/metrics/time-series/request_time?limit=10" \
  -H "Authorization: Bearer ADMIN_KEY"
```

### Model Management

```bash
# List all models
curl http://localhost:8000/admin/models \
  -H "Authorization: Bearer ADMIN_KEY"

# Unload a specific model to free memory
curl -X POST http://localhost:8000/admin/models/mbart_translation/unload \
  -H "Authorization: Bearer ADMIN_KEY"

# Reload a model
curl -X POST http://localhost:8000/admin/models/mbart_translation/load \
  -H "Authorization: Bearer ADMIN_KEY"
```

## Common Error Codes

| Error Code | Description | Solution |
|------------|-------------|----------|
| `INVALID_LANGUAGE` | Unsupported language specified | Check supported languages list |
| `TEXT_TOO_LONG` | Input text exceeds maximum length | Break text into smaller chunks |
| `MODEL_UNAVAILABLE` | Required model is not available | Verify model is loaded |
| `UNAUTHORIZED` | Authentication failed | Check API key and permissions |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Reduce request frequency |
| `TRANSLATION_FAILED` | Translation operation failed | Check input and model status |
| `SIMPLIFICATION_FAILED` | Simplification operation failed | Check input and model status |
| `VERIFICATION_FAILED` | Verification operation failed | Check veracity auditor configuration |

## Getting Help

If you encounter issues not covered in this guide:

1. **Check Logs**: Most issues are recorded in the application logs
2. **Search Issues**: Check the GitHub issues for similar problems
3. **Community Forum**: Post a question on our community forum
4. **Contact Support**: For enterprise users, contact support@casalingua.example.com

## Reporting Bugs

When reporting bugs, please include:

1. Complete error message and stack trace if available
2. Steps to reproduce the issue
3. System environment (OS, Docker version, etc.)
4. CasaLingua version
5. Example input that causes the problem (if applicable)

## Updating and Upgrades

If you're experiencing persistent issues, consider updating:

```bash
# Pull latest changes
git pull

# Update dependencies
pip install -r requirements.txt

# Rebuild Docker container
docker-compose build --no-cache app
docker-compose up -d
```