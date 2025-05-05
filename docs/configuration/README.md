# CasaLingua Configuration

This section provides detailed documentation on configuring CasaLingua for different environments and use cases.

## Configuration Topics

1. [Database Configuration](./database.md) - Configure SQLite or PostgreSQL persistent storage
2. [Environment Variables](./environment-variables.md) - Environment variable reference
3. [Server Configuration](./server.md) - Web server and API configuration
4. [Model Configuration](./models.md) - Model loading and resource allocation
5. [Security Configuration](./security.md) - Authentication, authorization, and security settings
6. [Logging Configuration](./logging.md) - Logging and monitoring settings

## Configuration Files

CasaLingua uses a hierarchical configuration system with multiple configuration files:

1. **`config/default.json`**: Base configuration with default values
2. **`config/development.json`**: Development environment overrides
3. **`config/production.json`**: Production environment overrides
4. **`config/model_registry.json`**: Model-specific configuration

The configuration system loads these files in order, with each subsequent file overriding values from the previous files.

## Environment-Specific Configuration

You can specify which environment to use by setting the `CASALINGUA_ENV` environment variable:

```bash
# For development
export CASALINGUA_ENV=development

# For production
export CASALINGUA_ENV=production
```

## Dynamic Configuration Reloading

CasaLingua supports dynamic configuration reloading. When configuration files are modified, the changes are automatically detected and applied without requiring a restart.

Components can register for configuration change notifications:

```python
from app.utils.config import register_config_change_callback

def config_changed(old_config, new_config):
    print("Configuration changed!")
    
callback_id = register_config_change_callback(config_changed)
```

## Default Configuration

The default configuration values are defined in `config/default.json`. See the individual configuration topic pages for details on specific sections of the configuration.

## Environment Variables

Environment variables can be used to override configuration values. Environment variables are prefixed with `CASALINGUA_` and are converted to lowercase configuration keys.

For example:
- `CASALINGUA_SERVER_PORT=8080` overrides `server.port`
- `CASALINGUA_MODELS_USE_GPU=false` overrides `models.use_gpu`
- `CASALINGUA_DATABASE_URL=postgresql://localhost/casalingua` overrides `database.url`

## Configuration Validation

CasaLingua validates configuration values on startup and provides warnings or errors for invalid configuration values. Critical configuration errors will prevent the application from starting.

## Configuration Security

It's important to keep sensitive configuration values secure:

1. Never commit sensitive values (database passwords, API keys, etc.) to version control
2. Use environment variables for sensitive values in production
3. Consider using a secrets management system for production deployments

## Further Reading

- [Getting Started](../getting-started.md) - Basic setup and configuration
- [Architecture](../architecture/README.md) - System architecture overview
- [Deployment](../deployment/README.md) - Deployment configuration