# Database Configuration

CasaLingua supports multiple database backends for persistent storage, allowing you to choose between:

1. **SQLite** (default) - for development, testing, and small deployments
2. **PostgreSQL** - for production and larger deployments

This guide explains how to configure and manage database connections for CasaLingua.

## Configuration Options

Database settings are configured in the `database` section of your configuration file (`config/default.json`, `config/development.json`, or `config/production.json`).

### SQLite Configuration

SQLite is the default database engine and requires minimal configuration:

```json
"database": {
  "url": "sqlite:///data/casalingua.db",
  "pool_size": 5,
  "max_overflow": 10,
  "echo": false,
  "connect_args": {
    "check_same_thread": false
  }
}
```

Key parameters:
- `url`: SQLite connection string (format: `sqlite:///path/to/database.db`)
- `pool_size`: Maximum number of database connections to keep open
- `max_overflow`: Maximum number of connections to create beyond the pool size
- `echo`: Whether to log SQL statements (useful for debugging)
- `connect_args`: SQLite-specific connection arguments

### PostgreSQL Configuration

For larger deployments or when you need a more robust database, configure PostgreSQL:

```json
"database": {
  "url": "postgresql://username:password@hostname:5432/database_name",
  "pool_size": 10,
  "max_overflow": 20,
  "echo": false,
  "connect_args": {
    "connect_timeout": 10
  }
}
```

Key parameters:
- `url`: PostgreSQL connection string (format: `postgresql://username:password@hostname:port/database_name`)
- `pool_size`: Maximum number of database connections to keep open
- `max_overflow`: Maximum number of connections to create beyond the pool size
- `echo`: Whether to log SQL statements (useful for debugging)
- `connect_args`: PostgreSQL-specific connection arguments

## Database Management Scripts

CasaLingua provides several utility scripts for database management:

### Toggle Database Type

Switch between SQLite and PostgreSQL using the `toggle_db_config.py` script:

```bash
# Switch to PostgreSQL (example with Raspberry Pi)
python scripts/toggle_db_config.py --type postgres --host 192.168.1.105 --username pi --password raspberry --database casalingua

# Switch to PostgreSQL (general example)
python scripts/toggle_db_config.py --type postgres --host db.example.com --username dbuser --password dbpass --database casalingua

# Switch to SQLite
python scripts/toggle_db_config.py --type sqlite
```

### Initialize PostgreSQL Database

When using PostgreSQL for the first time, initialize the database tables:

```bash
python scripts/init_postgres.py
```

### Check Database Status

View detailed information about the current database configuration:

```bash
python scripts/check_db_status.py
```

### Database Persistence Demo

Test the database persistence with a simple demonstration script:

```bash
# Create and retrieve sample data
python scripts/demo_persistent_memory.py

# Create, retrieve, and then delete sample data
python scripts/demo_persistent_memory.py --cleanup
```

## Data Storage Details

CasaLingua's persistence layer manages three primary database files:

1. **users.db**: Stores user information, authentication data, and language preferences
2. **content.db**: Stores content items, translations, embeddings, and related metadata
3. **progress.db**: Stores usage data, learning progress, and activity logs

When using PostgreSQL, these are implemented as tables within a single database.

## Database Backup and Restore

### Creating a Backup

You can create a backup of your database using the persistence manager:

```python
from app.services.storage.casalingua_persistence.manager import PersistenceManager

# Initialize the persistence manager
data_dir = "./data"  # For SQLite
db_config = {...}    # Your database configuration
persistence_manager = PersistenceManager(data_dir, db_config)

# Backup all databases
backup_dir = "./data/backups"
backup_results = persistence_manager.backup_all(backup_dir)
```

This creates timestamped backup files in the specified backup directory.

### Restoring from Backup

To restore from a backup:

```python
# Restore all databases from a specific timestamp
timestamp = "20250505_103954"  # Example timestamp
restore_results = persistence_manager.restore_all(backup_dir, timestamp)
```

## Performance Optimization

The database manager includes optimization capabilities:

```python
# Optimize all databases
optimization_results = persistence_manager.optimize_all()
```

This performs maintenance operations like VACUUM for SQLite or VACUUM ANALYZE for PostgreSQL.

## Environment Variable Configuration

You can also configure the database using environment variables:

```bash
# Configure PostgreSQL
export CASALINGUA_DATABASE_URL="postgresql://username:password@hostname:5432/database_name"
export CASALINGUA_DATABASE_POOL_SIZE=10
export CASALINGUA_DATABASE_MAX_OVERFLOW=20

# Configure SQLite
export CASALINGUA_DATABASE_URL="sqlite:///data/casalingua.db"
```

## Troubleshooting

### Common Issues

1. **Connection Failures**:
   - Verify database server is running
   - Check credentials and hostname
   - Ensure network access to database server
   - Check database existence

2. **Permission Errors**:
   - Verify user has appropriate permissions
   - Check file permissions for SQLite
   - Ensure database user has necessary privileges for PostgreSQL

3. **Performance Issues**:
   - Optimize databases regularly
   - Adjust pool size and max overflow
   - Monitor connection usage
   - Consider increasing database resources

### Database Logs

For detailed database logging, enable the echo parameter in your configuration:

```json
"database": {
  "echo": true
}
```

This will log all SQL statements executed, which is useful for debugging but can generate significant log volume in production.