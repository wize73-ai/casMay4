# CasaLingua Architecture

This section provides an overview of CasaLingua's system architecture, components, and data flows.

## System Overview

CasaLingua is a comprehensive language processing application specializing in translation and simplification services with a focus on legal housing documents. The system is designed with a modular, scalable architecture to handle various language processing tasks efficiently.

![CasaLingua Architecture](./images/architecture_overview.png)

## Core Components

### 1. API Layer
- FastAPI-based RESTful API
- Request validation and authentication
- Rate limiting and security controls
- Response formatting and error handling

### 2. Pipeline Processing System
- UnifiedProcessor orchestrating all processing operations
- Specialized pipelines for translation, simplification, etc.
- Multi-stage processing with concurrent operations
- Caching system for improved performance

### 3. Model Management
- EnhancedModelManager for loading and using models
- Dynamic model loading and unloading
- GPU/CPU resource management
- Model registry for configuration and discovery

### 4. Veracity Auditing
- Quality verification for processing outputs
- Multi-layer assessment with semantic, content, and format checks
- Auto-fixing capabilities for detected issues
- Quality metrics collection and reporting

### 5. Audit and Logging
- Comprehensive audit logging system
- Performance metrics collection
- Tracing for request flows
- Structured logging for analysis

## Data Flow

1. **Input Processing**
   - Input text received through API endpoints
   - Request validation and parameter processing
   - Language detection if needed
   - Context and domain identification

2. **Core Processing**
   - Text routed to appropriate pipeline (translation, simplification, etc.)
   - Model selection based on task, language, and domain
   - Processing with selected model
   - Concurrent operations for efficiency

3. **Quality Verification**
   - Optional verification of processing output
   - Multi-dimensional quality assessment
   - Issue detection and classification
   - Auto-fixing when enabled

4. **Response Generation**
   - Formatting of processing results
   - Inclusion of metadata and quality information
   - Error handling for failed operations
   - Cache management for repeated requests

## Key Interfaces

### Internal Interfaces

1. **Processor Interface**
   ```python
   async def process(
       content: Union[str, bytes],
       options: Dict[str, Any] = None
   ) -> Dict[str, Any]
   ```

2. **Model Interface**
   ```python
   async def run_model(
       model_type: str,
       method: str,
       input_data: Dict[str, Any]
   ) -> Any
   ```

3. **Veracity Interface**
   ```python
   async def check(
       content: str,
       processed_text: str,
       options: Dict[str, Any]
   ) -> Dict[str, Any]
   ```

### External APIs

1. **Pipeline API**
   - `/pipeline/translate`: Translation endpoint
   - `/pipeline/simplify`: Simplification endpoint
   - `/pipeline/detect`: Language detection endpoint
   - Additional pipeline endpoints for other operations

2. **Health API**
   - `/health`: Basic health check
   - `/health/detailed`: Detailed system status
   - `/readiness` and `/liveness`: Kubernetes probes

3. **Admin API**
   - `/admin/system/info`: System information
   - `/admin/models`: Model management
   - `/admin/metrics`: System metrics

## Scaling and Performance

CasaLingua is designed to scale horizontally and vertically:

1. **Horizontal Scaling**
   - Stateless API containers for request handling
   - Shared model server for resource efficiency
   - Distributed caching for performance

2. **Vertical Scaling**
   - GPU utilization for model inference
   - Memory management for large models
   - Batch processing for efficiency

3. **Performance Optimizations**
   - Request caching
   - Concurrent processing
   - Lazy loading of models
   - Resource-aware scheduling

## Storage Architecture

CasaLingua uses a flexible storage system that supports multiple database backends:

1. **SQLite Storage**
   - Default for development and small deployments
   - File-based storage in the `data` directory
   - Separate database files for users, content, and progress data
   - Low setup overhead, useful for development and testing

2. **PostgreSQL Storage**
   - Recommended for production deployments
   - High-performance relational database
   - Supports concurrent access and higher load
   - Better scalability and reliability for production use
   - Can be configured to connect to any PostgreSQL server

The persistence layer abstracts these storage options, providing a consistent interface regardless of the underlying database technology.

## Deployment Architecture

CasaLingua can be deployed in various configurations:

1. **Standalone Deployment**
   - Single server with all components
   - SQLite or local PostgreSQL for data storage
   - Suitable for development and small-scale usage

2. **Microservices Deployment**
   - Separate services for API, models, and auxiliary functions
   - Kubernetes orchestration
   - Service mesh for communication
   - PostgreSQL for shared data storage

3. **Hybrid Cloud Deployment**
   - Model serving on GPU-enabled instances
   - API serving on scalable compute
   - Distributed caching and storage
   - PostgreSQL with connection pooling for database tier

## Security Architecture

1. **Authentication and Authorization**
   - API key and JWT-based authentication
   - Role-based access control
   - Scoped permissions

2. **Data Security**
   - Input validation and sanitization
   - Output validation
   - Encryption for sensitive data

3. **Infrastructure Security**
   - Container security
   - Network isolation
   - Regular security scanning

## Further Reading

1. [Component Details](./components.md)
2. [Data Flow Diagrams](./data-flow.md)
3. [Deployment Guide](./deployment.md)
4. [Security Architecture](./security.md)