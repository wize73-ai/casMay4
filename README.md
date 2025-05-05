
# CasaLingua

![CasaLingua Logo](app/static/img/casalingua-logo.svg)

## Breaking Down Language Barriers in Housing

CasaLingua is a comprehensive language translation and processing platform designed to improve accessibility to housing services and information for non-English speakers. By leveraging advanced machine learning models and retrieval-augmented generation (RAG) techniques, CasaLingua provides accurate, contextually-aware translations tailored to the housing domain.

## Key Features

- **Domain-Specific Translation**: Specialized in housing terminology and concepts
- **Retrieval-Augmented Generation**: Enhanced translations using contextual knowledge
- **Document Translation**: Support for PDF, Word, and other document formats
- **Multi-Language Support**: Primary focus on Spanish-English with expandable language pairs
- **Quality Verification**: Built-in translation quality checks
- **Audit Logging**: Comprehensive tracking for compliance and security
- **Flexible Deployment**: API-first design with cloud or on-premises options
- **High Performance**: Request-level caching, parallel processing, and smart batching
- **Streaming Responses**: Support for real-time, chunk-based responses for large documents
- **Advanced Error Handling**: Standardized error categorization with comprehensive fallbacks
- **Bloom Housing Integration**: Seamless compatibility with Bloom Housing API format

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL (for production) or SQLite (for development)
- 8GB+ RAM (16GB+ recommended)
- NVIDIA GPU or Apple Silicon with 8GB+ VRAM (optional but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:wize73-ai/casMay4.git
   cd casalingua
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. Initialize the database:
   ```bash
   python -m app.tools.db_init
   ```

6. Start the server:
   ```bash
   uvicorn app.main:app --reload
   ```

   > **Note**: If you're running a startup script like `./scripts/startdev.sh` and get a "permission denied" error, run:
   > ```bash
   > chmod +x scripts/startdev.sh
   > ```
   > This makes the script executable.

7. Open your browser to [http://localhost:8000/docs](http://localhost:8000/docs) to explore the API.

## Architecture

CasaLingua follows a modular architecture with the following components:

- **API Layer**: FastAPI-based RESTful endpoints
- **Model Management**: Dynamic loading and management of language models
- **Pipeline System**: Flexible processing workflows for different tasks
- **RAG Engine**: Context-enhanced translation using document retrieval
- **Audit System**: Comprehensive logging and verification
- **Metrics Collection**: Performance and usage tracking

## Configuration

CasaLingua uses a hierarchical configuration system:

1. Default settings in `config/default.json`
2. Environment-specific settings in `config/[environment].json`
3. Override with environment variables prefixed with `CASALINGUA_`

## Development

### Project Structure

```
casalingua/
├── app/                   # Main application code
│   ├── api/               # API routes and handlers
│   ├── audit/             # Audit logging and verification
│   ├── core/              # Core business logic
│   ├── model/             # Model management components
│   ├── services/          # Support services
│   ├── schemas/           # Data models and validation
│   ├── utils/             # Utility functions
│   └── main.py            # Application entry point
├── config/                # Configuration files
├── logs/                  # Log output
├── models/                # Model storage
├── tests/                 # Test suite
└── README.md              # This file
```

### Running Tests

Run all tests:
```bash
pytest
```

Run API optimization tests:
```bash
python app/tests/test_optimizations.py
```

Run individual component tests:
```bash
python app/tests/run_tests.py
```

### Style Guide

This project follows PEP 8 style guidelines with a line length of 100 characters. We use `black` for code formatting and `isort` for import sorting.

## Deployment

### Docker

Build and run with Docker:

```bash
docker build -t casalingua .
docker run -p 8000:8000 --env-file .env casalingua
```

### Cloud Deployment

For cloud deployment, we recommend:
- CPU-optimized instances with at least 16GB RAM
- GPU acceleration for production workloads
- Load balancing for high-availability setups
- Managed database service

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgements

CasaLingua is developed by Exygy, a digital services agency building technology for social impact. This project was created to improve language accessibility in housing services and decrease barriers for limited English speakers seeking housing resources.

## Support

For support, please contact:
- Email: support@exygy.com
- GitHub Issues: [https://github.com/exygy/casalingua/issues](https://github.com/exygy/casalingua/issues)

---

*CasaLingua: Language Should Never Be a Barrier to Finding Home*
