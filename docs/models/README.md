# CasaLingua Models

This section provides documentation on the language models used by CasaLingua for translation, simplification, and other language processing tasks.

## Model Architecture

CasaLingua uses a variety of state-of-the-art language models to power its capabilities:

| Category | Primary Models | Fallback Models | Description |
|----------|----------------|-----------------|-------------|
| **Translation** | MBART | MT5 | Neural machine translation models |
| **Simplification** | BART | T5 | Text simplification and rewriting models |
| **Language Detection** | XLM-RoBERTa | FastText | Language identification models |
| **Embedding** | Sentence Transformers | - | Text representation models for semantic comparison |

## Translation Models

### MBART (Primary)

The mbart-large-50-many-to-many-mmt model is our primary translation engine:

- **Model**: `facebook/mbart-large-50-many-to-many-mmt`
- **Architecture**: Sequence-to-sequence transformer with 680M parameters
- **Languages**: Supports 50 languages with direct translation between any pair
- **Features**: 
  - High-quality translations for most language pairs
  - Preserves document structure and formatting
  - Handles complex grammar and idiomatic expressions
  - Domain adaptation for legal text

### MT5 (Fallback)

The mt5-base model serves as a fallback translation option:

- **Model**: `google/mt5-base`
- **Architecture**: Sequence-to-sequence transformer with 580M parameters
- **Languages**: Supports 101 languages
- **Features**:
  - More languages than MBART but sometimes lower quality
  - Used when MBART doesn't support a language pair
  - Good performance for short to medium-length texts

## Simplification Models

### BART (Primary)

The bart-large-cnn model powers our text simplification capabilities:

- **Model**: `facebook/bart-large-cnn`
- **Architecture**: Seq2seq transformer with 400M parameters
- **Languages**: English (primary), limited support for other languages
- **Features**:
  - Fine-tuned for text simplification
  - Specialized prompting for legal document simplification
  - Adaptive simplification based on target complexity level

### T5 (Alternative)

The t5-base model provides alternative simplification capabilities:

- **Model**: `t5-base`
- **Architecture**: Encoder-decoder transformer with 220M parameters
- **Languages**: English with experimental support for other languages
- **Features**:
  - Task-oriented model with "simplify:" prefix
  - Good performance for short text simplification

## Language Detection Model

### XLM-RoBERTa

- **Model**: `papluca/xlm-roberta-base-language-detection`
- **Architecture**: Masked language model with 270M parameters
- **Languages**: 100+ languages
- **Features**:
  - Fast and accurate language identification
  - Works reliably with short texts (10+ words)
  - Returns confidence scores for language predictions

## Embedding Models

### Sentence Transformers

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Architecture**: Transformer-based sentence embedding model
- **Features**:
  - Creates semantic embeddings for text comparison
  - Used by the veracity auditing system
  - Enables semantic search and similarity measurement

## Model Configuration

Models are configured in the `config/model_registry.json` file with settings like:

```json
{
  "mbart_translation": {
    "model_name": "facebook/mbart-large-50-many-to-many-mmt",
    "model_type": "seq2seq",
    "tokenizer_name": "facebook/mbart-large-50-many-to-many-mmt",
    "device": "cuda:0",
    "precision": "float16",
    "max_length": 512
  }
}
```

## Model Performance

### Hardware Requirements

| Model | Minimum RAM | Recommended RAM | GPU Memory |
|-------|------------|-----------------|------------|
| MBART | 8 GB | 16 GB | 4+ GB |
| MT5 | 6 GB | 12 GB | 3+ GB |
| BART | 4 GB | 8 GB | 2+ GB |
| XLM-RoBERTa | 2 GB | 4 GB | 1+ GB |
| All models | 16 GB | 32 GB | 8+ GB |

### Inference Speed

Average processing times (may vary based on hardware):

| Model | Short Text | Medium Text | Long Text |
|-------|------------|-------------|-----------|
| MBART | 150ms | 350ms | 800ms |
| MT5 | 180ms | 400ms | 900ms |
| BART | 120ms | 300ms | 700ms |
| XLM-RoBERTa | 50ms | 100ms | 200ms |

## Supported Languages

### Translation Support

| Language | Code | MBART | MT5 | Quality Level |
|----------|------|-------|-----|---------------|
| English | en | ✓ | ✓ | High |
| Spanish | es | ✓ | ✓ | High |
| French | fr | ✓ | ✓ | High |
| German | de | ✓ | ✓ | High |
| Chinese (Simplified) | zh | ✓ | ✓ | Medium |
| Japanese | ja | ✓ | ✓ | Medium |
| Korean | ko | ✓ | ✓ | Medium |
| Russian | ru | ✓ | ✓ | Medium |
| ... | ... | ... | ... | ... |

### Simplification Support

| Language | BART | T5 | Quality Level |
|----------|------|-----|---------------|
| English | ✓ | ✓ | High |
| Spanish | Limited | Limited | Medium |
| French | Limited | Limited | Medium |
| Other languages | Experimental | Experimental | Low |

## Model Loading and Management

CasaLingua uses a sophisticated model management system:

1. **Lazy Loading**: Models are loaded only when needed
2. **Dynamic Unloading**: Lesser-used models can be unloaded to free memory
3. **GPU Offloading**: Models can be moved between GPU and CPU based on usage
4. **Quantization**: Reduced precision for lower memory footprint
5. **Batch Processing**: Efficient processing of multiple requests

## Model Updates

Models can be updated without restarting the application:

```bash
# Update model registry
curl -X POST http://localhost:8000/admin/models/update-registry \
  -H "Authorization: Bearer ADMIN_KEY" \
  -d '{}'

# Reload a specific model
curl -X POST http://localhost:8000/admin/models/mbart_translation/reload \
  -H "Authorization: Bearer ADMIN_KEY" \
  -d '{}'
```

## Advanced Configuration

For advanced model configuration, see:

1. [Model Customization](./model-customization.md)
2. [Language Support](./language-support.md)
3. [Hardware Optimization](./hardware-optimization.md)