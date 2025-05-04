# Model Loading Fixes for CasaLingua

This PR addresses critical model loading issues in the CasaLingua language processing pipeline.

## Issues Fixed

1. **T5Model Compatibility Issue**
   - Problem: The simplifier model was instantiated with T5Model class which is not compatible with the `.generate()` method
   - Fix: Changed model class to T5ForConditionalGeneration in model_registry.json

2. **UnboundLocalError During Model Loading**
   - Problem: Various model classes (AutoModel, AutoModelForSeq2SeqLM, etc.) were referenced before assignment
   - Fix: Moved import statements inside the _load_transformers_model method to ensure all classes are properly defined in local scope

3. **Missing Model Registry File**
   - Problem: model_registry.json file was missing, causing defaults to be used
   - Fix: Created proper model_registry.json with all required model configurations

4. **NER Model Class Issues**
   - Problem: In some code paths, NER detection model was trying to use AutoModelForSequenceClassification instead of AutoModelForTokenClassification
   - Fix: Ensured consistent use of AutoModelForTokenClassification for NER models

5. **Model Registry Initialization**
   - Problem: Model registry wasn't properly initialized in FastAPI app state
   - Fix: Added code to ensure the registry is available in app.state

## Implementation Details

### 1. Created Model Registry Configuration

Created a `model_registry.json` file with proper configurations for all models:

```json
{
    "language_detection": {
        "model_name": "papluca/xlm-roberta-base-language-detection",
        "model_type": "transformers",
        "tokenizer_name": "papluca/xlm-roberta-base-language-detection",
        "task": "language_detection",
        "framework": "transformers"
    },
    "translation": {
        "model_name": "google/mt5-small",
        "tokenizer_name": "google/mt5-small",
        "task": "translation",
        "type": "transformers",
        "framework": "transformers"
    },
    "ner_detection": {
        "model_name": "dslim/bert-base-NER",
        "tokenizer_name": "dslim/bert-base-NER",
        "task": "ner_detection",
        "type": "transformers",
        "framework": "transformers"
    },
    "simplifier": {
        "model_name": "t5-small",
        "tokenizer_name": "t5-small",
        "task": "simplification",
        "type": "transformers",
        "model_class": "T5ForConditionalGeneration",
        "framework": "transformers"
    },
    ...
}
```

### 2. Fixed Model Loading Method

Modified the `_load_transformers_model` method to properly import all required classes:

```python
def _load_transformers_model(self, model_config: ModelConfig, device: str) -> Any:
    """
    Load a Hugging Face Transformers model with multi-GPU support
    
    Args:
        model_config (ModelConfig): Model configuration
        device (str): Device to load model on (e.g., "cuda:0", "cuda:1", "mps", "cpu")
        
    Returns:
        Any: Loaded model
    """
    # Import all needed transformers components here to ensure they're available
    import transformers
    from transformers import (
        AutoModel, 
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        T5ForConditionalGeneration,
        MT5ForConditionalGeneration,
        BertForTokenClassification,
        BertModel
    )
    
    # ... existing code ...
```

### 3. Fixed App State Initialization

Added model registry to app state in main.py:

```python
# Create and assign ModelRegistry to app.state
from app.services.models.loader import ModelRegistry
app.state.model_registry = model_loader.registry
```

## Testing Done

1. Successfully tested server startup with all models loading correctly
2. Verified health/models endpoint showing all models active
3. Tested language detection and translation endpoints
4. Verified hardware detection and model configuration based on available resources

## Potential Follow-ups

1. Improve the /health/models endpoint handler to work with the current ModelRegistry
2. Add more unit tests for model loading to prevent future regressions
3. Create a more robust model loading strategy with better fallbacks