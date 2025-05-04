"""
Model Wrappers Module for CasaLingua
Provides standardized interfaces between models and the pipeline

These wrappers ensure that all models, regardless of their underlying implementation,
expose a consistent interface to the pipeline components. They also apply
model-specific optimizations and handle specialized processing.
"""

import os
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import torch
import numpy as np
from enum import Enum
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

# Model types
class ModelType(str, Enum):
    TRANSLATION = "translation"
    LANGUAGE_DETECTION = "language_detection"
    NER_DETECTION = "ner_detection"
    SIMPLIFIER = "simplifier"
    RAG_GENERATOR = "rag_generator"
    RAG_RETRIEVER = "rag_retriever"
    ANONYMIZER = "anonymizer"

@dataclass
class ModelInput:
    """Standard input format for all models"""
    text: Union[str, List[str]]
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    context: Optional[List[Dict[str, Any]]] = None
    parameters: Optional[Dict[str, Any]] = None

@dataclass
class ModelOutput:
    """Standard output format for all models"""
    result: Any
    metadata: Dict[str, Any] = None
    metrics: Dict[str, Any] = None
    status: str = "success"
    error: Optional[str] = None

class BaseModelWrapper:
    """Base class for all model wrappers"""
    
    def __init__(self, model: Any, tokenizer: Any = None, config: Dict[str, Any] = None):
        """
        Initialize base model wrapper
        
        Args:
            model: The model to wrap
            tokenizer: The tokenizer to use
            config: Configuration parameters
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        self.device = self._get_device()
        self.precision = self._get_precision()
        self.model_type = self._get_model_type()
        self._metrics = []
        
        # Apply model-specific optimizations
        self._apply_optimizations()
        
        logger.info(f"Initialized {self.__class__.__name__} for {self.model_type} on {self.device}")
    
    def _get_device(self) -> str:
        """Get the device for the model"""
        if hasattr(self.model, "device"):
            return str(self.model.device)
        elif self.config.get("device"):
            return self.config["device"]
        return "cpu"
    
    def _get_precision(self) -> str:
        """Get the precision for the model"""
        return self.config.get("precision", "float32")
    
    def _get_model_type(self) -> str:
        """Get the model type"""
        return self.config.get("task", "unknown")
    
    def _apply_optimizations(self) -> None:
        """Apply model-specific optimizations"""
        # Set model to evaluation mode if available
        if hasattr(self.model, "eval") and callable(self.model.eval):
            self.model.eval()
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """
        Preprocess input data
        
        Args:
            input_data: Input data
            
        Returns:
            Dict[str, Any]: Preprocessed data
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("Subclasses must implement _preprocess")
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """
        Postprocess model output
        
        Args:
            model_output: Raw model output
            input_data: Original input data
            
        Returns:
            ModelOutput: Standardized output
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("Subclasses must implement _postprocess")
    
    async def process_async(self, input_data: ModelInput) -> ModelOutput:
        """
        Process input data asynchronously
        
        Args:
            input_data: Input data
            
        Returns:
            ModelOutput: Processed output
        """
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process, input_data)
    
    def process(self, input_data: ModelInput) -> ModelOutput:
        """
        Process input data
        
        Args:
            input_data: Input data
            
        Returns:
            ModelOutput: Processed output
        """
        start_time = time.time()
        
        try:
            # Preprocess input
            preprocessed = self._preprocess(input_data)
            preprocess_time = time.time() - start_time
            
            # Run model inference
            inference_start = time.time()
            with torch.no_grad():
                model_output = self._run_inference(preprocessed)
            inference_time = time.time() - inference_start
            
            # Postprocess output
            postprocess_start = time.time()
            result = self._postprocess(model_output, input_data)
            postprocess_time = time.time() - postprocess_start
            
            # Add metrics
            total_time = time.time() - start_time
            
            metrics = {
                "preprocess_time": preprocess_time,
                "inference_time": inference_time,
                "postprocess_time": postprocess_time,
                "total_time": total_time
            }
            
            if hasattr(result, "metrics") and result.metrics is not None:
                result.metrics.update(metrics)
            else:
                result.metrics = metrics
            
            # Store metrics for later analysis
            self._metrics.append(metrics)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing input: {e}", exc_info=True)
            return ModelOutput(
                result=None,
                status="error",
                error=str(e),
                metrics={"total_time": time.time() - start_time}
            )
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """
        Run model inference
        
        Args:
            preprocessed: Preprocessed input data
            
        Returns:
            Any: Raw model output
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("Subclasses must implement _run_inference")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics
        
        Returns:
            Dict[str, Any]: Metrics
        """
        if not self._metrics:
            return {}
        
        # Calculate average metrics
        metrics = {}
        for key in self._metrics[0].keys():
            metrics[key] = sum(m[key] for m in self._metrics) / len(self._metrics)
        
        metrics["count"] = len(self._metrics)
        return metrics


class TranslationModelWrapper(BaseModelWrapper):
    """Wrapper for translation models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess translation input"""
        if isinstance(input_data.text, list):
            texts = input_data.text
        else:
            texts = [input_data.text]
        
        # Get parameters
        source_lang = input_data.source_language or "en"
        target_lang = input_data.target_language or "es"
        
        # Check for MBART-specific language codes in parameters
        parameters = input_data.parameters or {}
        mbart_source_lang = parameters.get("mbart_source_lang")
        mbart_target_lang = parameters.get("mbart_target_lang")
        
        # Determine if we're using MBART or MT5 model
        is_mbart = (self.model.__class__.__name__ == "MBartForConditionalGeneration" or 
                    (hasattr(self.model, "config") and 
                     hasattr(self.model.config, "model_type") and 
                     getattr(self.model.config, "model_type", "") == "mbart"))
                     
        is_mt5 = (self.model.__class__.__name__ == "MT5ForConditionalGeneration" or 
                 (hasattr(self.model, "config") and 
                  hasattr(self.model.config, "model_type") and 
                  getattr(self.model.config, "model_type", "") == "mt5"))
        
        logger.debug(f"Model type detection: is_mbart={is_mbart}, is_mt5={is_mt5}")
        
        # Handle MBART model specially
        if is_mbart:
            logger.debug(f"Detected MBART model, using special language token handling")
            
            # Prepare inputs for MBART model
            if self.tokenizer:
                # For MBART models, we need to set source language for tokenizer
                if mbart_source_lang and hasattr(self.tokenizer, "set_src_lang_special_tokens"):
                    self.tokenizer.set_src_lang_special_tokens(mbart_source_lang)
                    
                # Tokenize without translation prompt, as MBART uses language tokens
                inputs = self.tokenizer(
                    texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=self.config.get("max_length", 512)
                )
                
                # Move to device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                
                # Store the MBART target language for generation
                return {
                    "inputs": inputs,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "mbart_source_lang": mbart_source_lang,
                    "mbart_target_lang": mbart_target_lang,
                    "original_texts": texts,
                    "is_mbart": True,
                    "is_mt5": False
                }
            else:
                # Fall back to regular processing if tokenizer not available
                logger.warning("MBART model detected but tokenizer not available")
                
        # Handle MT5 model specially
        elif is_mt5:
            logger.debug(f"Detected MT5 model, using special prefix format")
            
            # MT5 models typically use a target language prefix for translation
            # Format: "translate [source] to [target]: [text]"
            prompts = []
            for text in texts:
                prompt = f"translate {source_lang} to {target_lang}: {text}"
                prompts.append(prompt)
            
            logger.debug(f"MT5 translation prompts: {prompts}")
            
            # Tokenize inputs
            if self.tokenizer:
                inputs = self.tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=self.config.get("max_length", 512)
                )
                
                # Move to device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                
                logger.debug(f"MT5 tokenized input shapes: {', '.join([f'{k}: {v.shape}' for k, v in inputs.items() if isinstance(v, torch.Tensor)])}")
            else:
                inputs = {"texts": prompts}
            
            return {
                "inputs": inputs,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "original_texts": texts,
                "is_mbart": False,
                "is_mt5": True,
                "prompts": prompts  # Store the prompts for debugging
            }
        
        # Standard processing for other models
        # Prepare translation prompt
        prompts = []
        for text in texts:
            prompt = f"translate from {source_lang} to {target_lang}: {text}"
            prompts.append(prompt)
        
        logger.debug(f"Standard translation prompts: {prompts}")
        
        # Tokenize inputs
        if self.tokenizer:
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.get("max_length", 512)
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
        else:
            inputs = {"texts": prompts}
        
        return {
            "inputs": inputs,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "original_texts": texts,
            "is_mbart": False,
            "is_mt5": False,
            "prompts": prompts  # Store the prompts for debugging
        }
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run translation inference"""
        inputs = preprocessed["inputs"]
        is_mbart = preprocessed.get("is_mbart", False)
        is_mt5 = preprocessed.get("is_mt5", False)
        
        # For transformers models
        if hasattr(self.model, "generate") and callable(self.model.generate):
            # Get generation parameters
            gen_kwargs = self.config.get("generation_kwargs", {})
            
            # Set defaults if not provided
            if "max_length" not in gen_kwargs:
                gen_kwargs["max_length"] = 512
                
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 5
            
            # For MBART models, we need to specify the target language
            if is_mbart and "mbart_target_lang" in preprocessed:
                mbart_target_lang = preprocessed["mbart_target_lang"]
                if mbart_target_lang and hasattr(self.tokenizer, "lang_code_to_id"):
                    # Get the token ID for the target language
                    try:
                        # Use forced_bos_token_id for MBART to set target language
                        gen_kwargs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[mbart_target_lang]
                        logger.debug(f"Setting MBART target language token: {mbart_target_lang}")
                    except KeyError:
                        logger.warning(f"Language code {mbart_target_lang} not found in MBART tokenizer")
            
            # For MT5 models
            if is_mt5:
                # MT5 doesn't expect 'texts' in generate
                if "texts" in inputs:
                    inputs = {k: v for k, v in inputs.items() if k != "texts"}
                
                # Add MT5-specific generation parameters
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = False
                
                # Set a minimum length for MT5 outputs to avoid empty translations
                if "min_length" not in gen_kwargs:
                    gen_kwargs["min_length"] = 10
                
                # For debugging, log the prompts and generation params
                logger.debug(f"MT5 prompts: {preprocessed.get('prompts', [])}")
                logger.debug(f"MT5 generation parameters: {gen_kwargs}")
            
            # For debugging
            logger.debug(f"Model input shapes: {', '.join([f'{k}: {v.shape}' for k, v in inputs.items() if isinstance(v, torch.Tensor)])}")
            logger.debug(f"Generation parameters: {gen_kwargs}")
            
            try:
                # Generate translations
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )
                logger.debug(f"Generated output shape: {outputs.shape}")
                return outputs
            except Exception as e:
                logger.error(f"Error during model generation: {str(e)}", exc_info=True)
                # Return an empty output as fallback
                if is_mt5:
                    logger.warning("MT5 generation failed, returning empty output")
                    # Create a dummy output with a special token indicating an error
                    return torch.tensor([[self.tokenizer.pad_token_id]])
                raise  # Re-raise for other models
        
        # For custom models
        elif hasattr(self.model, "translate") and callable(self.model.translate):
            return self.model.translate(
                preprocessed["original_texts"],
                source_lang=preprocessed["source_lang"],
                target_lang=preprocessed["target_lang"]
            )
        
        else:
            raise ValueError("Unsupported translation model")
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess translation output"""
        # Check if the input is a torch tensor and convert to a list of decoded outputs
        if isinstance(model_output, torch.Tensor) and self.tokenizer and hasattr(self.tokenizer, "batch_decode"):
            # Log the output shape and values for debugging
            logger.debug(f"Postprocessing model output shape: {model_output.shape}")
            
            try:
                # Decode outputs
                translations = self.tokenizer.batch_decode(
                    model_output, 
                    skip_special_tokens=True
                )
                logger.debug(f"Decoded translations: {translations}")
            except Exception as e:
                logger.error(f"Error decoding model output: {str(e)}", exc_info=True)
                translations = ["[Error decoding translation]"]
        else:
            # Output already in text format
            translations = model_output
            logger.debug(f"Non-tensor output type: {type(model_output)}")
        
        # Check if this is an MT5 model
        is_mt5 = (self.model.__class__.__name__ == "MT5ForConditionalGeneration" or 
                 (hasattr(self.model, "config") and 
                  hasattr(self.model.config, "model_type") and 
                  getattr(self.model.config, "model_type", "") == "mt5"))
        
        # Keep track of whether we needed to fall back to MBART
        used_mbart_fallback = False
        mbart_translations = None
        original_mt5_translations = None
        
        logger.debug(f"Processing MT5 model: {is_mt5}")
        
        # Clean up MT5 special tokens if present
        if is_mt5 and translations:
            # Store the original MT5 translations before processing for debugging
            original_mt5_translations = translations.copy() if isinstance(translations, list) else [str(translations)]
            
            # Process each translation
            for i, translation in enumerate(translations):
                logger.debug(f"Original MT5 translation {i}: '{translation}'")
                
                # MT5 has serious hallucination issues - sometimes it produces severely garbled text
                # Let's do a more aggressive cleanup
                
                # 1. Remove all special tokens including <extra_id_N>
                if isinstance(translation, str):
                    # Match all <extra_id_N> tokens with regex
                    import re
                    translation = re.sub(r'<extra_id_\d+>', '', translation)
                    
                    # Remove any other special tokens
                    special_tokens_to_remove = ["</s>", "<pad>", "<unk>", "困り", "уланулан", "困り", ". )"]
                    for token in special_tokens_to_remove:
                        translation = translation.replace(token, "")
                    
                    # Remove any language indicators like "en es en es en es"
                    translation = re.sub(r'\b([a-z]{2})\s+([a-z]{2})\b(\s+[a-z]{2}\s+[a-z]{2})*', '', translation)
                
                    # 2. Remove repeated translate instructions
                    translation = re.sub(r'(translate .* to .*:).*?\1', r'\1', translation)
                    
                    # 3. Remove prompt text patterns
                    prefixes = [
                        f"translate {input_data.source_language} to {input_data.target_language}:", 
                        "translate to:", 
                        f"translate {input_data.source_language} to {input_data.target_language}",
                        f"translate {input_data.source_language} to:",
                        "translate from",
                        "translation:",
                        f"{input_data.source_language} to {input_data.target_language}:",
                        f"{input_data.source_language} {input_data.target_language}",
                        f"Translate {input_data.source_language} to {input_data.target_language}:"
                    ]
                    
                    for prefix in prefixes:
                        if translation.startswith(prefix):
                            translation = translation[len(prefix):].strip()
                        # Also remove it from middle of text
                        translation = translation.replace(prefix, " ")
                    
                    # 4. If the output repeats the input, try to remove the input text
                    if input_data.text in translation:
                        parts = translation.split(input_data.text, 1)
                        if len(parts) > 1 and len(parts[1].strip()) > 0:
                            translation = parts[1].strip()
                    
                    # 5. Remove excessive punctuation
                    translation = re.sub(r'([,.?!:;])\1+', r'\1', translation)
                    
                    # 6. Final trimming
                    translation = translation.strip()
                    
                    # Update the translation
                    translations[i] = translation
                    
                # Function to check if this is a poor quality translation
                def is_poor_quality_translation(text):
                    if not text or (isinstance(text, str) and not text.strip()):
                        return True
                    
                    # Check if it's our placeholder
                    if text == "[Translation not available]":
                        return True
                    
                    # Check if it's just repeating the source language text
                    if input_data.text.strip() == text.strip():
                        return True
                    
                    # Check for too short translations (unless source was also short)
                    if len(text) < 10 and len(input_data.text) > 20:
                        return True
                    
                    # Check for hallucinations - if it still contains language codes
                    if re.search(r'\b[a-z]{2}\s+[a-z]{2}\b', text):
                        return True
                    
                    # Check for severe token repetition
                    words = text.split()
                    if len(words) >= 4:
                        # Check for repeated sequences of words
                        repeated_patterns = 0
                        for j in range(len(words) - 3):
                            pattern = " ".join(words[j:j+2])
                            if pattern in " ".join(words[j+2:]):
                                repeated_patterns += 1
                        
                        if repeated_patterns > 2:  # Multiple repeated patterns suggest hallucination
                            return True
                    
                    return False
                
                # If translation is empty or poor quality, mark it for MBART fallback
                if is_poor_quality_translation(translations[i]):
                    # For now, just add a placeholder. We'll try MBART fallback later.
                    if input_data.source_language == input_data.target_language:
                        # Same language, just use the original text
                        translations[i] = input_data.text
                    else:
                        # Different languages, use placeholder temporarily
                        fallback_text = "[Translation not available]"
                        translations[i] = fallback_text
                        # Mark for MBART fallback attempt
                
                logger.debug(f"Processed MT5 translation {i}: '{translations[i]}'")
            
            # Check if we need to attempt MBART fallback for any translations
            if any(translation == "[Translation not available]" for translation in translations) or \
               any(is_poor_quality_translation(translation) for translation in translations):
                logger.info("MT5 translation quality poor, attempting MBART fallback")
                
                try:
                    # Get MBART translation as fallback
                    mbart_translations = self._get_mbart_fallback_translation(
                        input_data.text, 
                        input_data.source_language, 
                        input_data.target_language
                    )
                    
                    if mbart_translations:
                        # Replace poor quality MT5 translations with MBART results
                        if isinstance(mbart_translations, list):
                            for i, (mt5_trans, mbart_trans) in enumerate(zip(translations, mbart_translations)):
                                if is_poor_quality_translation(mt5_trans):
                                    translations[i] = mbart_trans
                                    used_mbart_fallback = True
                        else:
                            # Single translation case
                            if is_poor_quality_translation(translations[0] if isinstance(translations, list) else translations):
                                translations = mbart_translations
                                used_mbart_fallback = True
                except Exception as e:
                    logger.error(f"MBART fallback failed: {str(e)}", exc_info=True)
                    # Keep the cleaned MT5 translation as is
        
        # If single input, return single output
        if isinstance(input_data.text, str):
            translations = translations[0] if translations else ""
        
        # Enhance metadata for debugging
        metadata = {
            "source_language": input_data.source_language,
            "target_language": input_data.target_language,
            "model_type": getattr(self.model.config, "model_type", "unknown") if hasattr(self.model, "config") else "unknown"
        }
        
        # Add model-specific metadata
        if is_mt5:
            metadata["is_mt5"] = True
            metadata["mt5_postprocessing_applied"] = True
            
            if used_mbart_fallback:
                metadata["used_mbart_fallback"] = True
                metadata["original_mt5_translation"] = original_mt5_translations
            
            # Add information about the parameters used
            parameters = input_data.parameters or {}
            for k, v in parameters.items():
                metadata[f"param_{k}"] = v
        
        return ModelOutput(
            result=translations,
            metadata=metadata
        )
        
    def _get_mbart_fallback_translation(self, text, source_language, target_language):
        """
        Get a translation from MBART as fallback when MT5 fails.
        
        Args:
            text: The source text to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            str or list: The translation from MBART, or None if failed
        """
        from app.services.models.manager import EnhancedModelManager
        import asyncio
        
        # Log the fallback attempt
        logger.info(f"Attempting MBART fallback translation from {source_language} to {target_language}")
        
        # Map source language and target language to MBART language codes
        # MBART uses different language codes than our standard codes
        mbart_lang_map = {
            "en": "en_XX", "es": "es_XX", "fr": "fr_XX", "de": "de_DE", "it": "it_IT", 
            "pt": "pt_XX", "nl": "nl_XX", "pl": "pl_PL", "ru": "ru_RU", "zh": "zh_CN", 
            "ja": "ja_XX", "ko": "ko_KR", "ar": "ar_AR", "hi": "hi_IN", "tr": "tr_TR",
            "vi": "vi_VN", "th": "th_TH", "id": "id_ID", "tl": "fil_PH", "ro": "ro_RO"
        }
        
        mbart_source_lang = mbart_lang_map.get(source_language, f"{source_language}_XX")
        mbart_target_lang = mbart_lang_map.get(target_language, f"{target_language}_XX")
        
        try:
            # Try to use an existing MBART model if available in our application state
            # This is a bit of a hack since we don't have direct access to the model manager from here
            
            # Create a ModelInput for MBART
            mbart_input = ModelInput(
                text=text,
                source_language=source_language,
                target_language=target_language,
                parameters={
                    "mbart_source_lang": mbart_source_lang,
                    "mbart_target_lang": mbart_target_lang,
                    "fallback": True
                }
            )
            
            # Check if we're in a model manager context where we can use another model
            if hasattr(self, 'config') and hasattr(self.config, 'get') and self.config.get('model_manager'):
                # Use the model manager to get an MBART model
                model_manager = self.config.get('model_manager')
                
                # Run in an event loop if we're not already in one
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're already in an event loop, run synchronously
                        mbart_model_info = model_manager.load_model("mbart_translation")
                        # Create wrapper and process
                        from app.services.models.wrapper import create_model_wrapper
                        wrapper = create_model_wrapper(
                            "translation",
                            mbart_model_info["model"],
                            mbart_model_info["tokenizer"],
                            {"task": "translation", "device": self.device, "precision": "float16"}
                        )
                        result = wrapper.process(mbart_input)
                        return result.result
                    else:
                        # Create a new event loop and run
                        async def get_mbart_translation():
                            mbart_model_info = await model_manager.load_model("mbart_translation")
                            from app.services.models.wrapper import create_model_wrapper
                            wrapper = create_model_wrapper(
                                "translation",
                                mbart_model_info["model"],
                                mbart_model_info["tokenizer"],
                                {"task": "translation", "device": self.device, "precision": "float16"}
                            )
                            return wrapper.process(mbart_input)
                        
                        result = asyncio.run(get_mbart_translation())
                        return result.result
                except Exception as e:
                    logger.error(f"Error accessing MBART via model manager: {str(e)}", exc_info=True)
            
            # If we couldn't use the model manager, try a more direct approach
            # This should only be used as a last resort - it's better to use the model manager
            if isinstance(text, list):
                # For batch translation
                results = []
                
                # Import transformers directly if needed
                import torch
                from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
                
                # Load MBART model and tokenizer
                model_name = "facebook/mbart-large-50-many-to-many-mmt"
                model = MBartForConditionalGeneration.from_pretrained(model_name)
                tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
                
                # Move model to device if available
                if torch.cuda.is_available():
                    model = model.to("cuda")
                
                # Process each text
                for item_text in text:
                    # Set source language
                    tokenizer.src_lang = mbart_source_lang
                    
                    # Encode text
                    encoded = tokenizer(item_text, return_tensors="pt")
                    if torch.cuda.is_available():
                        encoded = {k: v.to("cuda") for k, v in encoded.items()}
                    
                    # Generate translation
                    generated_tokens = model.generate(
                        **encoded, 
                        forced_bos_token_id=tokenizer.lang_code_to_id[mbart_target_lang]
                    )
                    
                    # Decode translation
                    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                    results.append(translation)
                
                return results
            else:
                # For single text translation
                import torch
                from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
                
                # Load MBART model and tokenizer
                model_name = "facebook/mbart-large-50-many-to-many-mmt"
                model = MBartForConditionalGeneration.from_pretrained(model_name)
                tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
                
                # Move model to device if available
                if torch.cuda.is_available():
                    model = model.to("cuda")
                
                # Set source language
                tokenizer.src_lang = mbart_source_lang
                
                # Encode text
                encoded = tokenizer(text, return_tensors="pt")
                if torch.cuda.is_available():
                    encoded = {k: v.to("cuda") for k, v in encoded.items()}
                
                # Generate translation
                generated_tokens = model.generate(
                    **encoded, 
                    forced_bos_token_id=tokenizer.lang_code_to_id[mbart_target_lang]
                )
                
                # Decode translation
                translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                return translation
                
        except Exception as e:
            logger.error(f"MBART fallback translation failed: {str(e)}", exc_info=True)
            return None


class LanguageDetectionWrapper(BaseModelWrapper):
    """Wrapper for language detection models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess language detection input"""
        if isinstance(input_data.text, list):
            texts = input_data.text
        else:
            texts = [input_data.text]
        
        # Tokenize inputs
        if self.tokenizer:
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.get("max_length", 512)
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
        else:
            inputs = {"texts": texts}
        
        return {
            "inputs": inputs,
            "original_texts": texts
        }
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run language detection inference"""
        inputs = preprocessed["inputs"]
        
        # For transformers models
        if hasattr(self.model, "forward") and callable(self.model.forward):
            outputs = self.model(**inputs)
            return outputs
        
        # For custom models
        elif hasattr(self.model, "detect_language") and callable(self.model.detect_language):
            return self.model.detect_language(preprocessed["original_texts"])
        
        else:
            raise ValueError("Unsupported language detection model")
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess language detection output"""
        # Get language mapping
        id2label = getattr(self.model.config, "id2label", None)
        
        if id2label and hasattr(model_output, "logits"):
            # Get predictions
            logits = model_output.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Map to language codes
            languages = [id2label[pred.item()] for pred in predictions]
            
            # Get confidences
            probs = torch.softmax(logits, dim=-1)
            confidences = [probs[i, pred.item()].item() for i, pred in enumerate(predictions)]
            
            # Combine results
            results = [
                {"language": lang, "confidence": conf}
                for lang, conf in zip(languages, confidences)
            ]
        else:
            # Model already returned processed output
            results = model_output
        
        # If single input, return single output
        if isinstance(input_data.text, str):
            results = results[0] if results else {"language": "unknown", "confidence": 0.0}
        
        return ModelOutput(
            result=results,
            metadata={}
        )


class NERDetectionWrapper(BaseModelWrapper):
    """Wrapper for Named Entity Recognition models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess NER input"""
        if isinstance(input_data.text, list):
            texts = input_data.text
        else:
            texts = [input_data.text]
        
        # Tokenize inputs
        if self.tokenizer:
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.get("max_length", 512)
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
        else:
            inputs = {"texts": texts}
        
        return {
            "inputs": inputs,
            "original_texts": texts
        }
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run NER inference"""
        inputs = preprocessed["inputs"]
        
        # For transformers models
        if hasattr(self.model, "forward") and callable(self.model.forward):
            outputs = self.model(**inputs)
            return outputs
        
        # For custom models
        elif hasattr(self.model, "extract_entities") and callable(self.model.extract_entities):
            return self.model.extract_entities(preprocessed["original_texts"])
        
        else:
            raise ValueError("Unsupported NER model")
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess NER output"""
        # Different processing for different model types
        if hasattr(model_output, "logits"):
            # Process outputs for token classification
            logits = model_output.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Get label mapping
            id2label = getattr(self.model.config, "id2label", None)
            
            # Extract entities
            entities = []
            
            for i, pred_seq in enumerate(predictions):
                doc_entities = []
                current_entity = None
                
                # Get original tokens
                input_ids = preprocessed["inputs"]["input_ids"][i]
                original_text = preprocessed["original_texts"][i]
                
                # Convert to tokens
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                
                for j, (token, pred) in enumerate(zip(tokens, pred_seq)):
                    # Skip special tokens
                    if token in self.tokenizer.special_tokens_map.values():
                        continue
                    
                    # Get label
                    label = id2label[pred.item()] if id2label else str(pred.item())
                    
                    # Skip "O" (Outside) labels
                    if label == "O":
                        if current_entity:
                            doc_entities.append(current_entity)
                            current_entity = None
                        continue
                    
                    # Extract entity type (B-PER, I-LOC, etc.)
                    entity_type = label[2:] if label.startswith("B-") or label.startswith("I-") else label
                    
                    # Start new entity on B- tag
                    if label.startswith("B-"):
                        if current_entity:
                            doc_entities.append(current_entity)
                        
                        current_entity = {
                            "text": token,
                            "type": entity_type,
                            "start": j,
                            "end": j
                        }
                    # Continue entity on I- tag
                    elif label.startswith("I-") and current_entity and current_entity["type"] == entity_type:
                        current_entity["text"] += token
                        current_entity["end"] = j
                    # Handle unexpected I- tag
                    else:
                        if current_entity:
                            doc_entities.append(current_entity)
                        
                        current_entity = {
                            "text": token,
                            "type": entity_type,
                            "start": j,
                            "end": j
                        }
                
                # Add final entity
                if current_entity:
                    doc_entities.append(current_entity)
                
                entities.append(doc_entities)
        else:
            # Model already returned processed output
            entities = model_output
        
        # If single input, return single output
        if isinstance(input_data.text, str):
            entities = entities[0] if entities else []
        
        return ModelOutput(
            result=entities,
            metadata={}
        )


class RAGGeneratorWrapper(BaseModelWrapper):
    """Wrapper for RAG generator models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess RAG input"""
        if isinstance(input_data.text, str):
            query = input_data.text
        else:
            # Join multiple texts
            query = " ".join(input_data.text)
        
        # Get context
        context = input_data.context or []
        context_texts = [item.get("text", "") for item in context]
        
        # Prepare prompt with context
        if context_texts:
            context_str = "\n".join(context_texts)
            prompt = f"Context: {context_str}\n\nQuestion: {query}\n\nAnswer:"
        else:
            prompt = f"Question: {query}\n\nAnswer:"
        
        # Tokenize input
        if self.tokenizer:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get("max_length", 1024)
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
        else:
            inputs = {"text": prompt}
        
        return {
            "inputs": inputs,
            "query": query,
            "context": context
        }
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run RAG generation inference"""
        inputs = preprocessed["inputs"]
        
        # For transformers models
        if hasattr(self.model, "generate") and callable(self.model.generate):
            # Get generation parameters
            gen_kwargs = self.config.get("generation_kwargs", {})
            
            # Set defaults if not provided
            if "max_length" not in gen_kwargs:
                gen_kwargs["max_length"] = 512
                
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 4
                
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = True
                
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.7
            
            # Generate output
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
            
            return outputs
        
        # For custom models
        elif hasattr(self.model, "generate") and callable(self.model.generate):
            return self.model.generate(
                preprocessed["query"],
                context=preprocessed["context"]
            )
        
        else:
            raise ValueError("Unsupported RAG generator model")
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess RAG generation output"""
        if self.tokenizer and hasattr(self.tokenizer, "batch_decode"):
            # Decode outputs
            text = self.tokenizer.batch_decode(
                model_output, 
                skip_special_tokens=True
            )[0]
        else:
            # Output already in text format
            text = model_output
        
        return ModelOutput(
            result=text,
            metadata={
                "sources": [item.get("source", "") for item in (input_data.context or [])]
            }
        )


class RAGRetrieverWrapper(BaseModelWrapper):
    """Wrapper for RAG retriever models"""
    
    def __init__(self, model: Any, tokenizer: Any = None, config: Dict[str, Any] = None, index=None):
        """
        Initialize RAG retriever wrapper
        
        Args:
            model: The model to wrap
            tokenizer: The tokenizer to use
            config: Configuration parameters
            index: Vector index for retrieval
        """
        super().__init__(model, tokenizer, config)
        self.index = index
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess retriever input"""
        if isinstance(input_data.text, str):
            query = input_data.text
        else:
            # Use first text
            query = input_data.text[0]
        
        # Get parameters
        top_k = input_data.parameters.get("top_k", 5) if input_data.parameters else 5
        
        # Tokenize query
        if self.tokenizer:
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get("max_length", 512)
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
        else:
            inputs = {"text": query}
        
        return {
            "inputs": inputs,
            "query": query,
            "top_k": top_k
        }
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run retriever inference"""
        inputs = preprocessed["inputs"]
        
        # For sentence transformers models
        if hasattr(self.model, "encode") and callable(self.model.encode):
            # Encode query
            with torch.no_grad():
                query_embedding = self.model.encode(
                    preprocessed["query"],
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
            
            # Convert to numpy if tensor
            if torch.is_tensor(query_embedding):
                query_embedding = query_embedding.cpu().numpy()
            
            # Search in index
            if self.index is not None:
                scores, indices = self.index.search(
                    np.array([query_embedding]), 
                    preprocessed["top_k"]
                )
                
                # Return search results
                return {"scores": scores[0], "indices": indices[0]}
            else:
                return {"embedding": query_embedding}
        
        # For transformers models
        elif hasattr(self.model, "forward") and callable(self.model.forward):
            # Get query embedding
            outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Search in index
            if self.index is not None:
                scores, indices = self.index.search(
                    query_embedding, 
                    preprocessed["top_k"]
                )
                
                # Return search results
                return {"scores": scores[0], "indices": indices[0]}
            else:
                return {"embedding": query_embedding[0]}
        
        # For custom models
        elif hasattr(self.model, "retrieve") and callable(self.model.retrieve):
            return self.model.retrieve(
                preprocessed["query"],
                top_k=preprocessed["top_k"]
            )
        
        else:
            raise ValueError("Unsupported RAG retriever model")
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess retriever output"""
        # For embedding-only output
        if "embedding" in model_output:
            return ModelOutput(
                result=model_output["embedding"],
                metadata={"query": input_data.text}
            )
        
        # For search results
        if "scores" in model_output and "indices" in model_output:
            results = []
            
            # Map indices to actual documents if available
            if hasattr(self, "documents") and self.documents:
                for i, (score, idx) in enumerate(zip(model_output["scores"], model_output["indices"])):
                    if idx < len(self.documents):
                        doc = self.documents[idx]
                        results.append({
                            "score": float(score),
                            "document": doc
                        })
            else:
                # Just return indices and scores
                for i, (score, idx) in enumerate(zip(model_output["scores"], model_output["indices"])):
                    results.append({
                        "score": float(score),
                        "index": int(idx)
                    })
            
            return ModelOutput(
                result=results,
                metadata={"query": input_data.text}
            )
        
        # Return raw output if it doesn't match expected formats
        return ModelOutput(
            result=model_output,
            metadata={"query": input_data.text}
        )


class SimplifierWrapper(BaseModelWrapper):
    """Wrapper for text simplification models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess simplification input"""
        if isinstance(input_data.text, list):
            texts = input_data.text
        else:
            texts = [input_data.text]
        
        # Prepare simplification prompt
        prompts = []
        for text in texts:
            prompt = f"simplify: {text}"
            prompts.append(prompt)
        
        # Tokenize inputs
        if self.tokenizer:
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.get("max_length", 512)
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
        else:
            inputs = {"texts": prompts}
        
        return {
            "inputs": inputs,
            "original_texts": texts
        }
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run simplification inference"""
        inputs = preprocessed["inputs"]
        
        # For transformers models
        if hasattr(self.model, "generate") and callable(self.model.generate):
            # Get generation parameters
            gen_kwargs = self.config.get("generation_kwargs", {})
            
            # Set defaults if not provided
            if "max_length" not in gen_kwargs:
                gen_kwargs["max_length"] = 512
                
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 4
            
            # Generate simplifications
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
            
            return outputs
        
        # For custom models
        elif hasattr(self.model, "simplify") and callable(self.model.simplify):
            return self.model.simplify(preprocessed["original_texts"])
        
        else:
            raise ValueError("Unsupported simplification model")
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess simplification output"""
        if self.tokenizer and hasattr(self.tokenizer, "batch_decode"):
            # Decode outputs
            simplifications = self.tokenizer.batch_decode(
                model_output, 
                skip_special_tokens=True
            )
        else:
            # Output already in text format
            simplifications = model_output
        
        # If single input, return single output
        if isinstance(input_data.text, str):
            simplifications = simplifications[0] if simplifications else ""
        
        return ModelOutput(
            result=simplifications,
            metadata={}
        )


class AnonymizerWrapper(BaseModelWrapper):
    """Wrapper for anonymization models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess anonymization input"""
        if isinstance(input_data.text, list):
            texts = input_data.text
        else:
            texts = [input_data.text]
        
        # Tokenize inputs
        if self.tokenizer:
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.get("max_length", 512)
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
        else:
            inputs = {"texts": texts}
        
        return {
            "inputs": inputs,
            "original_texts": texts
        }
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run anonymization inference"""
        inputs = preprocessed["inputs"]
        
        # For transformers models
        if hasattr(self.model, "forward") and callable(self.model.forward):
            outputs = self.model(**inputs)
            return outputs
        
        # For custom models
        elif hasattr(self.model, "anonymize") and callable(self.model.anonymize):
            return self.model.anonymize(preprocessed["original_texts"])
        
        else:
            raise ValueError("Unsupported anonymization model")
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """
        Postprocess anonymization output by replacing identified entities with placeholders.
        
        Args:
            model_output: Model prediction output, typically containing entity information
            input_data: Original input data
            
        Returns:
            ModelOutput with anonymized text and metadata
        """
        parameters = input_data.parameters or {}
        strategy = parameters.get("strategy", "mask")  # mask, replace, pseudonymize
        
        # Different processing for different model types
        if hasattr(model_output, "logits"):
            # Process transformer token classification models (like NER models)
            logits = model_output.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Get label mapping
            id2label = getattr(self.model.config, "id2label", None)
            if not id2label:
                logger.warning("No id2label mapping found in model config, using default O/B-/I- parsing")
                
            # Process each text separately
            anonymized_texts = []
            entity_counts = {}
            
            for i, pred_seq in enumerate(predictions):
                original_text = preprocessed["original_texts"][i] if "preprocessed" in locals() else input_data.text[i] if isinstance(input_data.text, list) else input_data.text
                
                # Get original tokens
                input_ids = None
                if "preprocessed" in locals() and "inputs" in preprocessed and "input_ids" in preprocessed["inputs"]:
                    input_ids = preprocessed["inputs"]["input_ids"][i]
                
                if input_ids is not None and self.tokenizer:
                    # Convert prediction indices to tokens and entities
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                    anonymized_text, entities = self._anonymize_with_tokens(tokens, pred_seq, original_text, id2label, strategy)
                else:
                    # Fallback to direct entity recognition and replacement
                    anonymized_text, entities = self._anonymize_with_regex(original_text, strategy)
                
                # Add to result
                anonymized_texts.append(anonymized_text)
                
                # Count entity types
                for entity in entities:
                    entity_type = entity.get("type", "UNKNOWN")
                    if entity_type not in entity_counts:
                        entity_counts[entity_type] = 0
                    entity_counts[entity_type] += 1
        else:
            # Model already returned processed output
            anonymized_texts = model_output
            entity_counts = {}
            
            # If we have more structured output with entity info
            if isinstance(model_output, dict) and "entities" in model_output:
                for entity_list in model_output.get("entities", []):
                    for entity in entity_list:
                        entity_type = entity.get("type", "UNKNOWN")
                        if entity_type not in entity_counts:
                            entity_counts[entity_type] = 0
                        entity_counts[entity_type] += 1
        
        # If single input, return single output
        if isinstance(input_data.text, str):
            result = anonymized_texts[0] if anonymized_texts else ""
        else:
            result = anonymized_texts
        
        return ModelOutput(
            result=result,
            metadata={
                "entity_counts": entity_counts,
                "anonymization_strategy": strategy
            }
        )
        
    def _anonymize_with_tokens(self, tokens, predictions, original_text, id2label, strategy):
        """
        Anonymize text using token-level predictions.
        
        Args:
            tokens: Tokenized text
            predictions: Token-level predictions
            original_text: Original input text
            id2label: Mapping from prediction IDs to entity labels
            strategy: Anonymization strategy
            
        Returns:
            Tuple of (anonymized_text, entities)
        """
        entities = []
        anonymized_text = original_text
        spans_to_replace = []
        
        current_entity = None
        token_offset = 0
        
        # Identify entities from predictions
        for j, (token, pred) in enumerate(zip(tokens, predictions)):
            # Skip special tokens
            if self.tokenizer and token in getattr(self.tokenizer, "special_tokens_map", {}).values():
                continue
            
            # Get label
            if id2label:
                label = id2label[pred.item()] if isinstance(pred, torch.Tensor) else id2label[pred]
            else:
                # Default parsing pattern: O, B-TYPE, I-TYPE
                label = str(pred.item() if isinstance(pred, torch.Tensor) else pred)
                
            # Skip "O" (Outside) labels
            if label == "O":
                if current_entity:
                    spans_to_replace.append(current_entity)
                    current_entity = None
                continue
            
            # Extract entity type (B-PER, I-LOC, etc.)
            entity_type = label[2:] if (label.startswith("B-") or label.startswith("I-")) else label
            
            # Start new entity on B- tag
            if label.startswith("B-"):
                if current_entity:
                    spans_to_replace.append(current_entity)
                
                # Find token position in original text
                token_pos = original_text.find(token, token_offset)
                if token_pos >= 0:
                    token_offset = token_pos + len(token)
                    
                    current_entity = {
                        "text": token,
                        "type": entity_type,
                        "start": token_pos,
                        "end": token_offset
                    }
            
            # Continue entity on I- tag
            elif label.startswith("I-") and current_entity and current_entity["type"] == entity_type:
                current_entity["text"] += token
                current_entity["end"] = current_entity["start"] + len(current_entity["text"])
        
        # Add final entity
        if current_entity:
            spans_to_replace.append(current_entity)
        
        # Apply replacements from end to start to maintain offsets
        spans_to_replace.sort(key=lambda x: x["start"], reverse=True)
        for entity in spans_to_replace:
            replacement = self._get_replacement(entity["text"], entity["type"], strategy)
            anonymized_text = anonymized_text[:entity["start"]] + replacement + anonymized_text[entity["end"]:]
            entities.append({
                "type": entity["type"],
                "original": entity["text"],
                "replacement": replacement
            })
        
        return anonymized_text, entities
    
    def _anonymize_with_regex(self, text, strategy):
        """
        Anonymize text using regex patterns when token-level processing is not available.
        
        Args:
            text: Original text
            strategy: Anonymization strategy
            
        Returns:
            Tuple of (anonymized_text, entities)
        """
        anonymized_text = text
        entities = []
        
        # Common regex patterns for PII
        patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE": r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            "SSN": r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "DATE": r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b',
            "URL": r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*\??[-\w=&]+',
            "IP_ADDRESS": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
        # Apply regex patterns
        for entity_type, pattern in patterns.items():
            import re
            matches = list(re.finditer(pattern, anonymized_text))
            
            # Process matches from end to start to avoid offset changes
            for match in reversed(matches):
                original = match.group()
                replacement = self._get_replacement(original, entity_type, strategy)
                anonymized_text = anonymized_text[:match.start()] + replacement + anonymized_text[match.end():]
                entities.append({
                    "type": entity_type,
                    "original": original,
                    "replacement": replacement
                })
        
        return anonymized_text, entities
    
    def _get_replacement(self, original, entity_type, strategy):
        """
        Generate a replacement for the identified entity.
        
        Args:
            original: Original entity text
            entity_type: Type of entity (PER, LOC, ORG, etc.)
            strategy: Anonymization strategy
            
        Returns:
            Replacement text
        """
        # Strategy: mask - Replace with a generic placeholder
        if strategy == "mask":
            return f"[{entity_type}]"
        
        # Strategy: redact - Replace with fixed length mask
        elif strategy == "redact":
            return "X" * len(original)
        
        # Strategy: pseudonymize - Replace with realistic but fake data
        elif strategy == "pseudonymize":
            # Use the entity text as a seed for deterministic generation
            import hashlib
            seed = hashlib.md5(original.encode('utf-8')).hexdigest()
            seed_number = int(seed, 16)
            
            if entity_type in ["PER", "PERSON", "NAME"]:
                # Generate a gender-appropriate name
                first_names_male = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph"]
                first_names_female = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan"]
                last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson"]
                
                # Guess gender from name
                is_female = any(name in original for name in first_names_female)
                
                # Choose first name based on gender
                if is_female:
                    first_name = first_names_female[seed_number % len(first_names_female)]
                else:
                    first_name = first_names_male[seed_number % len(first_names_male)]
                    
                last_name = last_names[(seed_number >> 8) % len(last_names)]
                return f"{first_name} {last_name}"
                
            elif entity_type in ["LOC", "LOCATION", "GPE"]:
                locations = ["Springfield", "Centerville", "Fairview", "Riverview", "Lakeside", "Mountainview",
                             "Oakland", "Greenville", "Riverside", "Brookside", "Westwood", "Eastwood"]
                return locations[seed_number % len(locations)]
                
            elif entity_type in ["ORG", "ORGANIZATION"]:
                prefixes = ["Universal", "Global", "National", "International", "Advanced", "United", "Superior"]
                suffixes = ["Corporation", "Associates", "Industries", "Solutions", "Technologies", "Group", "Partners"]
                
                prefix = prefixes[seed_number % len(prefixes)]
                suffix = suffixes[(seed_number >> 8) % len(suffixes)]
                
                return f"{prefix} {suffix}"
                
            elif entity_type == "DATE":
                # Generate a reasonable date
                import datetime
                base_date = datetime.date(2000, 1, 1)
                days = seed_number % 3650  # ~10 years
                new_date = base_date + datetime.timedelta(days=days)
                return new_date.strftime("%m/%d/%Y")
                
            elif entity_type == "EMAIL":
                domains = ["example.com", "anonymous.org", "private.net", "redacted.info", "mail.com"]
                usernames = ["user", "person", "contact", "info", "service", "anonymous"]
                
                username = usernames[seed_number % len(usernames)]
                domain = domains[(seed_number >> 8) % len(domains)]
                
                return f"{username}{seed_number % 1000}@{domain}"
                
            elif entity_type == "PHONE":
                # Format: (XXX) XXX-XXXX
                area_code = 100 + (seed_number % 900)
                prefix = 100 + ((seed_number >> 10) % 900)
                suffix = 1000 + ((seed_number >> 20) % 9000)
                
                return f"({area_code}) {prefix}-{suffix}"
                
            elif entity_type == "ID_NUMBER":
                # Generic ID format with consistent generation
                return f"ID-{seed[:8]}"
                
            elif entity_type == "CREDIT_CARD":
                return "XXXX-XXXX-XXXX-XXXX"
                
            elif entity_type == "IP_ADDRESS":
                # Generate a private IP address
                return f"192.168.{seed_number % 256}.{(seed_number >> 8) % 256}"
                
            elif entity_type == "URL":
                paths = ["privacy", "about", "contact", "index", "main", "home", "service"]
                path = paths[seed_number % len(paths)]
                return f"https://example.com/{path}"
                
            else:
                return f"[{entity_type}]"
        
        # Default: replace with type
        else:
            return f"[{entity_type}]"


# Factory function to create the appropriate wrapper
def create_model_wrapper(model_type: str, model: Any, tokenizer: Any = None, config: Dict[str, Any] = None, **kwargs) -> BaseModelWrapper:
    """
    Create a model wrapper for the specified model type
    
    Args:
        model_type: Type of model
        model: Model instance
        tokenizer: Tokenizer instance
        config: Configuration parameters
        **kwargs: Additional arguments
        
    Returns:
        BaseModelWrapper: Appropriate model wrapper
    """
    wrapper_map = {
        "translation": TranslationModelWrapper,
        "language_detection": LanguageDetectionWrapper,
        "ner_detection": NERDetectionWrapper,
        "rag_generator": RAGGeneratorWrapper,
        "rag_retriever": RAGRetrieverWrapper,
        "simplifier": SimplifierWrapper,
        "anonymizer": AnonymizerWrapper
    }
    
    if model_type in wrapper_map:
        wrapper_class = wrapper_map[model_type]
        return wrapper_class(model, tokenizer, config, **kwargs)
    else:
        logger.warning(f"No specific wrapper for model type: {model_type}, using base wrapper")
        return BaseModelWrapper(model, tokenizer, config, **kwargs)


# Pipeline integration function
def integrate_with_pipeline(model_info: Dict[str, Any], pipeline_registry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrate a model with the pipeline
    
    Args:
        model_info: Model information
        pipeline_registry: Pipeline registry
        
    Returns:
        Dict[str, Any]: Updated pipeline registry
    """
    model_type = model_info.get("type")
    model = model_info.get("model")
    tokenizer = model_info.get("tokenizer")
    config = model_info.get("config", {})
    
    # Create appropriate wrapper
    wrapper = create_model_wrapper(model_type, model, tokenizer, config)
    
    # Register wrapper with pipeline
    pipeline_registry[model_type] = wrapper
    
    logger.info(f"Integrated model {model_type} with pipeline")
    return pipeline_registry