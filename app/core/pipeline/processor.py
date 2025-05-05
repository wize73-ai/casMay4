"""
Unified Processor for CasaLingua

This module implements the main processing pipeline that handles
all input types and orchestrates various processing operations.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import time
import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

from app.core.pipeline.detector import InputDetector
from app.core.pipeline.translator import TranslationPipeline
from app.core.pipeline.simplifier import SimplificationPipeline
from app.core.pipeline.anonymizer import AnonymizationPipeline
from app.core.pipeline.summarizer import SummarizationPipeline
from app.core.pipeline.tts import TTSPipeline
from app.core.document.pdf import PDFProcessor
from app.core.document.docx import DOCXProcessor
from app.core.document.ocr import OCRProcessor
from app.core.rag.expert import RAGExpert
from app.audit.logger import AuditLogger
from app.audit.metrics import MetricsCollector
from app.audit.veracity import VeracityAuditor
from app.services.hardware.resource_monitor import get_resource_monitor
from app.utils.logging import get_logger
from app.api.schemas.translation import TranslationResult, TranslationRequest
from app.api.schemas.language import LanguageDetectionRequest

logger = get_logger("casalingua.core.processor")

class UnifiedProcessor:
    """
    Unified processing pipeline for all types of input.
    
    This processor handles:
    - Text processing (translation, simplification, anonymization)
    - Document processing (PDF, DOCX, etc.)
    - Image processing (OCR)
    - Audio processing (speech-to-text)
    - RAG-enhanced processing
    
    It automatically detects input types and routes them through
    the appropriate processing pipeline while maintaining consistent
    logging, auditing, and performance metrics.
    """
    
    def __init__(self, 
                model_manager,
                audit_logger: AuditLogger,
                metrics: MetricsCollector,
                config: Dict[str, Any] = None,
                registry_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified processor.
        
        Args:
            model_manager: Model manager for accessing ML models
            audit_logger: Audit logger for tracking operations
            metrics: Metrics collector for performance monitoring
            config: Configuration dictionary
            registry_config: Registry configuration for models
        """
        self.model_manager = model_manager
        self.audit_logger = audit_logger
        self.metrics = metrics
        self.config = config or {}
        self.registry_config = registry_config or {}
        
        # Initialize subcomponents
        self.input_detector = InputDetector(self.model_manager, self.config, registry_config=self.registry_config)
        
        # Initialize pipeline components to None - will be created on demand
        self.translation_pipeline = None
        self.simplification_pipeline = None
        self.anonymization_pipeline = None
        self.multipurpose_pipeline = None  # For summarization
        self.tts_pipeline = None
        
        # Initialize document processors to None - will be created on demand
        self.pdf_processor = None
        self.docx_processor = None
        self.ocr_processor = None
        self.rag_expert = None
        
        # Initialize verification
        self.veracity_checker = VeracityAuditor(model_manager=self.model_manager, config=self.config)
        
        # Initialize resource monitor
        self.resource_monitor = get_resource_monitor(self.config)
        
        # Processing cache setup
        self.cache_enabled = self.config.get("enable_cache", True)
        self.cache_size = self.config.get("cache_size", 1000)
        self.cache_ttl = self.config.get("cache_ttl_seconds", 3600)  # 1 hour
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_lock = asyncio.Lock()  # Concurrency lock for thread safety
        
        # Status tracking
        self.initialized = False
        self.startup_time = None
        
        # Pipeline execution metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Temp directory setup
        self.temp_dir = Path(self.config.get("temp_dir", "temp"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Unified processor created (not yet initialized)")
    
    async def _initialize_translation_pipeline(self) -> None:
        """Initialize the translation pipeline component."""
        logger.info("Initializing translation pipeline")
        self.translation_pipeline = TranslationPipeline(
            self.model_manager, 
            self.config, 
            registry_config=self.registry_config
        )
        await self.translation_pipeline.initialize()
        logger.debug("Translation pipeline initialized successfully")
        logger.info("Translation pipeline initialization complete")
        
    async def _initialize_simplification_pipeline(self) -> None:
        """Initialize the simplification pipeline component."""
        logger.info("Initializing simplification pipeline")
        self.simplification_pipeline = SimplificationPipeline(
            self.model_manager, 
            self.config, 
            registry_config=self.registry_config
        )
        await self.simplification_pipeline.initialize()
        logger.info("Simplification pipeline initialization complete")
        
    async def _initialize_multipurpose_pipeline(self) -> None:
        """Initialize the multipurpose (summarization) pipeline component."""
        logger.info("Initializing multipurpose pipeline")
        self.multipurpose_pipeline = SummarizationPipeline(
            self.model_manager, 
            self.config, 
            registry_config=self.registry_config
        )
        await self.multipurpose_pipeline.initialize()
        logger.info("Multipurpose pipeline initialization complete")
        
    async def _initialize_anonymization_pipeline(self) -> None:
        """Initialize the anonymization pipeline component."""
        logger.info("Initializing anonymization pipeline")
        self.anonymization_pipeline = AnonymizationPipeline(
            self.model_manager, 
            self.config, 
            registry_config=self.registry_config
        )
        await self.anonymization_pipeline.initialize()
        logger.info("Anonymization pipeline initialization complete")
        
    async def _initialize_tts_pipeline(self) -> None:
        """Initialize the TTS pipeline component."""
        logger.info("Initializing TTS pipeline")
        self.tts_pipeline = TTSPipeline(
            self.model_manager,
            self.config,
            registry_config=self.registry_config
        )
        await self.tts_pipeline.initialize()
        logger.info("TTS pipeline initialization complete")
        
    async def _initialize_rag_expert(self) -> None:
        """Initialize the RAG expert component (if enabled)."""
        logger.info("Initializing RAG expert")
        self.rag_expert = RAGExpert(
            self.model_manager, 
            self.config, 
            registry_config=self.registry_config
        )
        await self.rag_expert.initialize()
        logger.info("RAG expert initialization complete")
    
    async def initialize(self) -> None:
        """
        Initialize all processing components concurrently.
        
        This loads the necessary models and prepares all pipeline components
        in parallel for faster startup.
        """
        if self.initialized:
            logger.warning("Processor already initialized")
            return
        
        start_time = time.time()
        logger.info("Initializing unified processor")
        
        # Create initialization tasks for pipeline components
        init_tasks = [
            self._initialize_translation_pipeline(),
            self._initialize_simplification_pipeline(),
            self._initialize_multipurpose_pipeline(),
            self._initialize_anonymization_pipeline(),
            self._initialize_tts_pipeline()
        ]
        
        # If RAG is enabled, add it to initialization tasks
        if self.config.get("rag_enabled", True):
            init_tasks.append(self._initialize_rag_expert())
        
        # Run all initialization tasks concurrently
        await asyncio.gather(*init_tasks)
        
        # Initialize document processors (these are synchronous)
        logger.info("Initializing document processors")
        self.pdf_processor = PDFProcessor(self.model_manager, self.config)
        self.docx_processor = DOCXProcessor(self.model_manager, self.config)
        self.ocr_processor = OCRProcessor(self.model_manager, self.config)
        
        # Initialize veracity checker
        logger.info("Initializing veracity auditor")
        await self.veracity_checker.initialize()
        
        # Initialize resource monitor tasks
        logger.info("Initializing resource monitor")
        
        # Start cache cleanup task
        if self.cache_enabled:
            asyncio.create_task(self._cache_cleanup_task())
        
        self.initialized = True
        self.startup_time = time.time() - start_time
        logger.info(f"Unified processor initialization complete in {self.startup_time:.2f}s")
    
    async def process(self, 
                     content: Union[str, bytes],
                     options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input content through the appropriate pipeline.
        
        Args:
            content: Input content (text or binary)
            options: Processing options
                - source_language: Source language code
                - target_language: Target language code
                - simplify: Whether to simplify text
                - simplification_level: Level of simplification (1-5)
                - grade_level: Target grade level (1-12)
                - anonymize: Whether to anonymize personal information
                - use_rag: Whether to use RAG for context
                - verify_output: Whether to verify output veracity
                - tts_output: Whether to generate speech from result
                - request_id: Request identifier
                - session_id: Session identifier
                - cache_key: Custom cache key for this request
                
        Returns:
            Dict containing processing results
        """
        if not self.initialized:
            raise RuntimeError("Processor not initialized")
        
        options = options or {}
        start_time = time.time()
        request_id = options.get("request_id", str(uuid.uuid4()))
        session_id = options.get("session_id")
        
        # Increment request counter
        self.total_requests += 1
        
        logger.info(f"Processing request {request_id}" + 
                    (f" in session {session_id}" if session_id else ""))
        
        # Check cache if enabled
        cache_key = options.get("cache_key")
        if not cache_key and self.cache_enabled:
            # Generate cache key from content and options
            cache_key = self._generate_cache_key(content, options)
            
        # Look up in cache
        if self.cache_enabled and cache_key:
            cache_result = await self._get_from_cache(cache_key)
            if cache_result:
                logger.info(f"Cache hit for request {request_id}")
                # Add cache metadata
                cache_result["cache_hit"] = True
                return cache_result
        
        try:
            # Start resource monitoring for this request
            monitor_task_id = f"request_{request_id}"
            await self.resource_monitor.start_monitoring(monitor_task_id, {
                "request_id": request_id,
                "session_id": session_id,
                "operation": options.get("operation", "process")
            })
            
            # 1. Detect input type
            detection_result = await self.input_detector.detect(
                content,
                {k: v for k, v in options.items() if k != "content"}
            )
            
            input_type = detection_result["input_type"]
            pipeline = detection_result["pipeline"]
            metadata = detection_result["metadata"]
            
            logger.info(f"Detected input type: {input_type}, pipeline: {pipeline}")
            
            # 2. Process through appropriate pipeline
            if pipeline == "text":
                result = await self._process_text(content, options, metadata)
            elif pipeline == "document":
                result = await self._process_document(content, options, metadata)
            elif pipeline == "ocr":
                result = await self._process_ocr(content, options, metadata)
            elif pipeline == "audio":
                result = await self._process_audio(content, options, metadata)
            elif pipeline == "chat":
                result = await self._process_chat(content, options, metadata)
            else:
                raise ValueError(f"Unknown pipeline: {pipeline}")
            
            # 3. Check veracity if enabled
            if options.get("verify_output", self.config.get("verify_output", True)):
                source_text = content if isinstance(content, str) else result.get("original_text", "")
                source_lang = options.get("source_language", result.get("source_language", "en"))
                target_lang = options.get("target_language", result.get("target_language", "en"))
                
                veracity_result = await self.veracity_checker.verify_translation(
                    source_text=source_text,
                    translation=result.get("processed_text", ""),
                    source_lang=source_lang,
                    target_lang=target_lang,
                    metadata={
                        "request_id": options.get("request_id"),
                        "session_id": options.get("session_id"),
                        "pipeline": pipeline,
                        "domain": options.get("domain")
                    }
                )
                result["veracity"] = veracity_result
            
            # 4. Generate text-to-speech if requested
            if options.get("tts_output", False) and "processed_text" in result:
                tts_result = await self._generate_speech(
                    result["processed_text"],
                    options.get("target_language", "en"),
                    options
                )
                if tts_result:
                    result["tts_output"] = tts_result
            
            # 5. Update metrics
            processing_time = time.time() - start_time
            self.successful_requests += 1
            self.metrics.record_processing(
                pipeline, 
                options.get("source_language", "unknown"),
                options.get("target_language", "unknown"),
                processing_time
            )
            
            # 6. Log audit record
            user_id = options.get("user_id", "anonymous")
            source_ip = options.get("source_ip")
            
            # Log model operation for main processing
            await self.audit_logger.log_model_operation(
                model_id=result.get("model_used", "default"),
                operation=pipeline,
                user_id=user_id,
                input_metadata={
                    "type": input_type,
                    "source_language": source_language,
                    "target_language": options.get("target_language"),
                    "content_length": len(content) if isinstance(content, str) else (len(content) if content else 0),
                    "options": {k: v for k, v in options.items() if k not in self.audit_logger.sensitive_fields}
                },
                output_metadata={
                    "success": True,
                    "veracity_score": result.get("veracity", {}).get("score", 1.0) if result.get("veracity") else 1.0,
                    "processing_steps": list(result.keys()),
                },
                duration=processing_time,
                status="success",
                correlation_id=request_id
            )
            
            # Log API request
            await self.audit_logger.log_api_request(
                endpoint=options.get("endpoint", "process"),
                method=options.get("method", "POST"),
                user_id=user_id,
                source_ip=source_ip,
                request_id=request_id,
                request_params={
                    "pipeline": pipeline,
                    "source_language": source_language,
                    "target_language": options.get("target_language"),
                    "operation_type": options.get("operation", pipeline)
                },
                status_code=200,
                response_time=processing_time,
                correlation_id=request_id
            )
            
            # Add timing information
            result["processing_time"] = processing_time
            result["pipeline"] = pipeline
            result["input_type"] = input_type
            
            # 7. Cache result if enabled
            if self.cache_enabled and cache_key:
                await self._add_to_cache(cache_key, result)
            
            # Stop resource monitoring
            monitor_data = await self.resource_monitor.stop_monitoring(f"request_{request_id}")
            if monitor_data:
                # Add resource usage summary to metrics
                resource_analysis = await self.resource_monitor.analyze_task(f"request_{request_id}")
                self.metrics.record_resource_usage(
                    pipeline,
                    {
                        "cpu_avg": resource_analysis.get("cpu", {}).get("avg", 0),
                        "cpu_max": resource_analysis.get("cpu", {}).get("max", 0),
                        "memory_avg": resource_analysis.get("memory", {}).get("avg_percent", 0),
                        "memory_peak": resource_analysis.get("memory", {}).get("peak", 0),
                        "gpu_utilization": resource_analysis.get("gpu", [{}])[0].get("utilization", {}).get("avg", 0) if resource_analysis.get("gpu") else 0,
                        "request_id": request_id
                    }
                )
                
                # Add resource summary to result
                result["resource_usage"] = {
                    "cpu_percent_avg": resource_analysis.get("cpu", {}).get("avg", 0),
                    "memory_percent_avg": resource_analysis.get("memory", {}).get("avg_percent", 0),
                    "duration_seconds": resource_analysis.get("duration", processing_time),
                }
            
            logger.info(f"Processing complete for request {request_id} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing request {request_id}: {str(e)}", exc_info=True)
            
            # Update metrics
            self.failed_requests += 1
            self.metrics.record_error(
                options.get("operation", "unknown"),
                options.get("source_language", "unknown"),
                str(e),
                processing_time
            )
            
            # Log audit record for failure
            user_id = options.get("user_id", "anonymous")
            source_ip = options.get("source_ip")
            
            # Log model operation failure
            await self.audit_logger.log_model_operation(
                model_id=options.get("model_name", "default"),
                operation=options.get("operation", "unknown"),
                user_id=user_id,
                input_metadata={
                    "source_language": options.get("source_language", "unknown"),
                    "target_language": options.get("target_language", "unknown"),
                    "content_length": len(content) if isinstance(content, str) else (len(content) if content else 0),
                    "options": {k: v for k, v in options.items() if k not in self.audit_logger.sensitive_fields}
                },
                output_metadata={
                    "success": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                duration=processing_time,
                status="failure",
                error_message=str(e),
                correlation_id=request_id
            )
            
            # Log API request failure
            await self.audit_logger.log_api_request(
                endpoint=options.get("endpoint", "process"),
                method=options.get("method", "POST"),
                user_id=user_id,
                source_ip=source_ip,
                request_id=request_id,
                request_params={
                    "operation_type": options.get("operation", "unknown")
                },
                status_code=500,
                response_time=processing_time,
                error_message=str(e),
                correlation_id=request_id
            )
            
            # Stop resource monitoring if it was started
            try:
                await self.resource_monitor.stop_monitoring(f"request_{request_id}")
            except Exception:
                pass
                
            # Return error result
            return {
                "status": "error",
                "error": str(e),
                "request_id": request_id,
                "processing_time": processing_time
            }
    
    async def _process_text(self, 
                           text: str, 
                           options: Dict[str, Any],
                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text through the text pipeline with parallel operations.
        
        Args:
            text: Text to process
            options: Processing options
            metadata: Input metadata
            
        Returns:
            Dict with processing results
        """
        logger.debug(f"Processing text of length {len(text)}")
        result = {
            "original_text": text,
            "processed_text": text,
            "metadata": {}
        }
        
        # Set up concurrent tasks for independent operations
        tasks = {}
        source_language = options.get("source_language")
        
        # 1. Task for language detection (if needed)
        if not source_language:
            logger.debug("Starting language detection task")
            detection_request = LanguageDetectionRequest(
                text=text,
                detailed=False
            )
            tasks["language_detection"] = self.translation_pipeline.detect_language(detection_request)
        
        # 2. Task for RAG context retrieval (if needed)
        rag_source_lang = source_language or "en"  # Use default if not provided yet
        if options.get("use_rag", False) and self.rag_expert:
            target_grade_level = options.get("grade_level", 8)
            logger.debug(f"Starting RAG context retrieval task with grade level {target_grade_level}")
            tasks["rag_context"] = self.rag_expert.get_context(
                text,
                rag_source_lang,
                options.get("target_language", "en"),
                {"grade_level": target_grade_level}
            )
        
        # Execute concurrent tasks if any
        task_results = {}
        if tasks:
            # Run all tasks concurrently and handle exceptions
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            task_results = dict(zip(tasks.keys(), results))
            
            # Process language detection result
            if "language_detection" in task_results:
                detection_result = task_results["language_detection"]
                
                # Check for exceptions
                if isinstance(detection_result, Exception):
                    logger.warning(f"Language detection failed: {str(detection_result)}")
                    source_language = "en"  # Fallback to English
                else:
                    source_language = detection_result.get("language", "en")
                    confidence = detection_result.get("confidence", 0.0)
                    
                    result["detected_language"] = source_language
                    result["language_confidence"] = confidence
                    logger.debug(f"Detected language: {source_language} (confidence: {confidence:.2f})")
            else:
                # Store the provided language
                result["source_language"] = source_language
        
        # Process RAG context result
        context = None
        if "rag_context" in task_results:
            rag_result = task_results["rag_context"]
            
            # Check for exceptions
            if isinstance(rag_result, Exception):
                logger.warning(f"RAG context retrieval failed: {str(rag_result)}")
            else:
                context = rag_result
                if context:
                    result["rag_context_used"] = True
                    result["context_items"] = len(context)
                    logger.debug(f"Retrieved {len(context)} RAG context items")
        
        # 3. Apply anonymization if requested
        if options.get("anonymize", False):
            logger.debug("Anonymizing text")
            anonymization_options = {
                "strategy": options.get("anonymization_strategy", "mask"),
                "entities": options.get("anonymization_entities", None),
                "domain": options.get("domain"),
                "consistency": options.get("anonymization_consistency", True)
            }
            
            # Start timer for anonymization
            anonymize_start_time = time.time()
            
            # Execute anonymization
            anonymized_text, entities = await self.anonymization_pipeline.process(
                text,
                source_language,
                anonymization_options
            )
            
            # Calculate anonymization time
            anonymization_time = time.time() - anonymize_start_time
            
            # Update result
            result["processed_text"] = anonymized_text
            result["anonymized_entities"] = {
                "count": len(entities),
                "types": list(set(e["type"] for e in entities))
            }
            result["anonymization_applied"] = True
            result["anonymization_time"] = anonymization_time
            
            # Log anonymization operation
            request_id = options.get("request_id", str(uuid.uuid4()))
            user_id = options.get("user_id", "anonymous")
            
            # Determine PII severity level
            pii_severity = "low"
            entity_types = set(e["type"] for e in entities)
            
            # Check for high-risk PII types
            high_risk_types = {"SSN", "CREDIT_CARD", "PASSPORT_NUMBER", "BANK_ACCOUNT"}
            medium_risk_types = {"EMAIL", "PHONE_NUMBER", "ADDRESS", "PERSON", "DATE_OF_BIRTH"}
            
            if any(etype in high_risk_types for etype in entity_types):
                pii_severity = "high"
            elif any(etype in medium_risk_types for etype in entity_types):
                pii_severity = "medium"
            
            # Audit the anonymization operation
            await self.audit_logger.log_model_operation(
                model_id="anonymizer",
                operation="anonymization",
                user_id=user_id,
                input_metadata={
                    "language": source_language,
                    "strategy": anonymization_options["strategy"],
                    "text_length": len(text),
                    "domain": options.get("domain")
                },
                output_metadata={
                    "success": True,
                    "entities_found": len(entities),
                    "entity_types": list(entity_types),
                    "pii_severity": pii_severity,
                    "anonymization_time": anonymization_time
                },
                duration=anonymization_time,
                status="success",
                correlation_id=request_id
            )
            
            # Log security event for high severity PII
            if pii_severity == "high":
                await self.audit_logger.log_security_event(
                    event_type="high_risk_pii_detection",
                    severity="high",
                    user_id=user_id,
                    source_ip=options.get("source_ip"),
                    details={
                        "entity_types": list(entity_types),
                        "entities_count": len(entities),
                        "anonymization_strategy": anonymization_options["strategy"],
                        "pipeline": "anonymization"
                    },
                    correlation_id=request_id
                )
        
        # 4. Apply translation if needed
        target_language = options.get("target_language")
        if target_language and target_language != source_language:
            logger.debug(f"Translating from {source_language} to {target_language}")

            translation_request = TranslationRequest(
                text=result["processed_text"],
                source_language=source_language,
                target_language=target_language,
                context=context if context else None,
                domain=options.get("domain"),
                preserve_formatting=options.get("preserve_formatting", True),
                model_id=options.get("model_name")
            )

            # Start timer for translation
            translation_start_time = time.time()
            
            # Execute translation
            translation_result = await self.translation_pipeline.translate(translation_request)
            
            # Calculate translation time
            translation_time = time.time() - translation_start_time
            
            # Update result
            result["processed_text"] = translation_result.translated_text
            result["translation_applied"] = True
            result["translation_model"] = translation_result.model_used
            result["target_language"] = target_language
            result["translation_time"] = translation_time
            
            # Log translation operation specifically
            request_id = options.get("request_id", str(uuid.uuid4()))
            user_id = options.get("user_id", "anonymous")
            
            # Audit the translation operation
            await self.audit_logger.log_model_operation(
                model_id=translation_result.model_used,
                operation="translation",
                user_id=user_id,
                input_metadata={
                    "source_language": source_language,
                    "target_language": target_language,
                    "text_length": len(result["processed_text"]),
                    "domain": options.get("domain"),
                    "with_context": context is not None
                },
                output_metadata={
                    "success": True,
                    "translation_time": translation_time,
                },
                duration=translation_time,
                status="success",
                correlation_id=request_id
            )
            
            # Verify translation quality if enabled
            if options.get("verify_output", self.config.get("verify_translation", True)):
                veracity_result = await self.veracity_checker.verify_translation(
                    source_text=text,
                    translation=translation_result.translated_text,
                    source_lang=source_language,
                    target_lang=target_language,
                    metadata={
                        "request_id": request_id,
                        "model_used": translation_result.model_used,
                        "domain": options.get("domain")
                    }
                )
                result["translation_veracity"] = veracity_result
        else:
            # If no translation, target language is the same as source
            result["target_language"] = source_language
        
        # 5. Apply simplification if requested
        if options.get("simplify", False):
            logger.debug("Simplifying text")
            
            # Determine the language for simplification
            simplify_language = target_language if target_language else source_language
            
            # Prepare simplification options
            simplification_options = options.copy()
            if context:
                simplification_options["context"] = context
            
            # Start timer for simplification
            simplify_start_time = time.time()
            
            # Text before simplification for comparison
            text_before_simplification = result["processed_text"]
            
            # Execute simplification
            simplification_result = await self.simplification_pipeline.simplify(
                result["processed_text"],
                simplify_language,
                options.get("simplification_level", 1),
                options.get("grade_level"),
                simplification_options
            )
            
            # Calculate simplification time
            simplification_time = time.time() - simplify_start_time
            
            # Update result
            result["processed_text"] = simplification_result["simplified_text"]
            result["simplification_applied"] = True
            result["simplification_level"] = simplification_result.get("level")
            result["grade_level"] = simplification_result.get("grade_level")
            result["simplification_time"] = simplification_time
            
            # Add readability metrics if available
            if "metrics" in simplification_result:
                result["readability_metrics"] = simplification_result["metrics"]
            
            # Log simplification operation
            request_id = options.get("request_id", str(uuid.uuid4()))
            user_id = options.get("user_id", "anonymous")
            
            # Audit the simplification operation
            await self.audit_logger.log_model_operation(
                model_id=simplification_result.get("model_used", "simplifier"),
                operation="simplification",
                user_id=user_id,
                input_metadata={
                    "language": simplify_language,
                    "simplification_level": options.get("simplification_level", 1),
                    "grade_level": options.get("grade_level"),
                    "text_length": len(text_before_simplification)
                },
                output_metadata={
                    "success": True,
                    "simplification_time": simplification_time,
                    "metrics": simplification_result.get("metrics", {})
                },
                duration=simplification_time,
                status="success",
                correlation_id=request_id
            )
        
        # Add processing metadata
        result["metadata"].update(metadata)
        
        return result
    
    async def _process_document(self, 
                              document_content: bytes, 
                              options: Dict[str, Any],
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document.
        
        Args:
            document_content: Document content as bytes
            options: Processing options
            metadata: Input metadata
            
        Returns:
            Dict with processing results
        """
        # Determine document type from metadata or content
        document_type = metadata.get("detected_mime_type", "application/octet-stream")
        
        logger.debug(f"Processing document of type {document_type}")
        
        # Select appropriate document processor
        if "pdf" in document_type:
            logger.debug("Using PDF processor")
            text, doc_metadata = await self.pdf_processor.extract_text(document_content)
        elif "word" in document_type or "docx" in document_type or "doc" in document_type:
            logger.debug("Using DOCX processor")
            text, doc_metadata = await self.docx_processor.extract_text(document_content)
        else:
            # Fallback to basic text extraction
            logger.debug("Using generic text extraction")
            try:
                text = document_content.decode('utf-8')
                doc_metadata = {}
            except UnicodeDecodeError:
                raise ValueError(f"Unsupported document type: {document_type}")
        
        # Update metadata with document info
        metadata.update(doc_metadata)
        
        # Process the extracted text
        text_result = await self._process_text(text, options, metadata)
        
        # Generate processed document if needed
        if options.get("generate_document", True):
            logger.debug("Generating processed document")
            
            processed_text = text_result["processed_text"]
            
            if "pdf" in document_type:
                output_document = await self.pdf_processor.create_document(
                    processed_text, metadata
                )
            elif "word" in document_type or "docx" in document_type or "doc" in document_type:
                output_document = await self.docx_processor.create_document(
                    processed_text, metadata
                )
            else:
                # Fallback to basic text
                output_document = processed_text.encode('utf-8')
            
            text_result["processed_document"] = output_document
            text_result["document_type"] = document_type
        
        # Add document metadata
        text_result["document_metadata"] = metadata
        
        # Generate filename for processed document if original filename available
        if "filename" in metadata:
            original_filename = metadata["filename"]
            filename_parts = original_filename.rsplit(".", 1)
            
            if len(filename_parts) > 1:
                base_name, extension = filename_parts
                target_language = options.get("target_language", "")
                
                new_filename = f"{base_name}_{target_language}.{extension}"
                text_result["filename"] = new_filename
        
        return text_result
    
    async def _process_ocr(self, 
                         image_content: bytes, 
                         options: Dict[str, Any],
                         metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an image using OCR.
        
        Args:
            image_content: Image content as bytes
            options: Processing options
            metadata: Input metadata
            
        Returns:
            Dict with processing results
        """
        logger.debug("Processing image with OCR")
        
        # Extract text using OCR
        ocr_result = await self.ocr_processor.extract_text(
            image_content,
            options.get("source_language")
        )
        
        extracted_text = ocr_result["text"]
        metadata.update(ocr_result.get("metadata", {}))
        
        # Process the extracted text
        text_result = await self._process_text(extracted_text, options, metadata)
        
        # Add OCR-specific metadata
        text_result["ocr_applied"] = True
        text_result["ocr_confidence"] = ocr_result.get("confidence", 0.0)
        text_result["image_metadata"] = metadata
        
        return text_result
    
    async def _process_audio(self, 
                           audio_content: bytes, 
                           options: Dict[str, Any],
                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process audio content.
        
        Args:
            audio_content: Audio content as bytes
            options: Processing options
            metadata: Input metadata
            
        Returns:
            Dict with processing results
        """
        logger.debug("Processing audio content")
        
        # Prepare input for speech-to-text model
        input_data = {
            "audio_content": audio_content,
            "source_language": options.get("source_language"),
            "parameters": {
                "model_name": options.get("stt_model_name")
            }
        }
        
        try:
            # Run speech-to-text model through model manager
            stt_result = await self.model_manager.run_model(
                "speech_to_text",  # Model type
                "process",         # Method
                input_data         # Input data
            )
            
            # Extract transcript
            if isinstance(stt_result, dict) and "result" in stt_result:
                transcribed_text = stt_result["result"]
                metadata["stt_model"] = stt_result.get("metadata", {}).get("model_used", "speech_to_text")
                
                # Check for detected language in result
                if "metadata" in stt_result and "detected_language" in stt_result["metadata"]:
                    detected_language = stt_result["metadata"]["detected_language"]
                    if not options.get("source_language") and detected_language:
                        options["source_language"] = detected_language
                
                # Process the transcribed text
                if transcribed_text:
                    text_result = await self._process_text(transcribed_text, options, metadata)
                    
                    # Add audio-specific metadata
                    text_result["audio_transcribed"] = True
                    text_result["original_audio_text"] = transcribed_text
                    text_result["detected_language"] = options.get("source_language")
                    text_result["audio_metadata"] = metadata
                    
                    return text_result
                else:
                    return {
                        "status": "error",
                        "error": "No speech detected in audio",
                        "metadata": metadata
                    }
            else:
                return {
                    "status": "error",
                    "error": "Speech-to-text model returned invalid result",
                    "metadata": metadata
                }
                
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": f"Audio processing error: {str(e)}",
                "metadata": metadata
            }
    
    async def _process_chat(self, 
                          message: str, 
                          options: Dict[str, Any],
                          metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a chat message.
        
        Args:
            message: Chat message
            options: Processing options
            metadata: Input metadata
            
        Returns:
            Dict with processing results
        """
        logger.debug("Processing chat message")
        
        # Mark as chat type
        chat_options = options.copy()
        chat_options["is_chat"] = True
        
        # Process message text
        text_result = await self._process_text(message, chat_options, metadata)
        text_result["is_chat"] = True
        
        # Add any conversation context
        if "conversation_id" in options:
            text_result["conversation_id"] = options["conversation_id"]
        
        if "message_id" in options:
            text_result["message_id"] = options["message_id"]
        
        return text_result
    
    async def _generate_speech(self, 
                             text: str, 
                             language: str,
                             options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            language: Language code
            options: Additional options
            
        Returns:
            Dict with speech generation results
        """
        options = options or {}
        logger.debug(f"Generating speech for text in language: {language}")
        
        if not self.tts_pipeline:
            logger.warning("TTS pipeline not available")
            return None
        
        try:
            # Prepare parameters for TTS
            voice = options.get("voice")
            speed = options.get("speed", 1.0)
            pitch = options.get("pitch", 1.0)
            output_format = options.get("audio_format", "mp3")
            
            # Generate unique filename
            request_id = options.get("request_id", str(uuid.uuid4()))
            output_path = str(self.temp_dir / "audio" / f"tts_{request_id}.{output_format}")
            
            # Generate speech using TTS pipeline
            tts_result = await self.tts_pipeline.synthesize(
                text=text,
                language=language,
                voice=voice,
                speed=speed,
                pitch=pitch,
                output_format=output_format,
                output_path=output_path
            )
            
            # If successful, create result
            if "error" not in tts_result:
                # Get audio file path
                audio_file = tts_result.get("audio_file")
                
                # Create API URL for audio
                filename = os.path.basename(audio_file)
                audio_url = f"/api/audio/{filename}"
                
                return {
                    "audio_file": audio_file,
                    "audio_url": audio_url,
                    "format": output_format,
                    "duration": tts_result.get("duration", 0),
                    "model_used": tts_result.get("model_used", "tts")
                }
            else:
                logger.error(f"TTS error: {tts_result.get('error')}")
                return None
            
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}", exc_info=True)
            return None
    
    def _generate_cache_key(self, content: Union[str, bytes], options: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for a request.
        
        Args:
            content: Request content
            options: Request options
            
        Returns:
            Unique cache key
        """
        import hashlib
        
        # If content is bytes, use its hash
        if isinstance(content, bytes):
            content_hash = hashlib.md5(content).hexdigest()
        else:
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # Extract cache-relevant options
        cache_relevant = {
            "source_language": options.get("source_language"),
            "target_language": options.get("target_language"),
            "simplify": options.get("simplify", False),
            "simplification_level": options.get("simplification_level"),
            "grade_level": options.get("grade_level"),
            "anonymize": options.get("anonymize", False),
            "use_rag": options.get("use_rag", False)
        }
        
        # Create hash of options
        options_hash = hashlib.md5(str(cache_relevant).encode('utf-8')).hexdigest()
        
        # Combine for final key
        return f"{content_hash}_{options_hash}"
    
    async def _add_to_cache(self, key: str, result: Dict[str, Any]) -> None:
        """
        Add a result to the cache with thread safety.
        
        Args:
            key: Cache key
            result: Result to cache
        """
        if not self.cache_enabled:
            return
        
        # Make a deep copy to prevent modification after caching
        import copy
        result_copy = copy.deepcopy(result)
        
        # Use lock to ensure thread safety
        async with self.cache_lock:
            # Ensure cache doesn't grow too large
            if len(self.cache) >= self.cache_size:
                await self._evict_cache_entry()
            
            # Add to cache
            self.cache[key] = result_copy
            self.cache_timestamps[key] = time.time()
            logger.debug(f"Added item to cache with key {key[:8]}...")
    
    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a result from the cache with thread safety.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not in cache or expired
        """
        if not self.cache_enabled:
            return None
        
        # Use lock to ensure thread safety
        async with self.cache_lock:
            if key not in self.cache:
                return None
            
            # Check if entry is expired
            timestamp = self.cache_timestamps.get(key, 0)
            if time.time() - timestamp > self.cache_ttl:
                # Remove expired entry
                del self.cache[key]
                del self.cache_timestamps[key]
                logger.debug(f"Removed expired cache entry with key {key[:8]}...")
                return None
            
            # Return cached entry (make a copy to avoid modification)
            logger.debug(f"Cache hit for key {key[:8]}...")
            return copy.deepcopy(self.cache[key])
    
    async def _evict_cache_entry(self) -> None:
        """Evict the oldest cache entry with thread safety."""
        if not self.cache:
            return
        
        # Find oldest entry (lock already acquired in _add_to_cache)
        oldest_key = min(self.cache_timestamps.items(), key=lambda x: x[1])[0]
        
        # Remove entry
        del self.cache[oldest_key]
        del self.cache_timestamps[oldest_key]
        logger.debug(f"Evicted oldest cache entry with key {oldest_key[:8]}...")
    
    async def _cache_cleanup_task(self) -> None:
        """Background task for cache cleanup with thread safety."""
        if not self.cache_enabled:
            return
        
        while True:
            try:
                # Sleep for half the TTL time
                await asyncio.sleep(self.cache_ttl / 2)
                
                # Use lock to ensure thread safety
                async with self.cache_lock:
                    # Find expired entries
                    current_time = time.time()
                    expired_keys = [
                        key for key, timestamp in self.cache_timestamps.items()
                        if current_time - timestamp > self.cache_ttl
                    ]
                    
                    # Remove expired entries
                    for key in expired_keys:
                        if key in self.cache:
                            del self.cache[key]
                        if key in self.cache_timestamps:
                            del self.cache_timestamps[key]
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
            except asyncio.CancelledError:
                # Task was cancelled
                logger.info("Cache cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup task: {str(e)}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retrying
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the processor.
        
        Returns:
            Dict with processor statistics
        """
        return {
            "initialized": self.initialized,
            "startup_time": self.startup_time,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.cache) if self.cache_enabled else 0,
            "cache_max_size": self.cache_size,
            "supported_pipelines": ["text", "document", "ocr", "audio", "chat"],
            "rag_enabled": self.rag_expert is not None,
            "supported_languages": self.translation_pipeline.get_supported_languages() if self.translation_pipeline else [],
            "components": {
                "translation": self.translation_pipeline is not None,
                "simplification": self.simplification_pipeline is not None,
                "anonymization": self.anonymization_pipeline is not None,
                "pdf": self.pdf_processor is not None,
                "docx": self.docx_processor is not None,
                "ocr": self.ocr_processor is not None,
                "rag": self.rag_expert is not None,
                "tts": self.tts_pipeline is not None
            }
        }
    
    async def cleanup(self) -> None:
        """
        Clean up processor resources.
        
        This method should be called before shutdown.
        """
        logger.info("Cleaning up processor resources")
        
        # Clean up cache
        self.cache.clear()
        self.cache_timestamps.clear()
        
        # Clean up temporary files
        if self.temp_dir.exists():
            import shutil
            try:
                # Clean up audio files
                audio_dir = self.temp_dir / "audio"
                if audio_dir.exists():
                    shutil.rmtree(audio_dir)
                    logger.debug("Removed temporary audio files")
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {str(e)}", exc_info=True)
        
        # Perform component-specific cleanup
        if self.rag_expert:
            await self.rag_expert.cleanup()
            
        if self.tts_pipeline:
            await self.tts_pipeline.cleanup()
        
        logger.info("Processor cleanup complete")
    
    async def detect_language(
        self,
        text: str,
        detailed: Optional[bool] = False,
        model_id: Optional[str] = None,
    ) -> dict:
        """
        Detect the language of the given text using the translation pipeline.
        
        Args:
            text: The input text to detect language for.
            detailed: Whether to return detailed detection results.
            model_id: Optional model ID to use for detection.
        
        Returns:
            A dictionary with language detection results.
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.translation_pipeline:
            raise AttributeError("Translation pipeline not initialized")
        
        # Create detection request
        request = LanguageDetectionRequest(
            text=text, 
            detailed=detailed, 
            model_id=model_id
        )
        
        # Perform language detection
        return await self.translation_pipeline.detect_language(request)

    async def process_summarization(
        self,
        text: str,
        language: str = "en",
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle summarization using the multipurpose pipeline component.
        
        Args:
            text: Text to summarize
            language: Language code
            user_id: Optional user ID for tracking
            request_id: Optional request ID for tracking
            
        Returns:
            Dict with summarization results
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.multipurpose_pipeline:
            raise RuntimeError("Multipurpose (summarization) pipeline not initialized")

        result = await self.multipurpose_pipeline.summarize(
            text=text,
            language=language,
            user_id=user_id,
            request_id=request_id
        )

        if not result:
            logger.warning(f"Summarization output is empty for request {request_id}")
            return {
                "input_text": text,
                "summary": "",
                "language": language,
                "model_id": "default"
            }
            
        logger.info(f"Summarization successful for request {request_id}")
        
        # Extract summary from result
        if isinstance(result, dict):
            summary = result.get("summary", "")
            model_id = result.get("model_used", "default")
        else:
            summary = str(result)
            model_id = "default"

        return {
            "input_text": text,
            "summary": summary,
            "language": language,
            "model_id": model_id
        }

    async def process_translation(
        self,
        text: str,
        source_language: str,
        target_language: str,
        model_id: Optional[str] = None,
        glossary_id: Optional[str] = None,
        preserve_formatting: bool = True,
        formality: Optional[str] = None,
        verify: bool = False,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        use_mbart: bool = True  # Default to using MBART
    ) -> Dict[str, Any]:
        """
        Handle translation using the translation pipeline component.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            model_id: Optional specific model to use
            glossary_id: Optional glossary ID
            preserve_formatting: Whether to preserve formatting
            formality: Optional formality level
            verify: Whether to verify translation
            user_id: Optional user ID for tracking
            request_id: Optional request ID for tracking
            use_mbart: Whether to use MBART as the primary model
            
        Returns:
            Dict with translation results
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.translation_pipeline:
            raise RuntimeError("Translation pipeline not initialized")

        # If use_mbart is True and no specific model is requested, use MBART
        if use_mbart and model_id is None:
            model_id = "mbart_translation"
            logger.info(f"Using MBART as primary translation model for request {request_id}")

        result = await self.translation_pipeline.translate_text(
            text=text,
            source_language=source_language,
            target_language=target_language,
            model_id=model_id,
            glossary_id=glossary_id,
            preserve_formatting=preserve_formatting,
            formality=formality,
            verify=verify,
            user_id=user_id,
            request_id=request_id
        )

        translated = result.get("translated_text", "")
        if not translated:
            logger.warning(f"Translation output is empty for request {request_id}")
        else:
            logger.info(f"Translation successful for request {request_id}")

        # Create a result dictionary with all required fields
        result_dict = {
            "source_text": text,
            "translated_text": translated,
            "source_language": source_language or "auto",
            "target_language": target_language,
            "confidence": result.get("confidence", 1.0),
            "model_id": result.get("model_used", "default"),
            "model_used": result.get("model_used", "translation"),
            "word_count": len(text.split()),
            "character_count": len(text),
            "process_time": 0.0,
            "verified": False,
            "verification_score": None,
            "detected_language": None
        }
        
        # Add info about primary model
        if result.get("primary_model"):
            result_dict["primary_model"] = result.get("primary_model")
            
        # Add info about fallback model if used
        if result.get("used_fallback"):
            result_dict["used_fallback"] = True
            result_dict["fallback_model"] = result.get("fallback_model")
        
        logger.debug(f"Translation result: {result_dict}")
        
        return result_dict
        
    async def process_batch_translation(
        self,
        texts: List[str],
        source_language: str,
        target_language: str,
        model_id: Optional[str] = None,
        glossary_id: Optional[str] = None,
        preserve_formatting: bool = True,
        formality: Optional[str] = None,
        verify: bool = False,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        use_mbart: bool = True  # Default to using MBART
    ) -> List[Dict[str, Any]]:
        """
        Handle batch translation of multiple texts concurrently.
        
        Args:
            texts: List of texts to translate
            source_language: Source language code
            target_language: Target language code
            model_id: Optional specific model to use
            glossary_id: Optional glossary ID
            preserve_formatting: Whether to preserve formatting
            formality: Optional formality level
            verify: Whether to verify translation
            user_id: Optional user ID for tracking
            request_id: Optional request ID for tracking
            use_mbart: Whether to use MBART as the primary model
            
        Returns:
            List of dicts with translation results
        """
        if not texts:
            return []
            
        if not self.initialized:
            await self.initialize()
            
        if not self.translation_pipeline:
            raise RuntimeError("Translation pipeline not initialized")
            
        logger.info(f"Processing batch translation of {len(texts)} texts")
        
        # If use_mbart is True and no specific model is requested, use MBART
        if use_mbart and model_id is None:
            model_id = "mbart_translation"
            logger.info(f"Using MBART as primary translation model for batch request {request_id}")
        
        # Check if translation pipeline supports native batch processing
        if hasattr(self.translation_pipeline, "translate_batch") and callable(
                getattr(self.translation_pipeline, "translate_batch")):
            logger.debug("Using native batch translation")
            batch_results = await self.translation_pipeline.translate_batch(
                texts=texts,
                source_language=source_language,
                target_language=target_language,
                model_id=model_id,
                glossary_id=glossary_id,
                preserve_formatting=preserve_formatting,
                formality=formality,
                verify=verify,
                user_id=user_id,
                request_id=request_id
            )
            return batch_results
        
        # If native batch processing is not available, use parallel processing
        logger.debug("Using concurrent processing for batch translation")
        
        # Create translation tasks for all texts
        tasks = []
        for i, text in enumerate(texts):
            # Generate a unique request ID for each translation
            text_request_id = f"{request_id}_{i}" if request_id else str(uuid.uuid4())
            
            # Create a task for this translation
            tasks.append(self.process_translation(
                text=text,
                source_language=source_language,
                target_language=target_language,
                model_id=model_id,
                glossary_id=glossary_id,
                preserve_formatting=preserve_formatting,
                formality=formality,
                verify=verify,
                user_id=user_id,
                request_id=text_request_id,
                use_mbart=use_mbart
            ))
        
        # Execute all translation tasks concurrently
        batch_results = await asyncio.gather(*tasks)
        
        logger.info(f"Completed batch translation of {len(texts)} texts")
        return batch_results