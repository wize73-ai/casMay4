"""
Anonymization Pipeline for CasaLingua

This module handles identification and anonymization of personal identifiable
information (PII) in text content with multiple anonymization strategies and
support for domain-specific patterns.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import re
import uuid
import hashlib
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Pattern

from app.utils.logging import get_logger

logger = get_logger("casalingua.core.anonymizer")

class EntityType:
    """Entity types for PII detection."""
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN" 
    ID_NUMBER = "ID_NUMBER"
    CREDIT_CARD = "CREDIT_CARD"
    ADDRESS = "ADDRESS"
    DATE = "DATE"
    URL = "URL"
    IP_ADDRESS = "IP_ADDRESS"
    MEDICAL = "MEDICAL"
    FINANCIAL = "FINANCIAL"
    LEGAL = "LEGAL"
    DEFAULT = "DEFAULT"
    
    # Sensitive entity types that should always be anonymized
    SENSITIVE_TYPES = {SSN, CREDIT_CARD, ID_NUMBER}
    
    # Standard entity types recognized by most NER models
    STANDARD_TYPES = {PERSON, LOCATION, ORGANIZATION}
    
    # All supported entity types
    ALL_TYPES = {PERSON, LOCATION, ORGANIZATION, EMAIL, PHONE, SSN, ID_NUMBER, 
                CREDIT_CARD, ADDRESS, DATE, URL, IP_ADDRESS, MEDICAL, FINANCIAL,
                LEGAL, DEFAULT}

class AnonymizationStrategy:
    """Anonymization strategies."""
    MASK = "mask"         # Replace with [ENTITY_TYPE]
    REPLACE = "replace"   # Replace with similar but fake data
    REDACT = "redact"     # Replace with ████████
    REMOVE = "remove"     # Remove completely
    HASH = "hash"         # Replace with hash
    PARTIAL = "partial"   # Show partial information
    PSEUDONYMIZE = "pseudonymize"  # Similar to REPLACE but with consistent data
    
    # All supported strategies
    ALL_STRATEGIES = {MASK, REPLACE, REDACT, REMOVE, HASH, PARTIAL, PSEUDONYMIZE}

class AnonymizationPipeline:
    """
    Anonymization pipeline for PII detection and masking.
    
    Features:
    - Named entity recognition
    - PII detection using models and patterns
    - Multiple anonymization strategies
    - Domain-specific entity detection
    - Language-specific patterns
    - Preservation of text structure and formatting
    - Consistent entity replacement
    """
    
    def __init__(
        self, 
        model_manager, 
        config: Dict[str, Any] = None, 
        registry_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the anonymization pipeline.
        
        Args:
            model_manager: Model manager for accessing anonymization models
            config: Configuration dictionary
            registry_config: Registry configuration for models
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.registry_config = registry_config or {}
        self.initialized = False
        
        # Model type for anonymization
        self.model_type = "anonymizer"
        self.ner_model_type = "ner_detection"
        
        # Set default anonymization strategy
        self.default_strategy = self.config.get(
            "default_anonymization_strategy", 
            AnonymizationStrategy.MASK
        )
        
        # Entity replacement consistency map
        self.entity_replacements = {}
        
        # Initialize pattern libraries
        self._init_patterns()
        
        # Domain-specific patterns
        self.domain_patterns = {}
        
        # Language-specific patterns will be loaded on demand
        self.language_patterns = {}
        
        logger.info("Anonymization pipeline created (not yet initialized)")
    
    async def initialize(self) -> None:
        """
        Initialize the anonymization pipeline.
        
        This loads necessary models and prepares the pipeline.
        """
        if self.initialized:
            logger.warning("Anonymization pipeline already initialized")
            return
        
        logger.info("Initializing anonymization pipeline")
        
        # Load anonymization model
        try:
            logger.info(f"Loading anonymization model ({self.model_type})")
            await self.model_manager.load_model(self.model_type)
            logger.info("Anonymization model loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading anonymization model: {str(e)}")
            logger.warning("Will use pattern-based anonymization")
        
        # Load NER model
        try:
            logger.info(f"Loading NER model ({self.ner_model_type})")
            await self.model_manager.load_model(self.ner_model_type)
            logger.info("NER model loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading NER model: {str(e)}")
            logger.warning("Entity recognition capabilities will be limited")
        
        # Load domain-specific patterns if configured
        domain_patterns_path = self.config.get("domain_patterns_path")
        if domain_patterns_path:
            self._load_domain_patterns(domain_patterns_path)
        
        # Pre-load language-specific patterns for common languages
        for lang in ["en", "es", "fr", "de"]:
            self._load_language_patterns(lang)
        
        self.initialized = True
        logger.info("Anonymization pipeline initialization complete")
    
    async def process(self, 
                    text: str, 
                    language: str,
                    options: Dict[str, Any] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Detect and anonymize PII in text.
        
        Args:
            text: Text to process
            language: Language code
            options: Additional options
                - strategy: Anonymization strategy
                - entities: List of entity types to anonymize
                - preserve_formatting: Whether to preserve formatting
                - model_name: Specific model to use
                - domain: Specific domain for additional patterns
                - consistency: Whether to maintain consistent replacements
                
        Returns:
            Tuple of (anonymized_text, detected_entities)
        """
        if not self.initialized:
            await self.initialize()
        
        # Handle empty text
        if not text or len(text.strip()) == 0:
            return text, []

        options = options or {}
        start_time = time.time()
        
        # Get anonymization strategy
        strategy = options.get("strategy", self.default_strategy)
        if strategy not in AnonymizationStrategy.ALL_STRATEGIES:
            logger.warning(f"Unknown anonymization strategy: {strategy}, using default")
            strategy = self.default_strategy
        
        # Get entities to anonymize
        target_entities = options.get("entities", list(EntityType.ALL_TYPES))
        domain = options.get("domain")
        consistency = options.get("consistency", True)
        
        logger.debug(f"Anonymizing text with strategy: {strategy}, language: {language}")
        
        try:
            # Reset entity replacements if not maintaining consistency
            if not consistency:
                self.entity_replacements = {}
            
            # Step 1: Detect entities
            detection_start = time.time()
            entities = await self._detect_entities(text, language, domain, options)
            detection_time = time.time() - detection_start
            logger.debug(f"Entity detection took {detection_time:.3f}s")
            
            # Step 2: Filter entities by type
            if target_entities and "ALL" not in target_entities:
                entities = [e for e in entities if e["type"] in target_entities]
            
            # Always include sensitive entities
            sensitive_entities = [e for e in entities if e["type"] in EntityType.SENSITIVE_TYPES]
            other_entities = [e for e in entities if e["type"] not in EntityType.SENSITIVE_TYPES]
            
            # Sort other entities by confidence (descending)
            other_entities = sorted(other_entities, key=lambda e: e.get("confidence", 0.0), reverse=True)
            
            # Combine, with sensitive entities first
            entities = sensitive_entities + other_entities
            
            # Skip anonymization if no entities found
            if not entities:
                logger.debug("No entities to anonymize")
                return text, []
            
            logger.debug(f"Detected {len(entities)} entities to anonymize")
            
            # Step 3: Resolve overlapping entities (keep only highest confidence)
            entities = self._resolve_overlapping_entities(entities)
            
            # Step 4: Anonymize text
            anonymization_start = time.time()
            anonymized_text, processed_entities = self._anonymize_text(
                text, entities, strategy, consistency
            )
            anonymization_time = time.time() - anonymization_start
            logger.debug(f"Anonymization took {anonymization_time:.3f}s")
            
            # Log performance metrics
            total_time = time.time() - start_time
            logger.info(f"Text anonymized in {total_time:.3f}s "
                       f"(detection: {detection_time:.3f}s, anonymization: {anonymization_time:.3f}s)")
            
            return anonymized_text, processed_entities
            
        except Exception as e:
            logger.error(f"Anonymization error: {str(e)}", exc_info=True)
            return text, []
    
    async def _detect_entities(self, 
                            text: str, 
                            language: str,
                            domain: Optional[str],
                            options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect entities in text using multiple methods.
        
        Args:
            text: Text to analyze
            language: Language code
            domain: Optional domain for specific patterns
            options: Additional options
            
        Returns:
            List of detected entities
        """
        # Step 1: Use NER model for entity detection
        model_entities = []
        model_id = options.get("model_name", self.ner_model_type)
        
        try:
            # Prepare input for NER model
            input_data = {
                "text": text,
                "source_language": language,
                "parameters": {
                    "domain": domain
                }
            }
            
            # Run NER model
            result = await self.model_manager.run_model(
                model_id,
                "process",
                input_data
            )
            
            # Extract entities from result
            if isinstance(result, dict) and "result" in result:
                # Handle different result formats
                if isinstance(result["result"], list):
                    model_entities = result["result"]
                elif isinstance(result["result"], dict) and "entities" in result["result"]:
                    model_entities = result["result"]["entities"]
                
                logger.debug(f"Detected {len(model_entities)} entities using NER model")
            
        except Exception as e:
            logger.warning(f"Model-based entity detection failed: {str(e)}")
            logger.debug("Falling back to pattern-based detection")
        
        # Step 2: Use regex patterns
        regex_entities = self._detect_entities_with_regex(text, language)
        logger.debug(f"Detected {len(regex_entities)} entities using regex patterns")
        
        # Step 3: Use domain-specific patterns if applicable
        domain_entities = []
        if domain and domain in self.domain_patterns:
            domain_entities = self._detect_domain_entities(text, domain)
            logger.debug(f"Detected {len(domain_entities)} domain-specific entities")
        
        # Step 4: Combine all entities and handle overlaps
        combined_entities = model_entities.copy()
        
        for entity in regex_entities + domain_entities:
            # Check for overlap with model entities
            if not self._has_overlap(entity, model_entities):
                combined_entities.append(entity)
        
        logger.debug(f"Combined {len(combined_entities)} unique entities")
        return combined_entities
    
    def _detect_entities_with_regex(self, 
                                  text: str, 
                                  language: str) -> List[Dict[str, Any]]:
        """
        Detect entities using regex patterns.
        
        Args:
            text: Text to analyze
            language: Language code
            
        Returns:
            List of detected entities
        """
        entities = []
        
        # Get language-specific patterns
        patterns = self._get_language_patterns(language)
        
        # Apply each pattern
        for entity_type, pattern in patterns.items():
            if isinstance(pattern, str):
                pattern = re.compile(pattern, re.IGNORECASE)
            
            for match in pattern.finditer(text):
                start, end = match.span()
                
                # Extract matched text and groups
                matched_text = match.group()
                
                # Create entity
                entity = {
                    "type": entity_type,
                    "text": matched_text,
                    "start": start,
                    "end": end,
                    "confidence": 0.9,  # High confidence for regex matches
                    "method": "regex"
                }
                
                # Add additional match groups as metadata if available
                if match.groupdict():
                    entity["metadata"] = match.groupdict()
                
                entities.append(entity)
        
        return entities
    
    def _detect_domain_entities(self, 
                              text: str, 
                              domain: str) -> List[Dict[str, Any]]:
        """
        Detect domain-specific entities.
        
        Args:
            text: Text to analyze
            domain: Domain (e.g., 'legal', 'medical', 'financial')
            
        Returns:
            List of detected entities
        """
        entities = []
        
        if domain not in self.domain_patterns:
            return entities
        
        # Apply domain-specific patterns
        for entity_type, pattern in self.domain_patterns[domain].items():
            if isinstance(pattern, str):
                pattern = re.compile(pattern, re.IGNORECASE)
            
            for match in pattern.finditer(text):
                start, end = match.span()
                entities.append({
                    "type": entity_type,
                    "text": match.group(),
                    "start": start,
                    "end": end,
                    "confidence": 0.85,  # Slightly lower confidence than standard patterns
                    "method": f"domain_{domain}"
                })
        
        return entities
    
    def _resolve_overlapping_entities(self, 
                                    entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve overlapping entities by keeping only the highest confidence one.
        
        Args:
            entities: List of detected entities
            
        Returns:
            List of non-overlapping entities
        """
        # Sort by confidence (descending)
        sorted_entities = sorted(entities, key=lambda e: e.get("confidence", 0.0), reverse=True)
        
        # Track used positions
        used_positions = set()
        resolved_entities = []
        
        for entity in sorted_entities:
            start, end = entity["start"], entity["end"]
            
            # Check if this entity overlaps with any higher confidence entity
            if not any(p in used_positions for p in range(start, end)):
                resolved_entities.append(entity)
                # Mark these positions as used
                used_positions.update(range(start, end))
        
        return resolved_entities
    
    def _anonymize_text(self,
                       text: str,
                       entities: List[Dict[str, Any]],
                       strategy: str,
                       consistency: bool) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Anonymize entities in text.
        
        Args:
            text: Original text
            entities: Detected entities
            strategy: Anonymization strategy
            consistency: Whether to maintain consistent replacements
            
        Returns:
            Tuple of (anonymized_text, processed_entities)
        """
        # Sort entities by position (from end to start to avoid offset issues)
        sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)
        
        # Create a copy of the text for modification
        anonymized_text = text
        processed_entities = []
        
        # Apply anonymization
        for entity in sorted_entities:
            entity_text = text[entity["start"]:entity["end"]]
            entity_type = entity["type"]
            
            # Get anonymized value using appropriate strategy
            anonymized_value = self._anonymize_entity(
                entity_text, 
                entity_type, 
                strategy,
                consistency
            )
            
            # Replace in text
            anonymized_text = (
                anonymized_text[:entity["start"]] + 
                anonymized_value + 
                anonymized_text[entity["end"]:]
            )
            
            # Update entity with anonymized value
            entity_info = {
                "type": entity_type,
                "original_text": entity_text,
                "anonymized_text": anonymized_value,
                "start": entity["start"],
                "end": entity["end"],
                "strategy": strategy,
                "confidence": entity.get("confidence", 1.0)
            }
            
            # Add metadata if present
            if "metadata" in entity:
                entity_info["metadata"] = entity["metadata"]
            
            processed_entities.append(entity_info)
        
        return anonymized_text, processed_entities
    
    def _anonymize_entity(self,
                         text: str,
                         entity_type: str,
                         strategy: str,
                         consistency: bool) -> str:
        """
        Anonymize a single entity using the specified strategy.
        
        Args:
            text: Entity text
            entity_type: Entity type
            strategy: Anonymization strategy
            consistency: Whether to maintain consistent replacements
            
        Returns:
            Anonymized text
        """
        # Check if we already have a replacement for this entity (for consistency)
        if consistency and text in self.entity_replacements:
            return self.entity_replacements[text]
        
        # Apply appropriate anonymization strategy
        if strategy == AnonymizationStrategy.MASK:
            anonymized = f"[{entity_type}]"
        
        elif strategy == AnonymizationStrategy.REPLACE:
            anonymized = self._generate_replacement(text, entity_type)
        
        elif strategy == AnonymizationStrategy.REDACT:
            # Replace with block characters
            anonymized = "█" * min(len(text), 10)
        
        elif strategy == AnonymizationStrategy.REMOVE:
            anonymized = ""
        
        elif strategy == AnonymizationStrategy.HASH:
            # Generate a short hash
            hash_obj = hashlib.sha256(text.encode())
            anonymized = hash_obj.hexdigest()[:10]
        
        elif strategy == AnonymizationStrategy.PARTIAL:
            anonymized = self._generate_partial(text, entity_type)
        
        else:
            # Default to masking
            anonymized = f"[{entity_type}]"
        
        # Store for consistency
        if consistency:
            self.entity_replacements[text] = anonymized
        
        return anonymized
    
    def _generate_replacement(self, text: str, entity_type: str) -> str:
        """
        Generate a realistic but fake replacement for an entity.
        
        Args:
            text: Original entity text
            entity_type: Entity type
            
        Returns:
            Fake replacement
        """
        if entity_type == EntityType.PERSON:
            # Generate fake person name
            gender_hint = self._guess_gender(text)
            if gender_hint == "female":
                return self._get_random_name(gender="female")
            else:
                return self._get_random_name(gender="male")

        elif entity_type == EntityType.LOCATION:
            # Generate fake location
            return self._get_random_location()

        elif entity_type == EntityType.ORGANIZATION:
            # Generate fake organization
            return self._get_random_organization()

        elif entity_type == EntityType.EMAIL:
            # Generate fake email
            name_part = f"user-{uuid.uuid4().hex[:6]}"
            domain_part = "example.com"

            # Try to preserve domain if present
            if "@" in text:
                parts = text.split('@')
                if len(parts) == 2 and "." in parts[1]:
                    domain_part = parts[1]

            return f"{name_part}@{domain_part}"

        elif entity_type == EntityType.PHONE:
            # Generate fake phone number
            return f"555-{uuid.uuid4().hex[:3]}-{uuid.uuid4().hex[:4]}"

        elif entity_type == EntityType.SSN:
            # Generate fake SSN
            return "XXX-XX-XXXX"

        elif entity_type == EntityType.ID_NUMBER:
            # Generate fake ID
            return f"ID-{uuid.uuid4().hex[:8]}"

        elif entity_type == EntityType.CREDIT_CARD:
            # Generate fake credit card number
            return "XXXX-XXXX-XXXX-XXXX"

        elif entity_type == EntityType.ADDRESS:
            # Generate fake address
            return f"123 {self._get_random_street()} St., Anytown, XX 12345"

        elif entity_type == EntityType.DATE:
            # Generate fake date
            import datetime
            today = datetime.date.today()
            delta = datetime.timedelta(days=uuid.uuid4().int % 365)
            fake_date = today + delta
            return fake_date.strftime("%Y-%m-%d")

        elif entity_type == EntityType.URL:
            # Generate fake URL
            return f"https://example.com/{uuid.uuid4().hex[:8]}"

        elif entity_type == EntityType.IP_ADDRESS:
            # Generate fake IP
            return f"192.0.2.{uuid.uuid4().int % 256}"

        # Domain-specific entities
        elif entity_type == EntityType.MEDICAL:
            fake_terms = ["Condition X", "Procedure Y", "Medication Z"]
            return fake_terms[uuid.uuid4().int % len(fake_terms)]
        elif entity_type == EntityType.FINANCIAL:
            fake_terms = ["Account 1234", "Transaction 5678", "Invoice #A1B2"]
            return fake_terms[uuid.uuid4().int % len(fake_terms)]
        elif entity_type == EntityType.LEGAL:
            fake_terms = ["Case #XYZ", "Legal Notice", "Statute 42"]
            return fake_terms[uuid.uuid4().int % len(fake_terms)]
        else:
            # Default replacement
            return f"[{entity_type}-{uuid.uuid4().hex[:6]}]"
    
    def _generate_partial(self, text: str, entity_type: str) -> str:
        """
        Generate a partial anonymization of an entity.
        
        Args:
            text: Original entity text
            entity_type: Entity type
            
        Returns:
            Partially anonymized text
        """
        if entity_type == EntityType.PERSON:
            # Keep first initial
            parts = text.split()
            if len(parts) > 1:
                return f"{parts[0][0]}. {parts[-1][0]}."
            return f"{text[0]}."
            
        elif entity_type == EntityType.ORGANIZATION:
            # Keep abbreviation or first word
            parts = text.split()
            if all(len(part) == 1 for part in parts):
                return text  # Already an abbreviation
            return parts[0] + "..."
            
        elif entity_type == EntityType.EMAIL:
            # Keep domain only
            if "@" in text:
                return f"****@{text.split('@')[1]}"
            return "****@****"
            
        elif entity_type == EntityType.PHONE:
            # Keep last 4 digits
            digits = re.findall(r'\d', text)
            if len(digits) >= 4:
                return f"XXX-XXX-{''.join(digits[-4:])}"
            return "XXX-XXX-XXXX"
            
        elif entity_type == EntityType.CREDIT_CARD:
            # Keep last 4 digits
            digits = re.findall(r'\d', text)
            if len(digits) >= 4:
                return f"XXXX-XXXX-XXXX-{''.join(digits[-4:])}"
            return "XXXX-XXXX-XXXX-XXXX"
            
        elif entity_type == EntityType.SSN:
            # Always completely mask SSN
            return "XXX-XX-XXXX"
            
        elif entity_type == EntityType.DATE:
            # Keep just the year
            year_match = re.search(r'(19|20)\d{2}', text)
            if year_match:
                return year_match.group()
            return "YYYY-MM-DD"
            
        else:
            # Default partial anonymization
            if len(text) <= 1:
                return text
            visible_chars = max(1, min(3, len(text) // 3))
            return text[:visible_chars] + "*" * (len(text) - visible_chars)
    
    def _init_patterns(self) -> None:
        """Initialize regex pattern libraries."""
        # Common patterns for all languages
        self.common_patterns = {
            # Email addresses
            EntityType.EMAIL: re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # Credit card numbers (major card types)
            EntityType.CREDIT_CARD: re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
            
            # URLs
            EntityType.URL: re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'),
            
            # IP addresses
            EntityType.IP_ADDRESS: re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        }
    
    def _load_domain_patterns(self, pattern_path: str) -> None:
        """
        Load domain-specific patterns from JSON file.
        
        Args:
            pattern_path: Path to pattern file
        """
        try:
            import json
            
            with open(pattern_path, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
            
            # Compile patterns
            for domain, domain_patterns in patterns.items():
                self.domain_patterns[domain] = {}
                for entity_type, pattern in domain_patterns.items():
                    self.domain_patterns[domain][entity_type] = re.compile(pattern, re.IGNORECASE)
            
            logger.info(f"Loaded domain patterns for {len(self.domain_patterns)} domains")
            
        except Exception as e:
            logger.error(f"Error loading domain patterns: {str(e)}", exc_info=True)
            self.domain_patterns = {}
    
    def _load_language_patterns(self, language: str) -> Dict[str, Pattern]:
        """
        Load language-specific patterns.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary of patterns
        """
        if language in self.language_patterns:
            return self.language_patterns[language]
        
        patterns = {}
        
        # Add common patterns that work across languages
        patterns.update(self.common_patterns)
        
        # Get additional patterns from the comprehensive pattern collection
        additional_patterns = self._get_patterns_for_language(language)
        for entity_type, pattern in additional_patterns.items():
            if entity_type not in patterns:
                patterns[entity_type] = pattern
        
        # Add language-specific patterns with more comprehensive coverage
        if language == "en":
            # US Phone numbers
            patterns[EntityType.PHONE] = re.compile(r'\b(?:\+\d{1,2}\s?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b')
            
            # US Social Security Numbers
            patterns[EntityType.SSN] = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
            
            # US Addresses (simplified)
            patterns[EntityType.ADDRESS] = re.compile(r'\b\d+\s+[A-Za-z0-9\s,]+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Place|Pl|Court|Ct|Way|Terrace|Ter)[\s,]+[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b', re.IGNORECASE)
            
            # Dates (multiple formats)
            patterns[EntityType.DATE] = re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}(?:,)? \d{2,4})\b', re.IGNORECASE)
            
            # Credit card numbers
            patterns[EntityType.CREDIT_CARD] = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
            
            # URLs
            patterns[EntityType.URL] = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*\??[-\w=&]+')
            
            # IP addresses
            patterns[EntityType.IP_ADDRESS] = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
            
            # Email addresses
            patterns[EntityType.EMAIL] = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
            
        elif language == "de":
            # German phone numbers
            patterns[EntityType.PHONE] = re.compile(r'\b(?:\+49\s?|0)[1-9](?:[\s.-]?\d{1,4}){1,4}\b')
            
            # German dates
            patterns[EntityType.DATE] = re.compile(r'\b(?:\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember) \d{1,2}(?:,)? \d{2,4})\b', re.IGNORECASE)
            
            # German ID cards (Personalausweis)
            patterns[EntityType.ID_NUMBER] = re.compile(r'\b[A-Z]{2}[0-9]{7}(?:\s?[0-9])?D?<<[0-9]{7}[0-9]\b')
            
        elif language == "es":
            # Spanish national ID (DNI)
            patterns[EntityType.ID_NUMBER] = re.compile(r'\b[0-9XYZ][0-9]{7}[A-Z]\b')
            
            # Spanish phone numbers
            patterns[EntityType.PHONE] = re.compile(r'\b(?:\+34\s?)?(?:6\d{2}|7[1-9]\d)[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}\b')
            
            # Spanish dates
            patterns[EntityType.DATE] = re.compile(r'\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|(?:Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre) \d{1,2}(?:,)? \d{2,4})\b', re.IGNORECASE)
            
        elif language == "fr":
            # French social security number
            patterns[EntityType.ID_NUMBER] = re.compile(r'\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b')
            
            # French phone numbers
            patterns[EntityType.PHONE] = re.compile(r'\b(?:\+33\s?|0)[1-9](?:[\s.-]?\d{2}){4}\b')
        
        # Store for future use
        self.language_patterns[language] = patterns
        return patterns
    
    def _get_language_patterns(self, language: str) -> Dict[str, Pattern]:
        """
        Get patterns for a specific language, loading if necessary.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary of patterns
        """
        if language in self.language_patterns:
            return self.language_patterns[language]
        
        return self._load_language_patterns(language)
    
    def _has_overlap(self, entity: Dict[str, Any], entities: List[Dict[str, Any]]) -> bool:
        """
        Check if an entity overlaps with any entity in a list.
        
        Args:
            entity: Entity to check
            entities: List of entities to check against
            
        Returns:
            True if there is an overlap, False otherwise
        """
        start, end = entity["start"], entity["end"]
        
        for other in entities:
            other_start, other_end = other["start"], other["end"]
            
            # Check for overlap
            if (start <= other_end and end >= other_start):
                return True
        
        return False
    
    def _guess_gender(self, name: str) -> str:
        """
        Make a simple guess of gender from a name.
        
        Args:
            name: Person name
            
        Returns:
            'female', 'male', or 'unknown'
        """
        # Simple heuristic based on common name endings
        name = name.lower()
        
        # Get first name (assuming first word is first name)
        first_name = name.split()[0] if " " in name else name
        
        # Common female name endings
        female_endings = ["a", "e", "ie", "y", "i", "ine", "elle"]
        if any(first_name.endswith(ending) for ending in female_endings):
            return "female"
        
        return "male"  # Default to male if no female indicators
    
    # Comprehensive methods for generating realistic fake data
    def _get_random_name(self, gender: str = "male") -> str:
        """
        Generate a realistic random name based on gender.
        
        Args:
            gender: The gender to generate a name for ("male", "female", or None)
            
        Returns:
            A realistic random name
        """
        # Large dictionary of common names by gender
        male_first_names = [
            "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
            "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven",
            "Andrew", "Paul", "Joshua", "Kenneth", "Kevin", "Brian", "George", "Timothy",
            "Ronald", "Jason", "Edward", "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas",
            "Eric", "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon", "Benjamin",
            "Samuel", "Gregory", "Alexander", "Frank", "Patrick", "Raymond", "Jack", "Dennis"
        ]
        
        female_first_names = [
            "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica",
            "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra", "Ashley",
            "Dorothy", "Kimberly", "Emily", "Donna", "Michelle", "Carol", "Amanda", "Melissa",
            "Deborah", "Stephanie", "Rebecca", "Laura", "Sharon", "Cynthia", "Kathleen", "Amy",
            "Angela", "Shirley", "Anna", "Ruth", "Brenda", "Pamela", "Nicole", "Katherine",
            "Samantha", "Christine", "Emma", "Catherine", "Debra", "Virginia", "Rachel", "Carolyn"
        ]
        
        last_names = [
            "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson",
            "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin",
            "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee",
            "Walker", "Hall", "Allen", "Young", "Hernandez", "King", "Wright", "Lopez",
            "Hill", "Scott", "Green", "Adams", "Baker", "Gonzalez", "Nelson", "Carter",
            "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker", "Evans"
        ]
        
        # Select appropriate name by gender
        if gender == "female":
            first_name = female_first_names[self._deterministic_random(len(female_first_names))]
        else:
            first_name = male_first_names[self._deterministic_random(len(male_first_names))]
            
        last_name = last_names[self._deterministic_random(len(last_names))]
        
        return f"{first_name} {last_name}"
    
    def _get_random_location(self) -> str:
        """
        Generate a realistic random city name.
        
        Returns:
            A random city name
        """
        cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
            "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
            "Fort Worth", "Columbus", "San Francisco", "Charlotte", "Indianapolis", "Seattle",
            "Denver", "Washington", "Boston", "El Paso", "Nashville", "Portland", "Oklahoma City",
            "Las Vegas", "Detroit", "Memphis", "Louisville", "Baltimore", "Milwaukee", "Albuquerque",
            "Tucson", "Fresno", "Sacramento", "Mesa", "Kansas City", "Atlanta", "Omaha", "Colorado Springs",
            "Raleigh", "Miami", "Long Beach", "Virginia Beach", "Oakland", "Minneapolis", "Tampa", "Arlington"
        ]
        
        return cities[self._deterministic_random(len(cities))]
    
    def _get_random_organization(self) -> str:
        """
        Generate a realistic random organization name.
        
        Returns:
            A random company/organization name
        """
        prefixes = [
            "Global", "National", "United", "International", "Advanced", "Pacific", "Superior",
            "Elite", "Premium", "Precision", "Strategic", "Dynamic", "Innovative", "Integrated",
            "Digital", "Universal", "Premier", "Apex", "Quantum", "Synergy", "Nexus", "Atlas",
            "Pinnacle", "Vanguard", "Horizon", "Sentinel", "Sterling", "Prestige", "Paramount"
        ]
        
        midsections = [
            "Tech", "Data", "Health", "Financial", "Energy", "Communications", "Logistics",
            "Aerospace", "Development", "Research", "Investment", "Media", "Transport", "Security",
            "Medical", "Engineering", "Construction", "", "", "", "", "", "", ""  # Empty strings increase chance of 2-part names
        ]
        
        suffixes = [
            "Corporation", "Associates", "Industries", "Solutions", "Technologies", "Systems",
            "Group", "Partners", "Enterprises", "International", "Inc", "LLC", "Ltd", "Co",
            "Services", "Innovations", "Consultants", "Ventures", "Holdings", "Networks", "Dynamics"
        ]
        
        prefix = prefixes[self._deterministic_random(len(prefixes))]
        midsection = midsections[self._deterministic_random(len(midsections))]
        suffix = suffixes[self._deterministic_random(len(suffixes))]
        
        # Format based on whether a midsection is included
        if midsection:
            return f"{prefix} {midsection} {suffix}"
        else:
            return f"{prefix} {suffix}"
    
    def _get_random_street(self) -> str:
        """
        Generate a realistic random street address.
        
        Returns:
            A random street address
        """
        number = self._deterministic_random(1000) + 1
        
        street_names = [
            "Main", "Oak", "Maple", "Cedar", "Pine", "Elm", "Washington", "Park",
            "Lake", "Hill", "River", "View", "Highland", "Forest", "Sunset", "Ridge",
            "Willow", "Meadow", "Church", "Center", "Broad", "Mill", "Broadway", "Market",
            "Spring", "Front", "Water", "Union", "South", "North", "East", "West",
            "Franklin", "Jefferson", "Madison", "Adams", "Jackson", "Lincoln", "Wilson"
        ]
        
        street_types = [
            "Street", "Avenue", "Boulevard", "Drive", "Lane", "Road", "Place", "Court",
            "Circle", "Way", "Parkway", "Terrace", "Plaza", "Trail", "Crossing"
        ]
        
        street_name = street_names[self._deterministic_random(len(street_names))]
        street_type = street_types[self._deterministic_random(len(street_types))]
        
        return f"{number} {street_name} {street_type}"
    
    def _deterministic_random(self, max_value: int, seed_text: str = None) -> int:
        """
        Generate a deterministic random number based on the hash of the entity.
        This ensures consistent replacements for the same entities.
        
        Args:
            max_value: Maximum value (exclusive)
            seed_text: Optional text to use as seed for deterministic generation
            
        Returns:
            Random integer between 0 and max_value-1
        """
        if seed_text is None:
            # If no seed text is provided, use instance id + counter as a pseudo-random source
            if not hasattr(self, '_random_counter'):
                self._random_counter = 0
            self._random_counter += 1
            seed_text = f"{id(self)}_{self._random_counter}"
        
        # Use hashlib to create a deterministic hash from the seed text
        import hashlib
        hash_obj = hashlib.md5(seed_text.encode('utf-8'))
        hash_value = int(hash_obj.hexdigest(), 16)
        
        # Convert the hash to a value within the desired range
        return hash_value % max_value
    
    def _get_patterns_for_language(self, language="en"):
        """Get regex patterns for entity types in the specified language."""
        patterns = {}
        
        if language == "en":
            # English dates
            patterns[EntityType.DATE] = re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? \d{2,4})\b', re.IGNORECASE)
            
            # English phone numbers
            patterns[EntityType.PHONE] = re.compile(r'\b(?:\+1\s?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b')
            
            # Social Security Number
            patterns[EntityType.ID_NUMBER] = re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b')
            
        elif language == "es":
            # Spanish phone numbers
            patterns[EntityType.PHONE] = re.compile(r'\b(?:\+34\s?)?(?:6\d{2}|7[1-9]\d)[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}\b')
            
            # Spanish ID (DNI/NIE)
            patterns[EntityType.ID_NUMBER] = re.compile(r'\b[0-9XYZ][0-9]{7}[A-Z]\b')
            
            # Spanish dates
            patterns[EntityType.DATE] = re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre) \d{1,2}(?:,)? \d{2,4})\b', re.IGNORECASE)
            
        elif language == "fr":
            # French phone numbers
            patterns[EntityType.PHONE] = re.compile(r'\b(?:\+33\s?|0)[1-9](?:[\s.-]?\d{2}){4}\b')
            
            # French social security number
            patterns[EntityType.ID_NUMBER] = re.compile(r'\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b')
            
            # French dates
            patterns[EntityType.DATE] = re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Janvier|Février|Mars|Avril|Mai|Juin|Juillet|Août|Septembre|Octobre|Novembre|Décembre) \d{1,2}(?:,)? \d{2,4})\b', re.IGNORECASE)
        
        return patterns