# app/core/document/ocr.py
"""
OCR Processor for CasaLingua

This module handles optical character recognition for images,
extracting text from images and scanned documents.
"""

import io
import logging
import os
from typing import Dict, Any, List, Optional, Tuple, Union

# For OCR processing, we'll try to use pytesseract
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter, Image as PILImage
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

class OCRProcessor:
    """
    OCR processor for images and scanned documents.
    
    Features:
    - Text extraction from images
    - Support for multiple languages
    - Image preprocessing for better OCR results
    """
    
    def __init__(self, model_manager, config: Dict[str, Any] = None):
        """
        Initialize the OCR processor.
        
        Args:
            model_manager: Model manager for OCR processing
            config: Configuration dictionary
        """
        self.model_manager = model_manager
        self.config = config if config is not None else {}
        
        
        # Configure tesseract path if provided in config
        tesseract_path = self.config.get("tesseract_path")
        if tesseract_path and OCR_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        logger.info("OCR processor initialized")
    
    async def extract_text(self, image_content: bytes, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text from an image using OCR.
        
        Args:
            image_content: Image content as bytes
            language: Optional language code for OCR
            
        Returns:
            Dictionary with extracted text and metadata
        """
        logger.debug(f"Extracting text from image of size {len(image_content)} bytes")
        
        # Check if we have OCR capabilities
        if not OCR_AVAILABLE:
            # Try to use model-based OCR if available
            return await self._use_model_ocr(image_content, language)
        
        try:
            # Load image from bytes
            image = PILImage.open(io.BytesIO(image_content))
            
            # Preprocess image for better OCR results
            processed_image = self._preprocess_image(image)
            
            # Map language code to tesseract language
            ocr_language = self._map_language_code(language) if language else 'eng'
            
            # Extract text with OCR
            ocr_result = pytesseract.image_to_data(
                processed_image, 
                lang=ocr_language,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            extracted_text = []
            total_confidence = 0
            word_count = 0
            
            for i, text in enumerate(ocr_result['text']):
                if text.strip():
                    confidence = ocr_result['conf'][i]
                    try:
                        confidence = float(confidence)
                    except ValueError:
                        continue
                    if confidence > 0:  # Only count valid confidence scores
                        extracted_text.append(text)
                        total_confidence += confidence
                        word_count += 1
            
            # Calculate average confidence
            avg_confidence = total_confidence / word_count if word_count > 0 else 0
            
            # Format result
            result = {
                "text": " ".join(extracted_text),
                "confidence": avg_confidence / 100.0,  # Normalize to 0-1
                "metadata": {
                    "word_count": word_count,
                    "image_width": image.width,
                    "image_height": image.height,
                    "language": ocr_language,
                    "method": "tesseract"
                }
            }
            
            logger.debug(f"Extracted {word_count} words from image with confidence {avg_confidence:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}", exc_info=True)
            
            # Try to use model-based OCR as fallback
            return await self._use_model_ocr(image_content, language)
    
    async def _use_model_ocr(self, image_content: bytes, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Use model-based OCR when tesseract is not available.
        
        Args:
            image_content: Image content as bytes
            language: Optional language code
            
        Returns:
            Dictionary with OCR results
        """
        logger.debug("Using model-based OCR")
        
        try:
            # Try to get OCR model from model manager
            ocr_model = await self.model_manager.get_model("ocr")
            
            if not ocr_model:
                raise ValueError("No OCR model available")
            
            # Prepare input for model
            input_data = {
                "image": image_content,
                "language": language
            }
            
            # Run OCR model
            result = await self.model_manager.run_model(
                ocr_model,
                "extract_text",
                input_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in model-based OCR: {str(e)}", exc_info=True)
            
            # Return empty result if all OCR methods fail
            return {
                "text": "",
                "confidence": 0.0,
                "metadata": {
                    "error": str(e),
                    "method": "failed"
                }
            }
    
    def _preprocess_image(self, image: PILImage.Image) -> PILImage.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image object
            
        Returns:
            Processed PIL Image
        """
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Apply sharpen filter
            image = image.filter(ImageFilter.SHARPEN)
            
            # Scale image if it's very large
            max_dimension = 3000  # Maximum pixel dimension
            if image.width > max_dimension or image.height > max_dimension:
                if image.width > image.height:
                    ratio = max_dimension / image.width
                    new_size = (max_dimension, int(image.height * ratio))
                else:
                    ratio = max_dimension / image.height
                    new_size = (int(image.width * ratio), max_dimension)
                image = image.resize(new_size, Image.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
            return image  # Return original image if preprocessing fails
    
    def _map_language_code(self, language_code: str) -> str:
        """
        Map ISO language code to tesseract language code.
        
        Args:
            language_code: ISO language code (e.g., 'en', 'es')
            
        Returns:
            Tesseract language code
        """
        # Tesseract language mapping (ISO code to tesseract code)
        language_map = {
            'en': 'eng',  # English
            'es': 'spa',  # Spanish
            'fr': 'fra',  # French
            'de': 'deu',  # German
            'it': 'ita',  # Italian
            'pt': 'por',  # Portuguese
            'ru': 'rus',  # Russian
            'zh': 'chi_sim',  # Chinese Simplified
            'ja': 'jpn',  # Japanese
            'ko': 'kor',  # Korean
            'ar': 'ara',  # Arabic
            'hi': 'hin',  # Hindi
        }
        
        # Get tesseract language code or default to English
        return language_map.get(language_code.lower(), 'eng')


# Direct test block for OCRProcessor
if __name__ == "__main__":
    import asyncio
    import requests

    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    processor = OCRProcessor(model_manager=None)

    try:
        # Sample image URL (you can replace this with a local test image)
        sample_image_url = "https://tesseract.projectnaptha.com/img/eng_bw.png"
        response = requests.get(sample_image_url)
        image_bytes = response.content

        result = asyncio.run(processor.extract_text(image_bytes))
        print("✅ OCR Test Result:")
        print(result)
    except Exception as e:
        print(f"❌ OCR test failed: {e}")