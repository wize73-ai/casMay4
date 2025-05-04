"""
Tokenizer Pipeline for CasaLingua

Provides a shared tokenizer for use across translation, RAG, simplification, etc.

Author: CasaLingua Team
Version: 0.1.0
"""

import logging
from typing import List, Optional
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from app.services.models.loader import ModelRegistry

LANG_CODE_MAPPING = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
}

logger = logging.getLogger("casalingua.tokenizer")

class TokenizerPipeline:
    def __init__(self, model_name: str = "google/mt5-small", task_type: str = "generic"):
        try:
            self.model_name = model_name
            self.task_type = task_type
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"✓ Loaded tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer {model_name}: {e}")
            raise

    def _apply_task_prompt(self, text: str) -> str:
        # Normalize by model + task combo
        if "bart" in self.model_name and self.task_type == "summarization":
            return f"summarize: {text}"
        elif "mt5" in self.model_name and self.task_type == "summarization":
            return f"<extra_id_0> {text}"
        elif "nllb" in self.model_name and self.task_type == "translation":
            return text  # language_id set separately
        elif "opus-mt" in self.model_name and self.task_type == "translation":
            return text  # special tokens handled in tokenizer
        elif "mbart" in self.model_name and self.task_type == "translation":
            return text  # prefix lang-specific token
        else:
            return text  # default fallback

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        processed = self._apply_task_prompt(text)
        return self.tokenizer.encode(processed, add_special_tokens=add_special_tokens)

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        processed = [self._apply_task_prompt(text) for text in texts]
        return self.tokenizer(processed, padding=False, add_special_tokens=add_special_tokens)["input_ids"]

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    def prepare_translation_inputs(self, text: str, source_lang: str, target_lang: str) -> dict:
        source_code = LANG_CODE_MAPPING.get(source_lang, source_lang)
        target_code = LANG_CODE_MAPPING.get(target_lang, target_lang)

        if "nllb" in self.model_name:
            if hasattr(self.tokenizer, "src_lang"):
                self.tokenizer.src_lang = source_code
            model_inputs = self.tokenizer(text, return_tensors="pt")
            forced_bos_id = self.tokenizer.lang_code_to_id.get(target_code)
            return {
                "inputs": model_inputs,
                "forced_bos_token_id": forced_bos_id,
                "source_lang": source_code,
                "target_lang": target_code
            }

        elif "mbart" in self.model_name:
            model_inputs = self.tokenizer(text, return_tensors="pt")
            forced_bos_id = self.tokenizer.lang_code_to_id.get(target_code)
            model_inputs["decoder_start_token_id"] = forced_bos_id
            return {
                "inputs": model_inputs,
                "forced_bos_token_id": forced_bos_id,
                "source_lang": source_code,
                "target_lang": target_code
            }

        elif "opus-mt" in self.model_name:
            prefix = f">>{target_code}<< "
            processed = prefix + text
            model_inputs = self.tokenizer(processed, return_tensors="pt")
            return {
                "inputs": model_inputs,
                "forced_bos_token_id": None,
                "source_lang": source_code,
                "target_lang": target_code
            }

        else:
            model_inputs = self.tokenizer(text, return_tensors="pt")
            return {
                "inputs": model_inputs,
                "forced_bos_token_id": None,
                "source_lang": source_code,
                "target_lang": target_code
            }

    def prepare_input(
        self,
        text: str,
        task: str,
        model_name: str,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None
    ) -> dict:
        from transformers import PreTrainedTokenizerBase

        if task == "translation":
            return self.prepare_translation_inputs(
                text=text,
                source_lang=source_language or "en",
                target_lang=target_language or "es"
            )

        elif task == "summarization":
            # Apply prefix for summarization if needed
            processed = self._apply_task_prompt(text)
            model_inputs = self.tokenizer(processed, return_tensors="pt")
            return {
                "inputs": model_inputs,
                "forced_bos_token_id": None,
                "source_lang": None,
                "target_lang": None
            }

        else:
            processed = self._apply_task_prompt(text)
            model_inputs = self.tokenizer(processed, return_tensors="pt")
            return {
                "inputs": model_inputs,
                "forced_bos_token_id": None,
                "source_lang": None,
                "target_lang": None
            }

    def prepare_for_summarization(self, text: str) -> dict:
        processed = self._apply_task_prompt(text)
        model_inputs = self.tokenizer(processed, return_tensors="pt")
        return {
            "inputs": model_inputs,
            "forced_bos_token_id": None,
            "source_lang": None,
            "target_lang": None
        }
