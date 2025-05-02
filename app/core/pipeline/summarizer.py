"""
Summarization Pipeline Module
Provides text summarization using a multipurpose generative model
"""

import logging
from transformers import AutoModelForSeq2SeqLM
from app.core.pipeline.tokenizer import TokenizerPipeline

logger = logging.getLogger("casalingua.summarizer")

class SummarizationPipeline:
    def __init__(
        self,
        model_manager,
        config,
        *,
        model_name=None,
        tokenizer=None,
        model=None,
        registry_config: dict = None,
    ):
        registry_config = registry_config or {}
        model_info = registry_config.get("rag_generator", {})
        model_name = model_name or model_info.get("model_name")
        tokenizer_name = model_info.get("tokenizer_name")

        logger.info(f"🔄 Initializing summarization model: {model_name}")
        self.tokenizer = tokenizer or TokenizerPipeline(model_name=tokenizer_name, task_type="rag_generation")
        self.model = model or AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model_manager = model_manager
        self.config = config
        logger.info(f"✅ Summarization model '{model_name}' ready")

    async def summarize(
        self,
        text: str,
        max_length: int = 100,
        min_length: int = 30,
        language: str = "en",
        shared_tokenizer=None,
        user_id: str = None,
        model_id: str = None,
        request_id: str = None,
        **kwargs
    ) -> str:
        logger.debug(f"📦 Shared tokenizer provided: {bool(shared_tokenizer)}")
        logger.debug(f"✍️ Summarizing text: {text[:60]}...")

        try:
            inputs = self.tokenizer.tokenize(text)
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
            summary = self.tokenizer.detokenize(summary_ids[0])
            logger.debug(f"📄 Summary: {summary}")
            return {
                "summary": summary,
                "model_used": "multipurpose",
                "language": language
            }
        except Exception as e:
            logger.exception("❌ Summarization failed")
            raise RuntimeError("Summarization error: " + str(e))

    async def initialize(self):
        logger.info("🧠 Summarization pipeline initialized")
