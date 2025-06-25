"""
validator.py
Model validation logic for HuggingFace models and conversion parameters.
"""

import logging
from typing import Dict, Any
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)

class Validator:
    """Validator for HuggingFace models and conversion parameters."""
    def validate_huggingface_model(self, model_name: str) -> Dict[str, Any]:
        """Validate HuggingFace model and tokenizer."""
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
            logger.info(f"Validated model: {model_name}")
            return {
                "model_name": model_name,
                "config": config.to_dict(),
                "tokenizer_class": tokenizer.__class__.__name__,
                "status": "valid"
            }
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {"model_name": model_name, "status": "invalid", "error": str(e)}

    def validate_conversion_params(self, target_format: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate conversion parameters for a given format."""
        # For now, just return valid for all
        return {"target_format": target_format, "params": params, "status": "valid"}

validator = Validator() 