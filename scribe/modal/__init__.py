"""Modal GPU integration utilities."""

from scribe.modal.images import hf_image, ml_image
from scribe.modal.model_service import create_model_service_class, get_base_model_service_template

__all__ = ["hf_image", "ml_image", "create_model_service_class", "get_base_model_service_template"]
