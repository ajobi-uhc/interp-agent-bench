"""Modal GPU integration utilities."""

from scribe.modal.images import hf_image, ml_image
from scribe.modal.model_service import create_model_service_class, get_base_model_service_template
from scribe.modal.interp_backend import create_interp_backend
from scribe.modal.interp_client import InterpClient

__all__ = [
    "hf_image",
    "ml_image",
    "create_model_service_class",
    "get_base_model_service_template",
    "create_interp_backend",
    "InterpClient",
]
