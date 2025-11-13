"""GPU deployment via Modal."""

# Lazy imports to avoid requiring 'modal' package in containers
def __getattr__(name):
    if name == "ModalImageBuilder":
        from .modal_image_builder import ModalImageBuilder
        return ModalImageBuilder
    elif name == "ModalClient":
        from .modal_client import ModalClient
        return ModalClient
    elif name == "ModalDeploymentInfo":
        from .modal_client import ModalDeploymentInfo
        return ModalDeploymentInfo
    elif name == "initialize_session":
        from .session_init import initialize_session
        return initialize_session
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ModalImageBuilder",
    "ModalClient",
    "ModalDeploymentInfo",
    "initialize_session",
]
