"""Modal image builder - compatibility wrapper."""

from typing import Optional, TYPE_CHECKING

import modal

from .image_builder import StandardImageBuilder, DockerImageBuilder

if TYPE_CHECKING:
    from .sandbox import ExecutionMode


class ModalImageBuilder:
    """
    Compatibility wrapper for image builders.

    Routes to StandardImageBuilder or DockerImageBuilder based on docker_in_docker flag.
    """

    def __init__(
        self,
        python_packages: list[str] = None,
        system_packages: list[str] = None,
        python_version: str = "3.11",
        docker_in_docker: bool = False,
        execution_mode: Optional["ExecutionMode"] = None,
        notebook_packages: list[str] = None,
        custom_setup_commands: list[str] = None,
    ):
        """Initialize image builder."""
        if docker_in_docker:
            self._builder = DockerImageBuilder(
                python_packages=python_packages,
                system_packages=system_packages,
                execution_mode=execution_mode,
                notebook_packages=notebook_packages,
                custom_setup_commands=custom_setup_commands,
            )
        else:
            self._builder = StandardImageBuilder(
                python_packages=python_packages,
                system_packages=system_packages,
                python_version=python_version,
                execution_mode=execution_mode,
                notebook_packages=notebook_packages,
                custom_setup_commands=custom_setup_commands,
            )

    def build(self) -> modal.Image:
        """Build Modal Image."""
        return self._builder.build()

    def get_sandbox_options(self) -> dict:
        """Get sandbox options."""
        return self._builder.get_sandbox_options()

    def get_sandbox_entrypoint(self) -> Optional[str]:
        """Get sandbox entrypoint."""
        return self._builder.get_sandbox_entrypoint()

    def cleanup(self):
        """Clean up resources."""
        self._builder.cleanup()
