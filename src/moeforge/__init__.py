"""MoE Forge package."""

__all__ = ["__version__"]

__version__ = "0.1.0"


def _register_transformers() -> None:
    try:
        from .hf_runtime import register_transformers_auto_classes
    except ImportError:
        return
    register_transformers_auto_classes()


_register_transformers()
