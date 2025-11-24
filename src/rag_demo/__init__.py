"""Gemini RAG example package."""

from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("gemini-rag-example")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
