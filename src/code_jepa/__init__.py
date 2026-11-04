"""Hierarchical Code-JEPA: span-grounded multi-level code representations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("code-jepa")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
