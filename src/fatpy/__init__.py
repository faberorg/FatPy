"""A collection of general utility functions."""

try:
    from ._version import __version__
except ImportError:
    # Default version during development
    __version__ = "0.0.0.dev0"
