"""FatPy - A Python package for fatigue life evaluation of materials."""

try:
    from ._version import __version__
except ImportError:
    # Default version during development
    __version__ = "0.0.0.dev0"
