"""Chakra execution trace tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlc-chakra")
except PackageNotFoundError:
    __version__ = "0+unknown"
