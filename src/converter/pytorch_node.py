"""Moved to host_trace.node. This module re-exports it, so existing imports keep working until the clean-up."""

from ..host_trace.node import PyTorchNode, PyTorchNodeType

__all__ = ["PyTorchNode", "PyTorchNodeType"]
