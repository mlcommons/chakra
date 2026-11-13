"""Moved to host_trace.node. This module re-exports it, so existing imports keep working until the clean-up."""

from ..host_trace.node import HostNode, HostNodeType

# The names before the move, kept as aliases until the clean-up.
PyTorchNode = HostNode
PyTorchNodeType = HostNodeType

__all__ = ["HostNode", "HostNodeType", "PyTorchNode", "PyTorchNodeType"]
