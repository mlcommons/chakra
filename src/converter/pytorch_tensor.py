"""Moved to host_trace.tensor. This module re-exports it, so existing imports keep working until the clean-up."""

from ..host_trace.tensor import HostTensor, list_to_host_tensor

# The names before the move, kept as aliases until the clean-up.
PyTorchTensor = HostTensor
list_to_pytorch_tensor = list_to_host_tensor

__all__ = ["HostTensor", "PyTorchTensor", "list_to_host_tensor", "list_to_pytorch_tensor"]
