"""Moved to host_trace.tensor. This module re-exports it, so existing imports keep working until the clean-up."""

from ..host_trace.tensor import PyTorchTensor, list_to_pytorch_tensor

__all__ = ["PyTorchTensor", "list_to_pytorch_tensor"]
