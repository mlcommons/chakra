"""Shared fixtures: small, valid host traces built in code, so each test can change exactly one thing."""

import copy
from typing import Any, Callable, Dict, List, Optional

import pytest
from chakra.src.host_trace.names import CORE_ATTRS, KERNEL_ATTRS, PROCESS_ANNOTATION, THREAD_ANNOTATION

SCHEMA = "1.1.1-chakra.0.0.4"


def host_node(node_id: int, name: str, parent: int, *, cat: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """Return one valid node of schema 1.1.1: every core and kernel attribute, inputs and outputs with strides."""
    io = {"values": [[1, 2, 0, 4, 4, "cpu"]], "shapes": [[2, 2]], "types": ["Tensor(float)"], "strides": [[2, 1]]}
    attrs: List[Dict[str, Any]] = [
        {"name": a, "type": "uint64", "value": node_id} for a in CORE_ATTRS if a != "op_schema"
    ]
    attrs.append({"name": "op_schema", "type": "string", "value": f"aten::{name}"})
    attrs += [{"name": a, "type": "string", "value": ""} for a in KERNEL_ATTRS]
    node = {
        "id": node_id,
        "name": name,
        "ctrl_deps": parent,
        "inputs": copy.deepcopy(io),
        "outputs": copy.deepcopy(io),
        "attrs": attrs,
    }
    if cat is not None:
        node["cat"] = cat
    node.update(extra)
    return node


def host_trace_data(nodes: Optional[List[Dict[str, Any]]] = None, **top: Any) -> Dict[str, Any]:
    """Return a valid host trace: a process root (1), a thread root (2), two ops (3, 4) and a GPU kernel (5)."""
    if nodes is None:
        nodes = [
            host_node(1, PROCESS_ANNOTATION, 1),
            host_node(2, THREAD_ANNOTATION, 1),
            host_node(3, "linear", 2),
            host_node(4, "addmm", 3),
            host_node(5, "gemm_kernel", 4, cat="kernel"),
        ]
    data = {
        "schema": SCHEMA,
        "pid": 1234,
        "time": "2026-01-01 00:00:00",
        "start_ts": 1000,
        "finish_ts": 2000,
        "nodes": nodes,
    }
    data.update(top)
    return data


@pytest.fixture
def make_trace_data() -> Callable[..., Dict[str, Any]]:
    """Build a valid host trace as a dict (see host_trace_data); tests change one thing in it."""
    return host_trace_data


@pytest.fixture
def make_node() -> Callable[..., Dict[str, Any]]:
    """Build one valid node as a dict (see host_node)."""
    return host_node
