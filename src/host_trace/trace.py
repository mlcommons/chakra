import logging
from typing import Any, Dict, List

from . import versions
from .errors import HostTraceError
from .names import PROCESS_ANNOTATION, THREAD_ANNOTATION
from .node import HostNode
from .reader import read_trace_json

logger = logging.getLogger(__name__)

# Top-level keys the converter writes into the .et metadata.
METADATA_KEYS = ("schema", "pid", "time", "start_ts", "finish_ts")


class HostTrace:
    """
    A Chakra host trace (raw from PyTorch, or linked by chakra_trace_link), parsed once and shared by both tools.

    Reading a trace: the schema is checked (versions.py), every node becomes a HostNode, and each node is attached to
    its parent. Each tool then takes its own view: the linker the ops under the process root (linker_ops), the
    converter all nodes, the thread roots and the metadata.

    Attributes
        raw (Dict[str, Any]): The JSON as read. The parser never modifies it.
        source (str): Where the trace was read from, used in messages.
        schema (versions.Schema): The parsed schema; schema.text is the string as written in the trace.
        nodes (Dict[int, HostNode]): Every node by id, in file order. A later node with the same id replaces an
            earlier one.
    """

    def __init__(self, raw: Dict[str, Any], source: str = "host trace") -> None:
        self.raw = raw
        self.source = source
        for key in ("schema", "nodes"):
            if key not in raw:
                raise HostTraceError(f"{source}: missing top-level key {key!r}; is this a host trace?")
        self.schema = versions.parse_schema(raw["schema"], source)
        versions.check_supported(self.schema, source)
        self._check_layout(raw["nodes"])

        self.nodes: Dict[int, HostNode] = {}
        for node_data in raw["nodes"]:
            node = HostNode(self.schema.text, node_data, source)
            self.nodes[node.id] = node
        attach_children(self.nodes)

    @classmethod
    def from_file(cls, path: str) -> "HostTrace":
        """Read and parse a host trace file (plain or gzip-compressed JSON)."""
        return cls(read_trace_json(path), str(path))

    def _check_layout(self, nodes: List[Dict[str, Any]]) -> None:
        """Warn once if nodes lack keys of their schema version (PyTorch changed the format once without a bump)."""
        attrs, tensor_keys = versions.expected_keys(self.schema)
        mismatched, first = 0, ""
        for node in nodes:
            names = {attr.get("name") for attr in node.get("attrs", []) if isinstance(attr, dict)}
            lacking = [a for a in attrs if a not in names]
            for side in ("inputs", "outputs"):
                value = node.get(side)
                if isinstance(value, dict):
                    lacking += [f"{side}.{k}" for k in tensor_keys if k not in value]
            if lacking:
                mismatched += 1
                first = first or f"node {node.get('id')} lacks {', '.join(lacking)}"
        if mismatched:
            logger.warning(
                f"{self.source}: {mismatched} of {len(nodes)} nodes do not have every key of schema {self.schema.text} "
                f"(first: {first}); reading them anyway"
            )

    def metadata(self) -> Dict[str, Any]:
        """Return the five top-level keys the converter writes into the .et. Raise HostTraceError if one is missing."""
        missing = [key for key in METADATA_KEYS if key not in self.raw]
        if missing:
            raise HostTraceError(
                f"{self.source}: missing top-level key(s) {', '.join(missing)}, which chakra_converter writes into "
                "the .et metadata"
            )
        return {key: self.raw[key] for key in METADATA_KEYS}

    def process_root(self) -> HostNode:
        """
        Return the root of the whole trace: the node named PROCESS_ANNOTATION whose only parent is itself.

        It is found by name, never by id: PyTorch gives it id 1, but chakra_trace_link renumbers ids from 0.
        """
        roots = [
            n
            for n in self.nodes.values()
            if n.name == PROCESS_ANNOTATION and (n.parent == n.id or n.parent not in self.nodes)
        ]
        if len(roots) != 1:
            found = "none" if not roots else f"{len(roots)} (ids {', '.join(str(n.id) for n in roots)})"
            raise HostTraceError(f"{self.source}: expected one process root named {PROCESS_ANNOTATION}, found {found}")
        return roots[0]

    def thread_roots(self) -> List[HostNode]:
        """Return the root of each thread: the nodes named THREAD_ANNOTATION, in file order."""
        return [n for n in self.nodes.values() if n.name == THREAD_ANNOTATION]

    def linker_ops(self) -> List[HostNode]:
        """
        Return the host ops chakra_trace_link links: the process root and every node under it, sorted by id.

        Raise HostTraceError if there is not exactly one process root, or if an op has no rf_id (the linker matches
        host ops to Kineto ops by rf_id).
        """
        ops = ops_reachable_from(self.process_root())
        for op in ops:
            if op.rf_id is None:
                raise HostTraceError(
                    f"{op.where()}rf_id is missing or null; chakra_trace_link matches host ops to Kineto ops by rf_id"
                )
        return ops


def attach_children(nodes: Dict[int, HostNode]) -> None:
    """
    Attach every node to its parent, if the parent is in the trace and is not the node itself. Sort children by id.

    The process root is its own parent, so it is not its own child.
    """
    for node in nodes.values():
        if node.parent in nodes and node.parent != node.id:
            nodes[node.parent].add_child(node)
    for node in nodes.values():
        node.children.sort(key=lambda child: child.id)


def ops_reachable_from(root: Any) -> List[Any]:
    """
    Return the node and every node under it, sorted by id.

    The walk is iterative and visits each id once, so deep traces need no recursion limit, and a parent cycle cannot
    make it loop forever. Nodes in a cycle are not under the root, and are left out like orphans.
    """
    seen = set()
    found = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.id in seen:
            continue
        seen.add(node.id)
        found.append(node)
        stack.extend(node.children)
    return sorted(found, key=lambda n: n.id)
