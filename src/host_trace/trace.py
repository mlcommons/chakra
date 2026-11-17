import logging
from typing import Any, Dict, Generic, List, Mapping, Type, TypeVar, cast

from . import versions
from .errors import HostTraceError
from .names import PROCESS_ANNOTATION
from .node import HostNode
from .reader import read_trace_json

logger = logging.getLogger(__name__)

# The node class of a trace: HostNode, or a tool's subclass of it (the converter's ConverterNode).
N = TypeVar("N", bound=HostNode)
_HOST_NODE: Any = HostNode  # the default node_class, typed Any so that it fits every N


class HostTrace(Generic[N]):
    """
    A Chakra host trace (raw from PyTorch, or linked by chakra_trace_link), read and checked once.

    Reading a trace: the schema is checked (versions.py), every node becomes a node_class instance (HostNode, or a
    tool's subclass of it), and each node is attached to its parent. The trace knows nothing about the tools: each
    tool takes what it needs from it and checks that itself.

    Attributes
        raw (Dict[str, Any]): The JSON as read. Never modified, by the parser or by the tools.
        source (str): Where the trace was read from, used in messages.
        schema (versions.Schema): The parsed schema; schema.text is the string as written in the trace.
        nodes (Dict[int, N]): Every node by id, in file order. Ids are unique: PyTorch assigns them from a
            counter, so a duplicate means a broken file and is an error.
    """

    def __init__(self, raw: Dict[str, Any], source: str = "host trace", node_class: Type[N] = _HOST_NODE) -> None:
        self.raw = raw
        self.source = source
        for key in ("schema", "nodes"):
            if key not in raw:
                raise HostTraceError(f"{source}: missing top-level key {key!r}; is this a host trace?")
        if not isinstance(raw["nodes"], list) or not all(isinstance(n, dict) for n in raw["nodes"]):
            raise HostTraceError(f"{source}: 'nodes' must be a list of node objects")
        self.schema = versions.parse_schema(raw["schema"], source)
        versions.check_supported(self.schema, source)
        self._check_layout(raw["nodes"])

        self.nodes: Dict[int, N] = {}
        for node_data in raw["nodes"]:
            node = node_class(self.schema.text, node_data, source)
            if node.id in self.nodes:
                raise HostTraceError(
                    f"{source}: duplicate node id {node.id} ({self.nodes[node.id].name} and {node.name}); "
                    "node ids must be unique"
                )
            self.nodes[node.id] = node
        _attach_children(self.nodes)

    @classmethod
    def from_file(cls, path: str, node_class: Type[N] = _HOST_NODE) -> "HostTrace[N]":
        """Read and parse a host trace file (plain or gzip-compressed JSON)."""
        return cls(read_trace_json(path), str(path), node_class)

    def _check_layout(self, nodes: List[Dict[str, Any]]) -> None:
        """Warn once if nodes lack keys of their schema version (PyTorch changed the format once without a bump)."""
        attrs, tensor_keys = versions.expected_keys(self.schema)
        mismatched, first = 0, ""
        for node in nodes:
            node_attrs = node.get("attrs", [])
            names = (
                {a.get("name") for a in node_attrs if isinstance(a, dict)} if isinstance(node_attrs, list) else set()
            )
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

    def process_root(self) -> N:
        """
        Return the root of the whole trace: the node named PROCESS_ANNOTATION whose only parent is itself.

        It is found by name, never by id: PyTorch gives it id 1, but chakra_trace_link renumbers ids from 0. Raise
        HostTraceError unless there is exactly one.
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

    def reachable_from(self, root: N) -> List[N]:
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
            stack.extend(cast(List[N], node.children))  # a node's children are nodes of the same trace, of class N
        return sorted(found, key=lambda n: n.id)

    def unreachable_from(self, root: N) -> List[N]:
        """Return the nodes that are not under the node (orphans, and nodes in a parent cycle), sorted by id."""
        under = {n.id for n in self.reachable_from(root)}
        return sorted((n for n in self.nodes.values() if n.id not in under), key=lambda n: n.id)


def _attach_children(nodes: Mapping[int, HostNode]) -> None:
    """
    Attach every node to its parent, if the parent is in the trace and is not the node itself. Sort children by id.

    The process root is its own parent, so it is not its own child.
    """
    for node in nodes.values():
        if node.parent in nodes and node.parent != node.id:
            nodes[node.parent]._add_child(node)
    for node in nodes.values():
        node.children.sort(key=lambda child: child.id)
