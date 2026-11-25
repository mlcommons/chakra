import logging
from typing import Any, Dict, Generic, List, Mapping, Tuple, Type, TypeVar, cast, overload

from . import versions
from .errors import HostTraceError
from .names import PROCESS_ANNOTATION
from .node import HostNode
from .reader import read_trace_json

logger = logging.getLogger(__name__)

# The node class of a trace: HostNode, or a tool's subclass of it (the converter's ConverterNode).
N = TypeVar("N", bound=HostNode)
M = TypeVar("M", bound=HostNode)


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

    @overload
    def __init__(self: "HostTrace[HostNode]", raw: Dict[str, Any], source: str = ...) -> None: ...
    @overload
    def __init__(self, raw: Dict[str, Any], source: str = ..., *, node_class: Type[N]) -> None: ...

    def __init__(self, raw: Dict[str, Any], source: str = "host trace", node_class: Type[Any] = HostNode) -> None:
        self.raw = raw
        self.source = source
        for key in ("schema", "nodes"):
            if key not in raw:
                raise HostTraceError(f"{source}: missing top-level key {key!r}; is this a host trace?")
        if not isinstance(raw["nodes"], list) or not all(isinstance(n, dict) for n in raw["nodes"]):
            raise HostTraceError(f"{source}: 'nodes' must be a list of node objects")
        self.schema = versions.parse_schema(raw["schema"], source)
        versions.check_supported(self.schema, source)

        attrs, tensor_keys = versions.expected_keys(self.schema)
        mismatched, first = 0, ""
        self.nodes: Dict[int, N] = {}
        for node_data in raw["nodes"]:
            node = node_class(self.schema.text, node_data, source)
            if node.id in self.nodes:
                raise HostTraceError(
                    f"{source}: duplicate node id {node.id} ({self.nodes[node.id].name} and {node.name}); "
                    "node ids must be unique"
                )
            self.nodes[node.id] = node
            lacking = _lacking_keys(node, attrs, tensor_keys)
            if lacking:
                mismatched += 1
                first = first or f"node {node.id} lacks {', '.join(lacking)}"
        if mismatched:
            logger.warning(
                f"{source}: {mismatched} of {len(raw['nodes'])} nodes do not have every key of schema "
                f"{self.schema.text} (first: {first}); reading them anyway"
            )
        _attach_children(self.nodes)

    @overload
    @classmethod
    def from_file(cls, path: str) -> "HostTrace[HostNode]": ...
    @overload
    @classmethod
    def from_file(cls, path: str, node_class: Type[M]) -> "HostTrace[M]": ...

    @classmethod
    def from_file(cls, path: str, node_class: Any = HostNode) -> "HostTrace[Any]":
        """Read and parse a host trace file (plain or gzip-compressed JSON)."""
        return cls(read_trace_json(path), str(path), node_class=node_class)

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


def _lacking_keys(node: HostNode, attrs: Tuple[str, ...], tensor_keys: Tuple[str, ...]) -> List[str]:
    """Return the attributes and inputs/outputs keys of the node's schema version that the node lacks."""
    lacking = [a for a in attrs if a not in node.attrs]
    for side in ("inputs", "outputs"):
        value = getattr(node, side)
        if isinstance(value, dict):
            lacking += [f"{side}.{k}" for k in tensor_keys if k not in value]
    return lacking
