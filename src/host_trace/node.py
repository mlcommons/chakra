from typing import Any, Callable, Dict, Iterable, List, Optional

from .errors import HostTraceError

# Keys every node must have, whatever the schema version.
REQUIRED_KEYS = ("id", "name", "ctrl_deps", "inputs", "outputs")


class HostNode:
    """
    A node of a Chakra host trace (a PyTorch execution trace, raw or linked): its parsed data and its children.

    The node is read the same way for every supported schema version; the version is checked once for the whole
    trace (see versions.py). Every entry of the node's attrs list is kept, unconverted, in the attrs dictionary. The
    attributes in names.CORE_ATTRS are also set as fields, converted (int, or str for op_schema), and None when
    missing: each tool checks the ones it needs. The node holds no tool's state; a tool that needs some subclasses it
    (the converter's ConverterNode) and passes the subclass to HostTrace.

    Attributes
        schema (str): Schema string of the trace the node belongs to (information only).
        source (Optional[str]): File the node was read from, used in error messages.
        id (int): Identifier of the node.
        name (str): Name of the node.
        parent (Any): Id of the parent node (ctrl_deps).
        inputs (Dict): Inputs of the node (values, types, shapes and, from schema 1.1.1, strides).
        outputs (Dict): Outputs of the node, like inputs.
        attrs (Dict[str, Any]): Every attribute of the node, by name, unconverted.
        rf_id, fw_parent, seq_id, scope, tid, fw_tid (Optional[int]): Attributes, converted to int.
        op_schema (Optional[str]): Attribute, converted to str.
        inclusive_dur (Optional[float]): Inclusive duration of the node.
        exclusive_dur (float): Exclusive duration of the node.
        ts (Optional[float]): Timestamp of the node.
        inter_thread_dep (Any): Inter-thread dependency of the node.
        sync_dep (Any): Synchronization dependencies of the node.
        cat (Any): Category of the node (set on GPU nodes).
        stream (int): Stream associated with the node.
        pg_name (str): Process Group name for the inter-GPU communication.
        children (List[HostNode]): Child nodes, sorted by id once the trace's tree is built.
    """

    def __init__(self, schema: str, node_data: Dict[str, Any], source: Optional[str] = None) -> None:
        """
        Initialize a HostNode object from the node data.

        Args:
            schema (str): The schema string of the trace (kept as information; parsing does not depend on it).
            node_data (Dict[str, Any]): Dictionary containing the data of the node.
            source (Optional[str]): File the node comes from, used in error messages.
        """
        self.schema = schema
        self.source = source
        self.children: List["HostNode"] = []
        self._parse(node_data)

    def __repr__(self) -> str:
        """
        Provide a string representation of the HostNode.

        Returns
            str: String representation of the node.
        """
        return (
            f"{type(self).__name__}(id={self.id}, name={self.name}, timestamp={self.ts}, "
            f"inclusive_duration={self.inclusive_dur}, exclusive_duration={self.exclusive_dur})"
        )

    def _parse(self, node_data: Dict[str, Any]) -> None:
        """Parse the node data. Raise HostTraceError for a missing required key or a malformed attribute."""
        prefix = f"{self.source}: " if self.source else ""
        missing = [key for key in REQUIRED_KEYS if key not in node_data]
        if missing:
            raise HostTraceError(f"{prefix}node {node_data.get('id', '(no id)')}: missing key(s) {', '.join(missing)}")

        self.id = node_data["id"]
        self.name = node_data["name"]
        self.parent = node_data["ctrl_deps"]
        # inputs and outputs are the dicts inside HostTrace.raw, shared rather than copied: never write into them.
        self.inputs = node_data["inputs"]
        self.outputs = node_data["outputs"]
        self.inclusive_dur = node_data.get("inclusive_dur")
        self.exclusive_dur = node_data.get("exclusive_dur", 0)
        self.ts = node_data.get("ts")
        self.inter_thread_dep = node_data.get("inter_thread_dep")
        self.sync_dep = node_data.get("sync_dep")
        self.cat = node_data.get("cat")
        self.stream = node_data.get("stream", 0)

        self.attrs: Dict[str, Any] = self._read_attrs(node_data.get("attrs", []))
        # The attributes in names.CORE_ATTRS, converted. Each is None when the node lacks it.
        self.rf_id: Optional[int] = self._converted("rf_id", int)
        self.fw_parent: Optional[int] = self._converted("fw_parent", int)
        self.seq_id: Optional[int] = self._converted("seq_id", int)
        self.scope: Optional[int] = self._converted("scope", int)
        self.tid: Optional[int] = self._converted("tid", int)
        self.fw_tid: Optional[int] = self._converted("fw_tid", int)
        self.op_schema: Optional[str] = self._converted("op_schema", str)
        # pg_name is a top-level key on collective kernels (from Kineto) and an attribute on record_param_comms nodes.
        # The attribute takes precedence when both exist.
        self.pg_name = self.attrs["pg_name"] if "pg_name" in self.attrs else node_data.get("pg_name", "")

    def _read_attrs(self, attrs: Any) -> Dict[str, Any]:
        """Return the attrs list as {name: value}. Raise HostTraceError unless it is a list of name/value entries."""
        if not isinstance(attrs, list):
            raise HostTraceError(f"{self.where()}attrs is a {type(attrs).__name__}, not a list")
        for index, attr in enumerate(attrs):
            if not isinstance(attr, dict) or "name" not in attr or "value" not in attr:
                raise HostTraceError(f"{self.where()}attrs entry {index} is not a name/value entry: {attr!r:.80}")
        return {attr["name"]: attr["value"] for attr in attrs}

    def _converted(self, name: str, convert: Callable[[Any], Any]) -> Any:
        value = self.attrs.get(name)
        if value is None:
            return None
        try:
            return convert(value)
        except (TypeError, ValueError) as e:
            raise HostTraceError(f"{self.where()}attribute {name!r} has the value {value!r} ({e})") from None

    def where(self) -> str:
        """Return the start of an error message about this node: its file (if known), id and name."""
        prefix = f"{self.source}: " if self.source else ""
        return f"{prefix}node {getattr(self, 'id', '?')} ({getattr(self, 'name', '?')}): "

    def require_attrs(self, names: Iterable[str], tool: str) -> None:
        """Raise HostTraceError naming the node if any of these attributes is missing from its attrs list."""
        missing = [name for name in names if name not in self.attrs]
        if missing:
            raise HostTraceError(f"{self.where()}missing attribute(s) {', '.join(missing)}, which {tool} needs")

    def require_tensor_keys(self, keys: Iterable[str], tool: str) -> None:
        """Raise HostTraceError naming the node if its inputs or outputs lack any of these keys."""
        keys = list(keys)
        for side in ("inputs", "outputs"):
            value = getattr(self, side)
            missing = [key for key in keys if not isinstance(value, dict) or key not in value]
            if missing:
                raise HostTraceError(f"{self.where()}{side} has no {', '.join(missing)}, which {tool} needs")

    def is_gpu_op(self) -> bool:
        """Return True if the node has a category: chakra_trace_link sets it on the GPU nodes it adds."""
        return self.cat is not None

    def _add_child(self, child_node: "HostNode") -> None:
        self.children.append(child_node)
