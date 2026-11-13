import traceback
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional

from .errors import HostTraceError
from .tensor import HostTensor

# Keys every node must have, whatever the schema version.
REQUIRED_KEYS = ("id", "name", "ctrl_deps", "inputs", "outputs")

# Attributes (from a node's attrs list) that Chakra reads, with their conversion. Each is None when the node lacks it.
TYPED_ATTRS: Dict[str, Callable[[Any], Any]] = {
    "rf_id": int,
    "fw_parent": int,
    "seq_id": int,
    "scope": int,
    "tid": int,
    "fw_tid": int,
    "op_schema": str,
}


class HostNodeType(Enum):
    """
    Enum representing the type of a node in a Chakra host trace.

    Attributes
        CPU_OP (int): Represents a CPU operation.
        GPU_OP (int): Represents a GPU operation.
        LABEL (int): Represents a non-operator node (e.g., labels).
        METADATA (int): Represents a metadata node (e.g., process group initialization).
    """

    CPU_OP = 1
    GPU_OP = 2
    LABEL = 3  # Non-operator nodes
    METADATA = 4  # Metadata nodes


class HostNode:
    """
    Represents a node in a Chakra host trace (a PyTorch execution trace, raw or linked).

    The node is read the same way for every supported schema version; the version is checked once for the whole
    trace (see versions.py). Every entry of the node's attrs list is kept, unconverted, in the attrs dictionary. The
    attributes Chakra reads are also set as fields, converted (TYPED_ATTRS), and None when missing: each tool checks
    the ones it needs.

    Attributes
        schema (str): Schema string of the trace the node belongs to (information only).
        source (Optional[str]): File the node was read from, used in error messages.
        data_deps (List[HostNode]): List of data-dependent parent nodes.
        children (List[HostNode]): List of child nodes, sorted by id once the trace's tree is built.
        gpu_children (List[HostNode]): List of GPU-specific child nodes (registered by the converter).
        record_param_comms_node (Optional[HostNode]): Corresponding record_param_comms node (converter-registered).
        nccl_node (Optional[HostNode]): Corresponding NCCL node (converter-registered).
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
        cat (Any): Category of the node.
        stream (int): Stream associated with the node.
        pg_name (str): Process Group name for the inter-GPU communication.
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
        self.data_deps: List["HostNode"] = []
        self.children: List["HostNode"] = []
        self.gpu_children: List["HostNode"] = []
        self.record_param_comms_node: Optional["HostNode"] = None
        self.nccl_node: Optional["HostNode"] = None

        self.parse_data(node_data)

    def __repr__(self) -> str:
        """
        Provide a string representation of the HostNode.

        Returns
            str: String representation of the node.
        """
        return (
            f"HostNode(id={self.id}, name={self.name}, op_type={self.get_op_type()}, timestamp={self.ts}, "
            f"inclusive_duration={self.inclusive_dur}, exclusive_duration={self.exclusive_dur})"
        )

    def parse_data(self, node_data: Dict[str, Any]) -> None:
        """
        Parse the node data.

        Args:
            node_data (Dict[str, Any]): The node data to be parsed.

        Raises:
            HostTraceError: If a required key is missing, or a read attribute cannot be converted.
        """
        missing = [key for key in REQUIRED_KEYS if key not in node_data]
        if missing:
            where = f"{self.source}: " if self.source else ""
            raise HostTraceError(f"{where}node {node_data.get('id', '(no id)')}: missing key(s) {', '.join(missing)}")

        self.id = node_data["id"]
        self.name = node_data["name"]
        self.parent = node_data["ctrl_deps"]
        self.inputs = node_data["inputs"]
        self.outputs = node_data["outputs"]
        self.inclusive_dur = node_data.get("inclusive_dur")
        self.exclusive_dur = node_data.get("exclusive_dur", 0)
        self.ts = node_data.get("ts")
        self.inter_thread_dep = node_data.get("inter_thread_dep")
        self.sync_dep = node_data.get("sync_dep")
        self.cat = node_data.get("cat")
        self.stream = node_data.get("stream", 0)

        self.attrs: Dict[str, Any] = {attr["name"]: attr["value"] for attr in node_data.get("attrs", [])}
        for name, convert in TYPED_ATTRS.items():
            setattr(self, name, self._converted(name, convert))
        # pg_name is a top-level key on collective kernels (from Kineto) and an attribute on record_param_comms nodes.
        # The attribute takes precedence when both exist.
        self.pg_name = self.attrs["pg_name"] if "pg_name" in self.attrs else node_data.get("pg_name", "")

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
        return f"{self.source + ': ' if self.source else ''}node {self.id} ({self.name}): "

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

    def get_op_type(self) -> HostNodeType:
        """
        Determine the type of PyTorch operation.

        Returns
            HostNodeType: The type of the PyTorch operation.
        """
        if "process_group:init" in self.name:
            return HostNodeType.METADATA
        elif self.is_gpu_op():
            return HostNodeType.GPU_OP
        elif hasattr(self, "op_schema") or hasattr(self, "outputs"):
            return HostNodeType.CPU_OP
        else:
            return HostNodeType.LABEL

    def is_metadata_op(self) -> bool:
        """
        Check if the node is a METADATA operator.

        Returns
            bool: True if the node is a METADATA operator, False otherwise.
        """
        return self.get_op_type() == HostNodeType.METADATA

    def is_cpu_op(self) -> bool:
        """
        Check if the node is a CPU operator.

        Returns
            bool: True if the node is a CPU operator, False otherwise.
        """
        return self.get_op_type() == HostNodeType.CPU_OP

    def is_gpu_op(self) -> bool:
        """
        Check if the node is a GPU operator.

        Returns
            bool: True if the node is a GPU operator, False otherwise.
        """
        return self.cat is not None

    def add_data_dep(self, parent_node: "HostNode") -> None:
        """
        Add a data-dependent parent node to this node.

        Args:
            parent_node (HostNode): The parent node to be added.
        """
        self.data_deps.append(parent_node)

    def add_child(self, child_node: "HostNode") -> None:
        """
        Add a child node to this node.

        Args:
            child_node (HostNode): The child node to be added.
        """
        self.children.append(child_node)

    def add_gpu_child(self, gpu_child_node: "HostNode") -> None:
        """
        Add a child GPU node for this node.

        Args:
            gpu_child_node (Optional[HostNode]): The child GPU node to be added.
        """
        self.gpu_children.append(gpu_child_node)

    def is_record_param_comms_op(self) -> bool:
        """
        Check if the node is a record_param_comms operator.

        Returns
            bool: True if the node is a record_param_comms operator, False otherwise.
        """
        return "record_param_comms" in self.name

    def is_nccl_op(self) -> bool:
        """
        Check if the node is a NCCL operator.

        Returns
            bool: True if the node is a NCCL operator, False otherwise.
        """
        return "nccl:" in self.name

    @property
    def comm_size(self) -> int:
        """
        Calculate the communication size for the given input types and shapes.

        Returns
            int: The calculated communication size.
        """
        comm_size = 0
        for input_value, input_type in zip(self.inputs["values"], self.inputs["types"], strict=False):
            if "Tensor" in input_type:
                if input_type.startswith("GenericList[Tensor"):
                    for inner_value in input_value:
                        tensor = HostTensor(inner_value)
                        input_size = tensor.num_elem * tensor.elem_bytes
                        comm_size += input_size
                else:
                    tensor = HostTensor(input_value)
                    input_size = tensor.num_elem * tensor.elem_bytes
                    comm_size += input_size
        return comm_size

    @staticmethod
    def get_data_type_size(data_type: str) -> int:
        """
        Return the data type size of a given data type in string.

        Args:
            data_type (str): The data type as a string.

        Returns:
            int: The size of the data type in bytes.

        Raises:
            ValueError: If the data type is not supported.
        """
        data_type_size_map = {
            "Tensor(float32)": 4,
            "Tensor(float)": 4,
            "Tensor(float64)": 8,
            "Tensor(double)": 8,
            "Tensor(float16)": 2,
            "Tensor(half)": 2,
            "Tensor(bfloat16)": 2,
            "Tensor(complex64)": 8,
            "Tensor(complex128)": 16,
            "Tensor(uint8)": 1,
            "Tensor(int8)": 1,
            "Tensor(int16)": 2,
            "Tensor(short)": 2,
            "Tensor(int32)": 4,
            "Tensor(int)": 4,
            "Tensor(int64)": 8,
            "Tensor(long)": 8,
            "Tensor(c10::Half)": 2,
            "Tensor(c10::BFloat16)": 2,
            "Tensor(unsigned char)": 1,
            "Tensor(long int)": 8,
            # TODO: Add more types
        }
        try:
            return data_type_size_map[data_type]
        except KeyError as e:
            traceback_str = traceback.format_exc()
            raise ValueError(
                f"Unsupported data type: {data_type}. The data_type_size_map dictionary is used for mapping the "
                f"number of bytes for a given tensor data type. This dictionary may be incomplete. Please update the "
                f"data_type_size_map or report this issue to the maintainer by creating an issue. Traceback:\n"
                f"{traceback_str}"
            ) from e
