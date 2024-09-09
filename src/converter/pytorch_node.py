import traceback
from enum import Enum
from typing import Any, Dict, List, Optional

from .pytorch_tensor import PyTorchTensor


class PyTorchNodeType(Enum):
    """
    Enum representing the type of a PyTorch node in an execution trace.

    Attributes
        CPU_OP (int): Represents a CPU operation.
        GPU_OP (int): Represents a GPU operation.
        LABEL (int): Represents a non-operator node (e.g., labels).
        METADATA (int): Represents a metadata node (e.g., process group initialization).
    """

    CPU_OP = 1
    GPU_OP = 2
    LABEL = 3  # Non-operator nodes
    METADATA = 4 # Metadata nodes


class PyTorchNode:
    """
    Represents a node in a PyTorch execution trace, initialized based on a schema version.

    Attributes
        schema (str): Schema version used for initialization.
        data_deps (List[PyTorchNode]): List of data-dependent parent nodes.
        children (List[PyTorchNode]): List of child nodes.
        gpu_children (List[PyTorchNode]): List of GPU-specific child nodes.
        record_param_comms_node (Optional[PyTorchNode]): Corresponding record_param_comms node.
        nccl_node (Optional[PyTorchNode]): Corresponding NCCL node.
        id (str): Identifier of the node.
        name (str): Name of the node.
        parent (Any): Parent of the node.
        inputs (Any): Inputs of the node.
        outputs (Any): Outputs of the node.
        inclusive_dur (Optional[float]): Inclusive duration of the node.
        exclusive_dur (float): Exclusive duration of the node.
        ts (Optional[float]): Timestamp of the node.
        inter_thread_dep (Any): Inter-thread dependency of the node.
        cat (Any): Category of the node.
        stream (int): Stream associated with the node.
        pg_name (str): Process Group name for the inter-GPU communication.
    """

    SUPPORTED_VERSIONS = ["1.0.2-chakra.0.0.4", "1.0.3-chakra.0.0.4", "1.1.0-chakra.0.0.4"]

    def __init__(self, schema: str, node_data: Dict[str, Any]) -> None:
        """
        Initialize a PyTorchNode object using the node data and schema version provided.

        Args:
            schema (str): The schema version based on which the node will be initialized.
            node_data (Dict[str, Any]): Dictionary containing the data of the PyTorch node.
        """
        self.schema = schema
        self.data_deps: List["PyTorchNode"] = []
        self.children: List["PyTorchNode"] = []
        self.gpu_children: List["PyTorchNode"] = []
        self.record_param_comms_node: Optional["PyTorchNode"] = None
        self.nccl_node: Optional["PyTorchNode"] = None

        self.parse_data(node_data)

    def __repr__(self) -> str:
        """
        Provide a string representation of the PyTorchNode.

        Returns
            str: String representation of the node.
        """
        return (
            f"PyTorchNode(id={self.id}, name={self.name}, op_type={self.get_op_type()}, timestamp={self.ts}, "
            f"inclusive_duration={self.inclusive_dur}, exclusive_duration={self.exclusive_dur})"
        )

    def parse_data(self, node_data: Dict[str, Any]) -> None:
        """
        Parse node data based on the provided schema version.

        Args:
            node_data (Dict[str, Any]): The node data to be parsed.
        """
        if self.schema in self.SUPPORTED_VERSIONS:
            if self.schema in ["1.0.2-chakra.0.0.4", "1.0.3-chakra.0.0.4", "1.1.0-chakra.0.0.4"]:
                self._parse_data_1_0_3_chakra_0_0_4(node_data)
        else:
            raise ValueError(
                f"Unsupported schema version '{self.schema}'. Please check if the schema version is in the list of "
                f"supported versions: {self.SUPPORTED_VERSIONS}. The schema version of the trace is not supported by "
                f"the converter. The schema version is determined by the PyTorch version used to collect Chakra host "
                f"execution traces. Please consider changing the PyTorch version you are using. For more details, you "
                f"can follow the git history of the relevant file: "
                f"https://github.com/pytorch/pytorch/blob/7cd48df2dae7e2194438b162968c47d1f05bf20e/torch/csrc/"
                f"profiler/standalone/execution_trace_observer.cpp#L308. Check which PyTorch versions generate Chakra "
                f"host traces that are supported by the converter."
            )

    def _parse_data_1_0_3_chakra_0_0_4(self, node_data: Dict[str, Any]) -> None:
        self.id = node_data["id"]
        self.name = node_data["name"]
        self.parent = node_data["ctrl_deps"]
        self.inputs = node_data["inputs"]
        self.outputs = node_data["outputs"]
        self.inclusive_dur = node_data.get("inclusive_dur")
        self.exclusive_dur = node_data.get("exclusive_dur", 0)
        self.ts = node_data.get("ts")
        self.inter_thread_dep = node_data.get("inter_thread_dep")
        self.cat = node_data.get("cat")
        self.stream = node_data.get("stream", 0)
        # In Colletive comms nodes, pg_name is in node_data if exists.
        # In SendRecv nodes, pg_name is in the attrs if exists.
        # Otherwise, pg_name is not present.
        self.pg_name = node_data.get("pg_name", "")

        for attr in node_data.get("attrs", []):
            setattr(self, attr["name"], attr["value"])

    def get_op_type(self) -> PyTorchNodeType:
        """
        Determine the type of PyTorch operation.

        Returns
            PyTorchNodeType: The type of the PyTorch operation.
        """
        if "process_group:init" in self.name:
            return PyTorchNodeType.METADATA
        elif self.is_gpu_op():
            return PyTorchNodeType.GPU_OP
        elif hasattr(self, "op_schema") or hasattr(self, "outputs"):
            return PyTorchNodeType.CPU_OP
        else:
            return PyTorchNodeType.LABEL

    def is_cpu_op(self) -> bool:
        """
        Check if the node is a CPU operator.

        Returns
            bool: True if the node is a CPU operator, False otherwise.
        """
        return self.get_op_type() == PyTorchNodeType.CPU_OP

    def is_gpu_op(self) -> bool:
        """
        Check if the node is a GPU operator.

        Returns
            bool: True if the node is a GPU operator, False otherwise.
        """
        return self.cat is not None

    def add_data_dep(self, parent_node: "PyTorchNode") -> None:
        """
        Add a data-dependent parent node to this node.

        Args:
            parent_node (PyTorchNode): The parent node to be added.
        """
        self.data_deps.append(parent_node)

    def add_child(self, child_node: "PyTorchNode") -> None:
        """
        Add a child node to this node.

        Args:
            child_node (PyTorchNode): The child node to be added.
        """
        self.children.append(child_node)

    def add_gpu_child(self, gpu_child_node: "PyTorchNode") -> None:
        """
        Add a child GPU node for this node.

        Args:
            gpu_child_node (Optional[PyTorchNode]): The child GPU node to be added.
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
        for input_value, input_type in zip(self.inputs["values"], self.inputs["types"]):
            if "Tensor" in input_type:
                if input_type.startswith("GenericList[Tensor"):
                    for inner_value in input_value:
                        tensor = PyTorchTensor(inner_value)
                        input_size = tensor.num_elem * tensor.elem_bytes
                        comm_size += input_size
                else:
                    tensor = PyTorchTensor(input_value)
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
