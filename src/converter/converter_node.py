from enum import Enum
from typing import Any, Dict, List, Optional

from ..host_trace.node import HostNode
from ..host_trace.tensor import HostTensor


class ConverterNodeType(Enum):
    """
    The converter's classification of a host-trace node.

    Attributes
        CPU_OP (int): Represents a CPU operation.
        GPU_OP (int): Represents a GPU operation (a node chakra_trace_link added for a kernel: it has a category).
        METADATA (int): Represents a metadata node (e.g., process group initialization).
    """

    CPU_OP = 1
    GPU_OP = 2
    METADATA = 4  # Metadata nodes


class ConverterNode(HostNode):
    """
    A host-trace node with what chakra_converter adds to it: its type, the links it registers, the communication size.

    PyTorchConverter reads traces with HostTrace(..., node_class=ConverterNode). The linker uses plain HostNode. The
    converter writes only these converter-owned attributes, never the parsed fields of HostNode.

    Attributes
        op_type (ConverterNodeType): The node's type, fixed when the node is read.
        gpu_children (List[ConverterNode]): GPU nodes among the children, registered by the converter.
        record_param_comms_node (Optional[ConverterNode]): The record_param_comms child, registered by the converter.
        nccl_node (Optional[ConverterNode]): The NCCL child, registered by the converter.
        comm_pg_name (str): The process group name the .et gets for a communication node. It starts as the parsed
            pg_name; for a SendRecv kernel under a record_param_comms node, the converter sets it to that node's.
    """

    def __init__(self, schema: str, node_data: Dict[str, Any], source: Optional[str] = None) -> None:
        super().__init__(schema, node_data, source)
        self.op_type = self._classify()
        self.gpu_children: List["ConverterNode"] = []
        self.record_param_comms_node: Optional["ConverterNode"] = None
        self.nccl_node: Optional["ConverterNode"] = None
        self.comm_pg_name: str = self.pg_name

    def _classify(self) -> ConverterNodeType:
        """
        Return the node's type. Called once, when the node is read: name and category don't change afterwards.

        METADATA if the name contains "process_group:init", else GPU_OP if the node has a category, else CPU_OP.
        """
        if "process_group:init" in self.name:
            return ConverterNodeType.METADATA
        if self.is_gpu_op():
            return ConverterNodeType.GPU_OP
        return ConverterNodeType.CPU_OP

    def is_metadata_op(self) -> bool:
        """Return True if the node is a METADATA node."""
        return self.op_type is ConverterNodeType.METADATA

    def is_cpu_op(self) -> bool:
        """Return True if the node is a CPU operator."""
        return self.op_type is ConverterNodeType.CPU_OP

    def add_gpu_child(self, gpu_child_node: "ConverterNode") -> None:
        """Register a child GPU node on this node."""
        self.gpu_children.append(gpu_child_node)

    def is_record_param_comms_op(self) -> bool:
        """Return True if the node is a record_param_comms operator."""
        return "record_param_comms" in self.name

    def is_nccl_op(self) -> bool:
        """Return True if the node is an NCCL operator."""
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
