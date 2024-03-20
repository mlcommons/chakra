#!/usr/bin/env python3

from enum import Enum
from typing import Any, Dict, List, Optional


class PyTorchNodeType(Enum):
    CPU_OP = 1
    GPU_OP = 2
    LABEL = 3  # Non-operator nodes


class PyTorchNode:
    """
    Represents a node in a PyTorch execution trace.

    This class initializes node fields from provided data based on a specific
    schema version and provides functionality to access various properties
    of the node.

    Attributes are dynamically initialized based on schema version
    "1.0.2-chakra.0.0.4".

    Raises:
        KeyError: If a required field is missing in the provided node data.
        ValueError: If an unsupported schema version is provided.
    """

    def __init__(self, schema_version: str, node_data: Dict[str, Any]) -> None:
        self.schema_version = schema_version
        if schema_version == "1.0.1":
            self.parse_1_0_1(node_data)
        elif schema_version == "1.0.2-chakra.0.0.4":
            self.parse_1_0_2_chakra_0_0_4(node_data)
        else:
            raise ValueError(f"Unsupported schema version: {schema_version}")
        self.data_deps: List['PyTorchNode'] = []
        self.children: List['PyTorchNode'] = []
        self.gpu_children: List['PyTorchNode'] = []
        self.record_param_comms_node: Optional['PyTorchNode'] = None
        self.nccl_node: Optional['PyTorchNode'] = None

    def parse_1_0_1(self, node_data: Dict[str, Any]):
        """
        Parses node data according to the "1.0.2-chakra.0.0.4" schema version.

        Args:
            node_data (Dict[str, Any]): The data dictionary for the node.
        """
        required_fields = [
            "name", "id", "rf_id", "parent", "fw_parent", "seq_id", "scope",
            "tid", "fw_tid", "op_schema", "inputs", "input_shapes",
            "input_types", "outputs", "output_shapes", "output_types"
        ]
        optional_fields = [
            "ts", "cat", "inclusive_dur", "exclusive_dur", "inter_thread_dep",
            "stream"
        ]

        for field in required_fields:
            if field not in node_data:
                raise KeyError(f"Field '{field}' is missing in node data "
                               f"for schema version {self.schema_version}")

        for field in required_fields + optional_fields:
            setattr(self, field, node_data.get(field))

    def parse_1_0_2_chakra_0_0_4(self, node_data: Dict[str, Any]):
        """
        Parses node data according to the "1.0.2-chakra.0.0.4" schema version.

        This version expects several node fields to be encapsulated within the 'attrs'
        field of the node data, and handles 'inputs' and 'outputs' as dictionaries
        containing 'values', 'shapes', and 'types'.

        Args:
            node_data (Dict[str, Any]): The data dictionary for the node.
        """
        # Directly set attributes from the node data
        self.id = node_data['id']
        self.name = node_data['name']

        # Handling 'ctrl_deps' as a direct attribute if present
        if 'ctrl_deps' in node_data:
            self.parent = node_data['ctrl_deps']

        # Process 'inputs' and 'outputs' dictionaries
        self.inputs = node_data['inputs']['values']
        self.input_shapes = node_data['inputs']['shapes']
        self.input_types = node_data['inputs']['types']
        self.outputs = node_data['outputs']['values']
        self.output_shapes = node_data['outputs']['shapes']
        self.output_types = node_data['outputs']['types']

        # Process attributes from 'attrs'
        for attr in node_data['attrs']:
            setattr(self, attr['name'], attr['value'])

        optional_fields = [
            "ts", "cat", "inclusive_dur", "exclusive_dur", "inter_thread_dep",
            "stream"
        ]

        for field in optional_fields:
            if field in node_data:
                setattr(self, field, node_data.get(field))

    def __repr__(self) -> str:
        """
        Represent the PyTorchNode as a string.
        Returns:
            str: A detailed string representation of the PyTorchNode.
        """
        return (
            f"PyTorchNode("
            f"id={self.id}, name={self.name}, "
            f"op_type={self.get_op_type()}"
        )

    def has_ts(self) -> bool:
        """
        Checks if the node has a timestamp field.

        Returns:
            bool: True if the node has a timestamp field, False otherwise.
        """
        return hasattr(self, 'ts')

    def has_cat(self) -> bool:
        """
        Checks if the node has a category field.

        Returns:
            bool: True if the node has a category field, False otherwise.
        """
        return hasattr(self, 'cat')

    def has_dur(self) -> bool:
        """
        Checks if the node has a duration field.

        Returns:
            bool: True if the node has a duration field, False otherwise.
        """
        return hasattr(self, 'inclusive_dur')

    def get_op_type(self) -> PyTorchNodeType:
        """
        Determines the type of PyTorch operation.

        Returns:
            PyTorchNodeType: The type of the PyTorch operation.
        """
        if self.is_gpu_op():
            return PyTorchNodeType.GPU_OP
        elif self.op_schema or self.outputs:
            return PyTorchNodeType.CPU_OP
        else:
            return PyTorchNodeType.LABEL

    def is_cpu_op(self) -> bool:
        """
        Checks if the node is a CPU operator.

        Returns:
            bool: True if the node is a CPU operator, False otherwise.
        """
        return self.get_op_type() == PyTorchNodeType.CPU_OP

    def is_gpu_op(self) -> bool:
        """
        Checks if the node is a GPU operator.

        Returns:
            bool: True if the node is a GPU operator, False otherwise.
        """
        return self.has_cat()

    def add_data_dep(self, parent_node: 'PyTorchNode') -> None:
        """
        Adds a data-dependent parent node to this node.

        Args:
            parent_node (PyTorchNode): The parent node to be added.
        """
        self.data_deps.append(parent_node)

    def add_child(self, child_node: 'PyTorchNode') -> None:
        """
        Adds a child node to this node.

        Args:
            child_node (PyTorchNode): The child node to be added.
        """
        self.children.append(child_node)

    def add_gpu_child(self, gpu_child_node: 'PyTorchNode') -> None:
        """
        Adds a child GPU node for this node.

        Args:
            gpu_child_node (Optional[PyTorchNode]): The child GPU node to be added.
        """
        self.gpu_children.append(gpu_child_node)

    def is_record_param_comms_op(self) -> bool:
        """
        Checks if the node is a record_param_comms operator.

        Returns:
            bool: True if the node is a record_param_comms operator, False otherwise.
        """
        return "record_param_comms" in self.name

    def is_nccl_op(self) -> bool:
        """
        Checks if the node is a NCCL operator.

        Returns:
            bool: True if the node is a NCCL operator, False otherwise.
        """
        return "nccl:" in self.name

    @property
    def comm_size(self) -> int:
        """
        Calculates the communication size for the given input types and shapes.

        Returns:
            int: The calculated communication size.
        """
        comm_size = 1
        for input_type, input_shape in zip(self.input_types, self.input_shapes):
            type_size = self.get_data_type_size(input_type)
            shape_size = 1
            for dim in input_shape:
                shape_size *= dim
            comm_size += type_size * shape_size
        return comm_size

    @staticmethod
    def get_data_type_size(data_type: str) -> int:
        """
        Returns the data type size of a given data type in string.

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
        except KeyError:
            raise ValueError(f"Unsupported data type: {data_type}")
