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

    Attributes:
        node_data (Dict[str, Any]): Data of the PyTorch node.
        data_deps (List[PyTorchNode]): List of data-dependent parent nodes.
        children (List[PyTorchNode]): List of child nodes.
    """

    def __init__(self, node_data: Dict[str, Any]) -> None:
        """
        Initializes a PyTorchNode object with the provided node data.

        Args:
            node_data (Dict[str, Any]): Dictionary containing the data of the
            PyTorch node.
        """
        self.node_data = node_data
        self.data_deps: List['PyTorchNode'] = []
        self.children: List['PyTorchNode'] = []
        self.child_gpu: Optional['PyTorchNode'] = None
        self.record_param_comms_node: Optional['PyTorchNode'] = None
        self.nccl_node: Optional['PyTorchNode'] = None

    def __repr__(self) -> str:
        """
        Represent the PyTorchNode as a string.
        Returns:
            str: A detailed string representation of the PyTorchNode.
        """
        return (
            f"PyTorchNode("
            f"id={self.id}, name={self.name}, "
            f"op_type={self.get_op_type()}, "
            f"timestamp={self.ts}, "
            f"inclusive_duration={self.inclusive_dur}, "
            f"exclusive_duration={self.exclusive_dur})"
        )

    @property
    def name(self) -> str:
        """
        Returns the name of the node.

        Returns:
            str: Name of the node.
        """
        return self.node_data["name"]

    @name.setter
    def name(self, value: str) -> None:
        """
        Sets the name of the node.

        Args:
            value (str): The new name of the node.
        """
        self.node_data["name"] = value

    @property
    def id(self) -> int:
        """
        Returns the node ID.

        Returns:
            int: ID of the node.
        """
        return self.node_data["id"]

    @id.setter
    def id(self, value: int) -> None:
        """
        Sets the node ID.

        Args:
            value (int): The new ID of the node.
        """
        self.node_data["id"] = value

    @property
    def rf_id(self) -> int:
        """
        Returns the unique record function ID.

        Returns:
            int: The unique record function ID.
        """
        return self.node_data["rf_id"]

    @rf_id.setter
    def rf_id(self, value: int) -> None:
        """
        Sets the unique record function ID.

        Args:
            value (int): The new unique record function ID.
        """
        self.node_data["rf_id"] = value

    @property
    def parent(self) -> int:
        """
        Returns the parent node ID.

        Returns:
            int: The parent node ID.
        """
        return self.node_data["parent"]

    @parent.setter
    def parent(self, value: int) -> None:
        """
        Sets the parent node ID.

        Args:
            value (int): The new parent node ID.
        """
        self.node_data["parent"] = value

    @property
    def fw_parent(self) -> int:
        """
        Returns the parent node ID from the forward thread.

        Returns:
            int: The parent node ID from the forward thread.
        """
        return self.node_data["fw_parent"]

    @fw_parent.setter
    def fw_parent(self, value: int) -> None:
        """
        Sets the parent node ID from the forward thread.

        Args:
            value (int): The new parent node ID from the forward thread.
        """
        self.node_data["fw_parent"] = value

    @property
    def seq_id(self) -> int:
        """
        Returns the record function sequence ID used to correlate forward and
        backward operators.

        Returns:
            int: The record function sequence ID.
        """
        return self.node_data["seq_id"]

    @seq_id.setter
    def seq_id(self, value: int) -> None:
        """
        Sets the record function sequence ID.

        Args:
            value (int): The new sequence ID.
        """
        self.node_data["seq_id"] = value

    @property
    def scope(self) -> int:
        """
        Returns the record scope.

        Returns:
            int: The record scope.
        """
        return self.node_data["scope"]

    @scope.setter
    def scope(self, value: int) -> None:
        """
        Sets the record scope.

        Args:
            value (int): The new scope value.
        """
        self.node_data["scope"] = value

    @property
    def tid(self) -> int:
        """
        Returns the record function thread ID.

        Returns:
            int: The record function thread ID.
        """
        return self.node_data["tid"]

    @tid.setter
    def tid(self, value: int) -> None:
        """
        Sets the record function thread ID.

        Args:
            value (int): The new thread ID.
        """
        self.node_data["tid"] = value

    @property
    def fw_tid(self) -> int:
        """
        Returns the thread ID of the forward execution thread.

        Returns:
            int: The thread ID of the forward execution thread.
        """
        return self.node_data["fw_tid"]

    @fw_tid.setter
    def fw_tid(self, value: int) -> None:
        """
        Sets the thread ID of the forward execution thread.

        Args:
            value (int): The new forward thread ID.
        """
        self.node_data["fw_tid"] = value

    @property
    def op_schema(self) -> str:
        """
        Returns the PyTorch operator schema.

        Returns:
            str: The PyTorch operator schema.
        """
        return self.node_data["op_schema"]

    @op_schema.setter
    def op_schema(self, value: str) -> None:
        """
        Sets the PyTorch operator schema.

        Args:
            value (str): The new operator schema.
        """
        self.node_data["op_schema"] = value

    @property
    def inputs(self) -> List[Any]:
        """
        Returns the array of input arguments.

        Returns:
            List[Any]: The array of input arguments.
        """
        return self.node_data["inputs"]

    @inputs.setter
    def inputs(self, value: List[Any]) -> None:
        """
        Sets the array of input arguments.

        Args:
            value (List[Any]): The new array of input arguments.
        """
        self.node_data["inputs"] = value

    @property
    def input_shapes(self) -> List[Any]:
        """
        Returns the array of input shapes.

        Returns:
            List[Any]: The array of input shapes.
        """
        return self.node_data["input_shapes"]

    @input_shapes.setter
    def input_shapes(self, value: List[Any]) -> None:
        """
        Sets the array of input shapes.

        Args:
            value (List[Any]): The new array of input shapes.
        """
        self.node_data["input_shapes"] = value

    @property
    def input_types(self) -> List[Any]:
        """
        Returns the array of input types.

        Returns:
            List[Any]: The array of input types.
        """
        return self.node_data["input_types"]

    @input_types.setter
    def input_types(self, value: List[Any]) -> None:
        """
        Sets the array of input types.

        Args:
            value (List[Any]): The new array of input types.
        """
        self.node_data["input_types"] = value

    @property
    def outputs(self) -> List[Any]:
        """
        Returns the array of output arguments.

        Returns:
            List[Any]: The array of output arguments.
        """
        return self.node_data["outputs"]

    @outputs.setter
    def outputs(self, value: List[Any]) -> None:
        """
        Sets the array of output arguments.

        Args:
            value (List[Any]): The new array of output arguments.
        """
        self.node_data["outputs"] = value

    @property
    def output_shapes(self) -> List[Any]:
        """
        Returns the array of output shapes.

        Returns:
            List[Any]: The array of output shapes.
        """
        return self.node_data["output_shapes"]

    @output_shapes.setter
    def output_shapes(self, value: List[Any]) -> None:
        """
        Sets the array of output shapes.

        Args:
            value (List[Any]): The new array of output shapes.
        """
        self.node_data["output_shapes"] = value

    @property
    def output_types(self) -> List[Any]:
        """
        Returns the array of output types.

        Returns:
            List[Any]: The array of output types.
        """
        return self.node_data["output_types"]

    @output_types.setter
    def output_types(self, value: List[Any]) -> None:
        """
        Sets the array of output types.

        Args:
            value (List[Any]): The new array of output types.
        """
        self.node_data["output_types"] = value

    @property
    def ts(self) -> int:
        """
        Returns the timestamp of the node.

        Returns:
            int: The timestamp of the node.
        """
        return self.node_data.get("ts", 0)

    @ts.setter
    def ts(self, value: int) -> None:
        """
        Sets the timestamp of the node.

        Args:
            value (int): The new timestamp of the node.
        """
        self.node_data["ts"] = value

    @property
    def cat(self) -> str:
        """
        Returns the category field of the node.

        Returns:
            str: The category field of the node.
        """
        return self.node_data.get("cat", "")

    @cat.setter
    def cat(self, value: str) -> None:
        """
        Sets the category field of the node.

        Args:
            value (str): The new category field of the node.
        """
        self.node_data["cat"] = value

    @property
    def inclusive_dur(self) -> int:
        """
        Returns the inclusive duration of the node.

        Returns:
            int: The inclusive duration of the node.
        """
        return self.node_data["inclusive_dur"]

    @inclusive_dur.setter
    def inclusive_dur(self, value: int) -> None:
        """
        Sets the inclusive duration of the node.

        Args:
            value (int): The new inclusive duration of the node.
        """
        self.node_data["inclusive_dur"] = value

    @property
    def exclusive_dur(self) -> int:
        """
        Returns the exclusive duration of the node.

        Returns:
            int: The exclusive duration of the node.
        """
        return self.node_data.get("exclusive_dur", 0)

    @exclusive_dur.setter
    def exclusive_dur(self, value: int) -> None:
        """
        Sets the exclusive duration of the node.

        Args:
            value (int): The new exclusive duration of the node.
        """
        self.node_data["exclusive_dur"] = value

    @property
    def inter_thread_dep(self) -> Optional[int]:
        """
        Returns the inter-thread dependency value of the node, if available.

        Returns:
            Optional[int]: The inter-thread dependency value or None if not
                           available.
        """
        return self.node_data.get("inter_thread_dep")

    @property
    def sync_dep(self) -> Optional[List[int]]:
        """
        Returns the synchronization dependency value of the node, if available.

        Returns:
            Optional[int]: The synchronization dependency value or None if not
                           available.
        """
        return self.node_data.get("sync_dep")

    @property
    def stream(self) -> int:
        return self.node_data["stream"]

    def has_ts(self) -> bool:
        """
        Checks if the node has a timestamp field.

        Returns:
            bool: True if the node has a timestamp field, False otherwise.
        """
        return "ts" in self.node_data

    def has_cat(self) -> bool:
        """
        Checks if the node has a category field.

        Returns:
            bool: True if the node has a category field, False otherwise.
        """
        return "cat" in self.node_data

    def has_dur(self) -> bool:
        """
        Checks if the node has a duration field.

        Returns:
            bool: True if the node has a duration field, False otherwise.
        """
        return "inclusive_dur" in self.node_data

    def get_op_type(self) -> PyTorchNodeType:
        """
        Determines the type of PyTorch operation.

        Returns:
            PyTorchNodeType: The type of the PyTorch operation.
        """
        if self.is_gpu_op():
            return PyTorchNodeType.GPU_OP
        elif self.node_data.get("op_schema") or self.node_data.get("outputs"):
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

    def set_child_gpu(self, child_gpu_node: Optional['PyTorchNode']) -> None:
        """
        Sets a child GPU node for this node.

        Args:
            child_gpu_node (Optional[PyTorchNode]): The child GPU node to be set.
        """
        self.child_gpu = child_gpu_node

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
            "Tensor(unsigned char)": 1,
            "Tensor(long int)": 8,
            # TODO: Add more types
        }
        try:
            return data_type_size_map[data_type]
        except KeyError:
            raise ValueError(f"Unsupported data type: {data_type}")
