#!/usr/bin/env python3

import bisect
import copy
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from chakra.third_party.utils.protolib import encodeMessage as encode_message
from chakra.et_converter.pytorch_node import PyTorchNodeType, PyTorchNode
from chakra.et_converter.pytorch_tensor import PyTorchTensor, list_to_pytorch_tensor
from chakra.et_def.et_def_pb2 import (
    GlobalMetadata,
    Node as ChakraNode,
    AttributeProto as ChakraAttr,
    INVALID_NODE,
    COMP_NODE,
    COMM_COLL_NODE,
    ALL_REDUCE,
    ALL_GATHER,
    BROADCAST,
    ALL_TO_ALL,
    REDUCE_SCATTER,
)


class UniqueIdAssigner:
    """
    Class for assigning unique IDs. Generates a new unique ID for each call,
    even with the same original ID, and keeps track of all assigned IDs.

    Attributes:
        next_id (int): The next available unique ID.
        original_to_assigned_ids (Dict[int, List[int]]): Mapping from original
            IDs to lists of assigned unique IDs.
    """

    def __init__(self) -> None:
        self.next_id = 0
        self.original_to_assigned_ids: Dict[int, List[int]] = {}

    def assign_unique_id(self, original_id: int) -> int:
        """
        Generates and tracks a new unique ID for each call for a given original ID.

        Args:
            original_id (int): The original ID to generate a unique ID for.

        Returns:
            int: A new unique ID for the original ID.
        """
        unique_id = self.next_id
        self.next_id += 1

        assigned_ids = self.original_to_assigned_ids.setdefault(original_id, [])
        assigned_ids.append(unique_id)

        return unique_id

    def get_assigned_ids(self, original_id: int) -> List[int]:
        """
        Retrieves all unique IDs assigned to a given original ID.

        Args:
            original_id (int): The original ID to retrieve unique IDs for.

        Returns:
            List[int]: List of unique IDs assigned to the original ID.
        """
        return self.original_to_assigned_ids.get(original_id, [])


class PyTorch2ChakraConverter:
    """
    Converter class for transforming PyTorch execution traces into Chakra format.

    This class is responsible for converting the execution traces collected
    from PyTorch into a format that is compatible with Chakra, a performance
    analysis tool. It handles the intricate mappings and transformations
    required to accurately represent the execution in a different format.

    Attributes:
        input_filename (str): Input file name containing PyTorch execution trace.
        output_filename (str): Output file name for the converted Chakra trace.
        num_dims (int): Number of dimensions involved in the conversion process.
        logger (logging.Logger): Logger for logging information during conversion.
        id_assigner (UniqueIdAssigner): Object to manage unique ID assignments.
        pytorch_schema (Optional[str]): Schema info of the PyTorch trace.
        pytorch_pid (Optional[int]): Process ID associated with the PyTorch trace.
        pytorch_time (Optional[str]): Time info of the PyTorch trace.
        pytorch_start_ts (Optional[int]): Start timestamp of the PyTorch trace.
        pytorch_finish_ts (Optional[int]): Finish timestamp of the PyTorch trace.
        pytorch_nodes (Dict[int, Any]): Map of PyTorch node IDs to nodes.
        pytorch_root_nids (List[int]): List of root node IDs in the PyTorch trace.
        pytorch_cpu_node_id_gpu_node_map (Dict[int, List[int]]): Map of PyTorch
            CPU node IDs to GPU node IDs.
        chakra_nodes (Dict[int, Any]): Map of Chakra node IDs to nodes.
        phase_end_nids (List[int]): List of node IDs for phase dependencies.
        input_storage_id_nid_map (Dict[int, int]): Map of input storage IDs to node IDs.
        output_storage_id_nid_map (Dict[int, int]): Map of output storage IDs to node IDs.
        input_tensor_id_nid_map (Dict[int, int]): Map of input tensor IDs to node IDs.
        output_tensor_id_nid_map (Dict[int, int]): Map of output tensor IDs to node IDs.
    """

    def __init__(
        self,
        input_filename: str,
        output_filename: str,
        num_dims: int,
        logger: logging.Logger
    ) -> None:
        """
        Initializes the PyTorch to Chakra converter. It sets up necessary
        attributes and prepares the environment for the conversion process.

        Args:
            input_filename (str): Name of the input file containing PyTorch execution trace.
            output_filename (str): Name of the output file for the converted Chakra trace.
            num_dims (int): Number of dimensions involved in the conversion process.
            logger (logging.Logger): Logger for logging information during the conversion.
        """
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_dims = num_dims
        self.logger = logger
        self.id_assigner = UniqueIdAssigner()
        self.initialize_attributes()

    def initialize_attributes(self) -> None:
        # Initialize file and trace-related attributes
        self.pytorch_schema = None
        self.pytorch_pid = None
        self.pytorch_time = None
        self.pytorch_start_ts = None
        self.pytorch_finish_ts = None
        self.pytorch_nodes = None
        self.pytorch_root_nids = []

        # Initialize node mapping dictionaries
        self.pytorch_cpu_node_id_gpu_node_map = {}
        self.chakra_nodes = {}

        # Initialize lists for phase dependencies and data dependency maps
        self.phase_end_nids = []

        # Map of input storage IDs to node IDs:
        # This dictionary tracks which nodes are consuming tensors based on their
        # storage ID, establishing a link between tensor storage and node consumption.
        self.input_storage_id_nid_map = {}

        # Map of output storage IDs to node IDs:
        # Similar to input_storage_id_nid_map, but this tracks the production of
        # tensors by nodes, associating tensor storage IDs with the nodes that
        # produce them.
        self.output_storage_id_nid_map = {}

        # Map of input tensor IDs to node IDs:
        # This dictionary is used when storage IDs are not applicable. It tracks
        # which nodes are consuming tensors by using tensor IDs, creating a link
        # between tensor IDs and the nodes that consume them.
        self.input_tensor_id_nid_map = {}

        # Map of output tensor IDs to node IDs:
        # Similar to input_tensor_id_nid_map, but for tracking the output of tensors
        # from nodes. It associates tensor IDs with the nodes that output them,
        # used when storage IDs are not available.
        self.output_tensor_id_nid_map = {}

    def convert(self) -> None:
        """
        Converts PyTorch execution traces into the Chakra format. Orchestrates
        the conversion process including trace loading, trace opening, phase
        end node construction, node splitting, and node conversion.
        """
        self.load_pytorch_execution_traces()

        self.open_chakra_execution_trace()

        self.construct_phase_end_nids()

        self.split_cpu_nodes_with_gpu_child()

        for pytorch_nid, pytorch_node in self.pytorch_nodes.items():
            if pytorch_node.is_cpu_op():
                self.update_input_tensor_map(pytorch_node.id, pytorch_node.inputs)
                self.update_output_tensor_map(pytorch_node.id, pytorch_node.outputs)

                if pytorch_node.child_gpu:
                    pytorch_gpu_node = pytorch_node.child_gpu
                    self.update_input_tensor_map(pytorch_gpu_node.id, pytorch_gpu_node.inputs)
                    # Ignoring GPU->CPU dependencies for now since it creates unwanted dependencies.

                chakra_node = self.convert_to_chakra_node(pytorch_node)
                self.chakra_nodes[chakra_node.id] = chakra_node

                if pytorch_node.child_gpu:
                    pytorch_gpu_node = pytorch_node.child_gpu
                    chakra_gpu_node = self.convert_to_chakra_node(pytorch_gpu_node)

                    if chakra_node.type == COMM_COLL_NODE:
                        pytorch_nccl_node = self.get_nccl_node(pytorch_node)
                        chakra_gpu_node.attr.extend([
                            ChakraAttr(name="comm_type",
                                       int64_val=pytorch_nccl_node.collective_comm_type),
                            ChakraAttr(name="comm_size",
                                       int64_val=pytorch_nccl_node.comm_size),
                            ChakraAttr(name="involved_dim",
                                       bool_list={"values": [True]*self.num_dims})
                        ])

                    chakra_gpu_node.data_deps.append(chakra_node.id)
                    self.chakra_nodes[chakra_gpu_node.id] = chakra_gpu_node

                for data_dep_pytorch_node in pytorch_node.data_deps:
                    chakra_node.data_deps.append(data_dep_pytorch_node.id)

                dep_nid = self.get_prev_phase_end_nid(chakra_node)
                if (dep_nid != -1) and (dep_nid not in chakra_node.data_deps):
                    chakra_node.data_deps.append(dep_nid)
>>>>>>> a4155fe (et_converter: Refactor PyTorch2ChakraConverter)

        self.identify_data_dependency()

        self.write_chakra_et()

        self.close_chakra_execution_trace()

    def load_pytorch_execution_traces(self) -> None:
        """
        Loads PyTorch execution traces from a file.

        Reads and parses the PyTorch execution trace data from a file, creating
        PyTorchNode objects and establishing node relationships.

        Raises:
            Exception: If there is an IOError in opening the file.
        """
        self.logger.info("Loading PyTorch execution traces from file.")
        try:
            with open(self.input_filename, "r") as pytorch_et:
                pytorch_et_data = json.load(pytorch_et)
            self._parse_and_instantiate_nodes(pytorch_et_data)
        except IOError as e:
            self.logger.error(f"Error opening file {self.input_filename}: {e}")
            raise Exception(f"Could not open file {self.input_filename}")

    def _parse_and_instantiate_nodes(self, pytorch_et_data: Dict) -> None:
        """
        Parses and instantiates PyTorch nodes from execution trace data.

        Args:
            pytorch_et_data (Dict): The execution trace data.

        Extracts node information, sorts nodes by timestamp, and establishes
        parent-child relationships among them.
        """
        self.logger.info("Extracting and processing node data from execution trace.")
        self.pytorch_schema = pytorch_et_data["schema"]
        self.pytorch_pid = pytorch_et_data["pid"]
        self.pytorch_time = pytorch_et_data["time"]
        self.pytorch_start_ts = pytorch_et_data["start_ts"]
        self.pytorch_finish_ts = pytorch_et_data["finish_ts"]

        pytorch_nodes = pytorch_et_data["nodes"]
        pytorch_node_objects = {
            node_data["id"]: PyTorchNode(node_data) for node_data in pytorch_nodes
        }
        self._establish_parent_child_relationships(pytorch_node_objects)

    def _establish_parent_child_relationships(
        self, pytorch_node_objects: Dict[int, PyTorchNode]
    ) -> None:
        """
        Establishes parent-child relationships among PyTorch nodes and counts
        the node types.

        Args:
            pytorch_node_objects (Dict[int, PyTorchNode]): Dictionary of PyTorch
            node objects.
        """
        # Initialize counters for different types of nodes
        node_type_counts = {
            "total_op": 0,
            "cpu_op": 0,
            "gpu_op": 0,
            "record_param_comms_op": 0,
            "nccl_op": 0,
            "root_op": 0
        }

        # Establish parent-child relationships
        for pytorch_node in pytorch_node_objects.values():
            parent_id = pytorch_node.parent
            if parent_id in pytorch_node_objects:
                parent_node = pytorch_node_objects[parent_id]
                parent_node.add_child(pytorch_node)

                if pytorch_node.is_gpu_op():
                    parent_node.set_child_gpu(pytorch_node)

                if pytorch_node.is_record_param_comms_op():
                    parent_node.record_param_comms_node = pytorch_node

                if pytorch_node.is_nccl_op():
                    parent_node.nccl_node = pytorch_node

            if pytorch_node.name in ["[pytorch|profiler|execution_graph|thread]",
                                     "[pytorch|profiler|execution_trace|thread]"]:
                self.pytorch_root_nids.append(pytorch_node.id)
                node_type_counts["root_op"] += 1

            # Collect statistics
            node_type_counts["total_op"] += 1
            if pytorch_node.is_cpu_op():
                node_type_counts["cpu_op"] += 1
            if pytorch_node.is_gpu_op():
                node_type_counts["gpu_op"] += 1
            if pytorch_node.is_record_param_comms_op():
                node_type_counts["record_param_comms_op"] += 1
            if pytorch_node.is_nccl_op():
                node_type_counts["nccl_op"] += 1

        # Log the counts of each node type
        for node_type, count in node_type_counts.items():
            self.logger.info(f"{node_type}: {count}")

        self.pytorch_nodes = pytorch_node_objects

    def open_chakra_execution_trace(self) -> None:
        """
        Opens the Chakra execution trace file for writing.

        Raises:
            Exception: If there is an IOError in opening the file.
        """
        self.logger.info(f"Opening Chakra execution trace file: {self.output_filename}")
        try:
            self.chakra_et = open(self.output_filename, "wb")
        except IOError as e:
            err_msg = f"Error opening file {self.output_filename}: {e}"
            self.logger.error(err_msg)
            raise Exception(err_msg)

    def construct_phase_end_nids(self) -> None:
        """
        Identifies the dependencies between phases in the execution trace.

        Uses a depth-first search (DFS) approach starting from phase root nodes to find
        the largest Node ID (NID) in each phase for dependency tracking.
        """
        self.logger.info("Constructing phase end node IDs.")
        for node in self.pytorch_nodes.values():
            if self.is_phase_root_op(node):
                largest_nid_within_phase = self.dfs(node)
                if largest_nid_within_phase != -1:
                    self.phase_end_nids.append(largest_nid_within_phase)
        self.phase_end_nids.sort()

    def is_phase_root_op(self, node: PyTorchNode) -> bool:
        """
        Determines if a node is a root node of a phase.

        Args:
            node (PyTorchNode): The node to be checked.

        Returns:
            bool: True if the node is a root node of a phase, False otherwise.
        """
        return node.parent in self.pytorch_root_nids

    def dfs(self, node: PyTorchNode) -> int:
        """
        Performs a depth-first search to find the largest Node ID (NID) in a subtree.

        Explores the subtree of the given node to find the largest NID among CPU operation nodes.

        Args:
            node (PyTorchNode): The node from which the search starts.

        Returns:
            int: The largest NID found in the subtree, or -1 if no CPU operation node is found.
        """
        if node.get_op_type() == PyTorchNodeType.GPU_OP:
            return -1
        elif node.get_op_type() == PyTorchNodeType.CPU_OP:
            return node.id
        else:  # PyTorchNodeType.LABEL or any other type
            largest_nid = -1
            for child_node in node.children:
                largest_nid = max(largest_nid, self.dfs(child_node))
            return largest_nid

        self.pytorch_nodes = updated_pytorch_nodes

    def split_cpu_nodes_with_gpu_child(self) -> None:
        """
        Decomposes CPU nodes with GPU child nodes to model execution overlap
        accurately. This method addresses scenarios where a CPU node has a GPU
        child node, with an overlap in their execution ending at the same time.
        The method splits the CPU node into:
        1. Non-Overlapping Part: Segment before the GPU node starts.
        2. Overlapping Part: Segment overlapping with the GPU node.

        Timeline Stages:
        Stage 1 - Original Scenario:
            |------------ CPU Node ------------|
                              |--- GPU Node ---|

        Stage 2 - After Split:
            |-- Non-Overlap --|--- Overlap ----|
                              |--- GPU Node ---|

        Raises:
            ValueError: If timestamps of GPU and CPU nodes are inconsistent.
        """
        self.logger.info("Decomposing CPU nodes with GPU child nodes.")
        updated_pytorch_nodes: Dict[int, PyTorchNode] = {}
        for cpu_node in self.pytorch_nodes.values():
            if cpu_node.child_gpu is None:
                new_cpu_node_id = self.id_assigner.assign_unique_id(cpu_node.id)
                cpu_node.id = new_cpu_node_id
                for child_node in cpu_node.children:
                    child_node.parent = cpu_node.id
                updated_pytorch_nodes[new_cpu_node_id] = cpu_node
            else:
                gpu_node = cpu_node.child_gpu
                if gpu_node.ts >= (cpu_node.ts + cpu_node.dur):
                    err_msg = f"Inconsistent timestamps for CPU node {cpu_node.id} and its GPU child"
                    self.logger.error(err_msg)
                    raise ValueError(err_msg)

                cpu_node_first, cpu_node_second, updated_gpu_node =\
                        self._split_cpu_node(cpu_node, gpu_node)
                updated_pytorch_nodes[cpu_node_first.id] = cpu_node_first
                updated_pytorch_nodes[cpu_node_second.id] = cpu_node_second
                updated_pytorch_nodes[updated_gpu_node.id] = updated_gpu_node

        self.pytorch_nodes = updated_pytorch_nodes

        self.update_phase_end_nids()

    def _split_cpu_node(
        self, cpu_node: PyTorchNode, gpu_node: PyTorchNode
    ) -> Tuple[PyTorchNode, PyTorchNode, PyTorchNode]:
        """
        Splits a CPU node based on the GPU node's timestamp.

        Args:
            cpu_node (PyTorchNode): Original CPU node to be split.
            gpu_node (PyTorchNode): GPU node dictating the split.

        Returns:
            Tuple[PyTorchNode, PyTorchNode, PyTorchNode]: Two split nodes and the updated GPU node.

        Raises:
            ValueError: For inconsistencies in the timestamps of the nodes.
        """
        original_cpu_info = f"Original CPU Node ID {cpu_node.id} ({cpu_node.name}), " \
                            f"Duration: {cpu_node.dur}."
        self.logger.debug(original_cpu_info)
        self.logger.debug(f"GPU Node ID {gpu_node.id} ({gpu_node.name}), "
                          f"Duration: {gpu_node.dur}.")

        cpu_node_first = copy.deepcopy(cpu_node)
        cpu_node_first.id = self.id_assigner.assign_unique_id(cpu_node.id)
        cpu_node_first.ts = cpu_node.ts
        cpu_node_first.dur = gpu_node.ts - cpu_node.ts
        cpu_node_first.set_child_gpu = gpu_node
        for child_node in cpu_node_first.children:
            child_node.parent = cpu_node_first.id
        if cpu_node_first.ts >= gpu_node.ts or cpu_node_first.dur <= 0:
            err_msg = (f"Invalid timestamps for the first split CPU node derived from {original_cpu_info}\n"
                       f"\tFirst Split CPU Node Timestamp: {cpu_node_first.ts}, \n"
                       f"\tGPU Node Timestamp: {gpu_node.ts}, \n"
                       f"\tFirst Split CPU Node Duration: {cpu_node_first.dur}.")
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        self.logger.debug(f"First Split CPU Node ID {cpu_node_first.id} ({cpu_node_first.name}), "
                          f"Duration: {cpu_node_first.dur}")

        gpu_node_id = self.id_assigner.assign_unique_id(gpu_node.id)
        gpu_node.id = gpu_node_id

        cpu_node_second = copy.deepcopy(cpu_node)
        cpu_node_second.id = self.id_assigner.assign_unique_id(cpu_node.id)
        cpu_node_second.ts = gpu_node.ts
        cpu_node_second.dur = cpu_node.dur - (gpu_node.ts - cpu_node.ts)
        cpu_node_second.set_child_gpu(None)
        cpu_node_second.add_data_dep(cpu_node_first)
        for child_node in cpu_node_second.children:
            child_node.parent = cpu_node_second.id
        if cpu_node_second.ts <= cpu_node_first.ts or cpu_node_second.dur <= 0:
            err_msg = (f"Invalid timestamps for the second split CPU node derived from {original_cpu_info}\n"
                       f"\tFirst Split Timestamp: {cpu_node_first.ts}, \n"
                       f"\tSecond Split Timestamp: {cpu_node_second.ts}, \n"
                       f"\tSecond Split Duration: {cpu_node_second.dur}.")
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        self.logger.debug(f"Second Split CPU Node ID {cpu_node_second.id} ({cpu_node_second.name}), "
                          f"Duration: {cpu_node_second.dur}.")

        return cpu_node_first, cpu_node_second, gpu_node

    def update_phase_end_nids(self) -> None:
        """
        Updates the phase end node IDs with the largest new node ID assigned
        during the splitting of CPU nodes with GPU children. Utilizes the
        get_assigned_ids function from UniqueIdAssigner to find all new IDs and
        selects the largest one for each original node ID.

        This ensures that the phase end boundaries are correctly maintained after
        splitting the nodes.
        """
        self.logger.info(
            "Updating phase end node IDs with the largest new IDs after node splitting."
        )
        updated_phase_end_nids = []
        for node_id in self.phase_end_nids:
            assigned_ids = self.id_assigner.get_assigned_ids(node_id)
            if assigned_ids:
                updated_phase_end_nids.append(max(assigned_ids))
        updated_phase_end_nids.sort()
        self.phase_end_nids = updated_phase_end_nids

    def update_input_tensor_map(self, nid: int, inputs: List[List[int]]) -> None:
        """
        Updates input_storage_id_nid_map and input_tensor_id_nid_map with input
        tensor information.

        Each dictionary is populated with mappings between storage ID (or tensor ID)
        and node IDs. For example, if node 0 takes tensor 10 as an input, a new
        mapping will be created like this `10: [0]`.

        Args:
            nid (int): Node ID associated with the input tensors.
            inputs (List[List[int]]): List of input tensor data.
        """
        for i in inputs:
            tensor = list_to_pytorch_tensor(i)
            if tensor.is_valid():
                if tensor.has_valid_storage_id():
                    storage_id = tensor.storage_id
                    self.input_storage_id_nid_map.setdefault(
                        storage_id, []
                    ).append(nid)
                else:
                    tensor_id = tensor.tensor_id
                    self.input_tensor_id_nid_map.setdefault(
                        tensor_id, []
                    ).append(nid)

    def update_output_tensor_map(self, nid: int, outputs: List[List[int]]) -> None:
        """
        Updates output_storage_id_nid_map and output_tensor_id_nid_map with output
        tensor information.

        Each dictionary is populated with mappings between storage ID (or tensor ID)
        and node IDs.  For example, if node 0 produces tensor 10 as an output,
        a new mapping will be created like this `10: [0]`.

        Args:
            nid (int): Node ID associated with the output tensors.
            outputs (List[List[int]]): List of output tensor data.
        """
        for o in outputs:
            tensor = list_to_pytorch_tensor(o)
            if tensor.is_valid():
                if tensor.has_valid_storage_id():
                    storage_id = tensor.storage_id
                    self.output_storage_id_nid_map.setdefault(
                        storage_id, []
                    ).append(nid)
                else:
                    tensor_id = tensor.tensor_id
                    self.output_tensor_id_nid_map.setdefault(
                        tensor_id, []
                    ).append(nid)

    def convert_to_chakra_node(self, pytorch_node: PyTorchNode) -> ChakraNode:
        """
        Converts a PyTorchNode to a ChakraNode.

        Args:
            pytorch_node (PyTorchNode): The PyTorch node to convert.

        Returns:
            ChakraNode: The converted Chakra node.
        """
        self.logger.debug(f"Converting PyTorch node ID {pytorch_node.id} to Chakra node.")

        chakra_node = ChakraNode()
        chakra_node.id = pytorch_node.id
        chakra_node.name = pytorch_node.name
        chakra_node.type = self.get_chakra_node_type_from_pytorch_node(pytorch_node)
        if pytorch_node.parent in self.chakra_nodes:
            chakra_node.ctrl_deps.append(pytorch_node.parent)
        chakra_node.duration_micros = pytorch_node.dur if pytorch_node.has_dur() else 0
        chakra_node.inputs.values = str(pytorch_node.inputs)
        chakra_node.inputs.shapes = str(pytorch_node.input_shapes)
        chakra_node.inputs.types = str(pytorch_node.input_types)
        chakra_node.outputs.values = str(pytorch_node.outputs)
        chakra_node.outputs.shapes = str(pytorch_node.output_shapes)
        chakra_node.outputs.types = str(pytorch_node.output_types)
        chakra_node.attr.extend([
            ChakraAttr(name="rf_id", int64_val=pytorch_node.rf_id),
            ChakraAttr(name="fw_parent", int64_val=pytorch_node.fw_parent),
            ChakraAttr(name="seq_id", int64_val=pytorch_node.seq_id),
            ChakraAttr(name="scope", int64_val=pytorch_node.scope),
            ChakraAttr(name="tid", int64_val=pytorch_node.tid),
            ChakraAttr(name="fw_tid", int64_val=pytorch_node.fw_tid),
            ChakraAttr(name="op_schema", string_val=pytorch_node.op_schema),
            ChakraAttr(name="is_cpu_op", int32_val=not pytorch_node.is_gpu_op()),
            ChakraAttr(name="ts", int64_val=pytorch_node.ts)
        ])
        return chakra_node

    def get_chakra_node_type_from_pytorch_node(self, pytorch_node: PyTorchNode) -> int:
        """
        Determines the Chakra node type from a PyTorch node.

        Args:
            pytorch_node (PyTorchNode): The PyTorch node to determine the type of.

        Returns:
            int: The corresponding Chakra node type.
        """
        if pytorch_node.is_gpu_op() and (
            "ncclKernel" in pytorch_node.name or "ncclDevKernel" in pytorch_node.name
        ):
            return COMM_COLL_NODE
        elif pytorch_node.is_gpu_op():
            return COMP_NODE
        elif ("c10d::" in pytorch_node.name) or ("nccl:" in pytorch_node.name):
            return COMM_COLL_NODE
        elif (pytorch_node.op_schema != "") or pytorch_node.outputs:
            return COMP_NODE
        return INVALID_NODE

    def get_nccl_node(self, node: PyTorchNode) -> PyTorchNode:
        """
        Returns a PyTorch NCCL node for a given Chakra CPU node.

        Critical for identifying communication type and size in communication nodes.
        There are two primary cases to consider: when the given node is a parent
        of a record_param_comms node or a NCCL node.

        Args:
            node (PyTorchNode): The parent node for which the NCCL node is needed.

        Returns:
            PyTorchNode: The corresponding NCCL node.

        Raises:
            ValueError: If no corresponding NCCL node is found.
        """
        self.logger.debug(f"Retrieving NCCL node for PyTorch node ID {node.id}.")
        if node.record_param_comms_node:
            record_param_comms_node = node.record_param_comms_node
            if record_param_comms_node.nccl_node:
                return record_param_comms_node.nccl_node
            else:
                err_msg = "No NCCL node found in the record_param_comms node."
                self.logger.error(err_msg)
                raise ValueError(err_msg)
        elif node.nccl_node:
            return node.nccl_node
        else:
            err_msg = "No NCCL node associated with the given PyTorch node."
            self.logger.error(err_msg)
            raise ValueError(err_msg)

    def get_prev_phase_end_nid(self, node: ChakraNode) -> int:
        """
        Returns the Node ID (NID) of the latest node of the previous phase for
        the given ChakraNode.

        This method is used to find the closest but smaller value from
        phase_end_nids compared to the given node's ID. It helps in
        determining the dependencies between different phases in the trace.

        Args:
            node (ChakraNode): The node to find the previous phase dependency for.

        Returns:
            int: NID of the latest node of the previous phase, or -1 if none.
        """
        self.logger.debug(
            f"Finding previous inter-phase dependency for node ID {node.id}."
        )
        index = bisect.bisect_left(self.phase_end_nids, node.id)

        if index == 0:
            # All elements in the list are greater than node.id;
            # no element satisfies the condition.
            return -1
        else:
            # The element at index-1 will be the closest, smaller value
            # compared to node.id.
            return self.phase_end_nids[index - 1]

    def identify_data_dependency(self) -> None:
        """
        Identifies data dependencies between nodes using tensor input/output
        relationships.

        Determines the relationships based on whether the tensors use storage IDs
        or tensor IDs.
        """
        self.logger.info("Identifying data dependencies among nodes.")
        self.identify_data_dependency_with_storage_id()
        self.identify_data_dependency_with_tensor_id()

    def identify_data_dependency_with_storage_id(self) -> None:
        """
        Identifies data dependency between nodes based on storage IDs.

        Uses the mapping of input and output tensors to their storage IDs to
        establish dependencies.
        """
        self.logger.info("Identifying data dependencies using storage IDs.")
        self.update_data_dependencies(
                self.input_storage_id_nid_map,
                self.output_storage_id_nid_map)

    def identify_data_dependency_with_tensor_id(self) -> None:
        """
        Identifies data dependency between nodes based on tensor IDs.

        Establishes dependencies using tensor IDs for tensors without valid
        storage IDs.
        """
        self.logger.info("Identifying data dependencies using tensor IDs.")
        self.update_data_dependencies(
                self.input_tensor_id_nid_map,
                self.output_tensor_id_nid_map)

    def update_data_dependencies(self, input_map: Dict[int, List[int]],
                                 output_map: Dict[int, List[int]]) -> None:
        """
        Updates data dependencies for nodes based on input and output tensor maps.

        Args:
            input_map (Dict[int, List[int]]): Map of input tensor IDs to node IDs.
            output_map (Dict[int, List[int]]): Map of output tensor IDs to node IDs.
        """
        self.logger.debug("Updating data dependencies for nodes.")
        for input_id, child_nids in input_map.items():
            if input_id in output_map:
                parent_nids = output_map[input_id]
                for child_nid in child_nids:
                    for parent_nid in parent_nids:
                        child_node = self.chakra_nodes[child_nid]
                        if (parent_nid not in child_node.data_deps)\
                                and (parent_nid < child_nid):
                            child_node.data_deps.append(parent_nid)

    def write_chakra_et(self) -> None:
        """
        Writes the Chakra execution trace by encoding global metadata and nodes.

        Encodes and writes both the metadata and individual nodes to create a
        complete execution trace.
        """
        self.logger.info("Writing Chakra execution trace.")
        self._write_global_metadata()
        self._encode_and_write_nodes()
        self.logger.info("Chakra execution trace writing completed.")

    def _write_global_metadata(self) -> None:
        """
        Encodes and writes global metadata for the Chakra execution trace.

        This process includes encoding metadata like schema, process ID, timestamps,
        and other relevant information for the Chakra execution trace.
        """
        self.logger.info("Encoding global metadata for Chakra execution trace.")
        global_metadata = GlobalMetadata(
            attr=[
                ChakraAttr(name="schema", string_val=self.pytorch_schema),
                ChakraAttr(name="pid", uint64_val=self.pytorch_pid),
                ChakraAttr(name="time", string_val=self.pytorch_time),
                ChakraAttr(name="start_ts", uint64_val=self.pytorch_start_ts),
                ChakraAttr(name="finish_ts", uint64_val=self.pytorch_finish_ts)
            ]
        )
        encode_message(self.chakra_et, global_metadata)

    def _encode_and_write_nodes(self) -> None:
        """
        Encodes and writes nodes for the Chakra execution trace.

        Each node from the PyTorch execution trace is encoded and written into the
        Chakra format. This includes node IDs, names, types, dependencies, and
        other attributes.
        """
        self.logger.info("Encoding and writing nodes for Chakra execution trace.")
        seen_nids = set()
        for nid in sorted(self.chakra_nodes.keys()):
            if nid in seen_nids:
                err_msg = f"Duplicate NID {nid} detected in Chakra nodes."
                self.logger.error(err_msg)
                raise ValueError(err_msg)
            seen_nids.add(nid)
            chakra_node = self.chakra_nodes[nid]
            encode_message(self.chakra_et, chakra_node)

    def close_chakra_execution_trace(self) -> None:
        """
        Closes the Chakra execution trace file if it is open.

        Ensures proper closure of the trace file to preserve data integrity.
        """
        self.logger.info("Closing Chakra execution trace file.")
        if self.chakra_et and not self.chakra_et.closed:
            self.chakra_et.close()
