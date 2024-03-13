#!/usr/bin/env python3

import copy
import json
import logging
from typing import Dict, List, Optional, Tuple, Set

from .pytorch_node import PyTorchNodeType, PyTorchNode
from chakra.third_party.utils.protolib import encodeMessage as encode_message
from chakra.et_def.et_def_pb2 import (
    GlobalMetadata,
    NodeType as ChakraNodeType,
    Node as ChakraNode,
    AttributeProto as ChakraAttr,
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

    def set_next_id(self, next_id: int) -> None:
        """
        Sets the starting next unique ID.

        Args:
            next_id (int): The starting next unique ID to set.
        """
        self.next_id = next_id

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
        chakra_et(IO[bytes]): File handle for the Chakra execution trace output file.
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
        parent_to_children_map (Dict[int, List[int]]): Map of Chakra parent node
                                                       IDs to their child node
                                                       IDs. Used to simulate
                                                       execution based on data
                                                       dependencies.
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
        self.chakra_et = None
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

        self.parent_to_children_map = {}

    def convert(self) -> None:
        """
        Converts PyTorch execution traces into the Chakra format. Orchestrates
        the conversion process including trace loading, trace opening, phase
        end node construction, node splitting, and node conversion.
        """
        self.load_pytorch_execution_traces()

        self.open_chakra_execution_trace()

        self.split_cpu_nodes_with_gpu_child()

        for pytorch_nid, pytorch_node in self.pytorch_nodes.items():
            if (pytorch_node.get_op_type() == PyTorchNodeType.CPU_OP)\
                    or (pytorch_node.get_op_type() == PyTorchNodeType.LABEL):
                chakra_node = self.convert_to_chakra_node(pytorch_node)
                self.chakra_nodes[chakra_node.id] = chakra_node

                for pytorch_gpu_node in pytorch_node.gpu_children:
                    chakra_gpu_node = self.convert_to_chakra_node(pytorch_gpu_node)

                    if chakra_node.type == COMM_COLL_NODE:
                        collective_comm_type = self.get_collective_comm_type(pytorch_node.name)
                        chakra_gpu_node.attr.extend([
                            ChakraAttr(name="comm_type",
                                       int64_val=collective_comm_type),
                            ChakraAttr(name="comm_size",
                                       int64_val=pytorch_gpu_node.comm_size),
                            ChakraAttr(name="involved_dim",
                                       bool_list={"values": [True] * self.num_dims})])

                    self.chakra_nodes[chakra_gpu_node.id] = chakra_gpu_node

        root_nodes = [node for node in self.chakra_nodes.values() if self.is_root_node(node)]
        for root_node in root_nodes:
            self.convert_ctrl_dep_to_data_dep(root_node)

        self.remove_dangling_nodes()

        self.update_parent_to_children_map()

        self.identify_cyclic_dependencies()

        self.write_chakra_et()

        self.close_chakra_execution_trace()

        self.simulate_execution()

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
            self.id_assigner.set_next_id(max(self.pytorch_nodes.keys()) + 1)
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
                    parent_node.add_gpu_child(pytorch_node)

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
        for pytorch_node in self.pytorch_nodes.values():
            if pytorch_node.is_cpu_op():
                cpu_node = pytorch_node
                gpu_children = cpu_node.gpu_children
                if gpu_children:
                    if cpu_node.exclusive_dur > 1:
                        split_nodes = self._split_cpu_node(cpu_node, gpu_children)
                        updated_pytorch_nodes.update(split_nodes)
                    else:
                        new_cpu_id = self.id_assigner.assign_unique_id(cpu_node.id)
                        cpu_node.id = new_cpu_id
                        updated_pytorch_nodes[new_cpu_id] = cpu_node
                        for child in cpu_node.children:
                            new_child_id = self.id_assigner.assign_unique_id(child.id)
                            child.id = new_child_id
                            child.parent = new_cpu_id
                            updated_pytorch_nodes[new_child_id] = gpu_child
                else:
                    new_id = self.id_assigner.assign_unique_id(cpu_node.id)
                    cpu_node.id = new_id
                    for child in cpu_node.children:
                        child.parent = new_id
                    updated_pytorch_nodes[new_id] = cpu_node
            elif not pytorch_node.is_gpu_op():
                new_id = self.id_assigner.assign_unique_id(pytorch_node.id)
                pytorch_node.id = new_id
                for child in pytorch_node.children:
                    child.parent = new_id
                updated_pytorch_nodes[new_id] = pytorch_node

        self.pytorch_nodes = updated_pytorch_nodes

    def _split_cpu_node(
        self,
        cpu_node: PyTorchNode,
        gpu_children: List[PyTorchNode]
    ) -> Dict[int, PyTorchNode]:
        """
        Splits a CPU node based on overlaps with GPU children. It correctly assigns
        the nearest preceding CPU node as the parent for each GPU child. It ensures
        non-GPU children of the original CPU node are preserved and added to the
        last part of the split CPU node.
        """
        updated_nodes = {}
        original_end_ts = cpu_node.ts + cpu_node.exclusive_dur

        # Preserving non-GPU children of the original CPU node
        non_gpu_children = [child for child in cpu_node.children if child not in gpu_children]

        split_points = sorted(set(gpu_child.ts for gpu_child in gpu_children if gpu_child.ts < original_end_ts))
        last_split_end = cpu_node.ts
        last_cpu_part_id = None

        for split_ts in split_points:
            split_duration = split_ts - last_split_end
            if split_duration > 0:
                new_cpu_part = copy.deepcopy(cpu_node)
                new_part_id = self.id_assigner.assign_unique_id(cpu_node.id)
                new_cpu_part.id = new_part_id
                new_cpu_part.ts = last_split_end
                new_cpu_part.exclusive_dur = split_duration
                new_cpu_part.inclusive_dur = split_duration
                new_cpu_part.parent = cpu_node.parent if last_cpu_part_id is None else last_cpu_part_id
                new_cpu_part.children = []
                new_cpu_part.gpu_children = []

                updated_nodes[new_part_id] = new_cpu_part
                last_cpu_part_id = new_part_id
                last_split_end = split_ts

        # Creating the final part if necessary
        if last_split_end < original_end_ts:
            final_part_duration = original_end_ts - last_split_end
            final_cpu_part = copy.deepcopy(cpu_node)
            final_part_id = self.id_assigner.assign_unique_id(cpu_node.id)
            final_cpu_part.id = final_part_id
            final_cpu_part.ts = last_split_end
            final_cpu_part.exclusive_dur = final_part_duration
            final_cpu_part.inclusive_dur = final_part_duration
            final_cpu_part.parent = last_cpu_part_id if last_cpu_part_id else cpu_node.parent
            final_cpu_part.children = []
            final_cpu_part.gpu_children = []

            updated_nodes[final_part_id] = final_cpu_part
            last_cpu_part_id = final_part_id

        # Adding non-GPU children to the last split of the CPU node
        if last_cpu_part_id:
            updated_nodes[last_cpu_part_id].children.extend(non_gpu_children)

        # Assigning GPU children to the closest CPU part
        for gpu_child in gpu_children:
            closest_cpu_parent_id = max(
                [id for id, node in updated_nodes.items() if node.ts < gpu_child.ts],
                key=lambda id: updated_nodes[id].ts,
                default=cpu_node.parent
            )

            gpu_child.parent = closest_cpu_parent_id
            new_gpu_id = self.id_assigner.assign_unique_id(gpu_child.id)
            gpu_child.id = new_gpu_id
            updated_nodes[new_gpu_id] = gpu_child

            # Add GPU child to the nearest CPU part's GPU children list
            updated_nodes[closest_cpu_parent_id].children.append(gpu_child)
            updated_nodes[closest_cpu_parent_id].gpu_children.append(gpu_child)

        return updated_nodes

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
        chakra_node.duration_micros = pytorch_node.exclusive_dur
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

    def get_chakra_node_type_from_pytorch_node(self, pytorch_node: PyTorchNode) -> ChakraNodeType:
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
        elif ("c10d::" in pytorch_node.name) or ("nccl:" in pytorch_node.name):
            return COMM_COLL_NODE
        return COMP_NODE

    def get_collective_comm_type(self, name: str) -> int:
        """
        Returns the collective communication type of the node.

        Args:
            name (str): The name of the node.

        Raises:
            ValueError: If the communication type is not found in the mapping.

        Returns:
            int: The collective communication type of the node.
        """
        comm_type_mapping = {
            "all_reduce": ALL_REDUCE,
            "all_to_all": ALL_TO_ALL,
            "all_gather": ALL_GATHER,
            "reduce_scatter": REDUCE_SCATTER,
            "broadcast": BROADCAST,
            "AllReduce": ALL_REDUCE,
            "Broadcast": BROADCAST,
            # Additional cases can be added here
        }

        for key, value in comm_type_mapping.items():
            if key.lower() in name.lower():
                return value

        raise ValueError(f"'{name}' not found in collective communication mapping. "
                         "Please add this collective communication name to the mapping.")

    def is_root_node(self, node):
        """
        Determines whether a given node is a root node in the execution trace.

        In the context of PyTorch execution traces, root nodes are the starting
        points of execution graphs or execution traces. These nodes typically do
        not have parent nodes and act as the original sources of execution flow.
        This method identifies such root nodes based on their names. Specifically,
        nodes with names indicating they are part of the PyTorch execution graph or
        execution trace threads are considered root nodes.

        Args:
            node (ChakraNode): The node to be evaluated.

        Returns:
            bool: True if the node is a root node, False otherwise.
        """
        if node.name in ["[pytorch|profiler|execution_graph|thread]",
                         "[pytorch|profiler|execution_trace|thread]"]:
            return True

    def convert_ctrl_dep_to_data_dep(self, chakra_node: ChakraNode) -> None:
        """
        Traverses nodes based on control dependencies (parent nodes) and encodes
        data dependencies appropriately. This method is crucial for converting the
        dependency structure from PyTorch execution traces to Chakra execution
        traces. In PyTorch traces, control dependencies are represented by a
        parent field in each node, denoting the parent node ID. This structure
        indicates which functions (operators) are called by a particular operator.

        In contrast, Chakra execution traces, while retaining control dependencies
        for compatibility, primarily rely on data dependencies to represent
        relationships between nodes. Data dependencies in Chakra are more broadly
        defined compared to those in PyTorch, where they are implicitly encoded in
        tensor input-output relationships. In Chakra, data dependencies are explicit
        and represent a general dependency between nodes.

        To convert PyTorch's control dependencies to Chakra's data dependencies, a
        Depth-First Search (DFS) is performed. The DFS traversal starts from a given
        Chakra node, traversing through its children (based on control
        dependencies). During traversal, data dependencies are encoded by linking
        nodes that have been visited in sequence. These dependencies form a chain,
        mirroring the function call order from the PyTorch trace.

        Special attention is given to the types of nodes involved. CPU and label
        nodes (non-GPU) in PyTorch can only depend on other CPU or label nodes.
        However, GPU nodes can depend on any type of node. Thus, while traversing,
        if a GPU node is encountered, it can establish a data dependency with the
        last visited node of any type. For CPU and label nodes, the dependency is
        only established with the last visited non-GPU node. This distinction
        ensures that the converted dependencies accurately reflect the execution
        dynamics of the original PyTorch trace within the Chakra framework.

        Additionally, this method enforces sequential dependencies between GPU
        operators within the same stream. It ensures that the execution order of
        GPU operators is preserved in the Chakra trace, reflecting the sequential
        execution within the same GPU stream in the original PyTorch trace.

        Furthermore, inter-thread dependencies are explicitly encoded in the Chakra
        execution traces. This feature allows for the representation of dependencies
        across different CPU threads, which are observed in Kineto traces via
        chrome://tracing. These dependencies are crucial for understanding the
        interaction between CPU threads and ensuring accurate modeling and analysis
        of concurrent operations within the Chakra framework.

        Args:
            chakra_node (ChakraNode): The starting node for the traversal and
            dependency processing.
        """
        visited: Set[int] = set()
        stack: List[ChakraNode] = [chakra_node]
        last_visited_non_gpu: Optional[ChakraNode] = None
        last_visited_any: Optional[ChakraNode] = None
        last_gpu_in_stream: Dict[int, ChakraNode] = {}

        while stack:
            current_node = stack.pop()
            if current_node.id in visited:
                continue

            visited.add(current_node.id)

            # Determine the operator type of the current node
            pytorch_node = self.pytorch_nodes.get(current_node.id)
            if not pytorch_node:
                continue

            node_op_type = pytorch_node.get_op_type()

            if node_op_type == PyTorchNodeType.GPU_OP:
                if last_visited_any:
                    if last_visited_any.id not in current_node.data_deps:
                        current_node.data_deps.append(last_visited_any.id)
                        self.logger.debug(
                            f"GPU Node ID {current_node.id} now has a data "
                            f"dependency on Node ID {last_visited_any.id}"
                        )

                stream_id = pytorch_node.stream
                if stream_id in last_gpu_in_stream:
                    if last_gpu_in_stream[stream_id].id not in current_node.data_deps:
                        current_node.data_deps.append(last_gpu_in_stream[stream_id].id)
                        self.logger.debug(
                            f"GPU Node ID {current_node.id} in stream {stream_id} now has a data "
                            f"dependency on GPU Node ID {last_gpu_in_stream[stream_id].id} in the same stream."
                        )
                last_gpu_in_stream[stream_id] = current_node
                last_visited_any = current_node
            else:
                if pytorch_node.inter_thread_dep:
                    for id in self.id_assigner.get_assigned_ids(pytorch_node.inter_thread_dep):
                        if id not in current_node.data_deps:
                            current_node.data_deps.append(id)
                            self.logger.debug(
                                f"CPU Node ID {current_node.id} now has an inter-thread data "
                                f"dependency on Node ID {id}"
                            )

                if last_visited_non_gpu:
                    if last_visited_non_gpu.id not in current_node.data_deps:
                        current_node.data_deps.append(last_visited_non_gpu.id)
                        self.logger.debug(
                            f"CPU Node ID {current_node.id} now has a data "
                            f"dependency on non-GPU Node ID {last_visited_non_gpu.id}"
                        )
                last_visited_non_gpu = current_node
                last_visited_any = current_node

            # Add children to the stack
            children_chakra_ids = [child.id for child in pytorch_node.children]
            for child_chakra_id in sorted(children_chakra_ids, reverse=True):
                child_chakra_node = self.chakra_nodes.get(child_chakra_id)
                if child_chakra_node and child_chakra_node.id not in visited:
                    stack.append(child_chakra_node)

    def remove_dangling_nodes(self) -> None:
        """
        Removes any dangling nodes from the chakra_nodes dictionary.
        A node is considered dangling if it has no parents and no children.
        """
        parent_ids = set()
        for node in self.chakra_nodes.values():
            parent_ids.update(node.data_deps)

        dangling_nodes = []
        for node_id, node in list(self.chakra_nodes.items()):
            if node_id not in parent_ids and not node.data_deps:
                dangling_nodes.append(node)
                del self.chakra_nodes[node_id]
                if node_id in self.pytorch_nodes:
                    del self.pytorch_nodes[node_id]

        if dangling_nodes:
            self.logger.info(f"Identified and removed {len(dangling_nodes)} dangling nodes:")
            for node in dangling_nodes:
                self.logger.info(f" - Node ID {node.id}: {node.name}")

    def update_parent_to_children_map(self) -> None:
        """
        Updates the parent_to_children_map based on the data dependencies of each node.
        This map is used to efficiently simulate node execution based on data dependencies.
        """
        for node_id, node in self.chakra_nodes.items():
            for dep_id in node.data_deps:
                # Ensure the dependency is registered as a parent of the current node
                if dep_id not in self.parent_to_children_map:
                    self.parent_to_children_map[dep_id] = []
                self.parent_to_children_map[dep_id].append(node_id)

    def identify_cyclic_dependencies(self) -> None:
        """
        Identifies if there are any cyclic dependencies among Chakra nodes.

        This method checks for cycles in the graph of Chakra nodes using a
        depth-first search (DFS) algorithm. It logs an error message and raises
        an exception if a cycle is detected, ensuring the graph is a Directed
        Acyclic Graph (DAG).

        Raises:
            Exception: If a cyclic dependency is detected among the Chakra nodes.
        """
        visited = set()
        stack = set()

        def dfs(node_id: int, path: List[int]) -> bool:
            """
            Depth-first search to detect cycles.

            Args:
                node_id (int): The node ID to start the DFS from.
                path (List[int]): The path traversed so far, for tracing the cycle.

            Returns:
                bool: True if a cycle is detected, False otherwise.
            """
            if node_id in stack:
                cycle_nodes = " -> ".join(
                    [self.chakra_nodes[n].name for n in path + [node_id]]
                )
                self.logger.error(f"Cyclic dependency detected: {cycle_nodes}")
                return True
            if node_id in visited:
                return False

            visited.add(node_id)
            stack.add(node_id)
            path.append(node_id)
            for child_id in self.chakra_nodes[node_id].data_deps:
                if dfs(child_id, path.copy()):
                    return True
            stack.remove(node_id)
            path.pop()
            return False

        for node_id in self.chakra_nodes:
            if dfs(node_id, []):
                raise Exception(
                    f"Cyclic dependency detected starting from node "
                    f"{self.chakra_nodes[node_id].name}"
                )

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

    def simulate_execution(self) -> None:
        """
        Simulates the execution of Chakra nodes based on data dependencies.

        This method considers both CPU and GPU nodes. Nodes are issued for
        execution based on the readiness determined by dependency resolution.
        A simplistic global clock is used to model the execution time.
        """
        self.logger.info("Simulating execution of Chakra nodes based on data "
                         "dependencies.")

        # Initialize queues for ready CPU and GPU nodes
        ready_cpu_nodes = [
            (node_id, self.chakra_nodes[node_id])
            for node_id in self.chakra_nodes
            if not self.chakra_nodes[node_id].data_deps and
            not self.pytorch_nodes[node_id].is_gpu_op()
        ]
        ready_gpu_nodes = [
            (node_id, self.chakra_nodes[node_id])
            for node_id in self.chakra_nodes
            if not self.chakra_nodes[node_id].data_deps and
            self.pytorch_nodes[node_id].is_gpu_op()
        ]
        ready_cpu_nodes.sort(key=lambda x: x[1].id)
        ready_gpu_nodes.sort(key=lambda x: x[1].id)

        issued_nodes: Set[int] = set()
        current_cpu_node: Optional[Tuple[int, int]] = None
        current_gpu_node: Optional[Tuple[int, int]] = None

        current_time: int = 0  # Simulated global clock in microseconds

        while any([ready_cpu_nodes, ready_gpu_nodes, current_cpu_node,
                   current_gpu_node]):
            if ready_cpu_nodes and not current_cpu_node:
                cpu_node_id, cpu_node = ready_cpu_nodes.pop(0)
                current_cpu_node = (cpu_node_id, current_time)
                issued_nodes.add(cpu_node_id)
                self.logger.info(
                    f"Issuing CPU Node ID {cpu_node_id} ({cpu_node.name}) at "
                    f"{current_time}us with duration {cpu_node.duration_micros}us"
                )

            if ready_gpu_nodes and not current_gpu_node:
                gpu_node_id, gpu_node = ready_gpu_nodes.pop(0)
                current_gpu_node = (gpu_node_id, current_time)
                issued_nodes.add(gpu_node_id)
                self.logger.info(
                    f"Issuing GPU Node ID {gpu_node_id} ({gpu_node.name}) at "
                    f"{current_time}us with duration {gpu_node.duration_micros}us"
                )

            current_time += 1

            if current_cpu_node and current_time - current_cpu_node[1] >= \
                    self.chakra_nodes[current_cpu_node[0]].duration_micros:
                self.logger.info(f"CPU Node ID {current_cpu_node[0]} completed "
                                 f"at {current_time}us")
                current_cpu_node = None

            if current_gpu_node and current_time - current_gpu_node[1] >= \
                    self.chakra_nodes[current_gpu_node[0]].duration_micros:
                self.logger.info(f"GPU Node ID {current_gpu_node[0]} completed "
                                 f"at {current_time}us")
                current_gpu_node = None

            for node_id in list(issued_nodes):
                children_ids = self.parent_to_children_map.get(node_id, [])
                for child_id in children_ids:
                    child_node = self.chakra_nodes[child_id]
                    child_node.data_deps.remove(node_id)
                    if not child_node.data_deps:
                        if not self.pytorch_nodes[child_id].is_gpu_op():
                            ready_cpu_nodes.append((child_id, child_node))
                        else:
                            ready_gpu_nodes.append((child_id, child_node))

            issued_nodes.clear()

        self.logger.info("Simulation of Chakra node execution completed.")
