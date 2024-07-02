import json
import logging
from typing import IO, Dict, List, Optional, Set, Tuple

from ...schema.protobuf.et_def_pb2 import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    BROADCAST,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    REDUCE_SCATTER,
    GlobalMetadata,
)
from ...schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr
from ...schema.protobuf.et_def_pb2 import Node as ChakraNode
from ..third_party.utils.protolib import encodeMessage as encode_message
from .pytorch_node import PyTorchNode, PyTorchNodeType


class PyTorchConverter:
    """
    Converter class for transforming PyTorch execution traces into Chakra format.

    This class is responsible for converting the execution traces collected from PyTorch into a format that is
    compatible with Chakra, a performance analysis tool. It handles the intricate mappings and transformations required
    to accurately represent the execution in a different format.

    Attributes
        input_filename (str): Input file name containing PyTorch execution trace.
        output_filename (str): Output file name for the converted Chakra trace.
    """

    def __init__(self, input_filename: str, output_filename: str) -> None:
        """
        Initialize the PyTorch to Chakra converter. It sets up necessary attributes and prepares the environment.

        Args:
            input_filename (str): Name of the input file containing PyTorch execution trace.
            output_filename (str): Name of the output file for the converted Chakra trace.
        """
        self.input_filename = input_filename
        self.output_filename = output_filename

    def convert(self) -> None:
        """Convert PyTorch execution traces into the Chakra format."""
        pytorch_et_data = self.load_pytorch_execution_traces()
        (
            pytorch_schema,
            pytorch_pid,
            pytorch_time,
            pytorch_start_ts,
            pytorch_finish_ts,
            pytorch_nodes,
        ) = self._parse_and_instantiate_nodes(pytorch_et_data)
        chakra_et = self.open_chakra_execution_trace(self.output_filename)
        chakra_nodes = {}
        self.convert_nodes(pytorch_nodes, chakra_nodes)
        root_nodes = [node for node in chakra_nodes.values() if self.is_root_node(node)]
        for root_node in root_nodes:
            self.convert_ctrl_dep_to_data_dep(pytorch_nodes, chakra_nodes, root_node)
        chakra_nodes = self.remove_dangling_nodes(chakra_nodes)
        parent_to_children_map = self.update_parent_to_children_map(chakra_nodes)
        self.identify_cyclic_dependencies(chakra_nodes)
        self.write_chakra_et(
            chakra_et,
            pytorch_schema,
            pytorch_pid,
            pytorch_time,
            pytorch_start_ts,
            pytorch_finish_ts,
            chakra_nodes,
        )
        self.close_chakra_execution_trace(chakra_et)
        self.simulate_execution(chakra_nodes, pytorch_nodes, parent_to_children_map)

    def load_pytorch_execution_traces(self) -> Dict:
        """
        Load PyTorch execution traces from a file.

        Read and parse the PyTorch execution trace data from a file, creating PyTorchNode objects and establishing
        node relationships.

        Raises
            Exception: If there is an IOError in opening the file.

        Returns
            Dict: The loaded PyTorch execution trace data.
        """
        logging.info("Loading PyTorch execution traces from file.")
        try:
            with open(self.input_filename, "r") as pytorch_et:
                return json.load(pytorch_et)
        except IOError as e:
            logging.error(f"Error opening file {self.input_filename}: {e}")
            raise Exception(f"Could not open file {self.input_filename}") from e

    def _parse_and_instantiate_nodes(
        self, pytorch_et_data: Dict
    ) -> Tuple[str, int, str, int, int, Dict[int, PyTorchNode]]:
        """
        Parse and instantiate PyTorch nodes from execution trace data.

        Args:
            pytorch_et_data (Dict): The execution trace data.

        Extract node information, sort nodes by timestamp, and establish parent-child relationships among them.

        Returns:
            Tuple: A tuple containing PyTorch schema, PID, time, start timestamp, finish timestamp, and dictionary of
                PyTorch node objects.
        """
        logging.info("Extracting and processing node data from execution trace.")
        pytorch_schema = pytorch_et_data["schema"]
        pytorch_pid = pytorch_et_data["pid"]
        pytorch_time = pytorch_et_data["time"]
        pytorch_start_ts = pytorch_et_data["start_ts"]
        pytorch_finish_ts = pytorch_et_data["finish_ts"]

        pytorch_nodes = pytorch_et_data["nodes"]
        pytorch_node_objects = {node_data["id"]: PyTorchNode(pytorch_schema, node_data) for node_data in pytorch_nodes}
        pytorch_root_nids = []
        pytorch_node_objects = self._establish_parent_child_relationships(pytorch_node_objects, pytorch_root_nids)
        return pytorch_schema, pytorch_pid, pytorch_time, pytorch_start_ts, pytorch_finish_ts, pytorch_node_objects

    def _establish_parent_child_relationships(
        self, pytorch_node_objects: Dict[int, PyTorchNode], pytorch_root_nids: List[int]
    ) -> Dict[int, PyTorchNode]:
        """
        Establish parent-child relationships among PyTorch nodes and count the node types.

        Args:
            pytorch_node_objects (Dict[int, PyTorchNode]): Dictionary of PyTorch node objects.
            pytorch_root_nids (List[int]): List to store root node IDs.

        Returns:
            Dict[int, PyTorchNode]: Dictionary of PyTorch nodes with established relationships.
        """
        node_type_counts = self._initialize_node_type_counts()

        for pytorch_node in pytorch_node_objects.values():
            parent_id = pytorch_node.parent
            if parent_id in pytorch_node_objects:
                self._process_parent_child_relationships(pytorch_node_objects, pytorch_node, parent_id)

            if self._is_root_node(pytorch_node):
                pytorch_root_nids.append(pytorch_node.id)
                node_type_counts["root_op"] += 1

            self._update_node_type_counts(node_type_counts, pytorch_node)

        for node_type, count in node_type_counts.items():
            logging.info(f"{node_type}: {count}")

        return pytorch_node_objects

    def _initialize_node_type_counts(self) -> Dict[str, int]:
        """
        Initialize counters for different types of nodes.

        Returns
            Dict[str, int]: A dictionary with node type counters initialized to zero.
        """
        return {
            "total_op": 0,
            "cpu_op": 0,
            "gpu_op": 0,
            "record_param_comms_op": 0,
            "nccl_op": 0,
            "root_op": 0,
        }

    def _is_root_node(self, pytorch_node: PyTorchNode) -> bool:
        """
        Check if a given PyTorch node is a root node.

        Args:
            pytorch_node (PyTorchNode): The PyTorch node to check.

        Returns:
            bool: True if the node is a root node, False otherwise.
        """
        return pytorch_node.name in [
            "[pytorch|profiler|execution_graph|thread]",
            "[pytorch|profiler|execution_trace|thread]",
        ]

    def _process_parent_child_relationships(
        self, pytorch_node_objects: Dict[int, PyTorchNode], pytorch_node: PyTorchNode, parent_id: int
    ) -> None:
        """
        Process the parent-child relationships for PyTorch nodes.

        Args:
            pytorch_node_objects (Dict[int, PyTorchNode]): Dictionary of PyTorch node objects.
            pytorch_node (PyTorchNode): The current PyTorch node being processed.
            parent_id (int): The ID of the parent node.
        """
        parent_node = pytorch_node_objects[parent_id]
        parent_node.add_child(pytorch_node)

        if pytorch_node.is_gpu_op():
            parent_node.add_gpu_child(pytorch_node)

        if pytorch_node.is_record_param_comms_op():
            parent_node.record_param_comms_node = pytorch_node

        if pytorch_node.is_nccl_op():
            parent_node.nccl_node = pytorch_node

    def _update_node_type_counts(self, node_type_counts: Dict[str, int], pytorch_node: PyTorchNode) -> None:
        """
        Update the node type counts based on the current PyTorch node.

        Args:
            node_type_counts (Dict[str, int]): Dictionary of node type counts.
            pytorch_node (PyTorchNode): The current PyTorch node being processed.
        """
        node_type_counts["total_op"] += 1
        if pytorch_node.is_cpu_op():
            node_type_counts["cpu_op"] += 1
        if pytorch_node.is_gpu_op():
            node_type_counts["gpu_op"] += 1
        if pytorch_node.is_record_param_comms_op():
            node_type_counts["record_param_comms_op"] += 1
        if pytorch_node.is_nccl_op():
            node_type_counts["nccl_op"] += 1

    def open_chakra_execution_trace(self, output_filename: str) -> IO[bytes]:
        """
        Open the Chakra execution trace file for writing.

        Args:
            output_filename (str): Name of the output file for the converted Chakra trace.

        Raises:
            Exception: If there is an IOError in opening the file.

        Returns:
            IO[bytes]: File handle for the Chakra execution trace output file.
        """
        logging.info(f"Opening Chakra execution trace file: {output_filename}")
        try:
            chakra_et = open(output_filename, "wb")  # noqa: SIM115
            return chakra_et
        except IOError as e:
            err_msg = f"Error opening file {output_filename}: {e}"
            logging.error(err_msg)
            raise Exception(err_msg) from e

    def convert_nodes(self, pytorch_nodes: Dict[int, PyTorchNode], chakra_nodes: Dict[int, ChakraNode]) -> None:
        """
        Convert PyTorch nodes to Chakra nodes.

        This method traverses through the PyTorch nodes and converts them to Chakra nodes. It also handles special
        cases for GPU nodes and collective communication types.
        """
        for _, pytorch_node in pytorch_nodes.items():
            if (pytorch_node.get_op_type() == PyTorchNodeType.CPU_OP) or (
                pytorch_node.get_op_type() == PyTorchNodeType.LABEL
            ):
                chakra_node = self.convert_to_chakra_node(pytorch_nodes, chakra_nodes, pytorch_node)
                chakra_nodes[chakra_node.id] = chakra_node

                for pytorch_gpu_node in pytorch_node.gpu_children:
                    chakra_gpu_node = self.convert_to_chakra_node(pytorch_nodes, chakra_nodes, pytorch_gpu_node)

                    if chakra_gpu_node.type == COMM_COLL_NODE:
                        collective_comm_type = self.get_collective_comm_type(pytorch_gpu_node.name)
                        chakra_gpu_node.attr.extend(
                            [
                                ChakraAttr(name="comm_type", int64_val=collective_comm_type),
                                ChakraAttr(name="comm_size", int64_val=pytorch_gpu_node.comm_size),
                            ]
                        )

                    elif chakra_gpu_node.type in {COMM_SEND_NODE, COMM_RECV_NODE}:
                        chakra_gpu_node.attr.extend(
                            [
                                ChakraAttr(name="comm_size", int64_val=pytorch_gpu_node.comm_size),
                            ]
                        )

                    chakra_nodes[chakra_gpu_node.id] = chakra_gpu_node

    def convert_to_chakra_node(
        self, pytorch_nodes: Dict[int, PyTorchNode], chakra_nodes: Dict[int, ChakraNode], pytorch_node: PyTorchNode
    ) -> ChakraNode:
        """
        Convert a PyTorchNode to a ChakraNode.

        Args:
            pytorch_nodes (Dict[int, PyTorchNode]): Dictionary of PyTorch nodes.
            chakra_nodes (Dict[int, ChakraNode]): Dictionary of existing Chakra nodes.
            pytorch_node (PyTorchNode): The PyTorch node to convert.

        Returns:
            ChakraNode: The converted Chakra node.
        """
        logging.debug(f"Converting PyTorch node ID {pytorch_node.id} to Chakra node.")
        chakra_node = ChakraNode()
        chakra_node.id = pytorch_node.id
        chakra_node.name = pytorch_node.name
        chakra_node.type = self.get_chakra_node_type_from_pytorch_node(pytorch_nodes, pytorch_node)
        if pytorch_node.parent in chakra_nodes:
            chakra_node.ctrl_deps.append(pytorch_node.parent)
        chakra_node.duration_micros = int(pytorch_node.exclusive_dur)

        """
        Quick and straightforward solution to identify an operator that covers more than 90% of the runtime. These are
        usually user_annotation operators and should be ignored. One such case is Optimizer.step, which we filter out
        in this code. Ideally, we should identify any user annotation nodes that cover more than 90% of the runtime and
        then set their runtime to 0.

        Note: We will cover this with a more general solution.
        """
        if "Optimizer.step" in pytorch_node.name:
            chakra_node.duration_micros = 0

        chakra_node.inputs.values = str(pytorch_node.inputs["values"])
        chakra_node.inputs.shapes = str(pytorch_node.inputs["shapes"])
        chakra_node.inputs.types = str(pytorch_node.inputs["types"])
        chakra_node.outputs.values = str(pytorch_node.outputs["values"])
        chakra_node.outputs.shapes = str(pytorch_node.outputs["shapes"])
        chakra_node.outputs.types = str(pytorch_node.outputs["types"])
        chakra_node.attr.extend(
            [
                ChakraAttr(name="rf_id", int64_val=pytorch_node.rf_id),
                ChakraAttr(name="fw_parent", int64_val=pytorch_node.fw_parent),
                ChakraAttr(name="seq_id", int64_val=pytorch_node.seq_id),
                ChakraAttr(name="scope", int64_val=pytorch_node.scope),
                ChakraAttr(name="tid", int64_val=pytorch_node.tid),
                ChakraAttr(name="fw_tid", int64_val=pytorch_node.fw_tid),
                ChakraAttr(name="op_schema", string_val=pytorch_node.op_schema),
                ChakraAttr(name="is_cpu_op", bool_val=not pytorch_node.is_gpu_op()),
            ]
        )
        return chakra_node

    def get_chakra_node_type_from_pytorch_node(
        self, pytorch_nodes: Dict[int, PyTorchNode], pytorch_node: PyTorchNode
    ) -> int:
        """
        Determine the Chakra node type from a PyTorch node.

        Args:
            pytorch_nodes (Dict[int, PyTorchNode]): Dictionary of PyTorch nodes.
            pytorch_node (PyTorchNode): The PyTorch node to determine the type of.

        Returns:
            int: The corresponding Chakra node type.
        """
        if pytorch_node.is_gpu_op():
            if "ncclDevKernel_SendRecv" in pytorch_node.name:
                parent_node = pytorch_nodes[pytorch_node.parent]
                keyword = (
                    pytorch_nodes[parent_node.parent].name
                    if parent_node.name == "record_param_comms"
                    else parent_node.name
                )
                if "send" in keyword:
                    return COMM_SEND_NODE
                if "recv" in keyword:
                    return COMM_RECV_NODE
            if "ncclKernel" in pytorch_node.name or "ncclDevKernel" in pytorch_node.name:
                return COMM_COLL_NODE
        return COMP_NODE

    def get_collective_comm_type(self, name: str) -> int:
        """
        Return the collective communication type of the node.

        Args:
            name (str): The name of the node.

        Raises:
            ValueError: If the communication type is not found in the mapping.

        Returns:
            int: The collective communication type of the node.
        """
        comm_type_mapping = {
            "allreduce": ALL_REDUCE,
            "alltoall": ALL_TO_ALL,
            "allgather": ALL_GATHER,
            "reducescatter": REDUCE_SCATTER,
            "broadcast": BROADCAST,
            # Additional cases can be added here
        }
        normalized_name = name.replace("_", "").replace("-", "").lower()
        for key in comm_type_mapping:
            if key in normalized_name:
                return comm_type_mapping[key]
        raise ValueError(
            f"'{name}' not found in collective communication mapping. "
            "Please add this collective communication name to the mapping."
        )

    def is_root_node(self, node: ChakraNode) -> bool:
        """
        Determine whether a given node is a root node in the execution trace.

        In the context of PyTorch execution traces, root nodes are the starting points of execution graphs or execution
        traces. These nodes typically do not have parent nodes and act as the original sources of execution flow. This
        method identifies such root nodes based on their names. Specifically, nodes with names indicating they are part
        of the PyTorch execution graph or execution trace threads are considered root nodes.

        Args:
            node (ChakraNode): The node to be evaluated.

        Returns:
            bool: True if the node is a root node, False otherwise.
        """
        return node.name in ["[pytorch|profiler|execution_graph|thread]", "[pytorch|profiler|execution_trace|thread]"]

    def convert_ctrl_dep_to_data_dep(  # noqa: C901
        self,
        pytorch_nodes: Dict[int, PyTorchNode],
        chakra_nodes: Dict[int, ChakraNode],
        chakra_node: ChakraNode,
    ) -> None:
        """
        Convert control dependencies to data dependencies in Chakra nodes.

        Traverse nodes based on control dependencies (parent nodes) and encode data dependencies appropriately. This
        method is crucial for converting the dependency structure from PyTorch execution traces to Chakra execution
        traces. In PyTorch traces, control dependencies are represented by a parent field in each node, denoting the
        parent node ID. This structure indicates which functions (operators) are called by a particular operator.

        In contrast, Chakra execution traces, while retaining control dependencies for compatibility, primarily rely on
        data dependencies to represent relationships between nodes. Data dependencies in Chakra are more broadly
        defined compared to those in PyTorch, where they are implicitly encoded in tensor input-output relationships.
        In Chakra, data dependencies are explicit and represent a general dependency between nodes.

        To convert PyTorch's control dependencies to Chakra's data dependencies, a Depth-First Search (DFS) is
        performed. The DFS traversal starts from a given Chakra node, traversing through its children (based on control
        dependencies). During traversal, data dependencies are encoded by linking nodes that have been visited in
        sequence. These dependencies form a chain, mirroring the function call order from the PyTorch trace.

        Special attention is given to the types of nodes involved. CPU and label nodes (non-GPU) in PyTorch can only
        depend on other CPU or label nodes. However, GPU nodes can depend on any type of node. Thus, while traversing,
        if a GPU node is encountered, it can establish a data dependency with the last visited node of any type. For
        CPU and label nodes, the dependency is only established with the last visited non-GPU node. This distinction
        ensures that the converted dependencies accurately reflect the execution dynamics of the original PyTorch trace
        within the Chakra framework.

        Additionally, this method enforces sequential dependencies between GPU operators within the same stream. It
        ensures that the execution order of GPU operators is preserved in the Chakra trace, reflecting the sequential
        execution within the same GPU stream in the original PyTorch trace.

        Furthermore, inter-thread dependencies are explicitly encoded in the Chakra execution traces. This feature
        allows for the representation of dependencies across different CPU threads, which are observed in Kineto traces
        via chrome://tracing. These dependencies are crucial for understanding the interaction between CPU threads and
        ensuring accurate modeling and analysis of concurrent operations within the Chakra framework.

        Args:
            pytorch_nodes (Dict[int, PyTorchNode]): Dictionary of PyTorch nodes.
            chakra_nodes (Dict[int, ChakraNode]): Dictionary of Chakra nodes.
            chakra_node (ChakraNode): The starting node for the traversal and dependency processing.
        """
        visited: Set[int] = set()
        stack: List[ChakraNode] = [chakra_node]
        last_visited_non_gpu: Optional[ChakraNode] = None
        last_visited_any: Optional[ChakraNode] = None

        while stack:
            current_node = stack.pop()
            if current_node.id in visited:
                continue

            visited.add(current_node.id)
            pytorch_node = pytorch_nodes.get(current_node.id)
            if not pytorch_node:
                continue

            node_op_type = pytorch_node.get_op_type()

            if node_op_type == PyTorchNodeType.GPU_OP:
                if last_visited_any and last_visited_any.id not in current_node.data_deps:
                    current_node.data_deps.append(last_visited_any.id)
                    logging.debug(
                        f"GPU Node ID {current_node.id} now has a data dependency on Node ID {last_visited_any.id}"
                    )
                last_visited_any = last_visited_non_gpu
            else:
                if pytorch_node.inter_thread_dep:
                    dep_id = pytorch_node.inter_thread_dep
                    if dep_id not in current_node.data_deps:
                        current_node.data_deps.append(dep_id)
                        logging.debug(
                            f"CPU Node ID {current_node.id} now has an inter-thread data dependency on Node ID {dep_id}"
                        )
                if last_visited_non_gpu and last_visited_non_gpu.id not in current_node.data_deps:
                    current_node.data_deps.append(last_visited_non_gpu.id)
                    logging.debug(
                        f"CPU Node ID {current_node.id} now has a data dependency on non-GPU Node ID "
                        f"{last_visited_non_gpu.id}"
                    )
                last_visited_non_gpu = current_node
                last_visited_any = current_node

            children_chakra_ids = [child.id for child in pytorch_node.children]
            for child_chakra_id in sorted(children_chakra_ids, reverse=True):
                child_chakra_node = chakra_nodes.get(child_chakra_id)
                if child_chakra_node and child_chakra_node.id not in visited:
                    stack.append(child_chakra_node)

    def remove_dangling_nodes(self, chakra_nodes: Dict[int, ChakraNode]) -> Dict[int, ChakraNode]:
        """
        Remove any dangling nodes from the chakra_nodes dictionary.

        A node is considered dangling if it has no parents and no children.

        Args:
            chakra_nodes (Dict[int, ChakraNode]): Dictionary of Chakra nodes.

        Returns:
            Dict[int, ChakraNode]: Updated dictionary of Chakra nodes with dangling nodes removed.
        """
        parent_ids = set()
        for node in chakra_nodes.values():
            parent_ids.update(node.data_deps)

        dangling_nodes = [
            node_id for node_id, node in chakra_nodes.items() if node_id not in parent_ids and not node.data_deps
        ]
        for node_id in dangling_nodes:
            del chakra_nodes[node_id]

        if dangling_nodes:
            logging.info(f"Identified and removed {len(dangling_nodes)} dangling nodes:")
            for node_id in dangling_nodes:
                logging.info(f" - Node ID {node_id}")

        return chakra_nodes

    def update_parent_to_children_map(self, chakra_nodes: Dict[int, ChakraNode]) -> Dict[int, List[int]]:
        """
        Update the parent_to_children_map based on the data dependencies of each node.

        This map is used to efficiently simulate node execution based on data dependencies.
        """
        parent_to_children_map = {}
        for node_id, node in chakra_nodes.items():
            for dep_id in node.data_deps:
                if dep_id not in parent_to_children_map:
                    parent_to_children_map[dep_id] = []
                parent_to_children_map[dep_id].append(node_id)
        return parent_to_children_map

    def identify_cyclic_dependencies(self, chakra_nodes: Dict[int, ChakraNode]) -> None:
        """
        Identify if there are any cyclic dependencies among Chakra nodes.

        This method checks for cycles in the graph of Chakra nodes using a depth-first search (DFS) algorithm. It logs
        an error message and raises an exception if a cycle is detected, ensuring the graph is a Directed Acyclic Graph
        (DAG).

        Raises
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
                cycle_nodes = " -> ".join([chakra_nodes[n].name for n in path + [node_id]])
                logging.error(f"Cyclic dependency detected: {cycle_nodes}")
                return True
            if node_id in visited:
                return False

            visited.add(node_id)
            stack.add(node_id)
            path.append(node_id)
            for child_id in chakra_nodes[node_id].data_deps:
                if dfs(child_id, path.copy()):
                    return True
            stack.remove(node_id)
            path.pop()
            return False

        for node_id in chakra_nodes:
            if dfs(node_id, []):
                raise Exception(f"Cyclic dependency detected starting from node {chakra_nodes[node_id].name}")

    def write_chakra_et(
        self,
        chakra_et: IO[bytes],
        pytorch_schema: str,
        pytorch_pid: int,
        pytorch_time: str,
        pytorch_start_ts: int,
        pytorch_finish_ts: int,
        chakra_nodes: Dict[int, ChakraNode],
    ) -> None:
        """
        Write the Chakra execution trace by encoding global metadata and nodes.

        Encode and write both the metadata and individual nodes to create a
        complete execution trace.
        """
        logging.info("Writing Chakra execution trace.")
        self._write_global_metadata(
            chakra_et, pytorch_schema, pytorch_pid, pytorch_time, pytorch_start_ts, pytorch_finish_ts
        )
        self._encode_and_write_nodes(chakra_et, chakra_nodes)
        logging.info("Chakra execution trace writing completed.")

    def _write_global_metadata(
        self,
        chakra_et: IO[bytes],
        pytorch_schema: str,
        pytorch_pid: int,
        pytorch_time: str,
        pytorch_start_ts: int,
        pytorch_finish_ts: int,
    ) -> None:
        """
        Encode and write global metadata for the Chakra execution trace.

        This process includes encoding metadata like schema, process ID, timestamps,
        and other relevant information for the Chakra execution trace.
        """
        logging.info("Encoding global metadata for Chakra execution trace.")
        global_metadata = GlobalMetadata(
            attr=[
                ChakraAttr(name="schema", string_val=pytorch_schema),
                ChakraAttr(name="pid", uint64_val=pytorch_pid),
                ChakraAttr(name="time", string_val=pytorch_time),
                ChakraAttr(name="start_ts", uint64_val=pytorch_start_ts),
                ChakraAttr(name="finish_ts", uint64_val=pytorch_finish_ts),
            ]
        )
        encode_message(chakra_et, global_metadata)

    def _encode_and_write_nodes(self, chakra_et: IO[bytes], chakra_nodes: Dict[int, ChakraNode]) -> None:
        """
        Encode and write nodes for the Chakra execution trace.

        Each node from the PyTorch execution trace is encoded and written into the Chakra format. This includes node
        IDs, names, types, dependencies, and other attributes.
        """
        logging.info("Encoding and writing nodes for Chakra execution trace.")
        seen_nids = set()
        for nid in sorted(chakra_nodes.keys()):
            if nid in seen_nids:
                err_msg = f"Duplicate NID {nid} detected in Chakra nodes."
                logging.error(err_msg)
                raise ValueError(err_msg)
            seen_nids.add(nid)
            chakra_node = chakra_nodes[nid]
            encode_message(chakra_et, chakra_node)

    def close_chakra_execution_trace(self, chakra_et: IO[bytes]) -> None:
        """
        Close the Chakra execution trace file if it is open.

        Ensure proper closure of the trace file to preserve data integrity.

        Args:
            chakra_et (IO[bytes]): File handle for the Chakra execution trace output file.
        """
        logging.info("Closing Chakra execution trace file.")
        if chakra_et and not chakra_et.closed:
            chakra_et.close()

    def simulate_execution(
        self,
        chakra_nodes: Dict[int, ChakraNode],
        pytorch_nodes: Dict[int, PyTorchNode],
        parent_to_children_map: Dict[int, List[int]],
    ) -> None:
        """
        Simulate the execution of Chakra nodes based on data dependencies.

        This method considers both CPU and GPU nodes. Nodes are issued for execution based on the readiness determined
        by dependency resolution. A simplistic global clock is used to model the execution time.

        Args:
            chakra_nodes (Dict[int, ChakraNode]): The Chakra nodes to be simulated.
            pytorch_nodes (Dict[int, PyTorchNode]): The PyTorch nodes to reference for additional information.
            parent_to_children_map (Dict[int, List[int]]): Mapping from parent node IDs to their child node IDs.
        """
        logging.info("Simulating execution of Chakra nodes based on data dependencies.")

        ready_cpu_nodes = [
            (node_id, chakra_nodes[node_id])
            for node_id in chakra_nodes
            if not chakra_nodes[node_id].data_deps and not pytorch_nodes[node_id].is_gpu_op()
        ]
        ready_gpu_nodes = [
            (node_id, chakra_nodes[node_id])
            for node_id in chakra_nodes
            if not chakra_nodes[node_id].data_deps and pytorch_nodes[node_id].is_gpu_op()
        ]
        ready_cpu_nodes.sort(key=lambda x: x[1].id)
        ready_gpu_nodes.sort(key=lambda x: x[1].id)

        issued_nodes: Set[int] = set()
        current_cpu_node: Optional[Tuple[int, int]] = None
        current_gpu_node: Optional[Tuple[int, int]] = None

        current_time: int = 0  # Simulated global clock in microseconds

        while any([ready_cpu_nodes, ready_gpu_nodes, current_cpu_node, current_gpu_node]):
            if ready_cpu_nodes and not current_cpu_node:
                cpu_node_id, cpu_node = ready_cpu_nodes.pop(0)
                current_cpu_node = (cpu_node_id, current_time)
                issued_nodes.add(cpu_node_id)
                logging.info(
                    f"Issuing CPU Node ID {cpu_node_id} ({cpu_node.name}) at {current_time}us with duration "
                    f"{cpu_node.duration_micros}us"
                )

            if ready_gpu_nodes and not current_gpu_node:
                gpu_node_id, gpu_node = ready_gpu_nodes.pop(0)
                current_gpu_node = (gpu_node_id, current_time)
                issued_nodes.add(gpu_node_id)
                logging.info(
                    f"Issuing GPU Node ID {gpu_node_id} ({gpu_node.name}) at {current_time}us with duration "
                    f"{gpu_node.duration_micros}us"
                )

            current_time += 1

            if (
                current_cpu_node
                and current_time - current_cpu_node[1] >= chakra_nodes[current_cpu_node[0]].duration_micros
            ):
                logging.info(f"CPU Node ID {current_cpu_node[0]} completed at {current_time}us")
                current_cpu_node = None

            if (
                current_gpu_node
                and current_time - current_gpu_node[1] >= chakra_nodes[current_gpu_node[0]].duration_micros
            ):
                logging.info(f"GPU Node ID {current_gpu_node[0]} completed at {current_time}us")
                current_gpu_node = None

            for node_id in list(issued_nodes):
                children_ids = parent_to_children_map.get(node_id, [])
                for child_id in children_ids:
                    child_node = chakra_nodes[child_id]
                    child_node.data_deps.remove(node_id)
                    if not child_node.data_deps:
                        if not pytorch_nodes[child_id].is_gpu_op():
                            ready_cpu_nodes.append((child_id, child_node))
                        else:
                            ready_gpu_nodes.append((child_id, child_node))

            issued_nodes.clear()

        logging.info("Simulation of Chakra node execution completed.")
