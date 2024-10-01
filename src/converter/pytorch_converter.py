import json
import logging
import os
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
    Converter for transforming Chakra host + device execution traces in JSON format into the Chakra protobuf format.

    This class is responsible for converting the execution traces collected from Chakra host + device in JSON format
    into the Chakra protobuf format. The input JSON traces are generated by trace_link and lack the proper dependencies
    for simulation. This converter handles the conversion of JSON nodes to protobuf nodes, identification and encoding
    of dependencies, removal of dangling nodes, and writing the final protobuf trace to the output file.
    """

    def convert(self, input_filename: str, output_filename: str, simulate: bool, dump_collective_nodes: bool) -> None:
        """
        Convert Chakra host + device execution traces in JSON format into the Chakra protobuf format.

        Args:
            input_filename (str): Input Chakra host + device execution trace in the JSON format.
            output_filename (str): Output Chakra host + device execution trace in the protobuf format.
            simulate (bool): Flag to indicate whether to simulate the execution of the converted trace. If True,
                the method will simulate the execution after writing the protobuf trace to the output file.
            dump_collective_nodes (bool): Flag to indicate whether to dump all collective opreations basic
                metadata to a csv file. If True, the method will dump the information after writing the protobuf 
                trace to the output file. The flag assumes the filename is {some-string}_{rank}.{some-string}
                because the dump file uses the rank to create unique file for each rank.
        """
        json_trace = self.load_json_execution_traces(input_filename)
        json_metadata, json_node_map = self.parse_json_trace(json_trace)

        protobuf_node_map = {}
        self.convert_json_to_protobuf_nodes(json_node_map, protobuf_node_map)
        root_node_list = [node for node in protobuf_node_map.values() if self.is_root_node(node.name)]
        for root_node in root_node_list:
            self.convert_ctrl_dep_to_data_dep(json_node_map, protobuf_node_map, root_node)

        protobuf_node_map = self.remove_dangling_nodes(protobuf_node_map)

        parent_to_children_map = self.update_parent_to_children_map(protobuf_node_map)

        self.identify_cyclic_dependencies(protobuf_node_map)

        self.write_protobuf_execution_trace(output_filename, json_metadata, protobuf_node_map)

        if dump_collective_nodes:
            self.dump_collective_operation(protobuf_node_map, input_filename)

        if simulate:
            self.simulate_execution(json_node_map, protobuf_node_map, parent_to_children_map)

    def load_json_execution_traces(self, input_filename: str) -> Dict:
        """
        Load Chakra host + device execution traces in JSON format from a file.

        Args:
            input_filename (str): Input Chakra host + device execution trace in the JSON format.

        Returns:
            Dict: The loaded Chakra host + device execution trace data.
        """
        logging.debug(f"Loading Chakra host + device execution traces in JSON format from file: {input_filename}")
        with open(input_filename, "r") as json_file:
            return json.load(json_file)

    def parse_json_trace(self, json_trace: Dict) -> Tuple[Dict, Dict[int, PyTorchNode]]:
        """
        Parse and instantiate PyTorch nodes from execution trace data.

        Args:
            json_trace (Dict): The execution trace data.

        Extract node information, sort nodes by timestamp, and establish parent-child relationships among them.

        Returns:
            Tuple: A tuple containing JSON metadata and dictionary of PyTorch node objects.
        """
        logging.debug("Extracting and processing node data from execution trace.")

        json_metadata = {
            "schema": json_trace["schema"],
            "pid": json_trace["pid"],
            "time": json_trace["time"],
            "start_ts": json_trace["start_ts"],
            "finish_ts": json_trace["finish_ts"],
        }

        logging.debug(f"Parsed JSON metadata: {json_metadata}")

        json_nodes = json_trace["nodes"]
        node_count = len(json_nodes)
        logging.debug(f"Number of nodes in JSON trace: {node_count}")

        json_node_map = {node_data["id"]: PyTorchNode(json_trace["schema"], node_data) for node_data in json_nodes}
        json_node_root_nids = []
        json_node_map = self.establish_parent_child_relationships(json_node_map, json_node_root_nids)
        return json_metadata, json_node_map

    def establish_parent_child_relationships(
        self, json_node_map: Dict[int, PyTorchNode], json_node_root_nids: List[int]
    ) -> Dict[int, PyTorchNode]:
        """
        Establish parent-child relationships among JSON nodes and count the node types.

        In Chakra host execution traces, the parent-child relationship is represented in the ctrl dep or parent field.
        The name of the field is determined by the schema version of the Chakra host execution traces. When a function
        calls multiple functions, the callee functions appear as children nodes in the control dependency. This method
        is responsible for reading such dependencies and updating the field accordingly.

        Args:
            json_node_map (Dict[int, PyTorchNode]): Dictionary of JSON node objects.
            json_node_root_nids (List[int]): List to store root node IDs.

        Returns:
            Dict[int, PyTorchNode]: Dictionary of JSON nodes with established relationships.
        """
        node_type_counts = {
            "total_op": 0,
            "cpu_op": 0,
            "gpu_op": 0,
            "record_param_comms_op": 0,
            "nccl_op": 0,
            "root_op": 0,
        }

        for json_node in json_node_map.values():
            parent_id = json_node.parent
            if parent_id in json_node_map:
                self.process_parent_child_relationships(json_node_map, json_node, parent_id)

            if self.is_root_node(json_node.name):
                json_node_root_nids.append(json_node.id)
                node_type_counts["root_op"] += 1

            node_type_counts["total_op"] += 1
            if json_node.is_cpu_op():
                node_type_counts["cpu_op"] += 1
            if json_node.is_gpu_op():
                node_type_counts["gpu_op"] += 1
            if json_node.is_record_param_comms_op():
                node_type_counts["record_param_comms_op"] += 1
            if json_node.is_nccl_op():
                node_type_counts["nccl_op"] += 1

        for node_type, count in node_type_counts.items():
            logging.debug(f"{node_type}: {count}")

        return json_node_map

    def is_root_node(self, node_name: str) -> bool:
        """
        Check if a given node name corresponds to a root node in the Chakra host execution trace.

        In the context of Chakra host execution traces, root nodes are the starting points of execution traces.
        These nodes typically do not have parent nodes and act as the original sources of execution flow.
        The execution trace has a call-stack-like structure in the ctrl-dep field (or parent field), and root
        nodes should be identified during the process of conversion.

        Chakra host execution traces may have multiple root nodes. These root nodes can be identified with specific
        keywords as shown in this method. Identifying root nodes is essential for correctly converting and representing
        the execution trace in the Chakra protobuf format.

        Args:
            node_name (str): The name of the node to check.

        Returns:
            bool: True if the node name corresponds to a root node, False otherwise.
        """
        return node_name in [
            "[pytorch|profiler|execution_graph|thread]",
            "[pytorch|profiler|execution_trace|thread]",
        ]

    def process_parent_child_relationships(
        self, json_node_map: Dict[int, PyTorchNode], json_node: PyTorchNode, parent_id: int
    ) -> None:
        """
        Process the parent-child relationships for Chakra JSON nodes.

        Args:
            json_node_map (Dict[int, PyTorchNode]): Dictionary of JSON node objects.
            json_node (PyTorchNode): The current JSON node being processed.
            parent_id (int): The ID of the parent node.
        """
        parent_node = json_node_map[parent_id]
        parent_node.add_child(json_node)

        if json_node.is_gpu_op():
            parent_node.add_gpu_child(json_node)

        if json_node.is_record_param_comms_op():
            # Add the record_param_comms node to the parent.
            # These operators act as metadata operators between the launcher and the actual communication operator.
            # This registration allows the converter to easily identify the communication operator to use.
            parent_node.record_param_comms_node = json_node

        if json_node.is_nccl_op():
            # Add the NCCL node to the parent.
            # NCCL operators are actual communication operators.
            # This registration allows the converter to easily identify the communication operator to use.
            parent_node.nccl_node = json_node

    def convert_json_to_protobuf_nodes(
        self, json_node_map: Dict[int, PyTorchNode], protobuf_node_map: Dict[int, ChakraNode]
    ) -> None:
        """
        Convert JSON nodes to Protobuf nodes.

        This method directly converts JSON nodes to Protobuf nodes without considering any dependencies. Dependencies
        will be handled by the convert_ctrl_dep_to_data_dep method.

        Args:
            json_node_map (Dict[int, PyTorchNode]): Dictionary of JSON nodes to be converted.
            protobuf_node_map (Dict[int, ChakraNode]): Dictionary where the converted Protobuf nodes will be stored.
        """
        for _, json_node in json_node_map.items():
            if (
                (json_node.get_op_type() == PyTorchNodeType.CPU_OP)
                or (json_node.get_op_type() == PyTorchNodeType.LABEL)
                or (json_node.get_op_type() == PyTorchNodeType.METADATA)
            ):
                chakra_node = self.convert_json_to_protobuf_node(json_node_map, protobuf_node_map, json_node)
                protobuf_node_map[chakra_node.id] = chakra_node

                for pytorch_gpu_node in json_node.gpu_children:
                    chakra_gpu_node = self.convert_json_to_protobuf_node(
                        json_node_map, protobuf_node_map, pytorch_gpu_node
                    )

                    if chakra_gpu_node.type == COMM_COLL_NODE:
                        collective_comm_type = self.get_collective_comm_type(pytorch_gpu_node.name)
                        chakra_gpu_node.attr.extend(
                            [
                                ChakraAttr(name="comm_type", int64_val=collective_comm_type),
                                ChakraAttr(name="comm_size", int64_val=pytorch_gpu_node.comm_size),
                                *( [ChakraAttr(name="pg_name", string_val=pytorch_gpu_node.pg_name)] if pytorch_gpu_node.pg_name != "" else [] ),
                            ]
                        )

                    elif chakra_gpu_node.type in {COMM_SEND_NODE, COMM_RECV_NODE}:
                        chakra_gpu_node.attr.extend(
                            [
                                ChakraAttr(name="comm_size", int64_val=pytorch_gpu_node.comm_size),
                                *( [ChakraAttr(name="pg_name", string_val=pytorch_gpu_node.pg_name)] if pytorch_gpu_node.pg_name != "" else [] ),
                            ]
                        )

                    protobuf_node_map[chakra_gpu_node.id] = chakra_gpu_node

    def convert_json_to_protobuf_node(
        self,
        json_node_map: Dict[int, PyTorchNode],
        protobuf_node_map: Dict[int, ChakraNode],
        json_node: PyTorchNode,
    ) -> ChakraNode:
        """
        Convert a JSON node (PyTorchNode) to a protobuf node (ChakraNode).

        This method takes a JSON node from the Chakra host + device execution trace and converts it to a protobuf node.
        The conversion includes transferring attributes, types, and dependencies from the JSON node to the protobuf
        node. Special handling is performed for nodes covering more than 90% of the runtime, such as Optimizer.step,
        to filter them out.

        Args:
            json_node_map (Dict[int, PyTorchNode]): Dictionary of JSON nodes.
            protobuf_node_map (Dict[int, ChakraNode]): Dictionary of protobuf nodes.
            json_node (PyTorchNode): The JSON node to convert.

        Returns:
            ChakraNode: The converted protobuf node.
        """
        logging.debug(f"Converting JSON node ID {json_node.id} to protobuf node.")

        protobuf_node = ChakraNode()
        protobuf_node.id = json_node.id
        protobuf_node.name = json_node.name
        protobuf_node.type = self.get_protobuf_node_type_from_json_node(json_node_map, json_node)
        if json_node.parent in protobuf_node_map:
            protobuf_node.ctrl_deps.append(json_node.parent)
        protobuf_node.duration_micros = int(json_node.exclusive_dur)

        # Handle nodes covering more than 90% of the runtime
        if "Optimizer.step" in json_node.name:
            protobuf_node.duration_micros = 0

        protobuf_node.inputs.values = str(json_node.inputs["values"])
        protobuf_node.inputs.shapes = str(json_node.inputs["shapes"])
        protobuf_node.inputs.types = str(json_node.inputs["types"])
        protobuf_node.outputs.values = str(json_node.outputs["values"])
        protobuf_node.outputs.shapes = str(json_node.outputs["shapes"])
        protobuf_node.outputs.types = str(json_node.outputs["types"])
        protobuf_node.attr.extend(
            [
                ChakraAttr(name="rf_id", int64_val=json_node.rf_id),
                ChakraAttr(name="fw_parent", int64_val=json_node.fw_parent),
                ChakraAttr(name="seq_id", int64_val=json_node.seq_id),
                ChakraAttr(name="scope", int64_val=json_node.scope),
                ChakraAttr(name="tid", int64_val=json_node.tid),
                ChakraAttr(name="fw_tid", int64_val=json_node.fw_tid),
                ChakraAttr(name="op_schema", string_val=json_node.op_schema),
                ChakraAttr(name="is_cpu_op", bool_val=not json_node.is_gpu_op()),
            ]
        )
        if json_node.stream is not None:
            protobuf_node.attr.append(ChakraAttr(name="stream", int64_val=json_node.stream))

        return protobuf_node

    def get_protobuf_node_type_from_json_node(
        self, json_node_map: Dict[int, PyTorchNode], json_node: PyTorchNode
    ) -> int:
        """
        Determine the Protobuf node type from a Chakra node.

        Args:
            json_node_map (Dict[int, PyTorchNode]): Dictionary of JSON nodes.
            json_node (PyTorchNode): The JSON node to determine the type of.

        Returns:
            int: The corresponding Chakra node type.
        """
        if json_node.is_gpu_op():
            if "ncclDevKernel_SendRecv" in json_node.name:
                parent_node = json_node_map[json_node.parent]
                keyword = (
                    json_node_map[parent_node.parent].name
                    if parent_node.name == "record_param_comms"
                    else parent_node.name
                )
                if "send" in keyword:
                    return COMM_SEND_NODE
                if "recv" in keyword:
                    return COMM_RECV_NODE
            if "ncclKernel" in json_node.name or "ncclDevKernel" in json_node.name:
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
            f"The name '{name}' does not correspond to a recognized collective communication type. "
            "The converter determines collective communication types based on the node name of a GPU operator. "
            f"However, it failed to identify the type for '{name}'. "
            "If this is a valid collective communication type, please update the converter code to include "
            "the appropriate mapping in comm_type_mapping. "
            "Investigate this issue or report it on GitHub for further assistance."
        )

    def convert_ctrl_dep_to_data_dep(
        self,
        json_node_map: Dict[int, PyTorchNode],
        protobuf_node_map: Dict[int, ChakraNode],
        chakra_node: ChakraNode,
    ) -> None:
        """
        Convert control dependencies to data dependencies in Chakra nodes.

        This method converts the control dependencies found in Chakra host traces collected from PyTorch
        into data dependencies, which are required by most simulators. In Chakra host traces, control dependencies
        represent the caller-callee relationship during execution. When an operator calls other operators,
        the caller becomes the parent, and the called operators become children. The order of these function calls
        is reflected in their node IDs, with lower IDs indicating earlier execution.

        Simulators, however, expect dependencies to represent the actual execution order, which is encoded in the
        data dependency field. This method performs this conversion by traversing the control dependencies and
        encoding them as data dependencies.

        Additionally, this method handles inter-thread dependencies. These dependencies are captured and encoded to
        ensure that the execution flow across multiple threads is correctly represented.

        Args:
            json_node_map (Dict[int, PyTorchNode]): Dictionary of PyTorch nodes.
            protobuf_node_map (Dict[int, ChakraNode]): Dictionary of Chakra nodes.
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
            json_node = json_node_map.get(current_node.id)
            if not json_node:
                continue

            node_op_type = json_node.get_op_type()

            if node_op_type == PyTorchNodeType.GPU_OP:
                if last_visited_any and last_visited_any.id not in current_node.data_deps:
                    current_node.data_deps.append(last_visited_any.id)
                    logging.debug(
                        f"GPU Node ID {current_node.id} now has a data dependency on Node ID {last_visited_any.id}"
                    )
                last_visited_any = last_visited_non_gpu
            else:
                if json_node.inter_thread_dep:
                    dep_id = json_node.inter_thread_dep
                    if dep_id not in current_node.data_deps:
                        current_node.data_deps.append(dep_id)
                        logging.debug(
                            f"CPU Node ID {current_node.id} now has an inter-thread data dependency on Node ID "
                            f"{dep_id}"
                        )
                if last_visited_non_gpu and last_visited_non_gpu.id not in current_node.data_deps:
                    current_node.data_deps.append(last_visited_non_gpu.id)
                    logging.debug(
                        f"CPU Node ID {current_node.id} now has a data dependency on non-GPU Node ID "
                        f"{last_visited_non_gpu.id}"
                    )
                last_visited_non_gpu = current_node
                last_visited_any = current_node

            children_chakra_ids = [child.id for child in json_node.children]
            for child_chakra_id in sorted(children_chakra_ids, reverse=True):
                child_chakra_node = protobuf_node_map.get(child_chakra_id)
                if child_chakra_node and child_chakra_node.id not in visited:
                    stack.append(child_chakra_node)

    def remove_dangling_nodes(self, protobuf_node_map: Dict[int, ChakraNode]) -> Dict[int, ChakraNode]:
        """
        Remove any dangling nodes from the protobuf_node_map dictionary.

        Dangling nodes are nodes that have neither children nor parents. These nodes are identified after the
        conversion and are typically unnecessary. Removing these nodes simplifies simulation and avoids potential
        complications.

        Args:
            protobuf_node_map (Dict[int, ChakraNode]): Dictionary of protobuf nodes.

        Returns:
            Dict[int, ChakraNode]: Updated dictionary of protobuf nodes with dangling nodes removed.
        """
        parent_ids = set()
        for node in protobuf_node_map.values():
            parent_ids.update(node.data_deps)

        dangling_nodes = [
            node_id for node_id, node in protobuf_node_map.items() if node_id not in parent_ids and not node.data_deps
        ]
        for node_id in dangling_nodes:
            del protobuf_node_map[node_id]

        if dangling_nodes:
            logging.debug(f"Identified and removed {len(dangling_nodes)} dangling nodes:")
            for node_id in dangling_nodes:
                logging.debug(f" - Node ID {node_id}")

        return protobuf_node_map

    def update_parent_to_children_map(self, protobuf_node_map: Dict[int, ChakraNode]) -> Dict[int, List[int]]:
        """
        Update the parent_to_children_map based on the data dependencies of each node.

        This map is used to efficiently simulate node execution based on data dependencies.
        """
        parent_to_children_map = {}
        for node_id, node in protobuf_node_map.items():
            for dep_id in node.data_deps:
                if dep_id not in parent_to_children_map:
                    parent_to_children_map[dep_id] = []
                parent_to_children_map[dep_id].append(node_id)
        return parent_to_children_map

    def identify_cyclic_dependencies(self, protobuf_node_map: Dict[int, ChakraNode]) -> None:
        """
        Identify if there are any cyclic dependencies among protobuf nodes.

        This method checks for cycles in the graph of protobuf nodes using a depth-first search (DFS) algorithm. It
        logs an error message and raises an exception if a cycle is detected, ensuring the graph is a Directed Acyclic
        Graph (DAG).

        Args:
            protobuf_node_map (Dict[int, ChakraNode]): Dictionary of protobuf nodes to check for cyclic dependencies.

        Raises:
            Exception: If a cyclic dependency is detected among the protobuf nodes.
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
                cycle_nodes = " -> ".join([protobuf_node_map[n].name for n in path + [node_id]])
                err_msg = (
                    f"Cyclic dependency detected: {cycle_nodes}. The conversion failed because a cyclic dependency "
                    f"was detected. Cyclic dependencies should not exist. The input and output traces must form a "
                    f"Directed Acyclic Graph (DAG). This is essential for simulation; otherwise, simulators cannot "
                    f"resolve the next dependency-free node and will hang. This indicates a bug in the conversion "
                    f"process. Please investigate or report this issue on GitHub."
                )
                logging.error(err_msg)
                return True
            if node_id in visited:
                return False

            visited.add(node_id)
            stack.add(node_id)
            path.append(node_id)
            for child_id in protobuf_node_map[node_id].data_deps:
                if dfs(child_id, path.copy()):
                    return True
            stack.remove(node_id)
            path.pop()
            return False

        for node_id in protobuf_node_map:
            if dfs(node_id, []):
                err_msg = (
                    "Cyclic dependency detected. The conversion failed because a cyclic dependency "
                    "was detected. Cyclic dependencies should not exist. The input and output traces must form a "
                    "Directed Acyclic Graph (DAG). This is essential for simulation; otherwise, simulators cannot "
                    "resolve the next dependency-free node and will hang. This indicates a bug in the conversion "
                    "process. Please investigate or report this issue on GitHub."
                )
                logging.error(err_msg)
                raise Exception(err_msg)

    def write_protobuf_execution_trace(
        self,
        output_filename: str,
        json_metadata: Dict,
        protobuf_node_map: Dict[int, ChakraNode],
    ) -> None:
        """
        Write the Chakra execution trace by encoding global metadata and nodes.

        Encode and write both the metadata and individual nodes to create a complete execution trace.

        Args:
            output_filename (str): The name of the output file for the protobuf execution trace.
            json_metadata (Dict): The metadata from the JSON trace.
            protobuf_node_map (Dict[int, ChakraNode]): The converted Chakra nodes.
        """
        logging.debug(f"Opening Chakra execution trace file: {output_filename}")
        with open(output_filename, "wb") as protobuf_et:
            logging.debug("Writing Chakra execution trace.")
            self.write_global_metadata(protobuf_et, json_metadata)
            self.encode_and_write_nodes(protobuf_et, protobuf_node_map)
            logging.debug("Chakra execution trace writing completed.")

    def write_global_metadata(
        self,
        protobuf_et: IO[bytes],
        metadata: Dict,
    ) -> None:
        """
        Encode and write global metadata for the Chakra execution trace.

        Args:
            protobuf_et (IO[bytes]): The output file handle for the protobuf execution trace.
            metadata (Dict): The metadata dictionary containing schema, pid, time, start_ts, and finish_ts.
        """
        logging.debug("Encoding global metadata for Chakra execution trace.")
        global_metadata = GlobalMetadata(
            attr=[
                ChakraAttr(name="schema", string_val=metadata["schema"]),
                ChakraAttr(name="pid", uint64_val=metadata["pid"]),
                ChakraAttr(name="time", string_val=metadata["time"]),
                ChakraAttr(name="start_ts", uint64_val=metadata["start_ts"]),
                ChakraAttr(name="finish_ts", uint64_val=metadata["finish_ts"]),
            ]
        )
        encode_message(protobuf_et, global_metadata)

    def encode_and_write_nodes(self, protobuf_et: IO[bytes], protobuf_node_map: Dict[int, ChakraNode]) -> None:
        """
        Encode and write nodes for the Chakra host + device execution trace in the protobuf format.

        Each node from the JSON execution trace is encoded and written into the protobuf format. This includes node
        IDs, names, types, dependencies, and other attributes.

        Args:
            protobuf_et (IO[bytes]): The output file handle for the protobuf execution trace.
            protobuf_node_map (Dict[int, ChakraNode]): Dictionary of protobuf nodes to be encoded and written.
        """
        logging.debug("Encoding and writing nodes for Chakra execution trace.")
        seen_nids = set()
        for nid in sorted(protobuf_node_map.keys()):
            if nid in seen_nids:
                err_msg = (
                    f"Duplicate NID {nid} detected in Chakra nodes while writing nodes to the file. "
                    f"Node IDs are expected to be unique, and encountering a duplicate indicates an issue in the "
                    f"conversion process. Please check if the same node was registered twice. If not, report this "
                    f"issue for further investigation."
                )
                logging.error(err_msg)
                raise ValueError(err_msg)
            seen_nids.add(nid)
            chakra_node = protobuf_node_map[nid]
            encode_message(protobuf_et, chakra_node)

    # ruff: noqa: C901
    def simulate_execution(
        self,
        json_node_map: Dict[int, PyTorchNode],
        protobuf_node_map: Dict[int, ChakraNode],
        parent_to_children_map: Dict[int, List[int]],
    ) -> None:
        """
        Simulate the execution of Chakra nodes based on data dependencies.

        This method considers both CPU and GPU nodes. Nodes are issued for execution based on the readiness determined
        by dependency resolution. A simplistic global clock is used to model the execution time.

        Args:
            json_node_map (Dict[int, PyTorchNode]): The PyTorch nodes to reference for additional debugrmation.
            protobuf_node_map (Dict[int, ChakraNode]): The Chakra nodes to be simulated.
            parent_to_children_map (Dict[int, List[int]]): Mapping from parent node IDs to their child node IDs.
        """
        logging.debug("Simulating execution of Chakra nodes based on data dependencies.")

        ready_cpu_nodes = [
            (node_id, protobuf_node_map[node_id])
            for node_id in protobuf_node_map
            if not protobuf_node_map[node_id].data_deps and not json_node_map[node_id].is_gpu_op()
        ]
        ready_gpu_nodes = [
            (node_id, protobuf_node_map[node_id])
            for node_id in protobuf_node_map
            if not protobuf_node_map[node_id].data_deps and json_node_map[node_id].is_gpu_op()
        ]
        ready_cpu_nodes.sort(key=lambda x: x[1].id)
        ready_gpu_nodes.sort(key=lambda x: x[1].id)

        issued_nodes: Set[int] = set()
        current_cpu_node: Optional[Tuple[int, int]] = None
        current_gpu_nodes: Dict[int, Tuple[int, int]] = {}

        current_time: int = 0  # Simulated global clock in microseconds

        while any([ready_cpu_nodes, ready_gpu_nodes, current_cpu_node, current_gpu_nodes]):
            if ready_cpu_nodes and not current_cpu_node:
                cpu_node_id, cpu_node = ready_cpu_nodes.pop(0)
                current_cpu_node = (cpu_node_id, current_time)
                issued_nodes.add(cpu_node_id)
                tid = json_node_map[cpu_node_id].tid
                logging.debug(
                    f"Issuing CPU Node ID {cpu_node_id} ({cpu_node.name}) at {current_time}us with duration "
                    f"{cpu_node.duration_micros}us, tid: {tid}"
                )

            if ready_gpu_nodes:
                for gpu_node_id, gpu_node in ready_gpu_nodes[:]:
                    json_node = json_node_map[gpu_node_id]
                    stream_id = json_node.stream
                    if stream_id not in current_gpu_nodes:
                        ready_gpu_nodes.remove((gpu_node_id, gpu_node))
                        current_gpu_nodes[stream_id] = (gpu_node_id, current_time)
                        issued_nodes.add(gpu_node_id)
                        tid = f"stream {stream_id}"
                        logging.debug(
                            f"Issuing GPU Node ID {gpu_node_id} ({gpu_node.name}) at {current_time}us on stream "
                            f"{stream_id} with duration {gpu_node.duration_micros}us, tid: {tid}"
                        )

            current_time += 1

            if (
                current_cpu_node
                and current_time - current_cpu_node[1] >= protobuf_node_map[current_cpu_node[0]].duration_micros
            ):
                cpu_node_id, _ = current_cpu_node
                tid = json_node_map[cpu_node_id].tid
                logging.debug(f"CPU Node ID {cpu_node_id} completed at {current_time}us, tid: {tid}")
                current_cpu_node = None

            completed_streams = []
            for stream_id, (gpu_node_id, start_time) in current_gpu_nodes.items():
                if current_time - start_time >= protobuf_node_map[gpu_node_id].duration_micros:
                    logging.debug(
                        f"GPU Node ID {gpu_node_id} on stream {stream_id} completed at {current_time}us, "
                        f"tid: stream {stream_id}"
                    )
                    completed_streams.append(stream_id)

            for stream_id in completed_streams:
                del current_gpu_nodes[stream_id]

            for node_id in list(issued_nodes):
                children_ids = parent_to_children_map.get(node_id, [])
                for child_id in children_ids:
                    child_node = protobuf_node_map[child_id]
                    child_node.data_deps.remove(node_id)
                    if not child_node.data_deps:
                        if not json_node_map[child_id].is_gpu_op():
                            ready_cpu_nodes.append((child_id, child_node))
                        else:
                            ready_gpu_nodes.append((child_id, child_node))

            issued_nodes.clear()

        logging.debug("Simulation of Chakra node execution completed.")

    def dump_collective_operation(self, protobuf_node_map, filename):
        def rank_from_file_name(filename_str):
            basename = os.path.basename(filename_str)
            parts = basename.split(sep='_')
            return parts[1]
        try:
            rank = rank_from_file_name(filename)
        except Exception as e:
            raise ValueError(f"dump_collective_operation: assuming the filename to be: et_$rank$.json, but instead: {filename}")

        output = f"chakra_collectives_dump.{rank}.csv"
        with open(output, 'w') as f:
            f.write("rank,node_id,coll_name,comm_size,og_comm_size,pg_name,og_pg_name,root_rank,og_root_rank\n")
            node: ChakraNode
            for node in protobuf_node_map.values():
                if node.type is COMM_COLL_NODE:
                    node_name = node.name.replace(',','_') # protect the csv
                    comm_size = None
                    pg_name = None
                    root_rank=None
                    for p in node.attr:
                        if p.name == 'pg_name':
                            pg_name = p.string_val
                        elif p.name == 'comm_size':
                            comm_size = p.int64_val
                        elif p.name == 'root_rank':
                            root_rank=p.int32_val

                    og_pg_name=True
                    if pg_name is None:
                        og_pg_name = False
                        pg_name = 0

                    og_comm_size=True
                    if comm_size is None:
                        comm_size=0
                        og_comm_size=False

                    og_root_rank=True
                    if root_rank is None:
                        og_root_rank=False
                        root_rank=0

                    f.write(f"{rank},{node.id},{node_name},{comm_size},{og_comm_size},{pg_name},{og_pg_name},{root_rank},{og_root_rank}\n")
