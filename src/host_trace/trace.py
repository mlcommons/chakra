import logging
from typing import Dict, List, Tuple

from .node import HostNode


def parse_json_trace(json_trace: Dict) -> Tuple[Dict, Dict[int, HostNode]]:
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

    json_node_map = {node_data["id"]: HostNode(json_trace["schema"], node_data) for node_data in json_nodes}
    json_node_root_nids = []
    json_node_map = establish_parent_child_relationships(json_node_map, json_node_root_nids)
    return json_metadata, json_node_map


def establish_parent_child_relationships(
    json_node_map: Dict[int, HostNode], json_node_root_nids: List[int]
) -> Dict[int, HostNode]:
    """
    Establish parent-child relationships among JSON nodes and count the node types.

    In Chakra host execution traces, the parent-child relationship is represented in the ctrl dep or parent field.
    The name of the field is determined by the schema version of the Chakra host execution traces. When a function
    calls multiple functions, the callee functions appear as children nodes in the control dependency. This method
    is responsible for reading such dependencies and updating the field accordingly.

    Args:
        json_node_map (Dict[int, HostNode]): Dictionary of JSON node objects.
        json_node_root_nids (List[int]): List to store root node IDs.

    Returns:
        Dict[int, HostNode]: Dictionary of JSON nodes with established relationships.
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
            process_parent_child_relationships(json_node_map, json_node, parent_id)

        if is_root_node(json_node.name):
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


def is_root_node(node_name: str) -> bool:
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


def process_parent_child_relationships(json_node_map: Dict[int, HostNode], json_node: HostNode, parent_id: int) -> None:
    """
    Process the parent-child relationships for Chakra JSON nodes.

    Args:
        json_node_map (Dict[int, HostNode]): Dictionary of JSON node objects.
        json_node (HostNode): The current JSON node being processed.
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
