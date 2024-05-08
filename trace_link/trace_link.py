import copy
import json
import logging
import os
import sys

import networkx as nx
from networkx.algorithms import isomorphism
from param_bench.train.compute.python.tools.execution_trace import (
    EXECUTION_TRACE_PROCESS_ANNOTATION,
    EXECUTION_TRACE_THREAD_ANNOTATION,
)
from param_bench.train.compute.python.tools.utility import (
    load_execution_trace_file,
    read_dictionary_from_json_file,
    write_dictionary_to_json_file,
)


# Increase recursion limit
sys.setrecursionlimit(10**6)

logger = logging.getLogger()


# Add and sort ET nodes from the execution trace
def collect_nodes(node):
    def traverse(node):
        nonlocal nodes
        nodes.append(node)
        for child in node.children:
            traverse(child)

    nodes = []
    traverse(node)
    sorted_nodes = sorted(nodes, key=lambda x: x.id)
    return sorted_nodes


class Kineto_node:
    def __init__(self, name, start, end, id):
        self.name = name
        self.start = start
        self.end = end
        self.id = id
        self.children = []


# Function to transform your self-defined tree to a directed graph with max depth
def transform_to_graph_depth(node, max_depth=100):
    graph = nx.DiGraph()
    add_node_to_graph_depth(node, graph, max_depth, 1)
    return graph


# Helper function to recursively add nodes and edges to the graph with max depth
def add_node_to_graph_depth(node, graph, max_depth, cur_depth):
    graph.add_node(node.id, label=node.name)
    if cur_depth == max_depth:
        return
    for child in node.children:
        graph.add_node(node.id, label=node.name)
        add_node_to_graph_depth(child, graph, max_depth, cur_depth + 1)


# Custom node comparison function for edit distance
def node_compare(n1, n2):
    return n1 == n2


# Find the segment that has a length closest to the target
def find_closest_segment(segs, target_length):
    closest_length = float("inf")
    closest_seg = None

    for seg in segs:
        length_difference = abs(len(seg) - target_length)
        if length_difference < closest_length:
            closest_length = length_difference
            closest_seg = seg

    return closest_seg


def has_category_field(op_dict):
    """
    All kineto node has category field.
    """
    if "cat" in op_dict:
        return True
    return False


def get_category_field(op_dict):
    if "cat" in op_dict:
        return op_dict["cat"]
    else:
        return None


def get_name_field(op_dict):
    if "name" in op_dict:
        return op_dict["name"]
    else:
        return None


def get_phase_field(op_dict):
    if "ph" in op_dict:
        return op_dict["ph"]
    else:
        return None


def get_duration_field(op_dict):
    if "dur" in op_dict:
        return op_dict["dur"]
    else:
        return 0


def get_timestamp_field(op_dict):
    if "ts" in op_dict:
        return op_dict["ts"]
    else:
        return 0


def find_closest_op(ops, ts):
    """
    Finds the operation that is closest in duration to a given timestamp.
    """
    closest_start = 0
    closest_op = {}
    for op in ops:
        if (
            has_category_field(op)
            and get_timestamp_field(op) < ts
            and (get_timestamp_field(op) + get_duration_field(op)) > ts
            and get_timestamp_field(op) > closest_start
        ):

            closest_start = get_timestamp_field(op)
            closest_op = op
    return closest_op


def find_parent_cpu_op(
    kineto_gpu_op,
    kineto_et_ops,
    kineto_ac2g_s_ops,
    kineto_ac2g_f_ops,
    kineto_cpu_launcher_ops,
):
    """
    Find the parent CPU operation for a given GPU operation based on
    the closest start time and duration that covers the GPU operation.
    """
    ts = -1

    # Find a CPU launch op which has the same external id with the GPU op.
    if kineto_gpu_op["args"]["External id"] in kineto_cpu_launcher_ops.keys():
        ts = get_timestamp_field(
            kineto_cpu_launcher_ops[kineto_gpu_op["args"]["External id"]]
        ) + get_duration_field(
            kineto_cpu_launcher_ops[kineto_gpu_op["args"]["External id"]]
        )
    # Find an arrow (CPU->GPU) start point which has the same external id with the GPU op.
    elif kineto_gpu_op["args"]["External id"] in kineto_ac2g_s_ops.keys():
        ts = get_timestamp_field(
            kineto_ac2g_s_ops[kineto_gpu_op["args"]["External id"]]
        )
    # Find an arrow (CPU->GPU) final destination point which has the same external id with the GPU op.
    elif kineto_gpu_op["args"]["External id"] in kineto_ac2g_f_ops.keys():
        ts = get_timestamp_field(
            kineto_ac2g_f_ops[kineto_gpu_op["args"]["External id"]]
        )
    assert ts != -1

    # Find a parent cpu launch op based on two conditions.
    # Condition 1: Parent op should be the op has the closest start time to the gpu op.
    # Condition 2: Parent op should be long enough to cover the gpu op.
    kineto_gpu_op["ts"] = ts
    parent_cpu_op = find_closest_op(kineto_et_ops, ts)

    if not parent_cpu_op:
        print(
            "Warning! the parent cpu_op for the following gpu_op is not found"
            + "and hence it is discarded. gpu_op name: "
            + str(kineto_gpu_op["name"])
            + ", ts: "
            + str(get_timestamp_field(kineto_gpu_op))
            + ", external_id: "
            + str(kineto_gpu_op["args"]["External id"])
        )

    return parent_cpu_op


def segment_ops_by_annotation(kineto_et_ops, annotation):
    """
    Segments a list of Kineto operations (kineto_et_ops) based on a specified annotation.
    Assume that an iteration ends with the specified annotation.
    """
    kineto_et_segs = []
    kineto_et_seg = []

    end_time = -1
    add_last_chunk = False
    for op in kineto_et_ops:
        if end_time > 0 and get_timestamp_field(op) >= end_time:
            kineto_et_segs.append(kineto_et_seg)
            kineto_et_seg = []
            end_time = -1

        if annotation in get_name_field(op):
            kineto_et_seg.append(op)
            end_time = get_timestamp_field(op) + get_duration_field(op)
            add_last_chunk = True
        else:
            kineto_et_seg.append(op)

    if add_last_chunk:
        kineto_et_segs.append(kineto_et_seg)
    logger.info(f"Kineto trace has {len(kineto_et_segs)} segments")

    return kineto_et_segs


def is_valid_op(op, category, name_exception="ProfilerStep", phase=None):
    """
    Check if the operation 'op' matches the given criteria.
    """
    return (
        has_category_field(op)
        and name_exception not in get_name_field(op)
        and get_category_field(op) == category
        and (phase is None or get_phase_field(op) == phase)
    )


def parse_kineto_ops(sorted_kineto_trace_ops):
    """
    Create a list of operations by filtering the input list `sorted_kineto_trace_ops`.
    """
    kineto_et_ops = [
        op
        for op in sorted_kineto_trace_ops
        if is_valid_op(op, "cpu_op") or is_valid_op(op, "user_annotation")
    ]

    kineto_ac2g_s_ops = {
        op["id"]: op
        for op in sorted_kineto_trace_ops
        if is_valid_op(op, "ac2g", phase="s")
    }

    kineto_ac2g_f_ops = {
        op["id"]: op
        for op in sorted_kineto_trace_ops
        if is_valid_op(op, "ac2g", phase="f")
    }

    kineto_cpu_launcher_ops = {
        op["args"]["External id"]: op
        for op in sorted_kineto_trace_ops
        if is_valid_op(op, "cuda_runtime")
        and (get_name_field(op) in ["cudaLaunchKernel", "cudaMemcpyAsync"])
    }

    kineto_gpu_ops = [
        op
        for op in sorted_kineto_trace_ops
        if is_valid_op(op, "kernel") or is_valid_op(op, "gpu_memcpy")
    ]

    return (
        kineto_et_ops,
        kineto_ac2g_s_ops,
        kineto_ac2g_f_ops,
        kineto_cpu_launcher_ops,
        kineto_gpu_ops,
    )


def trace_analysis(et_file, kineto_file, annotation="DataLoader"):
    """
    Extract operator info from raw traces
    """
    et = load_execution_trace_file(et_file)
    et.set_iterations(annotation)

    if et.iterations() > 1:
        logger.info(f"Execution trace has {et.iterations()} > 1 iterations.")
        # get an iteration further down the line
        trim_iter = min(2, et.iterations() - 1)
        et_ = et.clone_one_iteration(trim_iter)
    else:
        et_ = et

    nodes = et_.get_nodes()

    # Root node of execution trace is 1-based
    et_nodes = collect_nodes(nodes[1])

    logger.info(f"Number of original ops in execution trace: {len(et_nodes)}")

    kineto_trace_ops = read_dictionary_from_json_file(kineto_file)["traceEvents"]

    sorted_kineto_trace_ops = sorted(
        kineto_trace_ops, key=lambda kv: get_timestamp_field(kv)
    )

    (
        kineto_et_ops,
        kineto_ac2g_s_ops,
        kineto_ac2g_f_ops,
        kineto_cpu_launcher_ops,
        kineto_gpu_ops,
    ) = parse_kineto_ops(sorted_kineto_trace_ops)

    kineto_iteration_latencies = [
        get_duration_field(iteration)
        for iteration in sorted_kineto_trace_ops
        if "ProfilerStep" in get_name_field(iteration)
    ]

    # The choice below normally does not matter for approximate match since we rely on the isomorphism of
    # the graphs, but for exact match we will use the execution order and then we should be careful
    kineto_et_segs = segment_ops_by_annotation(kineto_et_ops, annotation)

    # In case of kineto only contains one iteration or the provided annotation is wrong, use the whole trace directly.
    # Otherwise find the iteration in kineto trace with the closest #ops to ET
    # (usually ET has 3 additional annotation ops for processes/threads)
    if kineto_et_segs:
        kineto_et_ops = find_closest_segment(kineto_et_segs, len(et_nodes) - 3)
    else:
        logger.warning(
            f"Could not find annotation {annotation} in kineto file"
            " using the whole file, processing could be very slow!!"
        )

    logger.info(f"Number of original cpu ops in kineto trace: {len(kineto_et_ops)}")
    logger.info(f"Number of original gpu ops in kineto trace: {len(kineto_gpu_ops)}")

    if len(kineto_iteration_latencies) > 0:
        average_iteration_latency = sum(kineto_iteration_latencies) / len(
            kineto_iteration_latencies
        )
        logger.info(f"Average iteration latency: {average_iteration_latency}")

    return (
        et_nodes,
        kineto_et_ops,
        kineto_ac2g_s_ops,
        kineto_ac2g_f_ops,
        kineto_cpu_launcher_ops,
        kineto_gpu_ops,
    )


def op_exists(name, kineto_et_ops, i):
    """
    This function checks if an op with the specified name exists at a certain distance from
    the given index i in the kineto_et_ops list. .

    Parameters:
    - name: The name of the op to be checked.
    - kineto_et_ops: A list of ops, each containing a name property.
    - i: The index in the kineto_et_ops list to start the check from.

    Returns:
    - A tuple containing a boolean that indicates whether the op with the specified name exists
      and the op where the name was found, or the op at index i if the name was not found.
    """

    MAX_DISTANCE = 20  # The default maximum distance from index i to look for the op.
    distance = 0  # The current distance from index i.

    while distance <= MAX_DISTANCE:
        # Calculate indices for forward and backward directions from the current position i.
        forward_index = i + distance
        backward_index = i - distance

        # Check forward within the bounds of the list.
        if (
            forward_index < len(kineto_et_ops)
            and get_name_field(kineto_et_ops[forward_index]) == name
        ):
            return True, kineto_et_ops[forward_index]

        # Check backward within the bounds of the list.
        if (
            backward_index >= 0
            and get_name_field(kineto_et_ops[backward_index]) == name
        ):
            return True, kineto_et_ops[backward_index]

        distance += 1  # Increment the distance and continue the search.

    return (
        False,
        kineto_et_ops[i],
    )  # Return False and the op at index i if the name is not found.


def find_op_shift(et_nodes, kineto_et_ops):
    """
    This function checks to see if the operations in Pytorch_et (et_nodes) and
    Kineto (kineto_et_ops) are shifter by a constant number. Sometimes it is
    possible that the operation in index i of et_nodes is mapped to index
    i+shift in kineto_et_ops. The objective of this function is to detect the shift value.
    To do this, this function picks N (max_pattern_length) number of consecutive
    ops is et_nodes, and compare it with N consecutive ops in kineto_et_ops.
    The maximum shift amount to check is determined by the max_shift_to_check variable.

    Parameters:
    - et_nodes: A list of Pytorch_et ops.
    - kineto_et_ops: A list of kineto ops.

    Returns:
    - The amount of shift between et_nodes and kineto_et_ops
    """
    # Number of consecutive ops to check
    max_pattern_length = 10
    # Number of consecutive ops to check
    max_shift_to_check = 1000
    # We pick the N consecutive ops starting from the index 5 in Pytorch_et
    start_index = 5
    for shift in range(max_shift_to_check):
        pattern_match = True
        for index in range(max_pattern_length):
            if start_index + index >= len(
                et_nodes
            ) or start_index + index + shift >= len(kineto_et_ops):
                return 0
            if (
                et_nodes[start_index + index].name
                != kineto_et_ops[start_index + index + shift]["name"]
            ):
                pattern_match = False
                break
        if pattern_match:
            return shift
    return 0


def exact_match(
    kineto_et_ops,
    kineto_ac2g_s_ops,
    kineto_ac2g_f_ops,
    kineto_cpu_launcher_ops,
    kineto_gpu_ops,
    et_nodes,
):
    # Since kineto trace is missing the annotations for processes/threads,
    # we add them back to match with ET
    kineto_op_per_thread = {}

    process_end_time = -1
    for i in range(len(kineto_et_ops)):
        op = kineto_et_ops[i]
        if op["tid"] not in kineto_op_per_thread:
            kineto_op_per_thread[op["tid"]] = {}
            kineto_op_per_thread[op["tid"]]["ts"] = get_timestamp_field(op)
            kineto_op_per_thread[op["tid"]]["end_ts"] = get_timestamp_field(
                op
            ) + get_duration_field(op)
            kineto_op_per_thread[op["tid"]]["index"] = i
        else:
            kineto_op_per_thread[op["tid"]]["end_ts"] = max(
                kineto_op_per_thread[op["tid"]]["end_ts"],
                get_timestamp_field(op) + get_duration_field(op),
            )
        process_end_time = max(
            process_end_time, get_timestamp_field(op) + get_duration_field(op)
        )

    process_op = {
        "name": EXECUTION_TRACE_PROCESS_ANNOTATION,
        "ts": get_timestamp_field(kineto_et_ops[0]),
        "dur": process_end_time - get_timestamp_field(kineto_et_ops[0]),
    }

    kineto_et_ops.insert(0, process_op)

    sorted_threads = dict(
        sorted(kineto_op_per_thread.items(), key=lambda x: x[1]["index"])
    )

    for index, (tid, thread_info) in enumerate(sorted_threads.items()):
        thread_op = {
            "name": EXECUTION_TRACE_THREAD_ANNOTATION,
            "ts": get_timestamp_field(thread_info),
            "dur": thread_info["end_ts"] - get_timestamp_field(thread_info),
        }
        # Be careful of the insertion position, note that we already inserted process op
        kineto_et_ops.insert(index + 1 + thread_info["index"], thread_op)

    # Duration of ET nodes
    et_enhanced_duration = {}
    # Timestamp of ET nodes
    et_enhanced_timestamp = {}

    et_gpu_ops_per_cpu_op_id = {}
    kineto_gpu_ops_per_cpu_op_idx = {}

    for gpu_op in kineto_gpu_ops:
        parent_cpu_op = find_parent_cpu_op(
            gpu_op,
            kineto_et_ops,
            kineto_ac2g_s_ops,
            kineto_ac2g_f_ops,
            kineto_cpu_launcher_ops,
        )

        if not parent_cpu_op:
            continue

        assert "Ev Idx" in parent_cpu_op["args"]
        if parent_cpu_op["args"]["Ev Idx"] not in kineto_gpu_ops_per_cpu_op_idx:
            kineto_gpu_ops_per_cpu_op_idx[parent_cpu_op["args"]["Ev Idx"]] = [gpu_op]
        else:
            kineto_gpu_ops_per_cpu_op_idx[parent_cpu_op["args"]["Ev Idx"]].append(
                gpu_op
            )
    shift = find_op_shift(et_nodes, kineto_et_ops)
    if shift:
        logger.info(
            "shift found between et_nodes, and kineto_et_events. Shift amount: "
            + str(shift)
        )
    # Link kineto trace and execution trace
    if len(kineto_et_ops) >= len(et_nodes):
        for i in range(len(et_nodes)):
            et_node = et_nodes[i]

            name_exist, kineto_et_op = op_exists(et_node.name, kineto_et_ops, i + shift)

            if (
                name_exist
                or (
                    "iteration#" in et_node.name
                    and "iteration#" in get_name_field(kineto_et_op)
                )
                or et_node.name.replace("execution_graph", "execution_trace")
                == get_name_field(kineto_et_op)
            ):

                et_enhanced_duration[et_node.id] = get_duration_field(kineto_et_op)
                et_enhanced_timestamp[et_node.id] = get_timestamp_field(kineto_et_op)

                if (
                    "args" in kineto_et_op
                    and "Ev Idx" in kineto_et_op["args"]
                    and kineto_et_op["args"]["Ev Idx"] in kineto_gpu_ops_per_cpu_op_idx
                ):
                    et_gpu_ops_per_cpu_op_id[et_node.id] = (
                        kineto_gpu_ops_per_cpu_op_idx[kineto_et_op["args"]["Ev Idx"]]
                    )
            else:  # If op_exists wasn't able to find the corresponding op.
                logger.info(
                    "Op mismatch between kineto and execution trace ( et size = "
                    + str(len(et_nodes))
                    + ", kineto size: "
                    + str(len(kineto_et_ops))
                    + " ):"
                )
                logger.info(
                    f"Op index: {i}, kineto op name: {get_name_field(kineto_et_op)},"
                    f"kineto op timestamp: {get_timestamp_field(kineto_et_op)}, "
                    f"execution trace op name: {et_node.name}, execution trace op id: {et_node.id}"
                )
                for i in range(len(kineto_et_ops)):
                    kineto_et_op = kineto_et_ops[i]
                    et_node = et_nodes[i]
                    logger.info(
                        "Index: "
                        + str(i)
                        + ", et name: "
                        + et_node.name
                        + ", kineto name: "
                        + get_name_field(kineto_et_op)
                    )

                exit(0)
    else:
        logger.info(
            "Ops count mismatch between kineto and execution trace ( et size = "
            + str(len(et_nodes))
            + ", kineto size: "
            + str(len(kineto_et_ops))
            + " )"
        )

    return et_enhanced_duration, et_enhanced_timestamp, et_gpu_ops_per_cpu_op_id


def approximate_match(kineto_et_ops, et_nodes):
    # Since kineto trace is missing the annotations for processes/threads, we add them back to match with ET
    kineto_op_per_thread = {}

    # Mapping node id to the corresponding node
    kineto_nodes_mapping = {}

    start_time = get_timestamp_field(kineto_et_ops[0])
    end_time = -1

    for op in kineto_et_ops:
        if op["tid"] not in kineto_op_per_thread:
            kineto_op_per_thread[op["tid"]] = []
        kineto_op_per_thread[op["tid"]].append(op)
        end_time = max(end_time, get_timestamp_field(op) + get_duration_field(op))

    process_node = Kineto_node(
        EXECUTION_TRACE_PROCESS_ANNOTATION, start_time, end_time, 0
    )
    kineto_nodes_mapping[0] = process_node

    cnt = 1
    for thread in kineto_op_per_thread:
        start_time = kineto_op_per_thread[thread][0]["ts"]
        end_time = -1
        for op in kineto_op_per_thread[thread]:
            end_time = max(end_time, get_timestamp_field(op) + get_duration_field(op))

        thread_node = Kineto_node(
            EXECUTION_TRACE_THREAD_ANNOTATION, start_time, end_time, cnt
        )
        print(
            f"thread {thread} thread_node start,end = {thread_node.start}, {thread_node.end}"
        )
        kineto_nodes_mapping[cnt] = thread_node
        cnt += 1

        process_node.children.append(thread_node)

        kineto_nodes = [thread_node]
        for op in kineto_op_per_thread[thread]:
            if get_timestamp_field(op) < kineto_nodes[-1].end:
                tmp = Kineto_node(
                    get_name_field(op),
                    get_timestamp_field(op),
                    get_timestamp_field(op) + get_duration_field(op),
                    cnt,
                )
                kineto_nodes_mapping[cnt] = tmp
                cnt += 1
                kineto_nodes[-1].children.append(tmp)
                kineto_nodes.append(tmp)
            else:
                while (
                    kineto_nodes[-1].end <= get_timestamp_field(op)
                    and len(kineto_nodes) > 1
                ):
                    kineto_nodes.pop()
                tmp = Kineto_node(
                    get_name_field(op),
                    get_timestamp_field(op),
                    get_timestamp_field(op) + get_duration_field(op),
                    cnt,
                )
                kineto_nodes_mapping[cnt] = tmp
                cnt += 1
                kineto_nodes[-1].children.append(tmp)
                kineto_nodes.append(tmp)

    # Max call stack depth when building the tree, the deeper the more accurate but takes longer time
    depth = 10

    # Build a tree from the kineto trace
    kineto_graph = transform_to_graph_depth(process_node, depth)
    logger.info(f"Kineto tree nodes number: {len(kineto_graph.nodes)}")

    # Build a tree from the execution trace
    et_graph = transform_to_graph_depth(et_nodes[0], depth)
    logger.info(f"ET tree nodes number: {len(et_graph.nodes)}")

    # Create the GraphMatcher
    GM = isomorphism.GraphMatcher(kineto_graph, et_graph)

    # Duration of ET nodes
    et_enhanced_duration = {}

    if GM.is_isomorphic():
        mapping = GM.mapping
        logger.info("Graphs are isomorphic")
        for kineto_id, et_id in mapping.items():
            et_enhanced_duration[et_id] = (
                kineto_nodes_mapping[kineto_id].end
                - kineto_nodes_mapping[kineto_id].start
            )
    else:
        logger.info("Graphs are not isomorphic")

        # # Compute the edit distance using the graph_edit_distance function with node comparison
        # paths, cost = nx.graph_edit_distance(kineto_graph, et_graph, node_compare)
        # logger.info(f"Tree edit distance: {cost}")

        # The problem of finding the exact Graph Edit Distance (GED) is NP-hard so it is often slow
        # and below is a sub-optimal approach

        edit_distance_generator = nx.optimize_graph_edit_distance(
            kineto_graph, et_graph, node_compare
        )
        cost = next(edit_distance_generator)

        paths_generator = nx.optimize_edit_paths(kineto_graph, et_graph, node_compare)
        node_edits, _, cost = next(paths_generator)

        logger.info(f"Sub-optimal tree edit distance: {cost}")

        for kineto_id, et_id in node_edits:
            if kineto_id is not None and et_id is not None:
                et_enhanced_duration[et_id] = (
                    kineto_nodes_mapping[kineto_id].end
                    - kineto_nodes_mapping[kineto_id].start
                )

    return et_enhanced_duration


def assign_et_ids(total_assigned_ids, assigned_ids, id):
    """
    Assigns a unique ET id to the operation, ensuring there are no duplicates.

    This function iterates through already assigned ET ids and assigns a new
    unique ET id by incrementing the given id until a unique one is found.
    The unique ET id is then stored for reference.
    """
    orig_id = id
    while True:
        if id in total_assigned_ids.keys():
            id += 1
        else:
            total_assigned_ids[id] = True
            if orig_id not in assigned_ids.keys():
                assigned_ids[orig_id] = id
            return id


def update_gpu_nodes(
    et_gpu_ops_per_cpu_op_id, node, total_assigned_ids, assigned_ids, orig_node_id
):
    """
    Update the GPU nodes with attributes from the corresponding CPU node
    and assign unique IDs.
    """
    gpu_nodes = sorted(
        et_gpu_ops_per_cpu_op_id[orig_node_id], key=lambda kv: get_timestamp_field(kv)
    )
    new_gpu_nodes = []

    # Assign the gpu_node's parent with cpu_node
    for gpu_node in gpu_nodes:
        copy_gpu_node = copy.deepcopy(gpu_node)
        copy_gpu_node["parent"] = node["id"]
        copy_gpu_node["id"] = assign_et_ids(
            total_assigned_ids, assigned_ids, orig_node_id
        )
        copy_gpu_node["inputs"] = node["inputs"]
        copy_gpu_node["input_shapes"] = node["input_shapes"]
        copy_gpu_node["input_types"] = node["input_types"]
        copy_gpu_node["outputs"] = node["outputs"]
        copy_gpu_node["output_shapes"] = node["output_shapes"]
        copy_gpu_node["output_types"] = node["output_types"]
        new_gpu_nodes.append(copy_gpu_node)
    return new_gpu_nodes


def dump_et_file(
    args, et_enhanced_duration, et_enhanced_timestamp, et_gpu_ops_per_cpu_op_id
):
    """
    Enhances and saves an execution trace file with additional information.

    This function reads an execution trace file, augments it with additional information
    such as updated IDs, durations, timestamps, and GPU operations. The enhanced data is then
    saved to a new file. The function ensures that nodes are assigned unique IDs and GPU nodes
    are updated with information from their corresponding CPU nodes.
    """

    # Check if the provided file path exists
    assert os.path.exists(args.et_file), f"The file {args.et_file} does not exist."

    with open(args.et_file, "r") as f:
        et = json.load(f)
        # assigned_ids: A dictionary mapping original ids to their corresponding unique ET ids.
        # total_assigned_ids: A Dict containing all ET ids that have already been assigned.
        assigned_ids = {}
        total_assigned_ids = {}

        for node in et["nodes"]:
            # Meaning that it is kineto node.
            if has_category_field(node.keys()):
                break

            orig_node_id = node["id"]
            node["id"] = assign_et_ids(total_assigned_ids, assigned_ids, orig_node_id)

            # Build CPU node
            if orig_node_id in et_enhanced_duration:
                node["dur"] = et_enhanced_duration[orig_node_id]
                node["ts"] = et_enhanced_timestamp[orig_node_id]

            # Build GPU node
            if orig_node_id in et_gpu_ops_per_cpu_op_id:
                gpu_nodes = update_gpu_nodes(
                    et_gpu_ops_per_cpu_op_id,
                    node,
                    total_assigned_ids,
                    assigned_ids,
                    orig_node_id,
                )
                et["nodes"].extend(gpu_nodes)
                et_gpu_ops_per_cpu_op_id.pop(orig_node_id)

        for node in et["nodes"]:
            if not has_category_field(node.keys()):
                # Make sure earlier CPU node's children have lower ID than this CPU node.
                # Update old ids into new ids.
                node["parent"] = assigned_ids[node["parent"]]

    et_plus_file = args.et_file.replace(".json", "_plus.json")
    logger.info(f"Enhanced execution trace dumped to {et_plus_file}.")
    with open(et_plus_file, "w") as f:
        json.dump(et, f, indent=4)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Link kineto trace with execution trace"
    )
    parser.add_argument(
        "--et-file", type=str, required=True, help="Path to the execution trace"
    )
    parser.add_argument(
        "--kineto-file", type=str, required=True, help="Path to the kineto trace"
    )
    parser.add_argument(
        "--annotation",
        default="DataLoader",
        type=str,
        help="Operator name to help slice multiple iterations in trace",
    )
    parser.add_argument(
        "--exact-match",
        default=False,
        action="store_true",
        help="Whether to match the traces exactly",
    )
    parser.add_argument("--log-level", default="INFO", help="Log output verbosity.")

    args = parser.parse_args()

    logger.setLevel(args.log_level)

    et_gpu_ops_per_cpu_op_id = {}
    (
        et_nodes,
        kineto_et_ops,
        kineto_ac2g_s_ops,
        kineto_ac2g_f_ops,
        kineto_cpu_launcher_ops,
        kineto_gpu_ops,
    ) = trace_analysis(args.et_file, args.kineto_file, args.annotation)

    if args.exact_match:
        (
            et_enhanced_duration,
            et_enhanced_timestamp,
            et_gpu_ops_per_cpu_op_id,
        ) = exact_match(
            kineto_et_ops,
            kineto_ac2g_s_ops,
            kineto_ac2g_f_ops,
            kineto_cpu_launcher_ops,
            kineto_gpu_ops,
            et_nodes,
        )
        # If linking works, add duration time to each ET node and dump as ET_plus
        dump_et_file(
            args, et_enhanced_duration, et_enhanced_timestamp, et_gpu_ops_per_cpu_op_id
        )
    else:
        raise AssertionError(
            "This script works only with the exact match mode,"
            + "please add the --exact-match flag to your script and run it again"
        )


if __name__ == "__main__":
    main()  # pragma: no cover
