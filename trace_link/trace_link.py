import argparse
import copy
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from param_bench.train.compute.python.tools.execution_trace import (
    EXECUTION_TRACE_PROCESS_ANNOTATION,
    EXECUTION_TRACE_THREAD_ANNOTATION,
    Node as PyTorchOperator
)
from param_bench.train.compute.python.tools.utility import (
    load_execution_trace_file,
    read_dictionary_from_json_file
)

# Increase the recursion limit for deep PyTorch execution traces.
sys.setrecursionlimit(10**6)


class KinetoOperator:
    """
    Represents a single operator extracted from the Kineto trace.

    Attributes:
        op_dict (Dict[str, Any]): Dictionary containing the operator data.
        category (Optional[str]): Category of the operator.
        name (Optional[str]): Name of the operator.
        phase (Optional[str]): Phase of the operator.
        duration (int): Duration of the operator in microseconds.
        timestamp (int): Timestamp of the operator in microseconds.
        external_id (Optional[str]): External ID associated with the operator.
        ev_idx (Optional[str]): Event index associated with the operator.
        tid (Optional[int]): Thread ID associated with the operator.
        parent_pytorch_op_id (Optional[int]): ID of the parent PyTorch operator.
    """

    def __init__(self, kineto_op: Dict[str, Any]) -> None:
        """
        Initializes a new instance of the KinetoOperator class.

        Args:
            kineto_op (Dict[str, Any]): The dictionary representing the
                                        operator data.
        """
        self.op_dict = kineto_op
        self.category = kineto_op.get("cat")
        self.name = kineto_op.get("name")
        self.phase = kineto_op.get("ph")
        self.duration = kineto_op.get("dur", 0)
        self.timestamp = kineto_op.get("ts", 0)
        self.external_id = None
        self.ev_idx = None
        self.tid = kineto_op.get("tid")
        self.parent_pytorch_op_id = None

        if "args" in kineto_op:
            self.external_id = kineto_op["args"].get("External id")
            self.ev_idx = kineto_op["args"].get("Ev Idx")

    def is_valid(self, category: str, name_exception: str = "ProfilerStep",
                 phase: Optional[str] = None) -> bool:
        """
        Checks if the operator matches specified filtering criteria.

        Args:
            category (str): The category to check against.
            name_exception (str): A name to exclude in the check.
            phase (Optional[str]): The phase to check against, if any.

        Returns:
            bool: True if the operator matches the criteria, False otherwise.
        """
        return (self.category is not None and
                name_exception not in self.name and
                self.category == category and
                (phase is None or self.phase == phase))

    def __repr__(self) -> str:
        """
        Represent the KinetoOperator as a string.

        Returns:
            str: A string representation of the KinetoOperator.
        """
        return (f"KinetoOperator(category={self.category}, "
                f"name={self.name}, phase={self.phase}, duration={self.duration}, "
                f"timestamp={self.timestamp}, external_id={self.external_id}, "
                f"ev_idx={self.ev_idx}, tid={self.tid}, "
                f"parent_pytorch_op_id={self.parent_pytorch_op_id})")


class UniqueIdAssigner:
    """
    Assigns unique IDs to items, ensuring each item gets a distinct ID.

    This class is used to maintain a consistent and unique mapping of original
    identifiers to new unique identifiers. It's particularly useful in scenarios
    where the uniqueness of IDs across different entities or iterations needs to
    be preserved.

    Attributes:
        next_id (int): The next unique ID to be assigned.
        original_to_new_ids (Dict[int, int]): A mapping from original IDs to their
            corresponding new unique IDs. This helps in retrieving already assigned
            unique IDs and ensures the same original ID always maps to the same
            unique ID.
    """

    def __init__(self) -> None:
        """
        Initializes the UniqueIdAssigner with a starting ID of 0.
        """
        self.next_id = 0
        self.original_to_new_ids: Dict[int, int] = {}

    def assign_or_retrieve_id(self, original_id: int) -> int:
        """
        Assigns a new unique ID to the given original ID if it doesn't have one already;
        otherwise, returns the previously assigned unique ID.

        Args:
            original_id (int): The original ID for which a unique ID is needed.

        Returns:
            int: A unique ID corresponding to the original ID.
        """
        if original_id not in self.original_to_new_ids:
            self.original_to_new_ids[original_id] = self.next_id
            self.next_id += 1

        return self.original_to_new_ids[original_id]

    def generate_new_id(self) -> int:
        """
        Generates a new unique ID without needing an original ID.

        This is useful for cases where new entities are created that do not
        have an existing identifier.

        Returns:
            int: A new unique ID.
        """
        unique_id = self.next_id
        self.next_id += 1
        return unique_id

    def lookup_new_id(self, original_id: int) -> int:
        """
        Retrieves the new unique ID for a given original ID, if it has been assigned.

        This method is useful for checking if a unique ID has already been
        assigned to an original ID and retrieving it.

        Args:
            original_id (int): The original ID to look up.

        Returns:
            int: The new unique ID if it has been assigned, otherwise returns
                the original ID.
        """
        return self.original_to_new_ids.get(original_id, original_id)


class TraceLinker:
    """
    Links PyTorch Execution Traces (ET) and Kineto Traces to generate PyTorch ET plus.

    This class handles the process of loading, processing, and linking
    PyTorch Execution Traces with Kineto Traces, enriching the PyTorch
    Execution Trace with detailed performance data.

    Attributes:
        pytorch_et_file (str): Path to the PyTorch execution trace file.
        kineto_file (str): Path to the Kineto trace file.
        pytorch_ops (List[PyTorchOperator]): PyTorch operators from ET trace.
        kineto_ops (List[KinetoOperator]): Kineto operators from the trace.
        kineto_ac2g_s_ops (Dict[str, KinetoOperator]): Start ops for CPU to GPU.
        kineto_ac2g_f_ops (Dict[str, KinetoOperator]): Final ops for CPU to GPU.
        kineto_cpu_launcher_ops (Dict[str, KinetoOperator]): CPU launcher ops.
        kineto_gpu_ops (List[KinetoOperator]): GPU operators.
        pytorch_op_id_to_kineto_ops_map (Dict[int, List[KinetoOperator]]):
            Map from PyTorch op IDs to Kineto GPU ops.
        pytorch_op_id_to_duration_map (Dict[int, int]): Duration map for PyTorch ops.
        pytorch_op_id_to_timestamp_map (Dict[int, int]): Timestamp map for PyTorch ops.
        id_assigner (UniqueIdAssigner): Assigns unique IDs to operators.
        pytorch_et_plus_data (Optional[Dict]): PyTorch ET plus data.
        logger (logging.Logger): Logger for the class.
    """

    def __init__(self, pytorch_et_file: str, kineto_file: str,
                 log_level: str = "INFO") -> None:
        """
        Initializes the TraceLinker with paths to the PyTorch and Kineto trace files,
        and a log level.

        Args:
            pytorch_et_file (str): Path to the PyTorch execution trace file.
            kineto_file (str): Path to the Kineto trace file.
            log_level (str): Logging level for the class.
        """
        self.pytorch_et_file = pytorch_et_file
        self.kineto_file = kineto_file
        self.pytorch_ops: List[PyTorchOperator] = []
        self.kineto_ops: List[KinetoOperator] = []
        self.kineto_ac2g_s_ops: Dict[str, KinetoOperator] = {}
        self.kineto_ac2g_f_ops: Dict[str, KinetoOperator] = {}
        self.kineto_cpu_launcher_ops: Dict[str, KinetoOperator] = {}
        self.kineto_gpu_ops: List[KinetoOperator] = []
        self.pytorch_op_id_to_kineto_ops_map: Dict[int, List[KinetoOperator]] = {}
        self.pytorch_op_id_to_duration_map: Dict[int, int] = {}
        self.pytorch_op_id_to_timestamp_map: Dict[int, int] = {}
        self.id_assigner = UniqueIdAssigner()
        self.pytorch_et_plus_data: Optional[Dict] = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level.upper())

    def load_traces(self) -> None:
        """
        Loads both PyTorch Execution Traces and Kineto Traces.
        This method is a high-level orchestrator that calls specific methods to load
        and process the PyTorch and Kineto traces individually.
        """
        self.load_pytorch_et()
        self.load_kineto_trace()

    def load_pytorch_et(self) -> None:
        """
        Loads and processes the PyTorch Execution Trace.
        This method handles multiple iterations in the trace and extracts the nodes,
        considering the specified annotation for segmenting the iterations.
        """
        self.logger.info("Starting to load PyTorch Execution Trace.")
        pytorch_et = load_execution_trace_file(self.pytorch_et_file)

        root_node = pytorch_et.get_nodes()[1]  # Root node is usually 1-based
        self.pytorch_ops = self.extract_pytorch_ops(root_node)
        self.logger.info(f"Original ops in PyTorch ET: {len(self.pytorch_ops)}")
        self.logger.info("PyTorch Execution Trace loaded successfully.")

    def extract_pytorch_ops(self, node: PyTorchOperator) -> List[PyTorchOperator]:
        """
        Extracts and sorts nodes from the PyTorch execution trace recursively.

        This method traverses the execution trace starting from the provided node,
        extracting all the operator nodes recursively, and then returns them sorted
        by their identifiers.

        Args:
            node (PyTorchOperator): Starting node for extraction.

        Returns:
            List[PyTorchOperator]: Sorted list of extracted PyTorchOperator nodes.
        """
        nodes = []

        def traverse(node: PyTorchOperator):
            nodes.append(node)
            for child in node.children:
                traverse(child)

        traverse(node)
        return sorted(nodes, key=lambda x: x.id)

    def load_kineto_trace(self) -> None:
        """
        Loads and processes the Kineto Trace.
        This method parses the Kineto trace file, creating KinetoOperator instances
        for each operator in the trace. It then categorizes and segments these
        operators for further processing and linking with PyTorch operators.
        """
        self.logger.info("Starting to load Kineto Trace.")
        kineto_trace_data = read_dictionary_from_json_file(self.kineto_file)
        sorted_kineto_ops = sorted(
            [KinetoOperator(op) for op in kineto_trace_data["traceEvents"]],
            key=lambda op: op.timestamp
        )

        self.categorize_kineto_ops(sorted_kineto_ops)

        self.logger.info(f"Processed Kineto trace with {len(self.kineto_ops)} CPU ops, "
                    f"{len(self.kineto_cpu_launcher_ops)} CPU launcher ops, "
                    f"and {len(self.kineto_gpu_ops)} GPU ops.")
        self.logger.info("Kineto Trace loaded successfully.")

    def categorize_kineto_ops(self, kineto_ops: List[KinetoOperator]) -> None:
        """
        Categorizes Kineto operators based on their properties and assigns them to
        corresponding groups for CPU, GPU, and other operations.

        Args:
            kineto_ops (List[KinetoOperator]): List of Kineto operators to categorize.
        """
        for op in kineto_ops:
            if op.is_valid("cpu_op") or op.is_valid("user_annotation"):
                self.kineto_ops.append(op)
                self.logger.debug(f"Added CPU or user annotation op: {op.name}")
            elif op.is_valid("ac2g", phase="s"):
                self._add_op_to_dict(op, self.kineto_ac2g_s_ops, "id")
            elif op.is_valid("ac2g", phase="f"):
                self._add_op_to_dict(op, self.kineto_ac2g_f_ops, "id")
            elif op.is_valid("cuda_runtime") and op.name in ["cudaLaunchKernel", "cudaMemcpyAsync"]:
                self._add_op_to_dict(op, self.kineto_cpu_launcher_ops, "args", "External id")
                self.logger.debug(f"Added CPU launcher op: {op.name}")
            elif op.is_valid("kernel") or op.is_valid("gpu_memcpy"):
                self.kineto_gpu_ops.append(op)
                self.logger.debug(f"Added GPU op: {op.name}")

    def _add_op_to_dict(self, op: KinetoOperator, target_dict: Dict,
                        *keys: str) -> None:
        """
        Adds an operator to a specific dictionary based on provided keys.
        The method navigates through the operator's dictionary using the keys
        and adds the operator to the target dictionary.

        Args:
            op (KinetoOperator): The operator to be added.
            target_dict (Dict): The dictionary to which the operator should be added.
            *keys (str): Keys used to navigate through the operator's dictionary.

        Raises:
            KeyError: If any of the keys are not found in the operator's dictionary.
        """
        value = op.op_dict
        for key in keys:
            if key not in value:
                error_msg = f"Key '{key}' not found in operator dictionary for op {op.name}."
                self.logger.error(error_msg)
                raise KeyError(error_msg)
            value = value[key]

        target_dict[value] = op

    def link_traces(self) -> None:
        """
        Initiates the linking process between PyTorch Execution Traces (ET) and
        Kineto Traces to produce an enhanced PyTorch Execution Trace (ET+). This
        process relies on the assumption of an 'exact match' between these traces.

        An 'exact match' implies that the PyTorch Execution Trace and the Kineto Trace
        were collected simultaneously. If this condition is not met, the tool may not
        function correctly, as the correlation between the two traces is essential for
        accurate linking and analysis.

        Currently, this tool supports only this 'exact_match' method for trace linking.
        """
        self.logger.info("Starting the process of linking PyTorch and Kineto traces.")
        self.exact_match()
        self.logger.info("Traces have been successfully linked.")

    def exact_match(self) -> None:
        """
        Performs the process of 'exact matching' between PyTorch Execution Trace nodes
        and Kineto operators. This method augments PyTorch nodes with timing and
        additional data extracted from Kineto's trace, under the key assumption that
        both traces were captured concurrently.

        In the context of this tool, 'exact match' denotes the precise alignment in
        time and sequence between the two types of traces. If the traces were not
        recorded in tandem, the alignment and thus the linking process would be
        inaccurate, leading to erroneous or misleading analytical conclusions.
        """
        self.add_thread_and_process_annotations()
        self.map_pytorch_to_kineto_ops()
        self.construct_et_plus_data()

    def add_thread_and_process_annotations(self) -> None:
        """
        Adds thread and process annotations to Kineto operators. These annotations
        are essential as they are present in PyTorch Execution Traces but missing
        in Kineto Traces. This method aligns Kineto operators with PyTorch ET nodes
        by considering thread IDs, ensuring compatibility and completeness of
        trace data for analysis.
        """
        kineto_op_per_thread = {}
        process_end_time = -1

        for i, op in enumerate(self.kineto_ops):
            if op.tid is not None:
                if op.tid not in kineto_op_per_thread:
                    kineto_op_per_thread[op.tid] = {
                        "ts": op.timestamp,
                        "end_ts": op.timestamp + op.duration,
                        "index": i
                    }
                else:
                    kineto_op_per_thread[op.tid]["end_ts"] = max(
                        kineto_op_per_thread[op.tid]["end_ts"],
                        op.timestamp + op.duration
                    )

                process_end_time = max(process_end_time, op.timestamp + op.duration)

        self.insert_process_annotation_op(process_end_time)
        self.insert_thread_annotation_ops(kineto_op_per_thread)

    def insert_process_annotation_op(self, end_time: int) -> None:
        """
        Inserts a process annotation operator at the beginning of the Kineto
        operators list. This annotation marks the start and end of the overall
        process in the trace.

        Args:
            end_time (int): End time of the last operator in the process.
        """
        self.logger.debug("Adding process annotation op with name %s and ts %d",
                          EXECUTION_TRACE_PROCESS_ANNOTATION,
                          self.kineto_ops[0].timestamp)
        process_annotation_op = KinetoOperator({
            "name": EXECUTION_TRACE_PROCESS_ANNOTATION,
            "ts": self.kineto_ops[0].timestamp,
            "dur": end_time - self.kineto_ops[0].timestamp
        })
        self.kineto_ops.insert(0, process_annotation_op)

    def insert_thread_annotation_ops(self, thread_info: Dict[int, Dict[str, int]]) -> None:
        """
        Inserts thread annotation operators into the Kineto operators list based
        on the provided thread information. Each annotation represents the start
        and duration of operations in a specific thread.

        Args:
            thread_info (Dict[int, Dict[str, int]]): Information about threads.
        """
        sorted_threads = dict(sorted(thread_info.items(),
                                     key=lambda x: x[1]["index"]))
        for index, (tid, info) in enumerate(sorted_threads.items()):
            self.logger.debug("Adding thread annotation op for tid %d with name %s and ts %d",
                              tid, EXECUTION_TRACE_THREAD_ANNOTATION, info["ts"])
            thread_annotation_op = KinetoOperator({
                "name": EXECUTION_TRACE_THREAD_ANNOTATION,
                "ts": info["ts"],
                "dur": info["end_ts"] - info["ts"]
            })
            self.kineto_ops.insert(index + 1 + info["index"], thread_annotation_op)

    def map_pytorch_to_kineto_ops(self) -> None:
        """
        Maps PyTorch ET nodes to corresponding Kineto operators, ensuring
        each PyTorch node has a matching Kineto operator.

        Raises:
            RuntimeError: If no corresponding Kineto operator is found for
                          a PyTorch operator.
        """
        self.logger.info("Mapping PyTorch ET nodes to Kineto operators.")
        cpu_ev_idx_to_gpu_ops_map = self.group_gpu_ops_by_cpu_launchers()

        if len(self.kineto_ops) < len(self.pytorch_ops):
            error_msg = ("Mismatch in operator count between PyTorch and "
                         "Kineto traces.")
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        for i, pytorch_op in enumerate(self.pytorch_ops):
            kineto_op = self.find_corresponding_kineto_op(pytorch_op, i)
            if kineto_op is None:
                error_msg = (f"No corresponding Kineto op found for PyTorch op "
                             f"{pytorch_op.name}")
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            self.link_ops(pytorch_op, kineto_op, cpu_ev_idx_to_gpu_ops_map)

        self.logger.info("Completed mapping of PyTorch operators to Kineto operators.")

    def group_gpu_ops_by_cpu_launchers(self) -> Dict[str, List[KinetoOperator]]:
        """
        Groups GPU operators based on their corresponding CPU launchers.

        This is determined by the 'ev_idx' which links GPU operators to their
        initiating CPU launcher events.

        Returns:
            Dict[str, List[KinetoOperator]]: Mapping from CPU launch event indices
                                             to GPU operators.

        Raises:
            ValueError: If 'ev_idx' is missing for any GPU operator.
        """
        cpu_ev_idx_to_gpu_ops_map = {}
        for gpu_op in self.kineto_gpu_ops:
            parent_cpu_op = self.find_parent_cpu_op(gpu_op)
            if not parent_cpu_op:
                warning_msg = (f"Missing parent CPU operator for GPU op "
                               f"'{gpu_op.name}'. Orphaned GPU operator.")
                self.logger.warning(warning_msg)
                continue

            if parent_cpu_op.ev_idx is None:
                error_msg = (f"Missing 'ev_idx' for CPU operator {parent_cpu_op.name}. "
                             f"Cannot link to GPU op {gpu_op.name} to {parent_cpu_op.name}.")
                self.logger.warning(error_msg)
                continue

            self.logger.debug(
                    f"group_gpu_ops_by_cpu_launchers "
                    f"'{parent_cpu_op.name}' -> '{gpu_op.name}'")

            cpu_ev_idx_to_gpu_ops_map.setdefault(parent_cpu_op.ev_idx, []).append(gpu_op)

        return cpu_ev_idx_to_gpu_ops_map

    def find_parent_cpu_op(self, kineto_gpu_op: KinetoOperator) -> Optional[KinetoOperator]:
        """
        Finds the parent CPU operator for a given GPU operator based on
        start time proximity and duration coverage.

        Args:
            kineto_gpu_op (KinetoOperator): The GPU operator.

        Returns:
            Optional[KinetoOperator]: The parent CPU operator if found.
        """
        ts = self.get_start_timestamp_for_gpu_op(kineto_gpu_op)
        kineto_gpu_op.timestamp = ts

        parent_cpu_op = self.find_closest_op(self.kineto_ops, ts)
        if not parent_cpu_op:
            self.logger.warning(
                f"No parent CPU operator found for GPU operator '{kineto_gpu_op.name}' "
                f"(ts: {kineto_gpu_op.timestamp})"
            )

        return parent_cpu_op

    def get_start_timestamp_for_gpu_op(self, kineto_gpu_op: KinetoOperator) -> int:
        """
        Determines the start timestamp for a GPU operator from various sources.

        Args:
            kineto_gpu_op (KinetoOperator): The GPU operator.

        Returns:
            int: The start timestamp.

        Raises:
            RuntimeError: If no valid timestamp is found for the GPU operator.
        """
        if kineto_gpu_op.external_id in self.kineto_cpu_launcher_ops:
            cpu_launcher_op = self.kineto_cpu_launcher_ops[kineto_gpu_op.external_id]
            return cpu_launcher_op.timestamp + cpu_launcher_op.duration
        if kineto_gpu_op.external_id in self.kineto_ac2g_s_ops:
            return self.kineto_ac2g_s_ops[kineto_gpu_op.external_id].timestamp
        if kineto_gpu_op.external_id in self.kineto_ac2g_f_ops:
            return self.kineto_ac2g_f_ops[kineto_gpu_op.external_id].timestamp
        raise RuntimeError(f"No valid timestamp found for GPU operator: {kineto_gpu_op.name}")

    def find_closest_op(self, kineto_ops: List[KinetoOperator],
                        ts: int) -> Optional[KinetoOperator]:
        """
        Finds the Kineto operator that is closest in start time to a given timestamp
        and has a duration that covers the timestamp.

        Args:
            kineto_ops (List[KinetoOperator]): List of Kineto operators.
            ts (int): The timestamp to compare against.

        Returns:
            Optional[KinetoOperator]: The closest Kineto operator if found.
        """
        closest_start = 0
        closest_op = None

        for op in kineto_ops:
            if (op.timestamp < ts) and \
               ((op.timestamp + op.duration) > ts) and \
               (op.timestamp > closest_start):
                closest_start = op.timestamp
                closest_op = op

        return closest_op

    def find_corresponding_kineto_op(self, pytorch_op: PyTorchOperator,
                                     index: int) -> Optional[KinetoOperator]:
        """
        Finds the corresponding Kineto operator for a given PyTorch operator.

        The search starts from the given index and expands gradually in both
        forward and backward directions until the end of the kineto_ops list is reached.

        Args:
            pytorch_op (PyTorchOperator): The PyTorch operator.
            index (int): The index to start the search from.

        Returns:
            Optional[KinetoOperator]: The corresponding Kineto operator, if found.
        """
        kineto_ops_length = len(self.kineto_ops)
        for distance in range(0, kineto_ops_length):
            forward_index = index + distance
            backward_index = index - distance

            if forward_index < kineto_ops_length:
                if self.kineto_ops[forward_index].name == pytorch_op.name:
                    return self.kineto_ops[forward_index]

            if (backward_index >= 0) and (backward_index < kineto_ops_length):
                if self.kineto_ops[backward_index].name == pytorch_op.name:
                    return self.kineto_ops[backward_index]

        return None

    def link_ops(self, pytorch_op: PyTorchOperator, kineto_op: KinetoOperator,
                 cpu_ev_idx_to_gpu_ops_map: Dict[str, List[KinetoOperator]]) -> None:
        """
        Links a PyTorch operator to its corresponding Kineto operator and any associated GPU operators.

        Args:
            pytorch_op (PyTorchOperator): PyTorch operator to link.
            kineto_op (KinetoOperator): Corresponding Kineto operator.
            cpu_ev_idx_to_gpu_ops_map (Dict[str, List[KinetoOperator]]): GPU ops mapping.
        """
        if kineto_op.ev_idx in cpu_ev_idx_to_gpu_ops_map:
            self.pytorch_op_id_to_kineto_ops_map[pytorch_op.id] =\
                    cpu_ev_idx_to_gpu_ops_map[kineto_op.ev_idx]
        self.pytorch_op_id_to_duration_map[pytorch_op.id] = kineto_op.duration
        self.pytorch_op_id_to_timestamp_map[pytorch_op.id] = kineto_op.timestamp
        if kineto_op.ev_idx in cpu_ev_idx_to_gpu_ops_map:
            self.link_gpu_ops(pytorch_op, cpu_ev_idx_to_gpu_ops_map[kineto_op.ev_idx])

    def link_gpu_ops(self, pytorch_op: PyTorchOperator,
                     kineto_gpu_ops: List[KinetoOperator]) -> None:
        """
        Links GPU operators to a PyTorch operator.

        Args:
            pytorch_op (PyTorchOperator): The PyTorch operator to link to.
            kineto_gpu_ops (List[KinetoOperator]): GPU operators to link.
        """
        for gpu_op in kineto_gpu_ops:
            gpu_op.parent_pytorch_op_id = pytorch_op.id

    def construct_et_plus_data(self) -> None:
        """
        Constructs the enhanced PyTorch Execution Trace (ET+) data structure by
        integrating Kineto data into the original PyTorch Execution Trace.

        This method enriches the PyTorch execution trace with detailed performance
        data from the Kineto trace, offering a comprehensive view of the execution.
        """
        self.logger.info("Constructing ET+ data.")
        with open(self.pytorch_et_file, "r") as file:
            pytorch_et_data = json.load(file)

        gpu_ops = []
        for op in pytorch_et_data["nodes"]:
            gpu_ops += self.process_op_and_dependents(op)
        pytorch_et_data["nodes"] += gpu_ops

        # Update parent-child relationships with new IDs
        for op in pytorch_et_data["nodes"]:
            if 'parent' in op:
                op["parent"] = self.id_assigner.assign_or_retrieve_id(op["parent"])

        self.pytorch_et_plus_data = pytorch_et_data
        self.logger.info("ET+ data construction completed.")

    def process_op_and_dependents(self, op: Dict) -> List[Dict]:
        """
        Processes a single operator in the PyTorch ET data, assigns a new unique ID,
        and processes any dependent GPU operators.

        Args:
            op (Dict): The operator to be processed.

        Returns:
            List[Dict]: A list of GPU operators processed and linked to the given
                       operator.
        """
        orig_op_id = op["id"]
        new_op_id = self.id_assigner.assign_or_retrieve_id(orig_op_id)
        op["id"] = new_op_id

        # Update operator with Kineto data if available
        if orig_op_id in self.pytorch_op_id_to_duration_map:
            op["dur"] = self.pytorch_op_id_to_duration_map[orig_op_id]
            op["ts"] = self.pytorch_op_id_to_timestamp_map[orig_op_id]

        # Process and append dependent GPU operators
        if orig_op_id in self.pytorch_op_id_to_kineto_ops_map:
            gpu_ops = self.process_dependent_gpu_ops(op, orig_op_id)
            self.pytorch_op_id_to_kineto_ops_map.pop(orig_op_id)
            return gpu_ops
        return []

    def process_dependent_gpu_ops(self, cpu_op: Dict,
                                  orig_op_id: int) -> List[Dict]:
        """
        Creates and returns a list of GPU operators that are dependent on a
        specific CPU operator, sorted by their timestamp. The GPU operators are
        deep copies of the existing operators with updated IDs and other relevant
        fields from the CPU operator.

        Args:
            cpu_op (Dict): The PyTorch CPU operator.
            orig_op_id (int): The original ID of the CPU operator.

        Returns:
            List[Dict]: A list of processed GPU operators.
        """
        updated_gpu_ops = []
        dependent_gpu_ops = self.pytorch_op_id_to_kineto_ops_map.get(orig_op_id, [])
        for gpu_op in sorted(dependent_gpu_ops, key=lambda x: x.timestamp):
            new_gpu_op = copy.deepcopy(cpu_op)
            new_gpu_op_id = self.id_assigner.generate_new_id()
            new_gpu_op.update({
                "id": new_gpu_op_id,
                "parent": orig_op_id,
                "inputs": cpu_op["inputs"],
                "input_shapes": cpu_op["input_shapes"],
                "input_types": cpu_op["input_types"],
                "outputs": cpu_op["outputs"],
                "output_shapes": cpu_op["output_shapes"],
                "output_types": cpu_op["output_types"],
                "cat": gpu_op.category,
                "name": gpu_op.name,
                "ph": gpu_op.phase,
                "dur": gpu_op.duration,
                "ts": gpu_op.timestamp
            })
            updated_gpu_ops.append(new_gpu_op)

        return updated_gpu_ops

    def dump_pytorch_execution_trace_plus(self, output_file: str) -> None:
        """
        Dumps the enhanced PyTorch Execution Trace (ET+) data to a file.

        Args:
            output_file (str): The file path where the ET+ data will be saved.
        """
        self.logger.info(f"Starting to dump ET+ data to {output_file}.")

        if self.pytorch_et_plus_data is None:
            self.logger.error("ET+ data not constructed. Please run construct_et_plus_data first.")
            return

        try:
            with open(output_file, "w") as file:
                json.dump(self.pytorch_et_plus_data, file, indent=4)
            self.logger.info(f"ET+ data dumped to {output_file}.")
        except IOError as e:
            self.logger.error(f"Failed to dump ET+ data to {output_file}. Error: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while dumping ET+ data. Error: {e}")


def main() -> None:
    """
    Main function to execute the trace linking process.

    For more detailed steps on collecting traces and converting them to Chakra
    traces, visit the guide at:
    https://github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection-%E2%80%90-A-Comprehensive-Guide-on-Merging-PyTorch-and-Kineto-Traces
    """
    parser = argparse.ArgumentParser(
        description="Link PyTorch execution trace with Kineto trace "
                    "to produce Chakra traces. For more information, "
                    "see the guide at https://github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection-%E2%80%90-A-Comprehensive-Guide-on-Merging-PyTorch-and-Kineto-Traces"
    )
    parser.add_argument("--pytorch-et-file", type=str, required=True,
                        help="Path to the PyTorch execution trace")
    parser.add_argument("--kineto-file", type=str, required=True,
                        help="Path to the Kineto trace")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path for the output PyTorch execution trace plus file")
    parser.add_argument("--log-level", default="INFO", type=str,
                        help="Log output verbosity level")

    args = parser.parse_args()

    linker = TraceLinker(
        args.pytorch_et_file,
        args.kineto_file,
        args.log_level
    )
    linker.load_traces()
    linker.link_traces()
    linker.dump_pytorch_execution_trace_plus(args.output_file)


if __name__ == "__main__":
    main()
