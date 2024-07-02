import bisect
import copy
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from et_replay.lib.execution_trace import (
    EXECUTION_TRACE_PROCESS_ANNOTATION,
    EXECUTION_TRACE_THREAD_ANNOTATION,
)
from et_replay.lib.execution_trace import (
    Node as PyTorchOperator,
)
from et_replay.lib.utils import (
    load_execution_trace_file,
    read_dictionary_from_json_file,
)

from .kineto_operator import KinetoOperator
from .unique_id_assigner import UniqueIdAssigner

# Increase the recursion limit for deep PyTorch execution traces.
sys.setrecursionlimit(10**6)


class TraceLinker:
    """
    Links PyTorch Execution Traces (ET) and Kineto Traces to generate PyTorch ET plus.

    Attributes
        id_assigner (UniqueIdAssigner): Assigns unique IDs to operators.
    """

    def __init__(self, log_level: str = "INFO") -> None:
        """
        Initialize the TraceLinker with a log level.

        Args:
            log_level (str): Logging level for the class.
        """
        self.id_assigner = UniqueIdAssigner()
        logging.basicConfig(level=log_level.upper())

    def link(self, pytorch_et_file: str, kineto_file: str, output_file: str) -> None:
        """
        High-level method to link traces and produce the ET+ file.

        Args:
            pytorch_et_file (str): Path to the PyTorch execution trace file.
            kineto_file (str): Path to the Kineto trace file.
            output_file (str): Path for the output PyTorch execution trace plus file.
        """
        pytorch_ops, kineto_data = self.load_traces(pytorch_et_file, kineto_file)

        (
            kineto_cpu_ops,
            kineto_tid_cpu_ops_map,
            kineto_correlation_cuda_runtime_map,
            kineto_gpu_ops,
            kineto_id_arrow_op_map,
            kineto_id_cuda_launch_op_map,
            kineto_process_start_time,
            kineto_process_end_time,
            kineto_thread_info,
            kineto_rf_id_to_kineto_op_map,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
        ) = self.update_kineto_data(kineto_data)

        kineto_tid_cpu_ops_map = self.enforce_inter_thread_order(kineto_tid_cpu_ops_map)
        pytorch_et_plus_data = self.link_traces(
            pytorch_et_file,
            pytorch_ops,
            kineto_cpu_ops,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
            kineto_correlation_cuda_runtime_map,
            kineto_rf_id_to_kineto_op_map,
            kineto_gpu_ops,
            kineto_thread_info,
            kineto_process_start_time,
            kineto_process_end_time,
        )
        self.dump_pytorch_execution_trace_plus(pytorch_et_plus_data, output_file)

    def load_traces(self, pytorch_et_file: str, kineto_file: str) -> Tuple[List[PyTorchOperator], Dict]:
        """
        Load both PyTorch Execution Traces and Kineto Traces.

        Args:
            pytorch_et_file (str): Path to the PyTorch execution trace file.
            kineto_file (str): Path to the Kineto trace file.

        Returns:
            Tuple: A tuple containing the list of PyTorch operators and the kineto data dictionary.
        """
        pytorch_ops = self.load_pytorch_et(pytorch_et_file)
        kineto_data = self.load_kineto_trace(kineto_file)
        return pytorch_ops, kineto_data

    def load_pytorch_et(self, pytorch_et_file: str) -> List[PyTorchOperator]:
        """
        Load and process the PyTorch Execution Trace.

        This method handles multiple iterations in the trace and extracts the nodes, considering the specified
        annotation for segmenting the iterations.

        Args:
            pytorch_et_file (str): Path to the PyTorch execution trace file.

        Returns:
            List[PyTorchOperator]: List of PyTorch operators.
        """
        logging.info("Starting to load PyTorch Execution Trace.")
        pytorch_et = load_execution_trace_file(pytorch_et_file)

        root_node = pytorch_et.get_nodes()[1]  # Root node is usually 1-based
        pytorch_ops = self.extract_pytorch_ops(root_node)
        logging.info(f"Original ops in PyTorch ET: {len(pytorch_ops)}")
        logging.info("PyTorch Execution Trace loaded successfully.")

        return pytorch_ops

    def extract_pytorch_ops(self, node: PyTorchOperator) -> List[PyTorchOperator]:
        """
        Extract and sort nodes from the PyTorch execution trace recursively.

        This method traverses the execution trace starting from the provided node, extracting all the operator nodes
        recursively, and then returns them sorted by their identifiers.

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

    def load_kineto_trace(self, kineto_file: str) -> Dict:
        """
        Load and process the Kineto Trace.

        This method parses the Kineto trace file, creating KinetoOperator instances for each operator in the trace.
        It then categorizes and segments these operators for further processing and linking with PyTorch operators.

        Args:
            kineto_file (str): Path to the Kineto trace file.

        Returns:
            Dict: Dictionary containing various data structures needed for linking traces.
        """
        logging.info("Starting to load Kineto Trace.")
        kineto_trace_data = read_dictionary_from_json_file(kineto_file)
        sorted_kineto_ops = sorted(
            [KinetoOperator(op) for op in kineto_trace_data["traceEvents"]],
            key=lambda op: op.timestamp,
        )

        kineto_data = self.construct_kineto_data_structures(sorted_kineto_ops)
        self.calculate_exclusive_dur(kineto_data["kineto_tid_cpu_ops_map"])

        kineto_data["sorted_kineto_cpu_ops"] = sorted(kineto_data["kineto_cpu_ops"], key=lambda op: op.timestamp)
        kineto_data["sorted_kineto_cpu_op_ts"] = [op.timestamp for op in kineto_data["sorted_kineto_cpu_ops"]]

        logging.info(
            f"Processed Kineto trace with {len(kineto_data['kineto_cpu_ops'])} CPU ops, "
            f"{len(kineto_data['kineto_id_cuda_launch_op_map'])} CPU launcher ops, "
            f"and {len(kineto_data['kineto_gpu_ops'])} GPU ops."
        )
        logging.info("Kineto Trace loaded successfully.")
        return kineto_data

    def construct_kineto_data_structures(self, kineto_ops: List[KinetoOperator]) -> Dict:
        """
        Construct necessary data structures required for trace linking from the provided Kineto operators.

        This method identifies process start time, end time, thread start time, and end time, and also categorizes
        operators into CPU, GPU, and other relevant groups.

        Args:
            kineto_ops (List[KinetoOperator]): List of Kineto operators to categorize.

        Returns:
            Dict: Dictionary containing categorized operators and timing boundaries.
        """
        logging.info("Categorizing Kineto operators and calculating timing boundaries.")
        process_start_time = sys.maxsize
        process_end_time = 0
        thread_info = {}

        kineto_cpu_ops = []
        kineto_tid_cpu_ops_map = {}
        kineto_correlation_cuda_runtime_map = {}
        kineto_gpu_ops = []
        kineto_id_arrow_op_map = {}
        kineto_id_cuda_launch_op_map = {}

        for op in kineto_ops:
            if op.is_cpu_op():
                kineto_cpu_ops.append(op)
                kineto_tid_cpu_ops_map.setdefault(op.tid, []).append(op)
                logging.debug(f"Added CPU or user annotation op: {op.name}")

            elif op.is_cuda_launch_op():
                kineto_id_cuda_launch_op_map[op.external_id] = op
                if op.correlation in kineto_correlation_cuda_runtime_map:
                    raise ValueError(
                        f"Duplicate correlation ID {op.correlation} found in self.kineto_id_cuda_launch_op_map."
                    )
                kineto_correlation_cuda_runtime_map[op.correlation] = op
                logging.debug(f"Added CPU launcher op: {op.name}")

            elif op.is_gpu_op():
                kineto_gpu_ops.append(op)
                logging.debug(f"Added GPU op: {op.name}")

            elif op.is_arrow_op():
                assert (op.phase == "s") or (op.phase == "f")
                if op.id is None:
                    error_msg = (
                        f"'id' field is None in Kineto operator, {op}. This is unexpected as 'id' "
                        "should generally be populated for 'ac2g' operators. Please verify the validity of "
                        "the Kineto trace and the 'op' data."
                    )
                    logging.error(error_msg)
                    raise KeyError(error_msg)

                kineto_id_arrow_op_map[op.id] = op

            # Update timing boundaries
            if op.tid is not None:
                process_start_time = min(process_start_time, op.timestamp)
                process_end_time = max(process_end_time, op.timestamp + op.inclusive_dur)
                thread_start_end = thread_info.setdefault(op.tid, [sys.maxsize, 0])
                thread_start_end[0] = min(thread_start_end[0], op.timestamp)
                thread_start_end[1] = max(thread_start_end[1], op.timestamp + op.inclusive_dur)

        kineto_rf_id_to_kineto_op_map = {op.rf_id: op for op in kineto_cpu_ops if op.rf_id is not None}

        return {
            "kineto_cpu_ops": kineto_cpu_ops,
            "kineto_tid_cpu_ops_map": kineto_tid_cpu_ops_map,
            "kineto_correlation_cuda_runtime_map": kineto_correlation_cuda_runtime_map,
            "kineto_gpu_ops": kineto_gpu_ops,
            "kineto_id_arrow_op_map": kineto_id_arrow_op_map,
            "kineto_id_cuda_launch_op_map": kineto_id_cuda_launch_op_map,
            "kineto_process_start_time": process_start_time,
            "kineto_process_end_time": process_end_time,
            "kineto_thread_info": thread_info,
            "kineto_rf_id_to_kineto_op_map": kineto_rf_id_to_kineto_op_map,
            "sorted_kineto_cpu_ops": [],
            "sorted_kineto_cpu_op_ts": [],
        }

    def calculate_exclusive_dur(self, kineto_tid_cpu_ops_map: Dict[int, List[KinetoOperator]]) -> None:
        """
        Calculate the exclusive duration of each operator in the Kineto traces in parallel.

        The exclusive duration is defined as the total duration of the operator minus any time spent in child operators,
        effectively representing the time spent exclusively in that operator.

        Args:
            kineto_tid_cpu_ops_map (Dict[int, List[KinetoOperator]]): Map of thread IDs to their corresponding Kineto
                operators.
        """
        logging.info("Calculating exclusive durations for Kineto operators in parallel.")

        def process_ops_for_thread(ops: List[KinetoOperator]) -> None:
            logging.info(f"Processing {len(ops)} operators in thread.")
            sorted_ops = sorted(ops, key=lambda op: (op.timestamp, op.inclusive_dur))
            for i, op in enumerate(sorted_ops):
                exclusive_dur = op.inclusive_dur
                overlapping_regions = []

                # Identify overlapping regions with child operators
                for child_op in sorted_ops[i + 1 :]:
                    if child_op.timestamp >= op.timestamp and (child_op.timestamp + child_op.inclusive_dur) <= (
                        op.timestamp + op.inclusive_dur
                    ):
                        overlap_start = child_op.timestamp
                        overlap_end = child_op.timestamp + child_op.inclusive_dur
                        overlapping_regions.append((overlap_start, overlap_end))
                    if (op.timestamp + op.inclusive_dur) < child_op.timestamp:
                        break

                # Merge overlapping regions and calculate exclusive duration
                merged_regions = self.merge_overlapping_intervals(overlapping_regions)
                for start, end in merged_regions:
                    exclusive_dur -= end - start

                # Check if exclusive_dur is not negative or zero
                if exclusive_dur < 0:
                    error_msg = (
                        f"Exclusive duration calculation error for node '{op.name}' "
                        f"(ts: {op.timestamp}, inclusive_dur: {op.inclusive_dur}, rf_id: {op.rf_id}): "
                        f"Duration cannot be less than zero."
                    )
                    logging.error(error_msg)
                    raise ValueError(error_msg)

                op.exclusive_dur = exclusive_dur
                logging.debug(
                    f"Node '{op.name}' (ts: {op.timestamp}, inclusive_dur: {op.inclusive_dur}, "
                    f"rf_id: {op.rf_id}) exclusive duration: {op.exclusive_dur} microseconds."
                )

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_ops_for_thread, ops) for ops in kineto_tid_cpu_ops_map.values()]

            for future in as_completed(futures):
                future.result()  # Wait for all threads to complete and handle any exceptions

        logging.info("Exclusive durations for Kineto operators calculated successfully.")

    @staticmethod
    def merge_overlapping_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge overlapping intervals into a single interval.

        Args:
            intervals (List[Tuple[int, int]]): List of intervals.

        Returns:
            List[Tuple[int, int]]: List of merged intervals.
        """
        if not intervals:
            return []

        # Sort intervals based on the start time
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]

        for current in intervals:
            prev = merged[-1]
            if current[0] <= prev[1]:
                # There is overlap, merge the current interval with the previous one
                merged[-1] = (prev[0], max(prev[1], current[1]))
            else:
                # No overlap, add the current interval
                merged.append(current)

        return merged

    def update_kineto_data(self, kineto_data: Dict) -> Tuple:
        """
        Update the variables of the TraceLinker class using the data structures from the kineto_data dictionary.

        Args:
            kineto_data (Dict): Dictionary containing categorized operators and timing boundaries.

        Returns:
            Tuple: Contains all updated variables from the kineto_data dictionary.
        """
        kineto_cpu_ops = kineto_data["kineto_cpu_ops"]
        kineto_tid_cpu_ops_map = kineto_data["kineto_tid_cpu_ops_map"]
        kineto_correlation_cuda_runtime_map = kineto_data["kineto_correlation_cuda_runtime_map"]
        kineto_gpu_ops = kineto_data["kineto_gpu_ops"]
        kineto_id_arrow_op_map = kineto_data["kineto_id_arrow_op_map"]
        kineto_id_cuda_launch_op_map = kineto_data["kineto_id_cuda_launch_op_map"]
        kineto_process_start_time = kineto_data["kineto_process_start_time"]
        kineto_process_end_time = kineto_data["kineto_process_end_time"]
        kineto_thread_info = kineto_data["kineto_thread_info"]
        kineto_rf_id_to_kineto_op_map = {op.rf_id: op for op in kineto_cpu_ops if op.rf_id is not None}
        sorted_kineto_cpu_ops = kineto_data["sorted_kineto_cpu_ops"]
        sorted_kineto_cpu_op_ts = kineto_data["sorted_kineto_cpu_op_ts"]

        return (
            kineto_cpu_ops,
            kineto_tid_cpu_ops_map,
            kineto_correlation_cuda_runtime_map,
            kineto_gpu_ops,
            kineto_id_arrow_op_map,
            kineto_id_cuda_launch_op_map,
            kineto_process_start_time,
            kineto_process_end_time,
            kineto_thread_info,
            kineto_rf_id_to_kineto_op_map,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
        )

    def enforce_inter_thread_order(
        self, kineto_tid_cpu_ops_map: Dict[int, List[KinetoOperator]], threshold: int = 1000
    ) -> Dict[int, List[KinetoOperator]]:
        """
        Enforce order between groups of operators in different threads.

        In Kineto traces with multiple threads, operators are executed in turns, creating groups. This function
        identifies these groups by detecting significant gaps in execution within each thread. It then establishes
        dependencies between these groups across different threads, ensuring the final Chakra execution traces reflect
        inter-thread dependencies realistically.

        An isolated group is formed when there's a significant gap in execution within a thread. Each new group relies
        on the last CPU operator from other threads, enforcing order and dependency across threads.

        Args:
            kineto_tid_cpu_ops_map (Dict[int, List[KinetoOperator]]): Kineto CPU operators grouped by thread ID.
            threshold (int): Threshold for significant gap detection in microseconds, used to define group boundaries.

        Returns:
            Dict[int, List[KinetoOperator]]: Updated map with enforced inter-thread order.
        """
        logging.info("Enforcing inter-thread order in Kineto traces.")

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.process_thread_inter_thread_order, tid, ops, kineto_tid_cpu_ops_map, threshold
                ): tid
                for tid, ops in kineto_tid_cpu_ops_map.items()
            }

            for future in as_completed(futures):
                tid = futures[future]
                try:
                    future.result()
                    logging.debug(f"Thread {tid} dependencies processed.")
                except Exception as e:
                    logging.error(f"Error processing thread {tid}: {e}")

        return kineto_tid_cpu_ops_map

    def process_thread_inter_thread_order(
        self, tid: int, ops: List[KinetoOperator], ops_by_tid: Dict[int, List[KinetoOperator]], threshold: int
    ) -> None:
        """
        Process a single thread's operators to enforce inter-thread order.

        Args:
            tid (int): Thread ID.
            ops (List[KinetoOperator]): List of Kineto operators for the thread.
            ops_by_tid (Dict[int, List[KinetoOperator]]): Kineto operators grouped by thread ID.
            threshold (int): Threshold for significant gap detection in microseconds.
        """
        logging.info(f"Thread {tid}: Identifying gaps for dependency linking with threshold {threshold}us.")
        sorted_ops = sorted(ops, key=lambda op: op.timestamp)
        last_cpu_node_rf_id = None

        for i, op in enumerate(sorted_ops):
            if (
                i == 0
                or (sorted_ops[i].timestamp - sorted_ops[i - 1].timestamp - sorted_ops[i - 1].inclusive_dur) > threshold
            ):
                last_cpu_node_rf_id = self.find_last_cpu_node_before_timestamp(ops_by_tid, tid, op.timestamp)
                if last_cpu_node_rf_id:
                    logging.debug(
                        f"Thread {tid}: Linking op '{op.name}' to CPU node before gap with rf_id "
                        f"'{last_cpu_node_rf_id}'."
                    )

            if last_cpu_node_rf_id:
                op.inter_thread_dep = last_cpu_node_rf_id

    def find_last_cpu_node_before_timestamp(
        self,
        ops_by_tid: Dict[int, List[KinetoOperator]],
        exclude_tid: int,
        timestamp: int,
    ) -> Optional[int]:
        """
        Find the last CPU node ID before a given timestamp in threads other than the excluded one.

        This ID is used to establish dependencies between groups across threads.

        Args:
            ops_by_tid (Dict[int, List[KinetoOperator]]): Operators grouped by thread ID.
            exclude_tid (int): Thread ID to exclude from the search.
            timestamp (int): Timestamp to compare against.

        Returns:
            Optional[int]: The ID of the last CPU node found, or None if not found.
        """
        logging.debug(f"Finding last CPU node before timestamp {timestamp} excluding thread {exclude_tid}.")
        last_cpu_node = None
        last_cpu_node_rf_id = None
        latest_timestamp = 0
        for tid, ops in ops_by_tid.items():
            if tid != exclude_tid:
                for op in sorted(ops, key=lambda op: op.timestamp):
                    if (
                        (op.category in ["cpu_op", "user_annotation"])
                        and (op.timestamp < timestamp)
                        and (op.timestamp > latest_timestamp)
                    ):
                        last_cpu_node = op
                        latest_timestamp = op.timestamp
                        last_cpu_node_rf_id = op.rf_id
        if last_cpu_node:
            logging.debug(f"Last CPU node before timestamp {timestamp} found: {last_cpu_node}")
        return last_cpu_node_rf_id

    def link_traces(
        self,
        pytorch_et_file: str,
        pytorch_ops: List[PyTorchOperator],
        kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
        kineto_correlation_cuda_runtime_map: Dict[int, KinetoOperator],
        kineto_rf_id_to_kineto_op_map: Dict[int, KinetoOperator],
        kineto_gpu_ops: List[KinetoOperator],
        kineto_thread_info: Dict[int, Tuple[int, int]],
        kineto_process_start_time: int,
        kineto_process_end_time: int,
    ) -> Dict:
        """
        Link PyTorch Execution Traces (ET) and Kineto Traces to produce an enhanced PyTorch Execution Trace (ET+).

        This process relies on the assumption of an 'exact match' between these traces.
        """
        logging.info("Starting the process of linking PyTorch and Kineto traces.")
        (
            kineto_cpu_ops,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
        ) = self.add_thread_and_process_annotations(
            kineto_cpu_ops,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
            kineto_thread_info,
            kineto_process_start_time,
            kineto_process_end_time,
        )
        (
            pytorch_op_id_to_kineto_ops_map,
            pytorch_op_id_to_inclusive_dur_map,
            pytorch_op_id_to_exclusive_dur_map,
            pytorch_op_id_to_timestamp_map,
            pytorch_op_id_to_inter_thread_dep_map,
        ) = self.map_pytorch_to_kineto_ops(
            pytorch_ops,
            kineto_cpu_ops,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
            kineto_correlation_cuda_runtime_map,
            kineto_rf_id_to_kineto_op_map,
            kineto_gpu_ops,
        )
        pytorch_et_plus_data = self.construct_et_plus_data(
            pytorch_et_file,
            pytorch_op_id_to_kineto_ops_map,
            pytorch_op_id_to_inclusive_dur_map,
            pytorch_op_id_to_exclusive_dur_map,
            pytorch_op_id_to_timestamp_map,
            pytorch_op_id_to_inter_thread_dep_map,
        )
        logging.info("Traces have been successfully linked.")
        return pytorch_et_plus_data

    def add_thread_and_process_annotations(
        self,
        kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
        kineto_thread_info: Dict[int, Tuple[int, int]],
        kineto_process_start_time: int,
        kineto_process_end_time: int,
    ) -> Tuple[List[KinetoOperator], List[KinetoOperator], List[int]]:
        """
        Add thread and process annotations to Kineto operators based on previously tracked timing information.

        These annotations are crucial for aligning Kineto operators with PyTorch ET nodes, ensuring completeness and
        compatibility of trace data for analysis. This method uses the process start and end times, as well as thread
        start and end times, collected during the categorization process to insert appropriate annotations directly
        into the Kineto operators list.
        """
        logging.info("Adding process and thread annotations to Kineto operators.")

        # Insert process annotation operator. This operator represents the
        # overall time span of the trace process.
        process_annotation_op = KinetoOperator(
            {
                "name": EXECUTION_TRACE_PROCESS_ANNOTATION,
                "ts": kineto_process_start_time,
                "inclusive_dur": kineto_process_end_time - kineto_process_start_time,
                "exclusive_dur": 0,  # Process exclusive duration not applicable
            }
        )
        kineto_cpu_ops.insert(0, process_annotation_op)
        logging.debug(
            "Process annotation added with start time {} and duration {}.".format(
                kineto_process_start_time,
                kineto_process_end_time - kineto_process_start_time,
            )
        )

        # Insert thread annotation operators for each thread. These annotations
        # are crucial for understanding thread-level execution within the trace.
        for tid, (start_ts, end_ts) in kineto_thread_info.items():
            inclusive_dur = end_ts - start_ts
            thread_annotation_op = KinetoOperator(
                {
                    "name": EXECUTION_TRACE_THREAD_ANNOTATION,
                    "ts": start_ts,
                    "inclusive_dur": inclusive_dur,
                    # Exclusive duration is set to zero in the final annotation. This is to avoid constraining
                    # the execution schedule to the original trace, allowing more flexibility in analyzing
                    # dependencies without being bound by specific execution timings.
                    "exclusive_dur": 0,
                }
            )
            # Find the correct position to insert the thread annotation
            position = next(
                (i for i, op in enumerate(kineto_cpu_ops) if op.tid == tid and op.timestamp >= start_ts),
                None,
            )
            if position is not None:
                kineto_cpu_ops.insert(position, thread_annotation_op)
            else:
                kineto_cpu_ops.append(thread_annotation_op)
            logging.debug(
                "Thread {} annotation added with start time {} and duration {}.".format(tid, start_ts, inclusive_dur)
            )

        sorted_kineto_cpu_ops = sorted(kineto_cpu_ops, key=lambda op: op.timestamp)
        sorted_kineto_cpu_op_ts = [op.timestamp for op in sorted_kineto_cpu_ops]

        return kineto_cpu_ops, sorted_kineto_cpu_ops, sorted_kineto_cpu_op_ts

    def map_pytorch_to_kineto_ops(
        self,
        pytorch_ops: List[PyTorchOperator],
        kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
        kineto_correlation_cuda_runtime_map: Dict[int, KinetoOperator],
        kineto_rf_id_to_kineto_op_map: Dict[int, KinetoOperator],
        kineto_gpu_ops: List[KinetoOperator],
    ) -> Tuple[Dict[int, List[KinetoOperator]], Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
        """Map PyTorch ET nodes to corresponding Kineto operators."""
        logging.info("Mapping PyTorch ET nodes to Kineto operators.")
        cpu_ev_idx_to_gpu_ops_map = self.group_gpu_ops_by_cpu_launchers(
            kineto_gpu_ops, kineto_correlation_cuda_runtime_map, sorted_kineto_cpu_ops, sorted_kineto_cpu_op_ts
        )

        pytorch_op_id_to_kineto_ops_map = {}
        pytorch_op_id_to_inclusive_dur_map = {}
        pytorch_op_id_to_exclusive_dur_map = {}
        pytorch_op_id_to_timestamp_map = {}
        pytorch_op_id_to_inter_thread_dep_map = {}

        pytorch_ops_count = len(pytorch_ops)
        kineto_ops_count = len(kineto_cpu_ops)
        if pytorch_ops_count > kineto_ops_count:
            # The specific comment is placed within the if block as requested.
            logging.warning(
                f"Number of PyTorch operators ({pytorch_ops_count}) is larger than the number of Kineto operators "
                f"({kineto_ops_count}). Expected PyTorch ops (CPU only) to be fewer than Kineto ops (CPU and GPU). "
                f"Logging this rare but possible scenario."
            )

        for _, pytorch_op in enumerate(pytorch_ops):
            if (pytorch_op.rf_id is not None) and (pytorch_op.rf_id in kineto_rf_id_to_kineto_op_map):
                kineto_op = kineto_rf_id_to_kineto_op_map[pytorch_op.rf_id]
                if kineto_op is None:
                    logging.warning(
                        f"No corresponding Kineto op found for PyTorch op ID: "
                        f"{pytorch_op.id}, Name: '{pytorch_op.name}'."
                    )
                    continue
                (
                    pytorch_op_id_to_kineto_ops_map[pytorch_op.id],
                    pytorch_op_id_to_inclusive_dur_map[pytorch_op.id],
                    pytorch_op_id_to_exclusive_dur_map[pytorch_op.id],
                    pytorch_op_id_to_timestamp_map[pytorch_op.id],
                    pytorch_op_id_to_inter_thread_dep_map[pytorch_op.id],
                ) = self.link_ops(pytorch_op, kineto_op, cpu_ev_idx_to_gpu_ops_map, kineto_rf_id_to_kineto_op_map)

        logging.info("Completed mapping of PyTorch operators to Kineto operators.")
        return (
            pytorch_op_id_to_kineto_ops_map,
            pytorch_op_id_to_inclusive_dur_map,
            pytorch_op_id_to_exclusive_dur_map,
            pytorch_op_id_to_timestamp_map,
            pytorch_op_id_to_inter_thread_dep_map,
        )

    def group_gpu_ops_by_cpu_launchers(
        self,
        kineto_gpu_ops: List[KinetoOperator],
        kineto_correlation_cuda_runtime_map: Dict[int, KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
    ) -> Dict[int, List[KinetoOperator]]:
        """
        Group GPU operators based on their corresponding CPU launchers.

        This is determined by the 'ev_idx' which links GPU operators to their initiating CPU launcher events.

        Args:
            kineto_gpu_ops (List[KinetoOperator]): List of Kineto GPU operators.
            kineto_correlation_cuda_runtime_map (Dict[int, KinetoOperator]): Mapping between correlation IDs and
                kernel-launching CUDA runtime operators.
            sorted_kineto_cpu_ops (List[KinetoOperator]): Sorted list of Kineto CPU operators.
            sorted_kineto_cpu_op_ts (List[int]): Sorted list of timestamps extracted from Kineto operators for
                efficient temporal queries.

        Returns:
            Dict[int, List[KinetoOperator]]: Mapping from CPU launch event indices to GPU operators.

        Raises:
            ValueError: If 'ev_idx' is missing for any GPU operator.
        """
        cpu_ev_idx_to_gpu_ops_map = {}
        for gpu_op in kineto_gpu_ops:
            parent_cpu_op = self.find_parent_cpu_op(
                gpu_op, kineto_correlation_cuda_runtime_map, sorted_kineto_cpu_ops, sorted_kineto_cpu_op_ts
            )
            if not parent_cpu_op:
                warning_msg = f"Missing parent CPU operator for GPU op '{gpu_op.name}'. Orphaned GPU operator."
                logging.warning(warning_msg)
                continue

            if parent_cpu_op.ev_idx == "":
                error_msg = (
                    f"Missing 'ev_idx' for CPU operator {parent_cpu_op.name}. "
                    f"Cannot link GPU op {gpu_op.name} to {parent_cpu_op.name}."
                )
                logging.warning(error_msg)
                continue

            logging.debug(f"group_gpu_ops_by_cpu_launchers '{parent_cpu_op.name}' -> '{gpu_op.name}'")

            cpu_ev_idx_to_gpu_ops_map.setdefault(parent_cpu_op.ev_idx, []).append(gpu_op)

        return cpu_ev_idx_to_gpu_ops_map

    def find_parent_cpu_op(
        self,
        kineto_gpu_op: KinetoOperator,
        kineto_correlation_cuda_runtime_map: Dict[int, KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
    ) -> Optional[KinetoOperator]:
        """
        Find the parent CPU operator for a given GPU operator by identifying the corresponding CUDA runtime operator.

        It then locates the closest preceding CPU operator based on the CUDA runtime's timestamp, considering the
        temporal distance between the GPU operation's start and the initiating CPU operation.

        Args:
            kineto_gpu_op (KinetoOperator): The GPU operator.
            kineto_correlation_cuda_runtime_map (Dict[int, KinetoOperator]): Mapping between correlation IDs and
                kernel-launching CUDA runtime operators.
            sorted_kineto_cpu_ops (List[KinetoOperator]): Sorted list of Kineto CPU operators.
            sorted_kineto_cpu_op_ts (List[int]): Sorted list of timestamps extracted from Kineto operators for
                efficient temporal queries.

        Returns:
            Optional[KinetoOperator]: The parent CPU operator if found.

        Raises:
            ValueError: If no CUDA runtime operator is found for the given correlation ID.
        """
        if kineto_gpu_op.correlation not in kineto_correlation_cuda_runtime_map:
            warning_msg = (
                f"No CUDA runtime operator found for correlation ID {kineto_gpu_op.correlation}. "
                "This is not a common case, and there should be a corresponding CUDA runtime operator for a given GPU "
                "kernel operator. It can be a case where CUDA runtime operators are not properly identified and added "
                "to the map, kineto_correlation_cuda_runtime_map. Please manually check if the corresponding CUDA "
                "runtime operator with the correlation is dropped by mistake. It is likely that it is because of "
                "incomplete map, cuda_launch_operations, in is_cuda_launch_op. Please update the map properly to cover"
                " all CUDA runtime launch operators."
            )
            logging.warning(warning_msg)
            return None

        kineto_runtime_op = kineto_correlation_cuda_runtime_map[kineto_gpu_op.correlation]
        kineto_gpu_op.tid = kineto_runtime_op.tid
        logging.debug(
            f"Found CUDA runtime operation '{kineto_runtime_op.name}' for GPU operator '{kineto_gpu_op.name}'."
        )

        kineto_gpu_op.timestamp = kineto_runtime_op.timestamp

        # Find the closest CPU operator that precedes the CUDA runtime operation
        parent_cpu_op = self.find_closest_op(
            kineto_gpu_op, sorted_kineto_cpu_ops, sorted_kineto_cpu_op_ts, kineto_runtime_op.timestamp
        )
        if not parent_cpu_op:
            logging.warning(
                f"No parent CPU operator found for GPU operator '{kineto_gpu_op.name}' "
                f"linked to CUDA runtime operation '{kineto_runtime_op.name}' "
                f"(ts: {kineto_runtime_op.timestamp})."
            )

        return parent_cpu_op

    def find_closest_op(
        self,
        kineto_gpu_op: KinetoOperator,
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
        ts: int,
    ) -> Optional[KinetoOperator]:
        """
        Find the Kineto operator that is closest in start time to a given timestamp and that covers the timestamp.

        Args:
            kineto_gpu_op (KinetoOperator): The GPU operator being compared.
            sorted_kineto_cpu_ops (List[KinetoOperator]): List of Kineto operators.
            sorted_kineto_cpu_op_ts (List[int]): List of timestamps for the sorted Kineto operators.
            ts (int): The timestamp to compare against.

        Returns:
            Optional[KinetoOperator]: The closest Kineto operator if found.
        """
        # Searching for the closest timestamp index
        index = bisect.bisect_left(sorted_kineto_cpu_op_ts, ts)

        if index == 0:
            # All operators are later than the timestamp
            return None
        else:
            # The operator immediately before the index is the closest one before the timestamp
            closest_op = sorted_kineto_cpu_ops[index - 1]

            # Check for NCCL specifics: if it's an NCCL operation and 'nccl:coalesced' should be skipped
            if "nccl" in kineto_gpu_op.name.lower() and closest_op.name == "nccl:coalesced":
                # Move back to find a non-'nccl:coalesced' operator, if available
                for new_index in range(index - 2, -1, -1):
                    potential_op = sorted_kineto_cpu_ops[new_index]
                    if potential_op.tid == kineto_gpu_op.tid and potential_op.name != "nccl:coalesced":
                        return potential_op
                # If no valid alternative found before 'nccl:coalesced', continue search forward
                index = index - 1  # Adjust index to skip 'nccl:coalesced'

            # After skipping 'nccl:coalesced', verify that the closest operation is on the same thread
            # as the GPU operation
            if closest_op.tid == kineto_gpu_op.tid:
                return closest_op

            # If the tids do not match, search forward to find the closest matching tid
            for i in range(index - 1, -1, -1):
                op = sorted_kineto_cpu_ops[i]
                if op.tid == kineto_gpu_op.tid:
                    if "nccl" in kineto_gpu_op.name.lower() and op.name == "nccl:coalesced":
                        continue  # Skip 'nccl:coalesced' if it's an NCCL-related GPU operation
                    if op.timestamp <= ts:
                        return op

            # If no matching tid is found going forward, return None
            return None

    def link_ops(
        self,
        pytorch_op: PyTorchOperator,
        kineto_op: KinetoOperator,
        cpu_ev_idx_to_gpu_ops_map: Dict[int, List[KinetoOperator]],
        kineto_rf_id_to_kineto_op_map: Dict[int, KinetoOperator],
    ) -> Tuple[List[KinetoOperator], int, int, int, Optional[int]]:
        """
        Link a PyTorch operator to its corresponding Kineto operator and any associated GPU operators.

        Args:
            pytorch_op (PyTorchOperator): PyTorch operator to link.
            kineto_op (KinetoOperator): Corresponding Kineto operator.
            cpu_ev_idx_to_gpu_ops_map (Dict[int, List[KinetoOperator]]): GPU ops mapping.
            kineto_rf_id_to_kineto_op_map (Dict[int, KinetoOperator]): Kineto operator mapping.

        Returns:
            Tuple containing:
                - List[KinetoOperator]: The list of linked Kineto GPU operators.
                - int: The inclusive duration of the linked Kineto operator.
                - int: The exclusive duration of the linked Kineto operator.
                - int: The timestamp of the linked Kineto operator.
                - Optional[int]: The inter-thread dependency ID if present.
        """
        kineto_op.pytorch_op = pytorch_op
        linked_gpu_ops = cpu_ev_idx_to_gpu_ops_map.get(kineto_op.ev_idx, [])
        inclusive_dur = kineto_op.inclusive_dur
        exclusive_dur = kineto_op.exclusive_dur
        timestamp = kineto_op.timestamp
        inter_thread_dep = None

        if kineto_op.inter_thread_dep:
            inter_thread_dep_kineto_op = kineto_rf_id_to_kineto_op_map[kineto_op.inter_thread_dep]
            if inter_thread_dep_kineto_op.pytorch_op:
                inter_thread_dep = inter_thread_dep_kineto_op.pytorch_op.id

        self.link_gpu_ops(pytorch_op, linked_gpu_ops)

        return linked_gpu_ops, inclusive_dur, exclusive_dur, timestamp, inter_thread_dep

    def link_gpu_ops(self, pytorch_op: PyTorchOperator, kineto_gpu_ops: List[KinetoOperator]) -> None:
        """
        Link GPU operators to a PyTorch operator.

        Args:
            pytorch_op (PyTorchOperator): The PyTorch operator to link to.
            kineto_gpu_ops (List[KinetoOperator]): GPU operators to link.
        """
        for gpu_op in kineto_gpu_ops:
            gpu_op.parent_pytorch_op_id = pytorch_op.id

    def construct_et_plus_data(
        self,
        pytorch_et_file: str,
        pytorch_op_id_to_kineto_ops_map: Dict[int, List[KinetoOperator]],
        pytorch_op_id_to_inclusive_dur_map: Dict[int, int],
        pytorch_op_id_to_exclusive_dur_map: Dict[int, int],
        pytorch_op_id_to_timestamp_map: Dict[int, int],
        pytorch_op_id_to_inter_thread_dep_map: Dict[int, int],
    ) -> Dict:
        """
        Construct the enhanced PyTorch Execution Trace (ET+) data structure.

        This method enriches the PyTorch execution trace with detailed performance data from the Kineto trace, offering
        a comprehensive view of the execution.

        Args:
            pytorch_et_file (str): Path to the PyTorch execution trace file.
            pytorch_op_id_to_kineto_ops_map (Dict[int, List[KinetoOperator]]): Map from PyTorch op IDs to Kineto GPU
                ops.
            pytorch_op_id_to_inclusive_dur_map (Dict[int, int]): Inclusive duration map for PyTorch ops.
            pytorch_op_id_to_exclusive_dur_map (Dict[int, int]): Exclusive duration map for PyTorch ops.
            pytorch_op_id_to_timestamp_map (Dict[int, int]): Timestamp map for PyTorch ops.
            pytorch_op_id_to_inter_thread_dep_map (Dict[int, int]): Mapping of PyTorch operator IDs to IDs of latest CPU
                node from other threads before the gap.

        Returns:
            Dict: The constructed ET+ data.
        """
        logging.info("Constructing ET+ data.")
        with open(pytorch_et_file, "r") as file:
            pytorch_et_data = json.load(file)

        sorted_nodes = sorted(pytorch_et_data["nodes"], key=lambda x: x["id"])
        gpu_ops = []
        for op in sorted_nodes:
            gpu_ops += self.process_op_and_dependents(
                op,
                pytorch_op_id_to_kineto_ops_map,
                pytorch_op_id_to_inclusive_dur_map,
                pytorch_op_id_to_exclusive_dur_map,
                pytorch_op_id_to_timestamp_map,
                pytorch_op_id_to_inter_thread_dep_map,
            )
        pytorch_et_data["nodes"] += gpu_ops

        # Update parent-child relationships with new IDs
        sorted_nodes = sorted(pytorch_et_data["nodes"], key=lambda x: x["id"])
        for op in sorted_nodes:
            if "ctrl_deps" in op:
                op["ctrl_deps"] = self.id_assigner.assign_or_retrieve_id(op["ctrl_deps"])

        return pytorch_et_data

    def process_op_and_dependents(
        self,
        op: Dict,
        pytorch_op_id_to_kineto_ops_map: Dict[int, List[KinetoOperator]],
        pytorch_op_id_to_inclusive_dur_map: Dict[int, int],
        pytorch_op_id_to_exclusive_dur_map: Dict[int, int],
        pytorch_op_id_to_timestamp_map: Dict[int, int],
        pytorch_op_id_to_inter_thread_dep_map: Dict[int, int],
    ) -> List[Dict]:
        """
        Process a single operator in the PyTorch ET data, assign a new unique ID, and process any dependent operators.

        Args:
            op (Dict): The operator to be processed.
            pytorch_op_id_to_kineto_ops_map (Dict[int, List[KinetoOperator]]): Map from PyTorch op IDs to Kineto GPU
                ops.
            pytorch_op_id_to_inclusive_dur_map (Dict[int, int]): Inclusive duration map for PyTorch ops.
            pytorch_op_id_to_exclusive_dur_map (Dict[int, int]): Exclusive duration map for PyTorch ops.
            pytorch_op_id_to_timestamp_map (Dict[int, int]): Timestamp map for PyTorch ops.
            pytorch_op_id_to_inter_thread_dep_map (Dict[int, int]): Mapping of PyTorch operator IDs to IDs of latest CPU
                node from other threads before the gap.

        Returns:
            List[Dict]: A list of GPU operators processed and linked to the given operator.
        """
        orig_op_id = op["id"]
        new_op_id = self.id_assigner.assign_or_retrieve_id(orig_op_id)
        op["id"] = new_op_id

        # Update operator with Kineto data if available
        if orig_op_id in pytorch_op_id_to_inclusive_dur_map:
            op["inclusive_dur"] = pytorch_op_id_to_inclusive_dur_map[orig_op_id]
            op["exclusive_dur"] = pytorch_op_id_to_exclusive_dur_map[orig_op_id]
            op["ts"] = pytorch_op_id_to_timestamp_map[orig_op_id]
            if orig_op_id in pytorch_op_id_to_inter_thread_dep_map:
                op["inter_thread_dep"] = self.id_assigner.lookup_new_id(
                    pytorch_op_id_to_inter_thread_dep_map[orig_op_id]
                )
            else:
                op["inter_thread_dep"] = None

        # Process and append dependent GPU operators
        if orig_op_id in pytorch_op_id_to_kineto_ops_map:
            gpu_ops = self.process_dependent_gpu_ops(op, orig_op_id, pytorch_op_id_to_kineto_ops_map)
            pytorch_op_id_to_kineto_ops_map.pop(orig_op_id)
            return gpu_ops
        return []

    def process_dependent_gpu_ops(
        self, cpu_op: Dict, orig_op_id: int, pytorch_op_id_to_kineto_ops_map: Dict[int, List[KinetoOperator]]
    ) -> List[Dict]:
        """
        Create and return a list of GPU operators that are dependent on a specific CPU operator.

        The GPU operators are deep copies of the existing operators with updated IDs and other relevant
        fields from the CPU operator.

        Args:
            cpu_op (Dict): The PyTorch CPU operator.
            orig_op_id (int): The original ID of the CPU operator.
            pytorch_op_id_to_kineto_ops_map (Dict[int, List[KinetoOperator]]): Map from PyTorch op IDs to Kineto GPU
                ops.

        Returns:
            List[Dict]: A list of processed GPU operators.
        """
        updated_gpu_ops = []
        dependent_gpu_ops = pytorch_op_id_to_kineto_ops_map.get(orig_op_id, [])
        for gpu_op in sorted(dependent_gpu_ops, key=lambda x: x.timestamp):
            new_gpu_op = copy.deepcopy(cpu_op)
            new_gpu_op_id = self.id_assigner.generate_new_id()
            new_gpu_op.update(
                {
                    "id": new_gpu_op_id,
                    "ctrl_deps": orig_op_id,
                    "inputs": cpu_op["inputs"],
                    "outputs": cpu_op["outputs"],
                    "cat": gpu_op.category,
                    "name": gpu_op.name,
                    "ph": gpu_op.phase,
                    "inclusive_dur": gpu_op.inclusive_dur,
                    "exclusive_dur": gpu_op.exclusive_dur,
                    "ts": gpu_op.timestamp,
                    "stream": gpu_op.stream,
                }
            )
            updated_gpu_ops.append(new_gpu_op)

        return updated_gpu_ops

    def dump_pytorch_execution_trace_plus(self, pytorch_et_plus_data: Dict, output_file: str) -> None:
        """
        Dump the enhanced PyTorch Execution Trace (ET+) data to a file.

        Args:
            pytorch_et_plus_data (Dict): The constructed ET+ data.
            output_file (str): The file path where the ET+ data will be saved.
        """
        logging.info(f"Starting to dump ET+ data to {output_file}.")

        if pytorch_et_plus_data is None:
            logging.error("ET+ data not constructed. Please run construct_et_plus_data first.")
            return

        if "nodes" in pytorch_et_plus_data:
            pytorch_et_plus_data["nodes"] = sorted(pytorch_et_plus_data["nodes"], key=lambda x: x["id"])

        try:
            with open(output_file, "w") as file:
                json.dump(pytorch_et_plus_data, file, indent=4)
            logging.info(f"ET+ data dumped to {output_file}.")
        except IOError as e:
            logging.error(f"Failed to dump ET+ data to {output_file}. Error: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while dumping ET+ data. Error: {e}")
