import bisect
import copy
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from et_replay.execution_trace import (
    EXECUTION_TRACE_PROCESS_ANNOTATION,
    EXECUTION_TRACE_THREAD_ANNOTATION,
)
from et_replay.execution_trace import Node as PyTorchOperator

from .chakra_device_trace_loader import ChakraDeviceTraceLoader
from .chakra_host_trace_loader import ChakraHostTraceLoader
from .kineto_operator import KinetoOperator
from .unique_id_assigner import UniqueIdAssigner


class TraceLinker:
    """
    Links Chakra host execution traces (ET) and Chakra device ET to generate Chakra host + device ET.

    Attributes
        chakra_host_trace_loader (ChakraHostTraceLoader): Loader for Chakra host execution traces.
        chakra_device_trace_loader (ChakraDeviceTraceLoader): Loader for Chakra device execution traces.
        id_assigner (UniqueIdAssigner): Assigns unique IDs to operators.
    """

    def __init__(self) -> None:
        """Initialize the TraceLinker with a log level."""
        self.chakra_host_trace_loader = ChakraHostTraceLoader()
        self.chakra_device_trace_loader = ChakraDeviceTraceLoader()
        self.id_assigner = UniqueIdAssigner()

    def link(self, chakra_host_trace: str, chakra_device_trace: str, output_file: str) -> None:
        """
        Links Chakra host execution traces (ET) and Chakra device ET to generate Chakra host + device ET.

        Args:
            chakra_host_trace (str): Path to the Chakra host execution trace file.
            chakra_device_trace (str): Path to the Kineto trace file.
            output_file (str): Path for the output nyTorch execution trace plus file.
        """
        host_ops = self.chakra_host_trace_loader.load(chakra_host_trace)

        (
            kineto_cpu_ops,
            kineto_tid_cpu_ops_map,
            kineto_correlation_cuda_runtime_map,
            kineto_gpu_ops,
            kineto_id_arrow_op_map,
            kineto_id_cuda_launch_op_map,
            kineto_process_start_time,
            kineto_process_end_time,
            kineto_thread_debug,
            kineto_rf_id_to_device_op_map,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
        ) = self.chakra_device_trace_loader.load(chakra_device_trace)

        kineto_tid_cpu_ops_map = self.enforce_inter_thread_order(kineto_tid_cpu_ops_map)

        chakra_execution_trace_plus_data = self.link_traces(
            chakra_host_trace,
            host_ops,
            kineto_cpu_ops,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
            kineto_correlation_cuda_runtime_map,
            kineto_rf_id_to_device_op_map,
            kineto_gpu_ops,
            kineto_thread_debug,
            kineto_process_start_time,
            kineto_process_end_time,
        )

        self.dump_chakra_execution_trace_plus(chakra_execution_trace_plus_data, output_file)

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
        logging.debug("Enforcing inter-thread order in Kineto traces.")

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.process_thread_inter_thread_order, tid, ops, kineto_tid_cpu_ops_map, threshold
                ): tid
                for tid, ops in kineto_tid_cpu_ops_map.items()
            }

            for future in as_completed(futures):
                tid = futures[future]
                future.result()
                logging.debug(f"Thread {tid} dependencies processed.")

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
        logging.debug(f"Thread {tid}: Identifying gaps for dependency linking with threshold {threshold}us.")
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
        chakra_host_trace: str,
        host_ops: List[PyTorchOperator],
        kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
        kineto_correlation_cuda_runtime_map: Dict[int, KinetoOperator],
        kineto_rf_id_to_device_op_map: Dict[int, KinetoOperator],
        kineto_gpu_ops: List[KinetoOperator],
        kineto_thread_debug: Dict[int, Tuple[int, int]],
        kineto_process_start_time: int,
        kineto_process_end_time: int,
    ) -> Dict:
        """
        Link Chakra Host ET and Chakra Device ET to produce an enhanced Chakra ET (ET +).

        Args:
            chakra_host_trace (str): Path to the Chakra host execution trace file.
            host_ops (List[PyTorchOperator]): List of Chakra host operators.
            kineto_cpu_ops (List[KinetoOperator]): List of Kineto CPU operators.
            sorted_kineto_cpu_ops (List[KinetoOperator]): Sorted list of Kineto CPU operators.
            sorted_kineto_cpu_op_ts (List[int]): Sorted list of timestamps for the Kineto CPU operators.
            kineto_correlation_cuda_runtime_map (Dict[int, KinetoOperator]): Mapping between correlation IDs and
                kernel-launching CUDA runtime operators.
            kineto_rf_id_to_device_op_map (Dict[int, KinetoOperator]): Mapping between rf_id and Kineto operators.
            kineto_gpu_ops (List[KinetoOperator]): List of Kineto GPU operators.
            kineto_thread_debug (Dict[int, Tuple[int, int]]): debugrmation about threads, mapping thread IDs to a tuple
                of start and end times.
            kineto_process_start_time (int): Start time of the process, based on the earliest operator timestamp.
            kineto_process_end_time (int): End time of the process, based on the latest operator timestamp.

        Returns:
            Dict: The enhanced Chakra Host Execution Trace (ET+).
        """
        logging.debug("Starting the process of linking Chakra host and device traces.")
        (
            kineto_cpu_ops,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
        ) = self.add_thread_and_process_annotations(
            kineto_cpu_ops,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
            kineto_thread_debug,
            kineto_process_start_time,
            kineto_process_end_time,
        )
        (
            host_op_id_to_kineto_ops_map,
            host_op_id_to_inclusive_dur_map,
            host_op_id_to_exclusive_dur_map,
            host_op_id_to_timestamp_map,
            host_op_id_to_inter_thread_dep_map,
        ) = self.map_host_to_device_ops(
            host_ops,
            kineto_cpu_ops,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
            kineto_correlation_cuda_runtime_map,
            kineto_rf_id_to_device_op_map,
            kineto_gpu_ops,
        )
        chakra_execution_trace_plus_data = self.construct_et_plus_data(
            chakra_host_trace,
            host_op_id_to_kineto_ops_map,
            host_op_id_to_inclusive_dur_map,
            host_op_id_to_exclusive_dur_map,
            host_op_id_to_timestamp_map,
            host_op_id_to_inter_thread_dep_map,
        )
        logging.debug("Traces have been successfully linked.")
        return chakra_execution_trace_plus_data

    def add_thread_and_process_annotations(
        self,
        kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
        kineto_thread_debug: Dict[int, Tuple[int, int]],
        kineto_process_start_time: int,
        kineto_process_end_time: int,
    ) -> Tuple[List[KinetoOperator], List[KinetoOperator], List[int]]:
        """
        Add thread and process annotations to Kineto operators based on previously tracked timing debugrmation.

        These annotations are crucial for aligning Kineto operators with Chakra host nodes, ensuring completeness and
        compatibility of trace data for analysis. This method uses the process start and end times, as well as thread
        start and end times, collected during the categorization process to insert appropriate annotations directly
        into the Kineto operators list.
        """
        logging.debug("Adding process and thread annotations to Kineto operators.")

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
        for tid, (start_ts, end_ts) in kineto_thread_debug.items():
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

    def map_host_to_device_ops(
        self,
        host_ops: List[PyTorchOperator],
        kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
        kineto_correlation_cuda_runtime_map: Dict[int, KinetoOperator],
        kineto_rf_id_to_device_op_map: Dict[int, KinetoOperator],
        kineto_gpu_ops: List[KinetoOperator],
    ) -> Tuple[
        Dict[int, List[KinetoOperator]],
        Dict[int, int],
        Dict[int, int],
        Dict[int, int],
        Dict[int, int],
    ]:
        """Map Chakra host operators to corresponding device operators."""
        logging.debug("Mapping Charka host operators to corresponding device operators.")
        cpu_ev_idx_to_gpu_ops_map = self.group_gpu_ops_by_cpu_launchers(
            kineto_gpu_ops, kineto_correlation_cuda_runtime_map, sorted_kineto_cpu_ops, sorted_kineto_cpu_op_ts
        )

        host_op_id_to_kineto_ops_map = {}
        host_op_id_to_inclusive_dur_map = {}
        host_op_id_to_exclusive_dur_map = {}
        host_op_id_to_timestamp_map = {}
        host_op_id_to_inter_thread_dep_map = {}

        for _, host_op in enumerate(host_ops):
            if (host_op.rf_id is not None) and (host_op.rf_id in kineto_rf_id_to_device_op_map):
                kineto_op = kineto_rf_id_to_device_op_map[host_op.rf_id]
                if kineto_op is None:
                    logging.warning(
                        f"No corresponding Kineto op found for Chakra host op ID: {host_op.id}, Name: "
                        f"'{host_op.name}'."
                    )
                    continue
                (
                    host_op_id_to_kineto_ops_map[host_op.id],
                    host_op_id_to_inclusive_dur_map[host_op.id],
                    host_op_id_to_exclusive_dur_map[host_op.id],
                    host_op_id_to_timestamp_map[host_op.id],
                    host_op_id_to_inter_thread_dep_map[host_op.id],
                ) = self.link_ops(
                    host_op,
                    kineto_op,
                    cpu_ev_idx_to_gpu_ops_map,
                    kineto_rf_id_to_device_op_map,
                )

        logging.debug("Completed mapping of Chakra host operators to Kineto operators.")
        return (
            host_op_id_to_kineto_ops_map,
            host_op_id_to_inclusive_dur_map,
            host_op_id_to_exclusive_dur_map,
            host_op_id_to_timestamp_map,
            host_op_id_to_inter_thread_dep_map,
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
                "incomplete map, cuda_launch_operations, in is_kernel_launch_op. Please update the map properly to "
                "cover all CUDA runtime launch operators."
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
        host_op: PyTorchOperator,
        kineto_op: KinetoOperator,
        cpu_ev_idx_to_gpu_ops_map: Dict[int, List[KinetoOperator]],
        kineto_rf_id_to_device_op_map: Dict[int, KinetoOperator],
    ) -> Tuple[List[KinetoOperator], int, int, int, Optional[int]]:
        """
        Link a Chakra host operator to its corresponding Kineto operator and any associated GPU operators.

        Args:
            host_op (PyTorchOperator): Chakra host operator to link.
            kineto_op (KinetoOperator): Corresponding Kineto operator.
            cpu_ev_idx_to_gpu_ops_map (Dict[int, List[KinetoOperator]]): GPU ops mapping.
            kineto_rf_id_to_device_op_map (Dict[int, KinetoOperator]): Kineto operator mapping.

        Returns:
            Tuple containing:
                - List[KinetoOperator]: The list of linked Kineto GPU operators.
                - int: The inclusive duration of the linked Kineto operator.
                - int: The exclusive duration of the linked Kineto operator.
                - int: The timestamp of the linked Kineto operator.
                - Optional[int]: The inter-thread dependency ID if present.
        """
        kineto_op.host_op = host_op
        linked_gpu_ops = cpu_ev_idx_to_gpu_ops_map.get(kineto_op.ev_idx, [])
        inclusive_dur = kineto_op.inclusive_dur
        exclusive_dur = kineto_op.exclusive_dur
        timestamp = kineto_op.timestamp

        inter_thread_dep = self.get_inter_thread_dep(kineto_op, kineto_rf_id_to_device_op_map)

        self.link_gpu_ops(host_op, linked_gpu_ops)

        return linked_gpu_ops, inclusive_dur, exclusive_dur, timestamp, inter_thread_dep

    def get_inter_thread_dep(self, kineto_op, kineto_rf_id_to_device_op_map):
        """
        Retrieve the inter-thread dependency ID for a given Kineto operator.

        This method finds the corresponding Chakra host operator ID for the inter-thread dependency if it exists.

        Args:
            kineto_op (KinetoOperator): The Kineto operator being processed.
            kineto_rf_id_to_device_op_map (Dict[int, KinetoOperator]): Mapping from rf_id to Kineto operators.

        Returns:
            Optional[int]: The Chakra host operator ID for the inter-thread dependency if it exists, otherwise None.
        """
        if kineto_op.inter_thread_dep:
            inter_thread_dep_kineto_op = kineto_rf_id_to_device_op_map[kineto_op.inter_thread_dep]
            if inter_thread_dep_kineto_op.host_op:
                return inter_thread_dep_kineto_op.host_op.id
        return None

    def link_gpu_ops(self, host_op: PyTorchOperator, kineto_gpu_ops: List[KinetoOperator]) -> None:
        """
        Link GPU operators to a Chakra host operator.

        Args:
            host_op (PyTorchOperator): The Chakra host operator to link to.
            kineto_gpu_ops (List[KinetoOperator]): GPU operators to link.
        """
        for gpu_op in kineto_gpu_ops:
            gpu_op.parent_host_op_id = host_op.id

    def construct_et_plus_data(
        self,
        chakra_host_trace: str,
        host_op_id_to_kineto_ops_map: Dict[int, List[KinetoOperator]],
        host_op_id_to_inclusive_dur_map: Dict[int, int],
        host_op_id_to_exclusive_dur_map: Dict[int, int],
        host_op_id_to_timestamp_map: Dict[int, int],
        host_op_id_to_inter_thread_dep_map: Dict[int, int],
    ) -> Dict:
        """
        Construct the enhanced Chakra Host Execution Trace (ET+) data structure.

        This method enriches the Chakra host execution trace with detailed performance data from the Kineto trace,
        offering a comprehensive view of the execution.

        Args:
            chakra_host_trace (str): Path to the Chakra host execution trace file.
            host_op_id_to_kineto_ops_map (Dict[int, List[KinetoOperator]]): Map from Chakra host op IDs to Kineto
                GPU ops.
            host_op_id_to_inclusive_dur_map (Dict[int, int]): Inclusive duration map for Chakra host ops.
            host_op_id_to_exclusive_dur_map (Dict[int, int]): Exclusive duration map for Chakra host ops.
            host_op_id_to_timestamp_map (Dict[int, int]): Timestamp map for Chakra host ops.
            host_op_id_to_inter_thread_dep_map (Dict[int, int]): Mapping of Chakra host operator IDs to IDs of
                latest CPU node from other threads before the gap.

        Returns:
            Dict: The constructed ET+ data.
        """
        logging.debug("Constructing ET+ data.")
        with open(chakra_host_trace, "r") as file:
            pytorch_et_data = json.load(file)

        sorted_nodes = sorted(pytorch_et_data["nodes"], key=lambda x: x["id"])
        gpu_ops = []
        for op in sorted_nodes:
            gpu_ops += self.process_op_and_dependents(
                op,
                host_op_id_to_kineto_ops_map,
                host_op_id_to_inclusive_dur_map,
                host_op_id_to_exclusive_dur_map,
                host_op_id_to_timestamp_map,
                host_op_id_to_inter_thread_dep_map,
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
        host_op_id_to_kineto_ops_map: Dict[int, List[KinetoOperator]],
        host_op_id_to_inclusive_dur_map: Dict[int, int],
        host_op_id_to_exclusive_dur_map: Dict[int, int],
        host_op_id_to_timestamp_map: Dict[int, int],
        host_op_id_to_inter_thread_dep_map: Dict[int, int],
    ) -> List[Dict]:
        """
        Process a single operator in the Chakra host trace, assign a unique ID, and process any dependent operators.

        Args:
            op (Dict): The operator to be processed.
            host_op_id_to_kineto_ops_map (Dict[int, List[KinetoOperator]]): Map from Chakra host op IDs to Kineto GPU
                ops.
            host_op_id_to_inclusive_dur_map (Dict[int, int]): Inclusive duration map for Chakra host ops.
            host_op_id_to_exclusive_dur_map (Dict[int, int]): Exclusive duration map for Chakra host ops.
            host_op_id_to_timestamp_map (Dict[int, int]): Timestamp map for Chakra host ops.
            host_op_id_to_inter_thread_dep_map (Dict[int, int]): Mapping of Chakra host operator IDs to IDs of latest
                CPU node from other threads before the gap.

        Returns:
            List[Dict]: A list of GPU operators processed and linked to the given operator.
        """
        orig_op_id = op["id"]
        new_op_id = self.id_assigner.assign_or_retrieve_id(orig_op_id)
        op["id"] = new_op_id

        # Update operator with Kineto data if available
        if orig_op_id in host_op_id_to_inclusive_dur_map:
            op["inclusive_dur"] = host_op_id_to_inclusive_dur_map[orig_op_id]
            op["exclusive_dur"] = host_op_id_to_exclusive_dur_map[orig_op_id]
            op["ts"] = host_op_id_to_timestamp_map[orig_op_id]
            if orig_op_id in host_op_id_to_inter_thread_dep_map:
                op["inter_thread_dep"] = self.id_assigner.lookup_new_id(host_op_id_to_inter_thread_dep_map[orig_op_id])
            else:
                op["inter_thread_dep"] = None

        # Process and append dependent GPU operators
        if orig_op_id in host_op_id_to_kineto_ops_map:
            gpu_ops = self.process_dependent_gpu_ops(op, orig_op_id, host_op_id_to_kineto_ops_map)
            host_op_id_to_kineto_ops_map.pop(orig_op_id)
            return gpu_ops
        return []

    def process_dependent_gpu_ops(
        self, cpu_op: Dict, orig_op_id: int, host_op_id_to_kineto_ops_map: Dict[int, List[KinetoOperator]]
    ) -> List[Dict]:
        """
        Create and return a list of GPU operators that are dependent on a specific CPU operator.

        The GPU operators are deep copies of the existing operators with updated IDs and other relevant
        fields from the CPU operator.

        Args:
            cpu_op (Dict): The Chakra host CPU operator.
            orig_op_id (int): The original ID of the CPU operator.
            host_op_id_to_kineto_ops_map (Dict[int, List[KinetoOperator]]): Map from host operator IDs to device
                operators

        Returns:
            List[Dict]: A list of processed GPU operators.
        """
        updated_gpu_ops = []
        dependent_gpu_ops = host_op_id_to_kineto_ops_map.get(orig_op_id, [])
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
                    **(
                        {"pg_name": gpu_op.pg_name}
                        if gpu_op.is_inter_gpu_comms_op() and gpu_op.pg_name is not None
                        else {}
                    ),
                }
            )

            updated_gpu_ops.append(new_gpu_op)

        return updated_gpu_ops

    def dump_chakra_execution_trace_plus(self, chakra_execution_trace_plus_data: Dict, output_file: str) -> None:
        """
        Dump the enhanced Chakra execution trace plus data to a file.

        Args:
            chakra_execution_trace_plus_data (Dict): The constructed ET+ data.
            output_file (str): The file path where the ET+ data will be saved.
        """
        logging.debug(f"Starting to dump ET+ data to {output_file}.")

        if chakra_execution_trace_plus_data is None:
            logging.error("ET+ data not constructed. Please run construct_et_plus_data first.")
            return

        if "nodes" in chakra_execution_trace_plus_data:
            chakra_execution_trace_plus_data["nodes"] = sorted(
                chakra_execution_trace_plus_data["nodes"], key=lambda x: x["id"]
            )

        with open(output_file, "w") as file:
            json.dump(chakra_execution_trace_plus_data, file, indent=4)
        logging.debug(f"ET+ data dumped to {output_file}.")
