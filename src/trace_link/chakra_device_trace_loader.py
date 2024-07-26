import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from et_replay.utils import read_dictionary_from_json_file

from .kineto_operator import KinetoOperator


class ChakraDeviceTraceLoader:
    """Loads Chakra device traces."""

    def load(
        self, chakra_device_trace: str
    ) -> Tuple[
        List[KinetoOperator],
        Dict[int, List[KinetoOperator]],
        Dict[int, KinetoOperator],
        List[KinetoOperator],
        Dict[int, KinetoOperator],
        Dict[int, KinetoOperator],
        int,
        int,
        Dict[int, Tuple[int, int]],
        Dict[int, KinetoOperator],
        List[KinetoOperator],
        List[int],
    ]:
        """
        Load and process the Chakra device trace.

        Args:
            chakra_device_trace (str): Path to the Chakra device trace file.

        Returns:
            Tuple containing various data structures needed for linking traces.
        """
        logging.debug(f"Starting to load Chakra device trace from file: {chakra_device_trace}.")
        chakra_trace_data = read_dictionary_from_json_file(chakra_device_trace)
        sorted_kineto_ops = sorted(
            [KinetoOperator(op) for op in chakra_trace_data["traceEvents"]],
            key=lambda op: op.timestamp,
        )

        dev_data = self.construct_dev_data_structures(sorted_kineto_ops, chakra_device_trace)
        self.calculate_exclusive_dur(dev_data["kineto_tid_cpu_ops_map"])

        dev_data["sorted_kineto_cpu_ops"] = sorted(dev_data["kineto_cpu_ops"], key=lambda op: op.timestamp)
        dev_data["sorted_kineto_cpu_op_ts"] = [op.timestamp for op in dev_data["sorted_kineto_cpu_ops"]]

        logging.debug(
            f"Processed Chakra device trace with {len(dev_data['kineto_cpu_ops'])} CPU ops, "
            f"{len(dev_data['kineto_id_cuda_launch_op_map'])} CPU launcher ops, "
            f"and {len(dev_data['kineto_gpu_ops'])} GPU ops."
        )
        logging.debug("Chakra device trace has been loaded and processed successfully.")
        return (
            dev_data["kineto_cpu_ops"],
            dev_data["kineto_tid_cpu_ops_map"],
            dev_data["kineto_correlation_cuda_runtime_map"],
            dev_data["kineto_gpu_ops"],
            dev_data["kineto_id_arrow_op_map"],
            dev_data["kineto_id_cuda_launch_op_map"],
            dev_data["kineto_process_start_time"],
            dev_data["kineto_process_end_time"],
            dev_data["kineto_thread_info"],
            dev_data["kineto_rf_id_to_kineto_op_map"],
            dev_data["sorted_kineto_cpu_ops"],
            dev_data["sorted_kineto_cpu_op_ts"],
        )

    def construct_dev_data_structures(self, kineto_ops: List[KinetoOperator], trace_file: str) -> Dict:
        """
        Construct necessary data structures required for trace linking from the provided Kineto operators.

        This method identifies process start time, end time, thread start time, and end time, and also categorizes
        operators into CPU, GPU, and other relevant groups.

        Args:
            kineto_ops (List[KinetoOperator]): List of Kineto operators to categorize.
            trace_file (str): Path to the trace file for logging purposes.

        Returns:
            Dict: Dictionary containing categorized operators and timing boundaries.
        """
        logging.debug("Categorizing Kineto operators and calculating timing boundaries.")
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

            elif op.is_kernel_launch_op():
                kineto_id_cuda_launch_op_map[op.external_id] = op
                if op.correlation in kineto_correlation_cuda_runtime_map:
                    error_msg = (
                        f"Duplicate correlation ID {op.correlation} found in kineto_id_cuda_launch_op_map. "
                        "The kineto_id_cuda_launch_op_map works as a mapping to link GPU operators with the launcher "
                        "CPU operator for the GPU operator. The correlation field works as a link, and this map has a "
                        "mapping between the correlation and the launcher operator. Each kernel launch operator "
                        "should have a unique correlation ID for linking it to a GPU operator. Therefore, duplicated "
                        "correlation is not expected in the map. Please review the file manually to see if the "
                        f"operator has an invalid correlation value in file: {trace_file}."
                    )
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                kineto_correlation_cuda_runtime_map[op.correlation] = op
                logging.debug(f"Added CPU launcher op: {op.name}")

            elif op.is_gpu_op():
                kineto_gpu_ops.append(op)
                logging.debug(f"Added GPU op: {op.name}")

            elif op.is_ac2g_op():  # arrow from CPU to GPU
                assert (op.phase == "s") or (op.phase == "f")
                if op.id is None:
                    error_msg = (
                        f"'id' field is None in Kineto operator: {op} in file: {trace_file}. This is unexpected as "
                        "'id' should generally be populated for 'ac2g' operators. Please verify the validity of "
                        "the Kineto trace and the operator data."
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
        logging.debug("Calculating exclusive durations for Kineto operators in parallel.")

        def process_ops_for_thread(ops: List[KinetoOperator]) -> None:
            logging.debug(f"Processing {len(ops)} operators in thread.")
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

        logging.debug("Exclusive durations for Kineto operators calculated successfully.")

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
