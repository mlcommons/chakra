import logging
import sys
from typing import Any, Callable, Dict, List, Tuple

from et_replay.execution_trace import Node as PyTorchOperator
from et_replay.execution_trace import ExecutionTrace as PyTorchTrace
from et_replay.execution_trace import EXECUTION_TRACE_THREAD_ANNOTATION
from et_replay.utils import read_dictionary_from_json_file

# Increase the recursion limit for deep Chakra host execution traces.
sys.setrecursionlimit(10**6)


class ChakraHostTraceLoader:
    """Loads Chakra host traces."""

    def load(self, chakra_host_trace_file: str) -> Tuple[List[PyTorchOperator], Dict[str, Any]]:
        """
        Load and process the Chakra Host Execution Trace.

        Args:
            chakra_host_trace_file (str): Path to the PyTorch execution trace file.

        Returns:
            Tuple[List[PyTorchOperator], Dict[str, Any]]: A tuple containing a list of PyTorch operators and a host trace.
        """
        logging.debug(f"Starting to load Chakra host execution trace from file: {chakra_host_trace_file}.")
        host_trace = read_dictionary_from_json_file(chakra_host_trace_file)

        schema: str = host_trace["schema"]
        pid: int = host_trace["pid"]
        nodes: List[Dict[str, Any]] = host_trace["nodes"]

        create_operator = self._get_operator_creation_method(schema)
        if create_operator is None:
            raise ValueError(
                f"No corresponding node creation function found for schema version {schema}"
            )
        
        host_ops: Dict[int, PyTorchOperator] = {}
        thread_roots: Dict[int, int] = {}
        for node in nodes:
            op = create_operator(pid, node)
            if op.parent_id == 1 and EXECUTION_TRACE_THREAD_ANNOTATION in op.name:
                thread_roots[op.tid] = op.id
            host_ops[op.id] = op

        for op in host_ops.values():
            if op.parent_id not in host_ops:             
                parent_id = thread_roots.get(op.tid, None)
                if parent_id is not None:
                    op.parent_id = parent_id
                    op.set_parent(host_ops[parent_id])
                    host_ops[parent_id].add_child(op)
                    node = next(filter(lambda n: n["id"] == op.id, nodes), None)
                    if node is not None:
                        node["ctrl_deps"] = parent_id
        
        for op in host_ops.values():
            op.sort_children()
        
        chakra_host_ops = sorted(host_ops.values(), key=lambda x: x.id)

        logging.debug(f"Extracted {len(chakra_host_ops)} operators from Chakra host execution trace.")
        logging.debug("Chakra host execution trace has been loaded and processed successfully.")

        return chakra_host_ops, host_trace

    def _get_operator_creation_method(self, schema: str) -> Callable[[int, Dict[str, Any]], PyTorchOperator] | None:
        node_creation_func = {
            "1.0.1": PyTorchTrace._create_node_v1_0_1,
            "1.0.2-chakra.0.0.4": PyTorchTrace._create_node_v1_0_2_chakra_0_0_4,
            # 1.0.3 expands pg name to <pg_name, pg_desc> so it use the same parser as 1.0.2
            "1.0.3-chakra.0.0.4": PyTorchTrace._create_node_v1_0_2_chakra_0_0_4,
            # 1.0.4 adds PT2 kernel backend and kernel file
            "1.0.4-chakra.0.0.4": PyTorchTrace._create_node_v1_0_2_chakra_0_0_4,
            # 1.1.0 includes new comm args in record_param_comms
            "1.1.0-chakra.0.0.4": PyTorchTrace._create_node_v1_0_2_chakra_0_0_4,
            # 1.1.1 includes tensor strides
            "1.1.1-chakra.0.0.4": PyTorchTrace._create_node_v1_1_1_chakra_0_0_4,
            # Add future versions here
        }
        return node_creation_func.get(schema, None)
