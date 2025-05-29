import logging
import sys
from typing import Any, Callable, Dict, List, Tuple

from et_replay.execution_trace import EXECUTION_TRACE_THREAD_ANNOTATION as THREAD_ANNOTATION
from et_replay.execution_trace import ExecutionTrace as PyTorchTrace
from et_replay.execution_trace import Node as PyTorchOperator
from et_replay.utils import read_dictionary_from_json_file

# Increase the recursion limit for deep Chakra host execution traces.
sys.setrecursionlimit(10**6)


class ChakraHostTraceLoader:
    """Loads Chakra host traces."""

    def load(self,
        chakra_host_trace_file: str,
        connect_host_trace: bool) -> Tuple[List[PyTorchOperator], Dict[str, Any]]:
        """
        Load and process the Chakra Host Execution Trace.

        Args:
            chakra_host_trace_file (str): Path to the PyTorch execution trace file.
            connect_host_trace (bool): Connect host nodes with missing parents to the corresponding thread root node.
        Returns:
            Tuple[List[PyTorchOperator], Dict[str, Any]]: Tuple containing list of PyTorch operators and host trace.
        """
        logging.debug(f"Starting to load Chakra host execution trace from file: {chakra_host_trace_file}.")
        host_trace = read_dictionary_from_json_file(chakra_host_trace_file)

        host_ops = self._create_host_ops(host_trace, connect_host_trace)
        root_node = host_ops.get(1) # Root node is usually 1-based
        
        chakra_host_ops = self.extract_chakra_host_ops(root_node)

        logging.debug(f"Extracted {len(chakra_host_ops)} operators from Chakra host execution trace.")
        logging.debug("Chakra host execution trace has been loaded and processed successfully.")

        return chakra_host_ops, host_trace

    def extract_chakra_host_ops(self, node: PyTorchOperator) -> List[PyTorchOperator]:
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
        logging.debug(f"Traversed {len(nodes)} nodes from root node ID: {node.id}")
        return sorted(nodes, key=lambda x: x.id)

    def _create_host_ops(self, host_trace: Dict[str, Any], connect_host_trace: bool) -> Dict[int, PyTorchOperator]:
        """
        Create host operators from the provided host trace.
        This method processes the host trace, extracts nodes, and creates PyTorchOperator instances based on the schema
        version specified in the host trace.

        Args:
            host_trace (Dict[str, Any]): The host trace dictionary.
            connect_host_trace (bool): Connect host nodes with missing parents to the corresponding thread root node.
        Returns:
            Dict[int, PyTorchOperator]: A dictionary mapping operator IDs to PyTorchOperator instances.
        """
        
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
            host_op = create_operator(pid, node)
            host_ops[host_op.id] = host_op
            if host_op.parent_id == 1 and THREAD_ANNOTATION in host_op.name:
                thread_roots[host_op.tid] = host_op.id

        for host_op in host_ops.values():
            if host_op.parent_id in host_ops and host_op.id != 1:
                parent = host_ops[host_op.parent_id]
                host_op.set_parent(parent)
                parent.add_child(host_op)
            elif connect_host_trace is True: # connect orphans to the thread root
                parent_id = thread_roots.get(host_op.tid, None)
                if parent_id is not None:
                    host_op.parent_id = parent_id
                    parent = host_ops[parent_id]
                    host_op.set_parent(parent)
                    parent.add_child(host_op)
                    node = next(filter(lambda n: n["id"] == host_op.id, nodes), None)
                    if node is not None:
                        node["ctrl_deps"] = parent_id
        
        for host_op in host_ops.values():
            host_op.sort_children()

        return host_ops
    
    def _get_operator_creation_method(self, schema: str) -> Callable[[int, Dict[str, Any]], PyTorchOperator] | None:
        """
        Get the operator creation method for the specified schema version.
        
        Args:
            schema (str): The schema version of the host trace.   
        Returns:
            Callable[[int, Dict[str, Any]], PyTorchOperator] | None: The operator creation functor for the schema version,
            or None if no functor is found.
        """
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
        return node_creation_func.get(schema)
