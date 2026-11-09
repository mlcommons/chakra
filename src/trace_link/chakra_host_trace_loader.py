import logging
import sys
from typing import List

from et_replay.execution_trace import Node as PyTorchOperator
from et_replay.utils import load_execution_trace_file

# Increase the recursion limit for deep Chakra host execution traces.
sys.setrecursionlimit(10**6)


class ChakraHostTraceLoader:
    """Loads Chakra host traces."""

    def load(self, chakra_host_trace_file: str) -> List[PyTorchOperator]:
        """
        Load and process the Chakra Host Execution Trace.

        Args:
            chakra_host_trace_file (str): Path to the PyTorch execution trace file.

        Returns:
            List[PyTorchOperator]: List of PyTorch operators.
        """
        logging.debug(f"Starting to load Chakra host execution trace from file: {chakra_host_trace_file}.")
        chakra_host_trace = load_execution_trace_file(chakra_host_trace_file)

        root_node = chakra_host_trace.get_nodes()[1]  # Root node is usually 1-based
        chakra_host_ops = self.extract_chakra_host_ops(root_node)
        logging.debug(f"Extracted {len(chakra_host_ops)} operators from Chakra host execution trace.")
        logging.debug("Chakra host execution trace has been loaded and processed successfully.")

        return chakra_host_ops

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
