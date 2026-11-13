import logging
from typing import List

from ..host_trace.node import HostNode
from ..host_trace.trace import HostTrace, ops_reachable_from


class ChakraHostTraceLoader:
    """Loads Chakra host traces."""

    def load(self, chakra_host_trace_file: str) -> List[HostNode]:
        """
        Load and process the Chakra Host Execution Trace.

        Args:
            chakra_host_trace_file (str): Path to the PyTorch execution trace file.

        Returns:
            List[HostNode]: The host ops under the process root, sorted by id.
        """
        logging.debug(f"Starting to load Chakra host execution trace from file: {chakra_host_trace_file}.")
        chakra_host_ops = HostTrace.from_file(chakra_host_trace_file).linker_ops()
        logging.debug(f"Extracted {len(chakra_host_ops)} operators from Chakra host execution trace.")
        logging.debug("Chakra host execution trace has been loaded and processed successfully.")

        return chakra_host_ops

    def extract_chakra_host_ops(self, node: HostNode) -> List[HostNode]:
        """
        Extract the node and every node under it, sorted by id.

        Args:
            node (HostNode): Starting node for extraction.

        Returns:
            List[HostNode]: Sorted list of the extracted nodes.
        """
        nodes = ops_reachable_from(node)
        logging.debug(f"Traversed {len(nodes)} nodes from root node ID: {node.id}")
        return nodes
