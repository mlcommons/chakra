import argparse
import re

import graphviz
import networkx as nx

from ...schema.protobuf.et_def_pb2 import GlobalMetadata, Node
from ..third_party.utils.protolib import decodeMessage as decode_message
from ..third_party.utils.protolib import openFileRd as open_file_rd


def escape_label(label: str) -> str:
    """
    Escapes special characters in labels for graph rendering.

    Args:
        label (str): The original label string.

    Returns:
        str: The escaped label string.
    """
    # Define special characters to escape
    special_chars = "{}()<>\\[\\]|&-"
    # Escape special characters
    return re.sub(f"([{special_chars}])", r"\\\1", label)


def main() -> None:
    """Generate an output graph file in the specified format (PDF, DOT, or GraphML)."""
    parser = argparse.ArgumentParser(description="Execution Trace Visualizer")
    parser.add_argument("--input_filename", type=str, required=True, help="Input Chakra execution trace filename")
    parser.add_argument(
        "--output_filename",
        type=str,
        required=True,
        help=(
            "Output graph filename. Supported extensions are pdf, dot, and graphml. "
            "Recommend using graphml for large graphs for rendering speed."
        ),
    )
    args = parser.parse_args()

    et = open_file_rd(args.input_filename)
    node = Node()
    gm = GlobalMetadata()

    # Determine the file type to be created based on the output filename
    if args.output_filename.endswith((".pdf", ".dot")):
        f = graphviz.Digraph()
        decode_message(et, gm)
        while decode_message(et, node):
            escaped_label = escape_label(node.name)
            f.node(name=f"{node.id}", label=escaped_label, id=str(node.id), shape="record")

            # Handling data dependencies
            for data_dep_id in node.data_deps:
                f.edge(str(data_dep_id), str(node.id), arrowhead="normal")  # using "normal" arrow for data_deps

            # Handling control dependencies
            for ctrl_dep_id in node.ctrl_deps:
                f.edge(str(ctrl_dep_id), str(node.id), arrowhead="tee")  # using "tee" arrow for ctrl_deps

        if args.output_filename.endswith(".pdf"):
            f.render(args.output_filename.replace(".pdf", ""), format="pdf", cleanup=True)
        else:  # ends with ".dot"
            f.render(args.output_filename.replace(".dot", ""), format="dot", cleanup=True)
    elif args.output_filename.endswith(".graphml"):
        G = nx.DiGraph()
        decode_message(et, gm)
        while decode_message(et, node):
            G.add_node(node.id, label=node.name)

            # Handling data dependencies
            for data_dep_id in node.data_deps:
                G.add_edge(data_dep_id, node.id, dependency="data")

            # Handling control dependencies
            for ctrl_dep_id in node.ctrl_deps:
                G.add_edge(ctrl_dep_id, node.id, dependency="control")

        nx.write_graphml(G, args.output_filename)
    else:
        print("Unknown output file extension. Must be one of pdf, dot, graphml.")

    et.close()


if __name__ == "__main__":
    main()
