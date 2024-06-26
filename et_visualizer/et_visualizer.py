#!/usr/bin/env python3

import argparse
import graphviz
import networkx as nx

from chakra.third_party.utils.protolib import (
    openFileRd as open_file_rd,
    decodeMessage as decode_message
)
from chakra.et_def.et_def_pb2 import Node


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execution Trace Visualizer"
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        default=None,
        required=True,
        help="Input Chakra execution trace filename"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        required=True,
        help="Output graph filename"
    )
    args = parser.parse_args()

    et = open_file_rd(args.input_filename)
    node = Node()

    # Determine the file type to be created based on the output filename
    if args.output_filename.endswith((".pdf", ".dot")):
        f = graphviz.Digraph()
        while decode_message(et, node):
            f.node(name=f"{node.id}",
                   label=f"{node.name}",
                   id=str(node.id),
                   shape="record")

            # Handling data dependencies
            for data_dep_id in node.data_deps:
                f.edge(str(data_dep_id), str(node.id), arrowhead="normal")  # using "normal" arrow for data_deps

            # Handling control dependencies
            for ctrl_dep_id in node.ctrl_deps:
                f.edge(str(ctrl_dep_id), str(node.id), arrowhead="tee")  # using "tee" arrow for ctrl_deps

        if args.output_filename.endswith(".pdf"):
            f.render(args.output_filename.replace(".pdf", ""),
                     format="pdf", cleanup=True)
        else:  # ends with ".dot"
            f.render(args.output_filename.replace(".dot", ""),
                     format="dot", cleanup=True)
    elif args.output_filename.endswith(".graphml"):
        G = nx.DiGraph()
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
