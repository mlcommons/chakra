#!/usr/bin/env python3

import argparse
import graphviz
import networkx as nx

from third_party.utils.protolib import (
    openFileRd as open_file_rd,
    decodeMessage as decode_message
)
from et_def.et_def_pb2 import Node


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
            for parent_id in node.parent:
                f.edge(str(parent_id), str(node.id))

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
            for parent_id in node.parent:
                G.add_edge(parent_id, node.id)
        nx.write_graphml(G, args.output_filename)

    et.close()


if __name__ == "__main__":
    main()
