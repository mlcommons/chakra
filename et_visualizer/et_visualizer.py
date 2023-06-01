#!/usr/bin/env python3

import argparse
import graphviz

from third_party.utils.protolib import (
    openFileRd as open_file_rd,
    decodeMessage as decode_message
)
from et_def.et_def_pb2 import Node

def main() -> None:
    parser = argparse.ArgumentParser(
            description="Execution Graph Visualizer"
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
            help="Output Graphviz graph filename"
    )
    args = parser.parse_args()

    eg = open_file_rd(args.input_filename)
    f = graphviz.Digraph()
    node = Node()
    while decode_message(eg, node):
        f.node(name=f"{node.id}",
               label=f"{node.name}",
               id=str(node.id),
               shape="record")
        for parent_id in node.parent:
            f.edge(str(parent_id), str(node.id))

    if args.output_filename.endswith(".pdf"):
        f.render(args.output_filename.replace(".pdf", ""),
                 format="pdf", cleanup=True)
    elif args.output_filename.endswith(".dot"):
        f.render(args.output_filename.replace(".dot", ""),
                 format="dot", cleanup=True)
    else:
        f.render(args.output_filename,
                 format="dot", cleanup=True)

    eg.close()

if __name__ == "__main__":
    main()
