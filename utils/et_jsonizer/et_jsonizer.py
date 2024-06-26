#!/usr/bin/env python3

import argparse

from google.protobuf.json_format import MessageToJson

from chakra.third_party.utils.protolib import (
    openFileRd as open_file_rd,
    decodeMessage as decode_message
)

from chakra.et_def.et_def_pb2 import (
    Node as ChakraNode,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execution Trace Jsonizer"
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
        help="Output filename"
    )
    args = parser.parse_args()

    et = open_file_rd(args.input_filename)
    node = ChakraNode()
    with open(args.output_filename, 'w') as f:
        while decode_message(et, node):
            f.write(MessageToJson(node))
    et.close()


if __name__ == "__main__":
    main()
