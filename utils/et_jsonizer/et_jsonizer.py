#!/usr/bin/env python3

import argparse

from google.protobuf.json_format import MessageToJson

from chakra.third_party.utils.protolib import (
    openFileRd as open_file_rd,
    decodeMessage as decode_message
)

from chakra.et_def.et_def_pb2 import (
    GlobalMetadata,
    Node as ChakraNode,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converts Chakra execution trace to JSON format."
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        required=True,
        help="Specifies the input filename of the Chakra execution trace."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        required=True,
        help="Specifies the output filename for the JSON data."
    )
    args = parser.parse_args()

    execution_trace = open_file_rd(args.input_filename)
    node = ChakraNode()
    with open(args.output_filename, 'w') as file:
        global_metadata = GlobalMetadata()
        decode_message(execution_trace, global_metadata)
        file.write(MessageToJson(global_metadata))
        while decode_message(execution_trace, node):
            file.write(MessageToJson(node))
    execution_trace.close()


if __name__ == "__main__":
    main()
