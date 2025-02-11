#!/usr/bin/env python3

import argparse
import os
import sys
import re
import json
from copy import deepcopy
from google.protobuf.json_format import MessageToJson

from ...schema.protobuf.et_def_pb2 import (
    COMM_COLL_NODE,
    COMP_NODE,
    COMM_SEND_NODE,
    COMM_RECV_NODE,
    GlobalMetadata,
)
from ...schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)
from ..third_party.utils.protolib import decodeMessage as decode_message
from ..third_party.utils.protolib import encodeMessage as encode_message
from ..third_party.utils.protolib import openFileRd as open_file_rd
from ..third_party.utils.protolib import openFileWt as open_file_wt
# from ..pg_extractor.pg_extractor import NetworkCommunicators,Communicator,CommunicatorNode

def main() -> None:
    parser = argparse.ArgumentParser(description="Read Chakra execution trace and extend parallelism.")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Specifies the directory of the Chakra execution trace."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Specify the directory for the generated Chakra execution trace files after expansion."
    )

    args = parser.parse_args()

    execution_trace = open_file_rd(args.input_file)
    et_outfile = open_file_wt(args.output_file)
    nodeslist=[]
    node = ChakraNode()

    global_metadata = GlobalMetadata()
    decode_message(execution_trace, global_metadata)
    while decode_message(execution_trace, node):
        nodeslist.append(node)
        node = ChakraNode()

    execution_trace.close()

    for node in nodeslist:
        if node.id == 133:
            copynode = node
        print(node.id)

    encode_message(et_outfile,global_metadata)
    for chakraNode in nodeslist:
        encode_message(et_outfile,chakraNode)
    copynode.id = len(nodeslist) + 1
    for attr in copynode.attr:
        if attr.name == 'pg_id':
            attr.int64_val += 1
        if attr.name == 'comm_type':
            attr.int64_val = 0
    encode_message(et_outfile,copynode)

if __name__=='__main__':
    main()