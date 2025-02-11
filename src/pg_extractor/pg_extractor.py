#!/usr/bin/env python3

import argparse
import os
import re
import json
from collections import defaultdict

from google.protobuf.json_format import MessageToJson

from ...schema.protobuf.et_def_pb2 import (
    COMM_COLL_NODE,
    COMP_NODE,
    COMM_SEND_NODE,
    COMM_RECV_NODE,
    ALL_REDUCE,
    REDUCE,
    ALL_GATHER,
    GATHER,
    SCATTER,
    BROADCAST,
    ALL_TO_ALL,
    REDUCE_SCATTER,
    CollectiveCommType,
    GlobalMetadata,
)
from ...schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)
from ..third_party.utils.protolib import decodeMessage as decode_message
from ..third_party.utils.protolib import encodeMessage as encode_message
from ..third_party.utils.protolib import openFileRd as open_file_rd
from ..third_party.utils.protolib import openFileWt as open_file_wt

class CommunicatorNode:
    def __init__(self, globalRank, rank, commList=None):
        self.globalRank = globalRank
        self.rank = rank
        if commList is None:
            self.commList = []
        else:
            self.commList = commList

    def to_dict(self):
        return {
            'globalRank': self.globalRank,
            'rank': self.rank,
            'commList': self.commList,
        }

class CollImplementation:
    def __init__(self, Coll, Impl):
        self.Coll = Coll
        self.Impl = Impl

    def to_dict(self):
        return {'Coll': self.Coll, 'Impl': self.Impl}

class Communicator:
    def __init__(self, communicatorId, type=None):
        self.communicatorId = communicatorId
        self.type = type
        self.nodes = []
        self.collImpl = []

    def addNode(self, arg1, commType=None):
        if isinstance(arg1, int):  # 如果第一个参数是整数
            globalRank = arg1
            node = next((n for n in self.nodes if n.globalRank == globalRank), None)
            if node is None:
                node = CommunicatorNode(globalRank, len(self.nodes))
                self.nodes.append(node)
            if commType is not None:
                node.commList.append(commType)
        elif isinstance(arg1, CommunicatorNode):  # 如果第一个参数是CommunicatorNode对象
            self.nodes.append(arg1)
        else:
            raise TypeError("Unsupported argument type for addNode: {}".format(type(arg1)))

    def to_dict(self):
        return {
            'communicatorId': self.communicatorId,
            'type': self.type,
            'nodes': [node.to_dict() for node in self.nodes],
            'collImpl': [ci.to_dict() for ci in self.collImpl]
        }

class NetworkCommunicators:
    def __init__(self, communicators : Communicator = None):
        if communicators == None:
            self.communicators = defaultdict(lambda: Communicator(None))
        else:
            self.communicators = communicators

    def append(self, pg_name, pg_desc, globalRank, commType=None, coll='All_Reduce', impl='RingAllreduceAlgo'):
        if self.communicators[pg_name].communicatorId is None:
            self.communicators[pg_name].communicatorId = pg_name
        if self.communicators[pg_name].type is None:
            self.communicators[pg_name].type = pg_desc
        if len(self.communicators[pg_name].collImpl) == 0:
            self.communicators[pg_name].collImpl.append(CollImplementation(coll,impl))

        self.communicators[pg_name].addNode(globalRank, commType)

    def to_dict(self):
        # 将communicators字典转换为字典列表
        return {
            'communicators': [comm.to_dict() for comm in self.communicators.values() if comm.communicatorId is not None]
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

def get_coll_type(comm_type: int) -> str:
    if comm_type == ALL_REDUCE:
        return "All_Reduce"
    elif comm_type == ALL_TO_ALL:
        return "All_to_All"
    elif comm_type == ALL_GATHER:
        return "All_Gather"
    elif comm_type == REDUCE_SCATTER:
        return "Reduce_Scatter"
    elif comm_type == BROADCAST:
        return "Bcast"
    elif comm_type == REDUCE:
        return "Reduce"
    elif comm_type == GATHER:
        return "Gather"
    elif comm_type == SCATTER:
        return "Scatter"
    return ""

def main() -> None:
    parser = argparse.ArgumentParser(description="Extracte process groups from Chakra execution trace.")
    parser.add_argument(
        "--input_filename", type=str, required=True, help="Specifies the input filename of the Chakra execution trace."
    )
    parser.add_argument(
        "--output_filename", type=str, required=True, help="Specifies the output filename for the communicator group JSON file."
    )
    args = parser.parse_args()

    dir_name = os.path.dirname(args.input_filename)
    et_name = os.path.basename(args.input_filename)

    # 正则表达式用于匹配id
    pattern = re.compile(f"^{re.escape(et_name)}\.(\\d+)\.et$")

    et_ids = list()
    for file in os.listdir(dir_name):
        match = pattern.match(file)
        if match:
            # 如果文件名符合要求，提取id
            et_ids.append(int(match.group(1)))

    et_ids.sort()

    communicators = NetworkCommunicators()
    for id in et_ids:
        execution_trace = open_file_rd(f'{args.input_filename}.{id}.et')
        node = ChakraNode()

        global_metadata = GlobalMetadata()
        decode_message(execution_trace, global_metadata)

        while decode_message(execution_trace, node):
            pg_name = ""
            pg_desc = ""
            is_cpu_op = True
            coll_type = -1
            comm_src = -1
            comm_dts = -1
            for attr in node.attr:
                if attr.name == 'pg_name':
                    pg_name = attr.string_val
                if attr.name == 'pg_desc':
                    pg_desc = attr.string_val
                if attr.name == 'is_cpu_op':
                    is_cpu_op = attr.bool_val
                if attr.name == 'comm_type':
                    coll_type = attr.int64_val
                if attr.name == 'comm_src':
                    comm_src = attr.int32_val
                if attr.name == 'comm_dst':
                    comm_dst = attr.int32_val
            if (not is_cpu_op):
                if node.type == COMM_COLL_NODE:
                    communicators.append(int(pg_name), pg_desc, id, get_coll_type(coll_type))
                elif node.type == COMM_SEND_NODE:
                    communicators.append(int(pg_name), pg_desc, id, f'Send to {comm_dst}')
                elif node.type == COMM_RECV_NODE:
                    communicators.append(int(pg_name), pg_desc, id, f'Recv from {comm_src}')

        execution_trace.close()

    with open(args.output_filename, 'w') as f:
        f.write(communicators.to_json())


if __name__ == "__main__":
    main()
