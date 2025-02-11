#!/usr/bin/env python3

import argparse
from collections import defaultdict

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

from enum import Enum
import pdb

class States(Enum):
    AVAIL = 1
    NOTAVAIL = 2
    MERGECPU = 3
    MERGEGPU = 4

dep_graph = {}
dep_replace = {}
node_num_child = defaultdict(set)
nodes_need_replacing = set()
nodes_need_merge_cpu = set()
nodes_need_merge_gpu = set()
node_addition_runtime = defaultdict(float)

def buildNodeDepReplace(node_id):
    if node_id in dep_replace:
        return
    dep_replace[node_id] = {'ctrl_deps': set(), 'data_deps': set()}
    dep_set = set()
    dep_set.update(dep_graph[node_id]['ctrl_deps'])
    dep_set.update(dep_graph[node_id]['data_deps'])
    for dep in dep_set:
        if dep in nodes_need_replacing:
            # 如果依赖的也是无效节点，确保替换前其dep_replace已被构建
            buildNodeDepReplace(dep)
            dep_replace[node_id]['ctrl_deps'].update(dep_replace[dep]['ctrl_deps'])
            dep_replace[node_id]['data_deps'].update(dep_replace[dep]['data_deps'])
            if node_id in nodes_need_merge_cpu and dep in nodes_need_merge_cpu:
                node_addition_runtime[node_id] += node_addition_runtime[dep]
            if node_id in nodes_need_merge_gpu and dep in nodes_need_merge_gpu:
                node_addition_runtime[node_id] += node_addition_runtime[dep]
        else:
            dep_replace[node_id]['ctrl_deps'].add(dep)
            dep_replace[node_id]['data_deps'].add(dep)

def buildDepReplace():
    for node_id in nodes_need_replacing:
        buildNodeDepReplace(node_id)


def checkAvailable(node: ChakraNode):
    num_ops = 0
    is_cpu_op = True
    for attr in node.attr:
        if attr.name == 'is_cpu_op':
            is_cpu_op = attr.bool_val
        if attr.name == 'num_ops':
            num_ops = attr.uint64_val
    if is_cpu_op:
        if node.duration_micros == 0 and num_ops == 0:
            return States.NOTAVAIL
        else:
            return States.MERGECPU
    elif (not is_cpu_op) and node.type == COMP_NODE:
        if node.duration_micros == 0 and num_ops == 0:
            return States.NOTAVAIL
        else:
            return States.MERGEGPU
    elif (not is_cpu_op) and (node.type == COMM_COLL_NODE or node.type == COMM_SEND_NODE or node.type == COMM_RECV_NODE):
        return States.AVAIL
    else:
        return States.NOTAVAIL

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Chakra execution trace.")
    parser.add_argument(
        "--input_filename", type=str, required=True, help="Specifies the input filename of the Chakra execution trace."
    )
    parser.add_argument(
        "--output_filename", type=str, required=True, help="Specifies the output filename for the cleaned Chakra execution trace."
    )
    args = parser.parse_args()

    execution_trace = open_file_rd(args.input_filename)
    cleaned_et = open_file_wt(args.output_filename)
    node = ChakraNode()
    nodes_not_avail = set()

    global_metadata = GlobalMetadata()
    decode_message(execution_trace, global_metadata)

    while decode_message(execution_trace, node):
        dep_graph.setdefault(node.id, {'ctrl_deps': [], 'data_deps': []})
        # 处理控制依赖
        if node.ctrl_deps:
            dep_graph[node.id]['ctrl_deps'].extend(node.ctrl_deps)
        for dep in dep_graph[node.id]['ctrl_deps']:
            node_num_child[dep].add(node.id)
        # 处理数据依赖
        if node.data_deps:
            dep_graph[node.id]['data_deps'].extend(node.data_deps)
        for dep in dep_graph[node.id]['data_deps']:
            node_num_child[dep].add(node.id)
        # 检查节点是否有效
        node_addition_runtime[node.id] = node.duration_micros
        state = checkAvailable(node)
        if state == States.MERGECPU:
            nodes_need_merge_cpu.add(node.id)
        elif state == States.MERGEGPU:
            nodes_need_merge_gpu.add(node.id)
        elif state == States.NOTAVAIL:
            nodes_not_avail.add(node.id)
    
    for node_id in nodes_need_merge_cpu:
        if len(dep_graph[node_id]['data_deps']) > 0 and len(node_num_child[node_id]) == 1 and next(iter(node_num_child[node_id])) in nodes_need_merge_cpu:
            nodes_need_replacing.add(node_id)
    
    for node_id in nodes_need_merge_gpu:
        if len(dep_graph[node_id]['data_deps']) > 0 and len(node_num_child[node_id]) == 1 and next(iter(node_num_child[node_id])) in nodes_need_merge_gpu:
            nodes_need_replacing.add(node_id)
    
    for node_id in nodes_not_avail:
        if len(dep_graph[node_id]['data_deps']) > 0:
            nodes_need_replacing.add(node_id)

    # 构建dep_replace, 按照无效节点依赖的有效节点设置其dep_replace的内容, 这样在后续程序中能够直接将每个节点所依赖的无效节点替换掉.
    buildDepReplace()

    execution_trace.seek(0)
    decode_message(execution_trace, global_metadata)
    encode_message(cleaned_et, global_metadata)

    print(f'nodes-num:{len(dep_graph)}, not-avail:{len(nodes_not_avail)}, need-merge:{len(nodes_need_merge_cpu), len(nodes_need_merge_gpu)}, need-replace:{len(nodes_need_replacing)}')
    while decode_message(execution_trace, node):
        # 检查节点是否有效，有效节点输出
        if node.id not in nodes_need_replacing:
            del node.ctrl_deps[:]
            del node.data_deps[:]
            my_dep = {'ctrl_deps': set(), 'data_deps': set()}
            for dep in dep_graph[node.id]['ctrl_deps']:
                if dep not in nodes_need_replacing:
                    my_dep['ctrl_deps'].add(dep)
            for dep in dep_graph[node.id]['data_deps']:
                if dep not in nodes_need_replacing:
                    my_dep['data_deps'].add(dep)
            for dep in set(dep_graph[node.id]['ctrl_deps']) | set(dep_graph[node.id]['data_deps']):
                if dep in nodes_need_replacing:
                    my_dep['ctrl_deps'].update(dep_replace[dep]['ctrl_deps'])
                    my_dep['data_deps'].update(dep_replace[dep]['data_deps'])
                    node_addition_runtime[node.id] += node_addition_runtime[dep]

            node.ctrl_deps.extend(my_dep['ctrl_deps'])
            node.data_deps.extend(my_dep['data_deps'])
            node.duration_micros = node_addition_runtime[node.id]
            encode_message(cleaned_et, node)

    execution_trace.close()
    cleaned_et.close()


if __name__ == "__main__":
    main()
