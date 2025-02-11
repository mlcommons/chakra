import argparse
import os
import sys
import re
import json
from copy import deepcopy
from google.protobuf.json_format import MessageToJson
from .DisjointSetForest import DisjointSetForest

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
from ..pg_extractor.pg_extractor import NetworkCommunicators, Communicator, CommunicatorNode


class Expander:
    """
    Expander 类用于扩展原有的小规模并行的 Chakra execution trace 到大规模并行的环境，支持数据并行（DP）和模型并行（TP）的扩展。

    Attributes:
        input_filepath (str): 输入Chakra execution trace文件的路径。
        output_filepath (str): 输出Chakra execution trace文件的路径。
        pg_descriptors (dict): 描述通信域的字典，包含与通信域相关的设置和配置。
        et_dict (dict): 用于存储 ET 文件信息的字典,初始化为空。key是globalRank:int, value是ChakraNode列表
        process_groups (list): 存储通信域列表
    """
    
    def __init__(self, input_filepath:str, output_filepath:str, pg_descriptors:str):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.pg_descriptors = pg_descriptors
        self.et_dict = {}
        self.process_groups = {}
        self.global_metadata = None
    
    def read_execution_trace_files(self):
        """
        read_execution_trace_files 方法：
            用于读取所有input_filepath目录下的chakra execution trace并将它们组织成chakraNode列表保存到et_dict中
        """
        dir_name = os.path.dirname(self.input_filepath)
        
        # 正则表达式用于匹配et文件
        pattern = re.compile(r'^[a-zA-Z]*(\d+)\.et$')
        for file in os.listdir(dir_name):
            nodeslist = []
            match = pattern.match(file)
            if match:
                et_file_path = os.path.join(self.input_filepath, file)
                execution_trace = open_file_rd(et_file_path)
                node = ChakraNode()

                global_metadata = GlobalMetadata()
                decode_message(execution_trace, global_metadata)
                if self.global_metadata == None:
                    self.global_metadata = global_metadata
                while decode_message(execution_trace, node):
                    nodeslist.append(node)

                globalRank = int(match.group(1))
                self.et_dict[globalRank] = nodeslist

                execution_trace.close()
                
    def config_pg_descriptors(self):
        """
        config_pg_descriptors 方法：
            用于读取所有pg_descriptor中的通信域描述配置,在后续并行扩展时要用到这些通信域相关信息
        """
        with open(self.pg_descriptors, "r") as pg_descriptors:
            pg_descriptors = json.load(pg_descriptors)
        for pg_descriptor in pg_descriptors['communicators']:
            pg_name = pg_descriptor['communicatorId']
            pg_type = pg_descriptor['type']
            pg = Communicator(pg_name, pg_type)
            for node in pg_descriptor['nodes']:
                global_rank = node['globalRank']
                rank = node['rank']
                commList = node['commList']
                communicator_node = CommunicatorNode(global_rank, rank, commList)
                pg.addNode(communicator_node)
            
            self.process_groups[pg_name] = pg


    def tensor_parallel_expand(self,factor:int):
        #根据globalRank寻找对应的DP域
        def find_DP_pg_name_for_rank(process_groups, globalRank):
            for pg_name, process_group in process_groups:
                if process_group.type == 'DP':
                    for node in process_group.nodes():
                        if getattr(node, 'globalRank') == globalRank:
                            return pg_name

        for globalRank, nodelist in self.et_dict.items():
            et_outfile = open_file_wt(f'{self.output_filepath}/{globalRank}.et')
            encode_message(et_outfile, self.global_metadata)
            for chakraNode in nodelist:
                if chakraNode.type == 'COMM_COLL_NODE':
                    for attr in chakraNode.attr:
                        if attr.name == 'pg_name':
                            pg_name = attr.string_val
                        if attr.name == 'is_cpu_op':
                            is_cpu_op = attr.bool_val
                    if self.process_groups[pg_name].type == 'TP' and (not is_cpu_op):
                        for attr in chakraNode.attr:
                            if attr.name == 'comm_size':
                                attr.int64_val /= factor
                elif chakraNode.type == 'COMP_NODE' :
                    for attr in chakraNode.attr:
                        if attr.name == 'num_ops':
                            attr.int64_val /= factor
                encode_message(et_outfile, chakraNode)

        comm_world_size = len(self.et_dict)

        # dsf = DisjointSetForest(8) 
        dsf = DisjointSetForest(comm_world_size * factor) 

        #每个globalRank复制factor-1份
        for i in range(factor - 1):
            #创建新的DP域
            pg_size = len(self.process_groups)
            for pg_name, process_group in self.process_groups.copy().items():
                if process_group.type == 'DP':
                    nodes = process_group.nodes
                    new_pg = Communicator(pg_size, 'DP')
                    for node in nodes:
                        ori_rank = getattr(node, 'globalRank')
                        new_rank = ori_rank + comm_world_size * (i + 1)
                        new_node = CommunicatorNode(new_rank, node.rank, node.commList)
                        print(f'{ori_rank}:{new_rank}')
                        dsf.union(ori_rank, new_rank)
                        new_pg.addNode(new_node)
                    self.process_groups[pg_size] = new_pg
                    pg_size += 1

            for fakeRank, nodelist in self.et_dict.items():
                realRank = fakeRank + (i + 1) * comm_world_size
                et_outfile = open_file_wt(f'{self.output_filepath}/{realRank}.et')
                encode_message(et_outfile, self.global_metadata)
                for chakraNode in nodelist:
                    if chakraNode.type == 'COMM_COLL_NODE' :
                        for attr in chakraNode.attr:
                            if attr.name == 'pg_name':
                                pg_name = attr.string_val
                                if self.process_groups[pg_name].type == 'DP':
                                    attr.string_val = find_DP_pg_name_for_rank(self.process_groups, realRank)
                    encode_message(et_outfile, chakraNode)
                    
        #更新TP域
        for pg_name, process_group in self.process_groups.items():
            if process_group.type == 'TP':
                node0_globalRank = process_group.nodes[0].globalRank
                tmprank = len(process_group.nodes)
                node0_commList = process_group.nodes[0].commList
                root = dsf.find(node0_globalRank)
                for i in range(comm_world_size * factor):
                    if dsf.find(i) == root and not self.isinclude(process_group.nodes, i):
                        node = CommunicatorNode(i, tmprank, node0_commList)
                        process_group.addNode(node)
                        tmprank += 1
                        
        #重写通信域描述
        writeout = {'communicators': [process_group.to_dict() for process_group in self.process_groups.values()] }
        with open(self.pg_descriptors, 'w') as f:
            f.write(json.dumps(writeout, indent=4))


    def data_parallel_expand(self, factor:int):
        #根据globalRank寻找对应的TP域
        def find_TP_pg_name_for_rank(process_groups, globalRank):
            for pg_name, process_group in process_groups:
                if process_group.type == 'TP':
                    for node in process_group.nodes():
                        if getattr(node, 'globalRank') == globalRank:
                            return pg_name

        for globalRank, nodelist in self.et_dict.items():
            et_outfile = open_file_wt(f'{self.output_filepath}/{globalRank}.et')
            encode_message(et_outfile, self.global_metadata)
            for chakraNode in nodelist:
                if chakraNode.type == 'COMM_COLL_NODE' :
                    for attr in chakraNode.attr:
                        if attr.name == 'pg_name':
                            pg_name = attr.string_val
                        if attr.name == 'is_cpu_op':
                            is_cpu_op = attr.bool_val
                    if self.process_groups[pg_name].type == 'DP' and not is_cpu_op:
                        for attr in chakraNode.attr:
                            if attr.name == 'comm_size':
                                attr.int64_val /= factor
                elif chakraNode.type == 'COMP_NODE' :
                    for attr in chakraNode.attr:
                        if attr.name == 'num_ops':
                            attr.int64_val /= factor
                encode_message(et_outfile, chakraNode)

        comm_world_size = len(self.et_dict)

        # dsf = DisjointSetForest(8) 
        dsf = DisjointSetForest(comm_world_size * factor) 

        #每个globalRank复制factor-1份
        for i in range(factor - 1):
            #创建新的DP域
            pg_size = len(self.process_groups)
            for pg_name, process_group in self.process_groups.copy().items():
                if process_group.type == 'TP':
                    nodes = process_group.nodes
                    new_pg = Communicator(pg_size, 'TP')
                    for node in nodes:
                        ori_rank = getattr(node, 'globalRank')
                        new_rank = ori_rank + comm_world_size * (i + 1)
                        new_node = CommunicatorNode(new_rank, node.rank, node.commList)
                        print(f'{ori_rank}:{new_rank}')
                        dsf.union(ori_rank, new_rank)
                        new_pg.addNode(new_node)
                    self.process_groups[pg_size] = new_pg
                    pg_size += 1

            for fakeRank, nodelist in self.et_dict.items():
                realRank = fakeRank + (i + 1) * comm_world_size
                et_outfile = open_file_wt(f'{self.output_filepath}/{realRank}.et')
                encode_message(et_outfile, self.global_metadata)
                for chakraNode in nodelist:
                    if chakraNode.type == 'COMM_COLL_NODE' :
                        for attr in chakraNode.attr:
                            if attr.name == 'pg_name':
                                pg_name = attr.string_val
                                if self.process_groups[pg_name].type == 'TP':
                                    attr.string_val = find_TP_pg_name_for_rank(self.process_groups, realRank)
                    encode_message(et_outfile,chakraNode)
                    
        #更新DP域
        for pg_name, process_group in self.process_groups.items():
            if process_group.type == 'DP':
                node0_globalRank = process_group.nodes[0].globalRank
                tmprank = len(process_group.nodes)
                node0_commList = process_group.nodes[0].commList
                root = dsf.find(node0_globalRank)
                for i in range(comm_world_size * factor):
                    if dsf.find(i) == root and not self.isinclude(process_group.nodes, i):
                        node = CommunicatorNode(i, tmprank, node0_commList)
                        process_group.addNode(node)
                        tmprank += 1
                        
        #重写通信域描述
        writeout = {'communicators': [process_group.to_dict() for process_group in self.process_groups.values()] }
        with open(self.pg_descriptors, 'w') as f:
            f.write(json.dumps(writeout, indent=4))

    def isinclude(nodes, id) -> bool:
        for node in nodes:
            if node.globalRank == id:
                return True
    

def main() -> None:
    parser = argparse.ArgumentParser(description="Read Chakra execution trace and extend parallelism.")
    parser.add_argument(
        "--input_filepath", type=str, required=True, help="Specifies the directory of the Chakra execution trace."
    )
    parser.add_argument(
        "--output_filepath", type=str, required=True, help="Specify the directory for the generated Chakra execution trace files after expansion."
    )
    parser.add_argument(
        "--pg_descriptors", type=str, required=True, help="Specify the process group descriptor file."
    )
    parser.add_argument(
        "--data_parallel", type=int,  help="Specify the factor for data parallel expansion."
    )
    parser.add_argument(
        "--tensor_parallel", type=int,  help="Specify the factor for tensor parallel expansion."
    )
    # parser.add_argument(
    #     "--jobs", type=str, help="Specify the number of cores for this script to execute."
    # )
    args = parser.parse_args()

    # if not os.path.exists(args.input_filepath):
    #     print(f"The specified path does not exist: {args.input_filepath}")
    #     return
    if not os.path.exists(args.output_filepath):
        os.makedirs(args.output_filepath)
    if not args.pg_descriptors:
        print(f"Please specify the process group descriptor file using --pg_descriptors filename.")
        sys.exit(1)
    if not os.path.exists(args.pg_descriptors):
        print(f"Pg_descriptors file does not exist.")
        sys.exit(1)
    expander = Expander(args.input_filepath, args.output_filepath, args.pg_descriptors)
    expander.read_execution_trace_files()
    expander.config_pg_descriptors()

    if args.tensor_parallel:
        # print(args.tensor_parallel)
        expander.tensor_parallel_expand(args.tensor_parallel)
    
    if args.data_parallel:
        expander.data_parallel_expand(args.data_parallel)


if __name__ == "__main__":
    main()
