#!/usr/bin/env python3

import copy
import logging
import pydot

from typing import Any
from third_party.utils.protolib import encodeMessage as encode_message
from et_def.et_def_pb2 import Node as ChakraNode
from et_def.et_def_pb2 import (
    COMP_NODE,
    COMM_SEND_NODE,
    COMM_RECV_NODE
)

class FlexFlow2ChakraConverter:
    def __init__(
        self,
        input_filename: str,
        output_filename: str,
        num_dims: int,
        npu_frequency: int,
        logger: logging.Logger
    ) -> None:
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_dims = num_dims
        self.num_cycles_per_sec = npu_frequency * 1_000 * 1_000
        self.logger = logger
        self.node_id_npu_id_dict = {}
        self.node_id_node_dict = {}

    def get_label(self, ff_node: Any) -> str:
        try:
            label = ff_node.get_attributes()["label"]
            return label.replace("\"", "")[1:-1]
        except:
            raise ValueError(f"Cannot retrieve label from a FlexFlow node")

    def get_id(self, ff_node: Any) -> int:
        ff_node_name = ff_node.get_name()
        try:
            return int(ff_node_name.replace("node", ""))
        except:
            raise ValueError(f"Cannot retrieve id from \"{ff_node_name}\"")

    def get_npu_id(self, ff_node: Any) -> int:
        label = self.get_label(ff_node)
        try:
            return int(label.split("|")[0].strip().split("=")[1])
        except:
            raise ValueError(f"Cannot retrieve npu_id from \"{label}\"")

    def get_name(self, ff_node: Any) -> str:
        label = self.get_label(ff_node)
        try:
            return label.split("|")[1].strip()
        except:
            raise ValueError(f"Cannot retrieve name from \"{label}\"")

    def get_node_type(self, ff_node: Any) -> int:
        label = self.get_label(ff_node)
        try:
            node_type = label.split("|")[3].strip()
            if node_type == "COMP_NODE":
                return COMP_NODE
            elif node_type == "COMM_SEND_RECV_NODE":
                return COMM_SEND_NODE
            else:
                raise ValueError(f"Unsupported node_type, \"{node_type}\"")
        except:
            raise ValueError(f"Cannot retrieve node_type from \"{label}\"")

    def get_simulated_run_time(self, ff_node: Any) -> int:
        label = self.get_label(ff_node)
        try:
            wall_clock_time = float(label.split("|")[4].strip().split("=")[1])
            return int(round(wall_clock_time * self.num_cycles_per_sec))
        except:
            raise ValueError(f"Cannot retrieve simulated_run_time from \"{label}\"")

    def get_comm_src(self, ff_node: Any) -> int:
        label = self.get_label(ff_node)
        try:
            return int(label.split("|")[4].strip().split("=")[1])
        except:
            raise ValueError(f"Cannot retrieve comm_src from \"{label}\"")

    def get_comm_dst(self, ff_node: Any) -> int:
        label = self.get_label(ff_node)
        try:
            return int(label.split("|")[5].strip().split("=")[1])
        except:
            raise ValueError(f"Cannot retrieve comm_dst from \"{label}\"")

    def get_comm_size(self, ff_node: Any) -> int:
        label = self.get_label(ff_node)
        try:
            return int(label.split("|")[6].strip().split("=")[1])
        except:
            raise ValueError(f"Cannot retrieve comm_size from \"{label}\"")

    def convert_FF_node_to_CK_node(self, ff_node: Any) -> Any:
        ck_node = ChakraNode()
        ck_node.id = self.get_id(ff_node)
        ck_node.name = self.get_name(ff_node)
        ck_node.node_type = self.get_node_type(ff_node)
        if ck_node.node_type == COMP_NODE:
            ck_node.simulated_run_time = self.get_simulated_run_time(ff_node)
        elif ck_node.node_type == COMM_SEND_NODE:
            ck_node.comm_src = self.get_comm_src(ff_node)
            ck_node.comm_dst = self.get_comm_dst(ff_node)
            ck_node.comm_size = self.get_comm_size(ff_node)
        self.node_id_npu_id_dict.update({ck_node.id: self.get_npu_id(ff_node)})
        return ck_node

    def convert(self) -> None:
        ff_graphs = pydot.graph_from_dot_file(self.input_filename)
        ff_graph = ff_graphs[0]
        if len(ff_graphs) != 1:
            raise ValueError("The input file has more than one FlexFlow graphs")

        # convert FlexFlow EG to Chakra EG
        npu_ids = set()
        num_ff_nodes = 0
        num_ff_edges = 0
        for ff_node in ff_graph.get_nodes():
            ck_node = self.convert_FF_node_to_CK_node(ff_node)
            self.node_id_node_dict.update({ck_node.id: ck_node})
            if ck_node.node_type == COMP_NODE:
                npu_ids.add(self.node_id_npu_id_dict[ck_node.id])
            num_ff_nodes += 1
        for edge in ff_graph.get_edges():
            src_id = int(edge.get_source().replace("node", ""))
            dst_id = int(edge.get_destination().replace("node", ""))
            ck_node = self.node_id_node_dict[dst_id]
            ck_node.parent.append(src_id)
            num_ff_edges += 1
        self.logger.info(f"Converted {num_ff_nodes} nodes and {num_ff_edges} edges")

        # generate per-NPU Chakra graphs
        next_comm_tag = 0
        npu_id_node_id_node_dict = {}
        comm_key_comm_tag_dict = {}
        total_comp_nodes = 0
        total_comm_nodes = 0
        for npu_id in npu_ids:
            npu_id_node_id_node_dict.update({npu_id: {}})
            per_npu_comp_nodes = 0
            per_npu_comm_nodes = 0
            for node_id in self.node_id_node_dict.keys():
                ck_node = copy.deepcopy(self.node_id_node_dict[node_id])

                # compute nodes
                if ck_node.node_type == COMP_NODE:
                    ck_node.name = f"COMP_NODE_{ck_node.name}"
                    if self.node_id_npu_id_dict[ck_node.id] == npu_id:
                        npu_id_node_id_node_dict[npu_id].update({node_id: ck_node})
                        per_npu_comp_nodes += 1
                        total_comp_nodes += 1

                # communication nodes
                elif (ck_node.node_type == COMM_SEND_NODE):
                    if (ck_node.comm_src == npu_id) or (ck_node.comm_dst == npu_id):
                        comm_key = f"{ck_node.id}_{ck_node.comm_src}_{ck_node.comm_dst}"
                        if comm_key not in comm_key_comm_tag_dict.keys():
                            comm_tag = next_comm_tag
                            comm_key_comm_tag_dict.update({comm_key: comm_tag})
                            next_comm_tag += 1
                        else:
                            comm_tag = comm_key_comm_tag_dict[comm_key]

                        # create a new communication node
                        ck_comm_node = ChakraNode()
                        ck_comm_node.id = ck_node.id
                        if ck_node.comm_src == npu_id:
                            ck_comm_node.name = "COMM_SEND_NODE"
                            ck_comm_node.node_type = COMM_SEND_NODE
                        elif ck_node.comm_dst == npu_id:
                            ck_comm_node.name = "COMM_RECV_NODE"
                            ck_comm_node.node_type = COMM_RECV_NODE
                        ck_comm_node.name += f"_{ck_node.name}"
                        ck_comm_node.comm_src = ck_node.comm_src
                        ck_comm_node.comm_dst = ck_node.comm_dst
                        ck_comm_node.comm_size = ck_node.comm_size
                        ck_comm_node.comm_tag = comm_tag
                        per_npu_comm_nodes += 1
                        total_comm_nodes += 1

                        # transfer dependencies
                        for parent_node_id in ck_node.parent:
                            parent_node = self.node_id_node_dict[parent_node_id]
                            if self.node_id_npu_id_dict[parent_node.id] == npu_id:
                                ck_comm_node.parent.append(parent_node_id)

                        npu_id_node_id_node_dict[npu_id].update({node_id: ck_comm_node})
            self.logger.info(f"NPU[{npu_id}]: {per_npu_comp_nodes} compute nodes and {per_npu_comm_nodes} communication nodes")
        self.logger.info(f"Total: {total_comp_nodes} compute nodes and {total_comm_nodes} communication nodes")

        # write per-NPU Chakra graphs
        for npu_id in sorted(npu_id_node_id_node_dict.keys()):
            filename = self.output_filename + f".{npu_id}.eg"
            with open(filename, "wb") as f:
                for node_id in sorted(npu_id_node_id_node_dict[npu_id].keys()):
                    ck_node = npu_id_node_id_node_dict[npu_id][node_id]
                    encode_message(f, ck_node)
        self.logger.info("All Chakra EGs are written to files")
