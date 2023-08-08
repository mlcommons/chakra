#!/usr/bin/env python3

import json
import logging
from typing import Any, Dict

from third_party.utils.protolib import encodeMessage as encode_message
from et_def.et_def_pb2 import (
    GlobalMetadata,
    Node as ChakraNode,
    AttributeProto as ChakraAttr,
    INVALID_NODE,
    COMP_NODE,
    COMM_COLL_NODE,
    BOOL,
    FLOAT,
    UINT,
    INT,
    STRING,
    BOOLS,
    FLOATS,
    UINTS,
    INTS,
    STRINGS,
    ALL_REDUCE,
    ALL_TO_ALL,
    ALL_GATHER,
    REDUCE_SCATTER,
    BROADCAST,
)


class PyTorch2ChakraConverter:
    def __init__(
            self,
            input_filename: str,
            output_filename: str,
            num_dims: int,
            logger: logging.Logger
    ) -> None:
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_dims = num_dims
        self.logger = logger

    @staticmethod
    def get_node_type(node: Dict[str, Any]) -> int:
        if "c10d::" in node["name"]:
            return COMM_COLL_NODE
        if node["op_schema"] != "" or node["outputs"]:
            return COMP_NODE
        return INVALID_NODE

    @staticmethod
    def get_attr(
            pt_node: Dict[str, Any],
            attr_name: str,
            attr_type: int
    ) -> ChakraAttr:
        attr = ChakraAttr(name=attr_name, type=attr_type)

        if attr_name in pt_node.keys():
            if attr_type == BOOL:
                attr.b = pt_node[attr_name]
            elif attr_type == FLOAT:
                attr.f = pt_node[attr_name]
            elif attr_type == UINT:
                attr.u = pt_node[attr_name]
            elif attr_type == INT:
                attr.i = pt_node[attr_name]
            elif attr_type == STRING:
                attr.s = pt_node[attr_name]
            elif attr_type == BOOLS:
                attr.bools = pt_node[attr_name]
            elif attr_type == FLOATS:
                attr.floats = pt_node[attr_name]
            elif attr_type == UINTS:
                attr.uints = pt_node[attr_name]
            elif attr_type == INTS:
                attr.ints = pt_node[attr_name]
            elif attr_type == STRINGS:
                attr.strings = pt_node[attr_name]

        return attr

    def detect_type(self, node: Dict[str, Any]) -> str:
        if node["op_schema"] or node["outputs"]:
            return 'operator'
        else:
            return 'label'

    def get_comm_type(self, node: Dict[str, Any]) -> int:
        if node["name"] == "nccl:all_reduce":
            return ALL_REDUCE
        elif node["name"] == "nccl:all_to_all":
            return ALL_TO_ALL
        elif (node["name"] == "nccl:all_gather")\
            or (node["name"] == "nccl:_all_gather_base"):
            return ALL_GATHER
        elif (node["name"] == "nccl:reduce_scatter")\
            or (node["name"] == "nccl:_reduce_scatter_base"):
            return REDUCE_SCATTER
        elif node["name"] == "nccl:broadcast":
            return BROADCAST
        else:
            node_name = node["name"]
            raise ValueError(f"{node_name} is not supported")
        return INVALID_COMM

    # https://pytorch.org/docs/stable/tensors.html
    # https://github.com/pytorch/pytorch/blob/master/c10/util/Half.h
    def get_data_type_size(self, data_type: str) -> int:
        data_type_size_dict = {
                "Tensor(float32)": 4,
                "Tensor(float)": 4,
                "Tensor(float64)": 8,
                "Tensor(double)": 8,
                "Tensor(float16)": 2,
                "Tensor(half)": 2,
                "Tensor(bfloat16)": 2,
                "Tensor(complex64)": 8,
                "Tensor(complex128)": 16,
                "Tensor(uint8)": 1,
                "Tensor(int8)": 1,
                "Tensor(int16)": 2,
                "Tensor(short)": 2,
                "Tensor(int32)": 4,
                "Tensor(int)": 4,
                "Tensor(int64)": 8,
                "Tensor(long)": 8,
                "Tensor(c10::Half)": 2,
                "Tensor(unsigned char)": 1,
                "Tensor(long int)": 8,
        }
        try:
            data_type_size = data_type_size_dict[data_type]
            return data_type_size
        except:
            raise ValueError(f"{data_type} is unsupported")

    def get_comm_size(self, node: Dict[str, Any]) -> int:
        comm_size = 1
        for input_types in node["input_types"]:
            comm_size *= self.get_data_type_size(input_types)
        for input_shape_outer in node["input_shapes"]:
            for input_shape_inner in input_shape_outer:
                comm_size = comm_size * input_shape_inner
        return comm_size

    def dfs(
            self,
            node: Dict[str, Any],
            pytorch_et_data: Dict[str, Any],
            pt_node_dict: Dict[int, Dict[str, Any]]
    ) -> None:
        if self.detect_type(node) == 'operator':
            pt_node_dict[node['id']] = node
        else:
            for pt_node in pytorch_et_data["nodes"]:
                if pt_node['parent'] == node['id']:
                    self.dfs(pt_node, pytorch_et_data, pt_node_dict)

    def convert(self) -> None:
        pt_node_dict = {}
        ck_node_dict = {}
        record_param_comms_pt_node_dict = {}
        nccl_pt_node_dict = {}
        input_storage_id_node_id_dict = {}
        input_tensor_id_node_id_dict = {}
        output_storage_id_node_id_dict = {}
        output_tensor_id_node_id_dict = {}

        with open(self.input_filename, "r") as pytorch_et, \
                open(self.output_filename, "wb") as chakra_et:
            pytorch_et_data = json.load(pytorch_et)

            md = GlobalMetadata(
              attribute=[
                ChakraAttr(name="schema", type=STRING, s=pytorch_et_data["schema"]),
                ChakraAttr(name="pid", type=UINT, u=pytorch_et_data["pid"]),
                ChakraAttr(name="time", type=STRING, s=pytorch_et_data["time"]),
                ChakraAttr(name="start_ts", type=UINT, u=pytorch_et_data["start_ts"]),
                ChakraAttr(name="finish_ts", type=UINT, u=pytorch_et_data["finish_ts"])
              ]
            )
            encode_message(chakra_et, md)

            self.dfs(pytorch_et_data["nodes"][0], pytorch_et_data, pt_node_dict)

            self.logger.info("Identify communication nodes")
            for pt_node in pytorch_et_data["nodes"]:
                if "record_param_comms" in pt_node["name"]:
                    record_param_comms_pt_node_dict.update({pt_node["parent"]: pt_node})
                if "nccl:" in pt_node["name"]:
                    nccl_pt_node_dict.update({pt_node["parent"]: pt_node})

            self.logger.info("Convert PyTorch nodes to Chakra nodes")
            for pt_node_id, pt_node in pt_node_dict.items():
                for i in pt_node["inputs"]:
                    if isinstance(i, list) and len(i) == 6:
                        tensor_id = i[0]
                        storage_id = i[1]
                        if storage_id > 0:
                            input_storage_id_node_id_dict.setdefault(storage_id, []).append(pt_node["id"])
                        else:
                            input_tensor_id_node_id_dict.setdefault(tensor_id, []).append(pt_node["id"])
                for o in pt_node["outputs"]:
                    if isinstance(o, list) and len(o) == 6:
                        tensor_id = o[0]
                        storage_id = o[1]
                        if storage_id > 0:
                            output_storage_id_node_id_dict.setdefault(storage_id, []).append(pt_node["id"])
                        else:
                            output_tensor_id_node_id_dict.setdefault(tensor_id, []).append(pt_node["id"])

                ck_node = ChakraNode()
                ck_node.id = pt_node["id"]
                ck_node.name = pt_node["name"]
                ck_node.type = self.get_node_type(pt_node)
                ck_node.inputs = str(pt_node["inputs"])
                ck_node.input_shapes = str(pt_node["input_shapes"])
                ck_node.input_types = str(pt_node["input_types"])
                ck_node.outputs = str(pt_node["outputs"])
                ck_node.output_shapes = str(pt_node["output_shapes"])
                ck_node.output_types = str(pt_node["output_types"])

                attrs = [("fw_parent", UINT), ("fw_tid", UINT), ("op_schema", STRING),
                        ("parent", UINT), ("seq_id", INT), ("rf_id", UINT), ("scope", UINT), ("tid", UINT)]
                for attr_name, attr_type in attrs:
                    attr = self.get_attr(pt_node, attr_name, attr_type)
                    ck_node.attribute.append(attr)

                # Convert compute nodes
                if ck_node.type == COMP_NODE:
                    attr = ChakraAttr(name="runtime", type=INT)
                    if "dur" in pt_node.keys():
                        attr.i = pt_node["dur"]
                    else:
                        attr.i = 0
                    ck_node.attribute.append(attr)

                # Convert collective communication nodes
                elif ck_node.type == COMM_COLL_NODE:
                    if ck_node.id in record_param_comms_pt_node_dict.keys():
                        record_param_comms_pt_node = record_param_comms_pt_node_dict[ck_node.id]
                        nccl_pt_node = nccl_pt_node_dict[record_param_comms_pt_node["id"]]
                    else:
                        nccl_pt_node = nccl_pt_node_dict[ck_node.id]

                    attr = ChakraAttr(name="comm_type", type=INT)
                    attr.i = self.get_comm_type(nccl_pt_node)
                    ck_node.attribute.append(attr)

                    attr = ChakraAttr(name="comm_size", type=INT)
                    attr.i = self.get_comm_size(nccl_pt_node)
                    ck_node.attribute.append(attr)

                    attr = ChakraAttr(name="involved_dim", type=BOOLS)
                    for _ in range(self.num_dims):
                        attr.bools.append(True)
                    ck_node.attribute.append(attr)

                ck_node_dict[ck_node.id] = ck_node

            self.logger.info("Encode data dependency with storage IDs")
            for input_storage_id, child_node_ids in input_storage_id_node_id_dict.items():
                if input_storage_id in output_storage_id_node_id_dict:
                    parent_node_ids = output_storage_id_node_id_dict[input_storage_id]
                    for child_node_id in child_node_ids:
                        for parent_node_id in parent_node_ids:
                            child_node = ck_node_dict[child_node_id]
                            if (parent_node_id not in child_node.parent)\
                            and child_node.id != parent_node_id:
                                child_node.parent.append(parent_node_id)

                                # remove cycles
                                parent_node = ck_node_dict[parent_node_id]
                                if (parent_node_id in child_node.parent) and\
                                   (child_node_id in parent_node.parent):
                                   if child_node_id < parent_node_id:
                                       child_node.parent.remove(parent_node_id)
                                   else:
                                       parent_node.parent.remove(child_node_id)

            self.logger.info("Encode data dependency with tensor IDs")
            for input_tensor_id, child_node_ids in input_tensor_id_node_id_dict.items():
                if input_tensor_id in output_tensor_id_node_id_dict:
                    parent_node_ids = output_tensor_id_node_id_dict[input_tensor_id]
                    for child_node_id in child_node_ids:
                        for parent_node_id in parent_node_ids:
                            child_node = ck_node_dict[child_node_id]
                            if (parent_node_id not in child_node.parent)\
                            and child_node.id != parent_node_id:
                                child_node.parent.append(parent_node_id)

                                # remove cycles
                                parent_node = ck_node_dict[parent_node_id]
                                if (parent_node_id in child_node.parent) and\
                                   (child_node_id in parent_node.parent):
                                   if child_node_id < parent_node_id:
                                       child_node.parent.remove(parent_node_id)
                                   else:
                                       parent_node.parent.remove(child_node_id)

            self.logger.info("Write Chakra traces")
            for ck_node_id in sorted(ck_node_dict.keys()):
                ck_node = ck_node_dict[ck_node_id]
                encode_message(chakra_et, ck_node)

        self.logger.info("All Chakra nodes are written to the output file")
