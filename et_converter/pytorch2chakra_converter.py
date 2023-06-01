#!/usr/bin/env python3

import json
import logging

from third_party.utils.protolib import encodeMessage as encode_message
from third_party.utils.eg_replay_utils import (
        is_backward_aten,
        is_backward_parent,
        is_fbgemm_backward,
        skip_op
)
from third_party.utils.execution_graph import (
        ExecutionGraph,
        Node as PyTorchNode,
        NodeType as PyTorchNodeType
)
from et_def.et_def_pb2 import Node as ChakraNode
from et_def.et_def_pb2 import (
        INVALID_NODE,
        COMP_NODE,
        COMM_COLL_NODE,
        INVALID_COMM,
        ALL_REDUCE,
        ALL_TO_ALL,
        ALL_GATHER,
        REDUCE_SCATTER
)

def is_op(node: PyTorchNode, strict: bool = False) -> bool:
    if not strict:
        return node.type == PyTorchNodeType.OPERATOR
    return node.type == PyTorchNodeType.OPERATOR and (
        node.parent is not None and node.parent.type != PyTorchNodeType.OPERATOR
    )

def is_qualified(op: PyTorchNode) -> bool:
    return not skip_op(op) and (
        is_backward_aten(op)
        or is_fbgemm_backward(op)
        or (is_op(op, strict=True) and not is_backward_parent(op))
    )

class PyTorch2ChakraConverter:
    def __init__(
        self,
        input_filename: str,
        output_filename: str,
        default_simulated_run_time: int,
        num_dims: int,
        logger: logging.Logger
    ) -> None:
        with open(input_filename, "r") as f:
            self.exgr = ExecutionGraph(json.load(f))
        self.output_filename = output_filename
        self.default_simulated_run_time = default_simulated_run_time
        self.num_dims = num_dims
        self.logger = logger

        # Nodes/Ops after preprocessing.
        self.sorted_nodes = []

        # Skip the node if their names contain any of the following strings.
        self.skip_node_names = [
            "DataLoader",
            "aten::set_"
        ]

        # This is used to pick out a single iteration when eg trace contains multiple iterations.
        # Basically this label should be captured at the beginning of each iteration so that one iteration
        # is between two consecutive label nodes.
        self.label = ""

        try:
            #pyre-ignore
            from param_bench.train.compute.python.tools.fb.internals import (
                add_internal_label,
                add_internal_parallel_nodes_parents,
                add_internal_skip_nodes,
            )
        except ImportError:
            logger.info("FB internals not present")
        else:
            self.skip_node_names = add_internal_skip_nodes(self.skip_node_names)
            self.label = add_internal_label()

        self.operators_count = [0]

        self.logger.info("skip_node_names=[" + ",".join(self.skip_node_names) + "]")
        self.logger.info(f"label={self.label}")

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
                "Tensor(c10::BFloat16)": 2,
                "Tensor(unsigned char)": 1,
        }
        try:
            data_type_size = data_type_size_dict[data_type]
            return data_type_size
        except:
            raise ValueError(f"{data_type} is unsupported")

    def get_nccl_node(self, node: PyTorchNode) -> PyTorchNode:
        return node.children[0]

    def get_node_type(self, node: PyTorchNode) -> int:
        if node.name == "record_param_comms":
            if len(node.children) == 1:
                return COMM_COLL_NODE
            else:
                return INVALID_NODE

        if node.op_schema or node.outputs:
            return COMP_NODE

        return INVALID_NODE

    def get_comm_type(self, node: PyTorchNode) -> int:
        nccl_node = self.get_nccl_node(node)
        if nccl_node.name == "nccl:all_reduce":
            return ALL_REDUCE
        elif nccl_node.name == "nccl:all_to_all":
            return ALL_TO_ALL
        elif nccl_node.name == "nccl:all_gather":
            return ALL_GATHER
        elif nccl_node.name == "nccl:reduce_scatter":
            return REDUCE_SCATTER
        elif nccl_node.name == "nccl:_reduce_scatter_base":
            return REDUCE_SCATTER
        else:
            raise ValueError(f"{nccl_node.name} is not supported")
        return INVALID_COMM

    def get_comm_size(self, node: PyTorchNode) -> int:
        nccl_node = self.get_nccl_node(node)
        comm_size = 1
        for input_types in nccl_node.input_types:
            comm_size *= self.get_data_type_size(input_types)
        for input_shape_outer in nccl_node.input_shapes:
            for input_shape_inner in input_shape_outer:
                comm_size = comm_size * input_shape_inner
        return comm_size

    def extract_subgraph(self, root: PyTorchNode) -> None:
        def dfs_traverse(root: PyTorchNode) -> None:
            for child in root.children:
                try:
                    if self.label and self.label in child.name:
                        self.sorted_nodes.append(child)

                    if any(x in child.name for x in self.skip_node_names):
                        continue

                    if is_qualified(child):
                        self.sorted_nodes.append(child)
                    else:
                        dfs_traverse(child)
                except Exception as e:
                    raise ValueError(f"Graph parse error: {e}, node id: {child.id}")

        dfs_traverse(root)
        self.sorted_nodes = sorted(self.sorted_nodes, key=lambda x: x.id)
        for i in range(len(self.sorted_nodes)):
            if self.label and self.label in self.sorted_nodes[i].name:
                self.operators_count.append(i)
        if len(self.operators_count) > 1:
            self.sorted_nodes = self.sorted_nodes[
                self.operators_count[1] + 1 : self.operators_count[2]
            ]
        self.logger.info(f"#Operators to execute: {len(self.sorted_nodes)}")

    def convert(self) -> None:
        nodes = self.exgr.get_nodes(clean=True)
        if len(nodes) > 1:
            root = nodes[1]  # 1-base
        else:
            raise ValueError("len(nodes) <= 1")

        self.extract_subgraph(root)

        num_comp_nodes = 0
        num_comm_nodes = 0
        num_invalid_nodes = 0
        output_filename = "%s.eg" % (self.output_filename)
        with open(output_filename, "wb") as g:
            prev_node_id = None
            for pt_node in self.sorted_nodes:
                ck_node = ChakraNode()
                ck_node.id = pt_node.id
                if prev_node_id != None:
                    ck_node.parent.append(prev_node_id)
                prev_node_id = pt_node.id
                ck_node.node_type = self.get_node_type(pt_node)
                if ck_node.node_type == COMP_NODE:
                    ck_node.name = f"COMP_NODE_{pt_node.name}"
                    try:
                        ck_node.simulated_run_time = pt_node.dur
                    except:
                        ck_node.simulated_run_time = self.default_simulated_run_time
                    num_comp_nodes += 1
                elif ck_node.node_type == COMM_COLL_NODE:
                    ck_node.name = f"COMM_COLL_NODE_{pt_node.children[0].name}"
                    ck_node.comm_type = self.get_comm_type(pt_node)
                    ck_node.comm_size = self.get_comm_size(pt_node)
                    for i in range(self.num_dims):
                        ck_node.involved_dim.append(True)
                    num_comm_nodes += 1
                elif ck_node.node_type == INVALID_NODE:
                    ck_node.name = "INVALID_NODE"
                    num_invalid_nodes += 1

                encode_message(g, ck_node)
        self.logger.info(f"Converted {num_comp_nodes} compute nodes, {num_comm_nodes} communication nodes, and {num_invalid_nodes} invalid nodes")
        self.logger.info("All Chakra nodes are written to the file")
