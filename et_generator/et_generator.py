#!/usr/bin/env python3

import argparse
from typing import List

from third_party.utils.protolib import encodeMessage as encode_message
from et_def.et_def_pb2 import (
    Node as ChakraNode,
    AttributeProto as ChakraAttr,
    COMP_NODE,
    COMM_COLL_NODE,
    INT,
    INTS,
    ALL_REDUCE,
    ALL_TO_ALL,
    ALL_GATHER,
    REDUCE_SCATTER,
)

NODE_ID = 0


def get_node(node_name: str, node_type: int) -> ChakraNode:
    global NODE_ID
    node = ChakraNode()
    node.id = NODE_ID
    node.name = node_name
    node.type = node_type
    NODE_ID += 1
    return node


def get_runtime_attr(runtime: int) -> ChakraAttr:
    attr = ChakraAttr(name="runtime", type=INT)
    attr.i = runtime
    return attr


def get_comm_type_attr(comm_type: int) -> ChakraAttr:
    attr = ChakraAttr(name="comm_type", type=INT)
    attr.i = comm_type
    return attr


def get_comm_size_attr(comm_size: int) -> ChakraAttr:
    attr = ChakraAttr(name="comm_size", type=INT)
    attr.i = comm_size
    return attr


def get_involved_dim_attr(num_dims: int) -> ChakraAttr:
    attr = ChakraAttr(name="involved_dim", type=INTS)
    for i in range(num_dims):
        attr.ints.append(1)
    return attr


def one_comp_node(num_npus: int, runtime: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comp_node.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("COMP_NODE", COMP_NODE)
            attr = get_runtime_attr(runtime)
            node.attribute.append(attr)
            encode_message(et, node)


def two_comp_nodes_independent(num_npus: int, runtime: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"two_comp_nodes_independent.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("COMP_NODE", COMP_NODE)
            attr = get_runtime_attr(runtime)
            node.attribute.append(attr)
            encode_message(et, node)

            node = get_node("COMP_NODE", COMP_NODE)
            attr = get_runtime_attr(runtime)
            node.attribute.append(attr)
            encode_message(et, node)


def two_comp_nodes_dependent(num_npus: int, runtime: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"two_comp_nodes_dependent.{npu_id}.et"
        with open(output_filename, "wb") as et:
            parent_node = get_node("COMP_NODE", COMP_NODE)
            attr = get_runtime_attr(runtime)
            parent_node.attribute.append(attr)
            encode_message(et, parent_node)

            child_node = get_node("COMP_NODE", COMP_NODE)
            attr = get_runtime_attr(runtime)
            child_node.attribute.append(attr)
            child_node.parent.append(parent_node.id)
            encode_message(et, child_node)


def one_comm_node_allreduce(num_npus: int, num_dims: int, comm_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_node_allreduce.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("ALL_REDUCE", COMM_COLL_NODE)
            attr = get_comm_type_attr(ALL_REDUCE)
            node.attribute.append(attr)
            attr = get_comm_size_attr(comm_size)
            node.attribute.append(attr)
            attr = get_involved_dim_attr(num_dims)
            node.attribute.append(attr)
            encode_message(et, node)


def one_comm_node_alltoall(num_npus: int, num_dims: int, comm_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_node_alltoall.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("ALL_TO_ALL", COMM_COLL_NODE)
            attr = get_comm_type_attr(ALL_TO_ALL)
            node.attribute.append(attr)
            attr = get_comm_size_attr(comm_size)
            node.attribute.append(attr)
            attr = get_involved_dim_attr(num_dims)
            node.attribute.append(attr)
            encode_message(et, node)


def one_comm_node_allgather(num_npus: int, num_dims: int, comm_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_node_allgather.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("ALL_GATHER", COMM_COLL_NODE)
            attr = get_comm_type_attr(ALL_GATHER)
            node.attribute.append(attr)
            attr = get_comm_size_attr(comm_size)
            node.attribute.append(attr)
            attr = get_involved_dim_attr(num_dims)
            node.attribute.append(attr)
            encode_message(et, node)


def one_comm_node_reducescatter(num_npus: int, num_dims: int, comm_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_node_reducescatter.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("REDUCE_SCATTER", COMM_COLL_NODE)
            attr = get_comm_type_attr(REDUCE_SCATTER)
            node.attribute.append(attr)
            attr = get_comm_size_attr(comm_size)
            node.attribute.append(attr)
            attr = get_involved_dim_attr(num_dims)
            node.attribute.append(attr)
            encode_message(et, node)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execution Trace Generator"
    )
    parser.add_argument(
        "--num_npus",
        type=int,
        default=64,
        help="Number of NPUs"
    )
    parser.add_argument(
        "--num_dims",
        type=int,
        default=2,
        help="Number of dimensions in the network topology"
    )
    parser.add_argument(
        "--default_runtime",
        type=int,
        default=5,
        help="Default runtime of compute nodes"
    )
    parser.add_argument(
        "--default_comm_size",
        type=int,
        default=65536,
        help="Default communication size of communication nodes"
    )
    args = parser.parse_args()

    one_comp_node(args.num_npus, args.default_runtime)
    two_comp_nodes_independent(args.num_npus, args.default_runtime)
    two_comp_nodes_dependent(args.num_npus, args.default_runtime)
    one_comm_node_allreduce(args.num_npus, args.num_dims, args.default_comm_size)
    one_comm_node_alltoall(args.num_npus, args.num_dims, args.default_comm_size)
    one_comm_node_allgather(args.num_npus, args.num_dims, args.default_comm_size)
    one_comm_node_reducescatter(args.num_npus, args.num_dims, args.default_comm_size)


if __name__ == "__main__":
    main()
