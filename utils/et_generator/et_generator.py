#!/usr/bin/env python3

import argparse

from chakra.third_party.utils.protolib import encodeMessage as encode_message
from chakra.et_def.et_def_pb2 import (
    Node as ChakraNode,
    AttributeProto as ChakraAttr,
    INVALID_NODE,
    METADATA_NODE,
    MEM_LOAD_NODE,
    MEM_STORE_NODE,
    COMP_NODE,
    COMM_SEND_NODE,
    COMM_RECV_NODE,
    COMM_COLL_NODE,
    BOOL,
    FLOAT,
    INT,
    STRING,
    BOOLS,
    FLOATS,
    INTS,
    STRINGS,
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


def get_attr(name: str, attr_type: int) -> ChakraAttr:
    attr = ChakraAttr(name=name, type=attr_type)
    return attr


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
    attr = ChakraAttr(name="involved_dim", type=BOOLS)
    for i in range(num_dims):
        attr.bools.append(True)
    return attr


def one_metadata_node_bool(num_npus: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_metadata_node_bool.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("METADATA_NODE", METADATA_NODE)
            attr = get_attr("bool", BOOL)
            attr.b = True
            attr.doc_string = "bool"
            node.attribute.append(attr)
            encode_message(et, node)


def one_metadata_node_float(num_npus: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_metadata_node_float.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("METADATA_NODE", METADATA_NODE)
            attr = get_attr("float", FLOAT)
            attr.f = 1.2345
            attr.doc_string = "float"
            node.attribute.append(attr)
            encode_message(et, node)


def one_metadata_node_int(num_npus: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_metadata_node_int.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("METADATA_NODE", METADATA_NODE)
            attr = get_attr("int", INT)
            attr.i = 12345
            attr.doc_string = "int"
            node.attribute.append(attr)
            encode_message(et, node)


def one_metadata_node_string(num_npus: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_metadata_node_string.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("METADATA_NODE", METADATA_NODE)
            attr = get_attr("string", INT)
            attr.s = "12345"
            attr.doc_string = "string"
            node.attribute.append(attr)
            encode_message(et, node)


def one_metadata_node_bools(num_npus: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_metadata_node_bools.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("METADATA_NODE", METADATA_NODE)
            attr = get_attr("bools", BOOLS)
            for i in range(10):
                if i % 2 == 0:
                    attr.bools.append(True)
                else:
                    attr.bools.append(False)
            attr.doc_string = "bools"
            node.attribute.append(attr)
            encode_message(et, node)


def one_metadata_node_floats(num_npus: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_metadata_node_floats.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("METADATA_NODE", METADATA_NODE)
            attr = get_attr("floats", FLOATS)
            val = 1.2345
            for i in range(10):
                attr.floats.append(val)
                val += 0.0001
            attr.doc_string = "floats"
            node.attribute.append(attr)
            encode_message(et, node)


def one_metadata_node_ints(num_npus: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_metadata_node_ints.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("METADATA_NODE", METADATA_NODE)
            attr = get_attr("ints", INTS)
            val = 12345
            for i in range(10):
                attr.ints.append(val)
                val += 1
            attr.doc_string = "ints"
            node.attribute.append(attr)
            encode_message(et, node)


def one_metadata_node_strings(num_npus: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_metadata_node_strings.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("METADATA_NODE", METADATA_NODE)
            attr = get_attr("strings", STRINGS)
            val = 12345
            for i in range(10):
                attr.strings.append(str(val))
                val += 1
            attr.doc_string = "strings"
            node.attribute.append(attr)
            encode_message(et, node)


def one_mem_load_node(num_npus: int, tensor_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_mem_load_node.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("MEM_LOAD_NODE", MEM_LOAD_NODE)
            attr = get_attr("tensor_size", INT)
            attr.i = tensor_size
            node.attribute.append(attr)
            encode_message(et, node)


def one_mem_store_node(num_npus: int, tensor_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_mem_store_node.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("MEM_STORE_NODE", MEM_STORE_NODE)
            attr = get_attr("tensor_size", INT)
            attr.i = tensor_size
            node.attribute.append(attr)
            encode_message(et, node)


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


def one_comm_send_node(num_npus: int, comm_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_send_node.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("COMM_SEND_NODE", COMM_SEND_NODE)
            attr = get_attr("src", INT)
            attr.i = npu_id
            node.attribute.append(attr)
            attr = get_attr("dst", INT)
            if npu_id == (num_npus - 1):
                attr.i = 0
            else:
                attr.i = npu_id + 1
            node.attribute.append(attr)
            attr = get_comm_size_attr(comm_size)
            node.attribute.append(attr)
            encode_message(et, node)


def one_comm_recv_node(num_npus: int, comm_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_recv_node.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("COMM_RECV_NODE", COMM_RECV_NODE)
            attr = get_attr("src", INT)
            if npu_id == 0:
                attr.i = num_npus - 1
            else:
                attr.i = npu_id - 1
            node.attribute.append(attr)
            attr = get_attr("dst", INT)
            attr.i = npu_id
            node.attribute.append(attr)
            attr = get_comm_size_attr(comm_size)
            node.attribute.append(attr)
            encode_message(et, node)


def one_comm_coll_node_allreduce(num_npus: int, num_dims: int, comm_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_coll_node_allreduce.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("ALL_REDUCE", COMM_COLL_NODE)
            attr = get_comm_type_attr(ALL_REDUCE)
            node.attribute.append(attr)
            attr = get_comm_size_attr(comm_size)
            node.attribute.append(attr)
            attr = get_involved_dim_attr(num_dims)
            node.attribute.append(attr)
            encode_message(et, node)


def one_comm_coll_node_alltoall(num_npus: int, num_dims: int, comm_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_coll_node_alltoall.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("ALL_TO_ALL", COMM_COLL_NODE)
            attr = get_comm_type_attr(ALL_TO_ALL)
            node.attribute.append(attr)
            attr = get_comm_size_attr(comm_size)
            node.attribute.append(attr)
            attr = get_involved_dim_attr(num_dims)
            node.attribute.append(attr)
            encode_message(et, node)


def one_comm_coll_node_allgather(num_npus: int, num_dims: int, comm_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_coll_node_allgather.{npu_id}.et"
        with open(output_filename, "wb") as et:
            node = get_node("ALL_GATHER", COMM_COLL_NODE)
            attr = get_comm_type_attr(ALL_GATHER)
            node.attribute.append(attr)
            attr = get_comm_size_attr(comm_size)
            node.attribute.append(attr)
            attr = get_involved_dim_attr(num_dims)
            node.attribute.append(attr)
            encode_message(et, node)


def one_comm_coll_node_reducescatter(num_npus: int, num_dims: int, comm_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_coll_node_reducescatter.{npu_id}.et"
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
        "--default_tensor_size",
        type=int,
        default=1024,
        help="Default tensor size of memory nodes"
    )
    parser.add_argument(
        "--default_comm_size",
        type=int,
        default=65536,
        help="Default communication size of communication nodes"
    )
    args = parser.parse_args()

    one_metadata_node_bool(args.num_npus)
    one_metadata_node_float(args.num_npus)
    one_metadata_node_int(args.num_npus)
    one_metadata_node_string(args.num_npus)
    one_metadata_node_bools(args.num_npus)
    one_metadata_node_floats(args.num_npus)
    one_metadata_node_ints(args.num_npus)
    one_metadata_node_strings(args.num_npus)

    one_mem_load_node(args.num_npus, args.default_tensor_size)
    one_mem_store_node(args.num_npus, args.default_tensor_size)

    one_comp_node(args.num_npus, args.default_runtime)
    two_comp_nodes_independent(args.num_npus, args.default_runtime)
    two_comp_nodes_dependent(args.num_npus, args.default_runtime)

    one_comm_send_node(args.num_npus, args.default_comm_size)
    one_comm_recv_node(args.num_npus, args.default_comm_size)

    one_comm_coll_node_allreduce(args.num_npus, args.num_dims, args.default_comm_size)
    one_comm_coll_node_alltoall(args.num_npus, args.num_dims, args.default_comm_size)
    one_comm_coll_node_allgather(args.num_npus, args.num_dims, args.default_comm_size)
    one_comm_coll_node_reducescatter(args.num_npus, args.num_dims, args.default_comm_size)


if __name__ == "__main__":
    main()
