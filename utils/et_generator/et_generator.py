#!/usr/bin/env python3

import argparse

from chakra.third_party.utils.protolib import encodeMessage as encode_message
from chakra.et_def.et_def_pb2 import (
    Node as ChakraNode,
    DoubleList,
    FloatList,
    Int32List,
    Int64List,
    Uint32List,
    Uint64List,
    Sint32List,
    Sint64List,
    Fixed32List,
    Fixed64List,
    Sfixed32List,
    Sfixed64List,
    BoolList,
    StringList,
    BytesList,
    GlobalMetadata,
    AttributeProto as ChakraAttr,
    METADATA_NODE,
    MEM_LOAD_NODE,
    MEM_STORE_NODE,
    COMP_NODE,
    COMM_COLL_NODE,
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


def get_comm_type_attr(comm_type: int) -> ChakraAttr:
    return ChakraAttr(name="comm_type", int64_val=comm_type)


def get_involved_dim_attr(num_dims: int) -> ChakraAttr:
    return ChakraAttr(name="involved_dim", bool_list=BoolList(values=[True] * num_dims))


def one_metadata_node_all_types(num_npus: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_metadata_node_all_types.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("METADATA_NODE", METADATA_NODE)

            node.attr.append(
                ChakraAttr(name="double", double_val=1.2345, doc_string="double")
            )
            double_list = DoubleList(values=[1.2345, 2.3456])
            node.attr.append(ChakraAttr(name="double_list", double_list=double_list))

            node.attr.append(
                ChakraAttr(name="float", float_val=1.2345, doc_string="float")
            )
            float_list = FloatList(values=[1.2345, 2.3456])
            node.attr.append(ChakraAttr(name="float_list", float_list=float_list))

            node.attr.append(
                ChakraAttr(name="int32", int32_val=12345, doc_string="int32")
            )
            int32_list = Int32List(values=[12345, 23456])
            node.attr.append(ChakraAttr(name="int32_list", int32_list=int32_list))

            node.attr.append(
                ChakraAttr(name="int64", int64_val=9876543210, doc_string="int64")
            )
            int64_list = Int64List(values=[9876543210, 1234567890])
            node.attr.append(ChakraAttr(name="int64_list", int64_list=int64_list))

            node.attr.append(
                ChakraAttr(name="uint32", uint32_val=12345, doc_string="uint32")
            )
            uint32_list = Uint32List(values=[12345, 23456])
            node.attr.append(ChakraAttr(name="uint32_list", uint32_list=uint32_list))

            node.attr.append(
                ChakraAttr(name="uint64", uint64_val=9876543210, doc_string="uint64")
            )
            uint64_list = Uint64List(values=[9876543210, 1234567890])
            node.attr.append(ChakraAttr(name="uint64_list", uint64_list=uint64_list))

            node.attr.append(
                ChakraAttr(name="sint32", sint32_val=-12345, doc_string="sint32")
            )
            sint32_list = Sint32List(values=[12345, -23456])
            node.attr.append(ChakraAttr(name="sint32_list", sint32_list=sint32_list))

            node.attr.append(
                ChakraAttr(name="sint64", sint64_val=-9876543210, doc_string="sint64")
            )
            sint64_list = Sint64List(values=[9876543210, -1234567890])
            node.attr.append(ChakraAttr(name="sint64_list", sint64_list=sint64_list))

            node.attr.append(ChakraAttr(name="fixed32", fixed32_val=12345))
            fixed32_list = Fixed32List(values=[12345, 23456])
            node.attr.append(ChakraAttr(name="fixed32_list", fixed32_list=fixed32_list))

            node.attr.append(ChakraAttr(name="fixed64", fixed64_val=9876543210))
            fixed64_list = Fixed64List(values=[9876543210, 1234567890])
            node.attr.append(ChakraAttr(name="fixed64_list", fixed64_list=fixed64_list))

            node.attr.append(ChakraAttr(name="sfixed32", sfixed32_val=-12345))
            sfixed32_list = Sfixed32List(values=[12345, -23456])
            node.attr.append(
                ChakraAttr(name="sfixed32_list", sfixed32_list=sfixed32_list)
            )

            node.attr.append(ChakraAttr(name="sfixed64", sfixed64_val=-9876543210))
            sfixed64_list = Sfixed64List(values=[9876543210, -1234567890])
            node.attr.append(
                ChakraAttr(name="sfixed64_list", sfixed64_list=sfixed64_list)
            )

            node.attr.append(ChakraAttr(name="bool", bool_val=True, doc_string="bool"))
            bool_list = BoolList(values=[i % 2 == 0 for i in range(10)])
            node.attr.append(ChakraAttr(name="bool_list", bool_list=bool_list))

            node.attr.append(
                ChakraAttr(name="string", string_val="12345", doc_string="string")
            )
            string_list = StringList(values=[str(12345 + i) for i in range(10)])
            node.attr.append(ChakraAttr(name="string_list", string_list=string_list))

            node.attr.append(
                ChakraAttr(name="bytes", bytes_val=bytes("12345", "utf-8"))
            )
            bytes_list = BytesList(
                values=[bytes(str(12345 + i), "utf-8") for i in range(10)]
            )
            node.attr.append(ChakraAttr(name="bytes_list", bytes_list=bytes_list))

            encode_message(et, node)


def one_remote_mem_load_node(num_npus: int, tensor_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_remote_mem_load_node.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("MEM_LOAD_NODE", MEM_LOAD_NODE)
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
            encode_message(et, node)


def one_remote_mem_store_node(num_npus: int, tensor_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_remote_mem_store_node.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("MEM_STORE_NODE", MEM_STORE_NODE)
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
            encode_message(et, node)


def one_comp_node(num_npus: int, runtime: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comp_node.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("COMP_NODE", COMP_NODE)
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.duration_micros = runtime
            encode_message(et, node)


def two_comp_nodes_independent(num_npus: int, runtime: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"two_comp_nodes_independent.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("COMP_NODE", COMP_NODE)
            node.duration_micros = runtime
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            encode_message(et, node)

            node = get_node("COMP_NODE", COMP_NODE)
            node.duration_micros = runtime
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            encode_message(et, node)


def two_comp_nodes_dependent(num_npus: int, runtime: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"two_comp_nodes_dependent.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            parent_node = get_node("COMP_NODE", COMP_NODE)
            parent_node.duration_micros = runtime
            parent_node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            encode_message(et, parent_node)

            child_node = get_node("COMP_NODE", COMP_NODE)
            child_node.duration_micros = runtime
            child_node.data_deps.append(parent_node.id)
            child_node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            encode_message(et, child_node)


def one_comm_coll_node_allreduce(num_npus: int, num_dims: int, comm_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_coll_node_allreduce.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("ALL_REDUCE", COMM_COLL_NODE)
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.append(get_comm_type_attr(ALL_REDUCE))
            node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))
            attr = get_involved_dim_attr(num_dims)
            node.attr.append(attr)
            encode_message(et, node)


def one_comm_coll_node_alltoall(num_npus: int, num_dims: int, comm_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_coll_node_alltoall.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("ALL_TO_ALL", COMM_COLL_NODE)
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.append(get_comm_type_attr(ALL_TO_ALL))
            node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))
            attr = get_involved_dim_attr(num_dims)
            node.attr.append(attr)
            encode_message(et, node)


def one_comm_coll_node_allgather(num_npus: int, num_dims: int, comm_size: int) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_coll_node_allgather.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("ALL_GATHER", COMM_COLL_NODE)
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.append(get_comm_type_attr(ALL_GATHER))
            node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))
            attr = get_involved_dim_attr(num_dims)
            node.attr.append(attr)
            encode_message(et, node)


def one_comm_coll_node_reducescatter(
    num_npus: int, num_dims: int, comm_size: int
) -> None:
    for npu_id in range(num_npus):
        output_filename = f"one_comm_coll_node_reducescatter.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("REDUCE_SCATTER", COMM_COLL_NODE)
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.append(get_comm_type_attr(REDUCE_SCATTER))
            node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))
            attr = get_involved_dim_attr(num_dims)
            node.attr.append(attr)
            encode_message(et, node)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execution Trace Generator")
    parser.add_argument("--num_npus", type=int, default=64, help="Number of NPUs")
    parser.add_argument(
        "--num_dims",
        type=int,
        default=2,
        help="Number of dimensions in the network topology",
    )
    parser.add_argument(
        "--default_runtime",
        type=int,
        default=5,
        help="Default runtime of compute nodes",
    )
    parser.add_argument(
        "--default_tensor_size",
        type=int,
        default=1024,
        help="Default tensor size of memory nodes",
    )
    parser.add_argument(
        "--default_comm_size",
        type=int,
        default=65536,
        help="Default communication size of communication nodes",
    )
    args = parser.parse_args()

    one_metadata_node_all_types(args.num_npus)

    one_remote_mem_load_node(args.num_npus, args.default_tensor_size)
    one_remote_mem_store_node(args.num_npus, args.default_tensor_size)

    one_comp_node(args.num_npus, args.default_runtime)
    two_comp_nodes_independent(args.num_npus, args.default_runtime)
    two_comp_nodes_dependent(args.num_npus, args.default_runtime)

    one_comm_coll_node_allreduce(args.num_npus, args.num_dims, args.default_comm_size)
    one_comm_coll_node_alltoall(args.num_npus, args.num_dims, args.default_comm_size)
    one_comm_coll_node_allgather(args.num_npus, args.num_dims, args.default_comm_size)
    one_comm_coll_node_reducescatter(
        args.num_npus, args.num_dims, args.default_comm_size
    )


if __name__ == "__main__":
    main()
