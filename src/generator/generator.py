import argparse

from ...schema.protobuf.et_def_pb2 import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    BARRIER,
    BROADCAST,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    MEM_LOAD_NODE,
    MEM_STORE_NODE,
    METADATA_NODE,
    REDUCE_SCATTER,
    BoolList,
    BytesList,
    DoubleList,
    Fixed32List,
    Fixed64List,
    FloatList,
    GlobalMetadata,
    Int32List,
    Int64List,
    Sfixed32List,
    Sfixed64List,
    Sint32List,
    Sint64List,
    StringList,
    Uint32List,
    Uint64List,
)
from ...schema.protobuf.et_def_pb2 import (
    AttributeProto as ChakraAttr,
)
from ...schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)
from ...schema.protobuf.et_def_pb2 import (
    NodeType as ChakraNodeType,
)
from ..third_party.utils.protolib import encodeMessage as encode_message

NODE_ID = 0


def get_node(node_name: str, node_type: ChakraNodeType) -> ChakraNode:
    """Generate a new ChakraNode with a unique ID."""
    global NODE_ID
    node = ChakraNode()
    node.id = NODE_ID
    node.name = node_name
    node.type = node_type
    NODE_ID += 1
    return node


def get_comm_type_attr(comm_type: int) -> ChakraAttr:
    """Create a communication type attribute."""
    return ChakraAttr(name="comm_type", int64_val=comm_type)


def one_metadata_node_all_types(num_npus: int) -> None:
    """Generate metadata nodes with all types of attributes."""
    for npu_id in range(num_npus):
        output_filename = f"one_metadata_node_all_types.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("METADATA_NODE", METADATA_NODE)
            node.attr.extend(
                [
                    ChakraAttr(name="double", double_val=1.2345, doc_string="double"),
                    ChakraAttr(name="double_list", double_list=DoubleList(values=[1.2345, 2.3456])),
                    ChakraAttr(name="float", float_val=1.2345, doc_string="float"),
                    ChakraAttr(name="float_list", float_list=FloatList(values=[1.2345, 2.3456])),
                    ChakraAttr(name="int32", int32_val=12345, doc_string="int32"),
                    ChakraAttr(name="int32_list", int32_list=Int32List(values=[12345, 23456])),
                    ChakraAttr(name="int64", int64_val=9876543210, doc_string="int64"),
                    ChakraAttr(name="int64_list", int64_list=Int64List(values=[9876543210, 1234567890])),
                    ChakraAttr(name="uint32", uint32_val=12345, doc_string="uint32"),
                    ChakraAttr(name="uint32_list", uint32_list=Uint32List(values=[12345, 23456])),
                    ChakraAttr(name="uint64", uint64_val=9876543210, doc_string="uint64"),
                    ChakraAttr(name="uint64_list", uint64_list=Uint64List(values=[9876543210, 1234567890])),
                    ChakraAttr(name="sint32", sint32_val=-12345, doc_string="sint32"),
                    ChakraAttr(name="sint32_list", sint32_list=Sint32List(values=[12345, -23456])),
                    ChakraAttr(name="sint64", sint64_val=-9876543210, doc_string="sint64"),
                    ChakraAttr(name="sint64_list", sint64_list=Sint64List(values=[9876543210, -1234567890])),
                    ChakraAttr(name="fixed32", fixed32_val=12345),
                    ChakraAttr(name="fixed32_list", fixed32_list=Fixed32List(values=[12345, 23456])),
                    ChakraAttr(name="fixed64", fixed64_val=9876543210),
                    ChakraAttr(name="fixed64_list", fixed64_list=Fixed64List(values=[9876543210, 1234567890])),
                    ChakraAttr(name="sfixed32", sfixed32_val=-12345),
                    ChakraAttr(name="sfixed32_list", sfixed32_list=Sfixed32List(values=[12345, -23456])),
                    ChakraAttr(name="sfixed64", sfixed64_val=-9876543210),
                    ChakraAttr(name="sfixed64_list", sfixed64_list=Sfixed64List(values=[9876543210, -1234567890])),
                    ChakraAttr(name="bool", bool_val=True, doc_string="bool"),
                    ChakraAttr(name="bool_list", bool_list=BoolList(values=[i % 2 == 0 for i in range(10)])),
                    ChakraAttr(name="string", string_val="12345", doc_string="string"),
                    ChakraAttr(name="string_list", string_list=StringList(values=[str(12345 + i) for i in range(10)])),
                    ChakraAttr(name="bytes", bytes_val=bytes("12345", "utf-8")),
                    ChakraAttr(
                        name="bytes_list",
                        bytes_list=BytesList(values=[bytes(str(12345 + i), "utf-8") for i in range(10)]),
                    ),
                ]
            )

            encode_message(et, node)


def one_remote_mem_load_node(num_npus: int, tensor_size: int) -> None:
    """Generate remote memory load nodes."""
    for npu_id in range(num_npus):
        output_filename = f"one_remote_mem_load_node.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("MEM_LOAD_NODE", MEM_LOAD_NODE)
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
            encode_message(et, node)


def one_remote_mem_store_node(num_npus: int, tensor_size: int) -> None:
    """Generate remote memory store nodes."""
    for npu_id in range(num_npus):
        output_filename = f"one_remote_mem_store_node.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("MEM_STORE_NODE", MEM_STORE_NODE)
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
            encode_message(et, node)


def one_comp_node(num_npus: int, runtime: int) -> None:
    """Generate computation nodes with a given runtime."""
    for npu_id in range(num_npus):
        output_filename = f"one_comp_node.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("COMP_NODE", COMP_NODE)
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.duration_micros = runtime
            encode_message(et, node)


def two_comp_nodes_independent(num_npus: int, runtime: int) -> None:
    """Generate two independent computation nodes."""
    for npu_id in range(num_npus):
        output_filename = f"two_comp_nodes_independent.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            for _ in range(2):
                node = get_node("COMP_NODE", COMP_NODE)
                node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
                node.duration_micros = runtime
                encode_message(et, node)


def two_comp_nodes_dependent(num_npus: int, runtime: int) -> None:
    """Generate two dependent computation nodes."""
    for npu_id in range(num_npus):
        output_filename = f"two_comp_nodes_dependent.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            parent_node = get_node("COMP_NODE", COMP_NODE)
            parent_node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            parent_node.duration_micros = runtime
            encode_message(et, parent_node)

            child_node = get_node("COMP_NODE", COMP_NODE)
            child_node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            child_node.duration_micros = runtime
            child_node.data_deps.append(parent_node.id)
            encode_message(et, child_node)


def generate_comm_coll_node(num_npus: int, comm_size: int, comm_type: int, node_name: str) -> None:
    """Generate communication collective nodes."""
    for npu_id in range(num_npus):
        output_filename = f"{node_name}.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node(node_name, COMM_COLL_NODE)
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.extend([get_comm_type_attr(comm_type), ChakraAttr(name="comm_size", int64_val=comm_size)])
            encode_message(et, node)


def one_comm_coll_node_allreduce(num_npus: int, comm_size: int) -> None:
    """Generate one AllReduce communication collective node."""
    generate_comm_coll_node(num_npus, comm_size, ALL_REDUCE, "ALL_REDUCE")


def one_comm_coll_node_alltoall(num_npus: int, comm_size: int) -> None:
    """Generate one AllToAll communication collective node."""
    generate_comm_coll_node(num_npus, comm_size, ALL_TO_ALL, "ALL_TO_ALL")


def one_comm_coll_node_allgather(num_npus: int, comm_size: int) -> None:
    """Generate one AllGather communication collective node."""
    generate_comm_coll_node(num_npus, comm_size, ALL_GATHER, "ALL_GATHER")


def one_comm_coll_node_reducescatter(num_npus: int, comm_size: int) -> None:
    """Generate one ReduceScatter communication collective node."""
    generate_comm_coll_node(num_npus, comm_size, REDUCE_SCATTER, "REDUCE_SCATTER")


def one_comm_coll_node_broadcast(num_npus: int, comm_size: int) -> None:
    """Generate one Broadcast communication collective node."""
    generate_comm_coll_node(num_npus, comm_size, BROADCAST, "BROADCAST")


def one_comm_coll_node_barrier(num_npus: int) -> None:
    """Generate one Barrier communication collective node."""
    generate_comm_coll_node(num_npus, comm_size=0, comm_type=BARRIER, node_name="BARRIER")


def one_comm_send_node(num_npus: int, tensor_size: int) -> None:
    """Generate communication send nodes."""
    for npu_id in range(num_npus):
        output_filename = f"one_comm_send_node.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("COMM_SEND_NODE", COMM_SEND_NODE)
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
            encode_message(et, node)


def one_comm_recv_node(num_npus: int, tensor_size: int) -> None:
    """Generate communication receive nodes."""
    for npu_id in range(num_npus):
        output_filename = f"one_comm_recv_node.{npu_id}.et"
        with open(output_filename, "wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = get_node("COMM_RECV_NODE", COMM_RECV_NODE)
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
            encode_message(et, node)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execution Trace Generator")
    parser.add_argument("--num_npus", type=int, default=64, help="Number of NPUs")
    parser.add_argument("--default_runtime", type=int, default=5, help="Default runtime of compute nodes")
    parser.add_argument("--default_tensor_size", type=int, default=1024, help="Default tensor size of memory nodes")
    parser.add_argument(
        "--default_comm_size", type=int, default=65536, help="Default communication size of communication nodes"
    )
    args = parser.parse_args()

    one_metadata_node_all_types(args.num_npus)
    one_remote_mem_load_node(args.num_npus, args.default_tensor_size)
    one_remote_mem_store_node(args.num_npus, args.default_tensor_size)
    one_comp_node(args.num_npus, args.default_runtime)
    two_comp_nodes_independent(args.num_npus, args.default_runtime)
    two_comp_nodes_dependent(args.num_npus, args.default_runtime)
    one_comm_coll_node_allreduce(args.num_npus, args.default_comm_size)
    one_comm_coll_node_alltoall(args.num_npus, args.default_comm_size)
    one_comm_coll_node_allgather(args.num_npus, args.default_comm_size)
    one_comm_coll_node_reducescatter(args.num_npus, args.default_comm_size)
    one_comm_coll_node_broadcast(args.num_npus, args.default_comm_size)
    one_comm_coll_node_barrier(args.num_npus)
    one_comm_send_node(args.num_npus, args.default_tensor_size)
    one_comm_recv_node(args.num_npus, args.default_tensor_size)


if __name__ == "__main__":
    main()
