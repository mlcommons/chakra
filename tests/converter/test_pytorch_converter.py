import json
import logging
from typing import Dict, Tuple
from unittest.mock import MagicMock, mock_open, patch

import pytest
from chakra.schema.protobuf.et_def_pb2 import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    BROADCAST,
    COMM_COLL_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    METADATA_NODE,
    REDUCE_SCATTER,
)
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode
from chakra.src.converter.converter_node import ConverterNode, ConverterNodeType
from chakra.src.converter.pytorch_converter import PyTorchConverter
from chakra.src.host_trace.errors import HostTraceError
from chakra.src.host_trace.trace import HostTrace


@pytest.fixture
def sample_pytorch_data() -> Dict:
    return {
        "schema": "1.0.2-chakra.0.0.4",
        "pid": 1234,
        "time": "2023-01-01 12:00:00",
        "start_ts": 1000,
        "finish_ts": 2000,
        "nodes": [
            {
                "id": 1,
                "name": "node1",
                "ctrl_deps": None,
                "exclusive_dur": 50,
                "inputs": {"values": "values", "shapes": "shapes", "types": "types"},
                "outputs": {"values": "values", "shapes": "shapes", "types": "types"},
                "attrs": [
                    {"name": "rf_id", "type": "uint64", "value": 0},
                    {"name": "fw_parent", "type": "uint64", "value": 0},
                    {"name": "seq_id", "type": "int64", "value": -1},
                    {"name": "scope", "type": "uint64", "value": 7},
                    {"name": "tid", "type": "uint64", "value": 1},
                    {"name": "fw_tid", "type": "uint64", "value": 0},
                    {"name": "op_schema", "type": "string", "value": ""},
                ],
            },
            {
                "id": 2,
                "name": "node2",
                "ctrl_deps": 1,
                "exclusive_dur": 30,
                "inputs": {"values": "values", "shapes": "shapes", "types": "types"},
                "outputs": {"values": "values", "shapes": "shapes", "types": "types"},
                "attrs": [
                    {"name": "rf_id", "type": "uint64", "value": 0},
                    {"name": "fw_parent", "type": "uint64", "value": 0},
                    {"name": "seq_id", "type": "int64", "value": -1},
                    {"name": "scope", "type": "uint64", "value": 7},
                    {"name": "tid", "type": "uint64", "value": 1},
                    {"name": "fw_tid", "type": "uint64", "value": 0},
                    {"name": "op_schema", "type": "string", "value": ""},
                ],
            },
        ],
    }


@pytest.fixture
def mock_chakra_node() -> ChakraNode:
    node = ChakraNode()
    node.id = 1
    node.name = "node1"
    node.type = COMP_NODE
    return node


def load_for_conversion(data: Dict) -> Tuple[Dict, Dict[int, ConverterNode]]:
    """Parse a host trace and add the converter's links, as PyTorchConverter.convert does."""
    converter = PyTorchConverter()
    trace = HostTrace(data, node_class=ConverterNode)
    converter.register_converter_links(trace.nodes)
    return converter.read_metadata(trace), trace.nodes


def test_host_trace_metadata_and_nodes(sample_pytorch_data: Dict) -> None:
    json_metadata, json_node_map = load_for_conversion(sample_pytorch_data)

    assert json_metadata["schema"] == "1.0.2-chakra.0.0.4"
    assert json_metadata["pid"] == 1234
    assert json_metadata["time"] == "2023-01-01 12:00:00"
    assert json_metadata["start_ts"] == 1000
    assert json_metadata["finish_ts"] == 2000
    assert len(json_node_map) == 2
    assert json_node_map[1].id == 1
    assert json_node_map[2].id == 2


def test_convert_json_to_protobuf_nodes(sample_pytorch_data: Dict) -> None:
    converter = PyTorchConverter()
    json_metadata, json_node_map = load_for_conversion(sample_pytorch_data)
    chakra_nodes = {}
    converter.convert_json_to_protobuf_nodes(json_node_map, chakra_nodes)
    assert len(chakra_nodes) == 2
    assert chakra_nodes[1].id == 1
    assert chakra_nodes[2].id == 2


def test_convert_ctrl_dep_to_data_dep(sample_pytorch_data: Dict) -> None:
    converter = PyTorchConverter()
    json_metadata, json_node_map = load_for_conversion(sample_pytorch_data)
    chakra_nodes = {}
    converter.convert_json_to_protobuf_nodes(json_node_map, chakra_nodes)
    root_node = chakra_nodes[1]
    converter.convert_ctrl_dep_to_data_dep(json_node_map, chakra_nodes, root_node)
    assert root_node.data_deps == []


@patch("builtins.open", new_callable=mock_open)
def test_write_chakra_et(mock_file: MagicMock, sample_pytorch_data: Dict) -> None:
    converter = PyTorchConverter()
    json_metadata, json_node_map = load_for_conversion(sample_pytorch_data)
    chakra_nodes = {}
    converter.convert_json_to_protobuf_nodes(json_node_map, chakra_nodes)
    converter.write_protobuf_execution_trace("output.et", json_metadata, chakra_nodes)
    assert mock_file().write.called


@pytest.mark.parametrize(
    "pytorch_node_data, expected_type",
    [
        ({"name": "process_group:init", "is_gpu_op": False, "is_metadata_op": True}, METADATA_NODE),
        ({"name": "ncclKernel", "is_gpu_op": True, "is_metadata_op": False}, COMM_COLL_NODE),
        ({"name": "ncclDevKernel", "is_gpu_op": True, "is_metadata_op": False}, COMM_COLL_NODE),
        ({"name": "c10d::all_reduce", "is_gpu_op": True, "is_metadata_op": False}, COMP_NODE),
        ({"name": "other_op", "is_gpu_op": False, "is_metadata_op": False}, COMP_NODE),
    ],
)
def test_get_protobuf_node_type_from_json_node(pytorch_node_data: Dict, expected_type: int) -> None:
    # Create a mock ConverterNode with the required attributes
    pytorch_node = MagicMock(spec=ConverterNode)
    pytorch_node.name = pytorch_node_data["name"]
    pytorch_node.is_gpu_op = MagicMock(return_value=pytorch_node_data["is_gpu_op"])
    pytorch_node.is_metadata_op = MagicMock(return_value=pytorch_node_data["is_metadata_op"])

    # Create a mock json_node_map dictionary with actual ConverterNode instances
    mock_pytorch_node_data = {
        "id": 0,
        "name": "mock_node",
        "ctrl_deps": None,
        "exclusive_dur": 0,
        "inputs": {"values": [], "shapes": [], "types": []},
        "outputs": {"values": [], "shapes": [], "types": []},
        "attrs": [],
    }
    mock_pytorch_node = ConverterNode("1.0.2-chakra.0.0.4", mock_pytorch_node_data)
    json_node_map = {0: mock_pytorch_node, 1: pytorch_node}

    converter = PyTorchConverter()
    node_type = converter.get_protobuf_node_type_from_json_node(json_node_map, pytorch_node)
    assert node_type == expected_type


@pytest.mark.parametrize(
    "name, expected_comm_type",
    [
        ("allreduce", ALL_REDUCE),
        ("alltoall", ALL_TO_ALL),
        ("allgather", ALL_GATHER),
        ("reducescatter", REDUCE_SCATTER),
        ("broadcast", BROADCAST),
    ],
)
def test_get_collective_comm_type(name: str, expected_comm_type: int) -> None:
    converter = PyTorchConverter()
    comm_type = converter.get_collective_comm_type(name)
    assert comm_type == expected_comm_type


@pytest.mark.parametrize("key", ["schema", "pid", "time", "start_ts", "finish_ts"])
def test_missing_metadata_key_is_an_error_naming_it(key: str, make_trace_data) -> None:
    data = make_trace_data()
    trace = HostTrace(data, node_class=ConverterNode)
    del trace.raw[key]
    with pytest.raises(HostTraceError, match=f"missing top-level key\\(s\\) {key}, which chakra_converter writes"):
        PyTorchConverter().read_metadata(trace)


def _convert_one(node_data: Dict) -> ChakraNode:
    node = ConverterNode("1.1.1-chakra.0.0.4", node_data)
    return PyTorchConverter().convert_json_to_protobuf_node({node.id: node}, {}, node)


@pytest.mark.parametrize("attr", ["rf_id", "fw_parent", "seq_id", "scope", "tid", "fw_tid", "op_schema"])
def test_converted_node_without_a_core_attribute_is_an_error_naming_it(attr: str, make_node) -> None:
    node_data = make_node(3, "linear", 2)
    node_data["attrs"] = [a for a in node_data["attrs"] if a["name"] != attr]
    with pytest.raises(
        HostTraceError, match=f"node 3 \\(linear\\): missing attribute\\(s\\) {attr}, which chakra_conv"
    ):
        _convert_one(node_data)


def test_converted_node_without_a_tensor_key_is_an_error_naming_it(make_node) -> None:
    node_data = make_node(3, "linear", 2)
    del node_data["outputs"]["shapes"]
    with pytest.raises(
        HostTraceError, match="node 3 \\(linear\\): outputs has no shapes, which chakra_converter needs"
    ):
        _convert_one(node_data)


def test_null_attribute_is_converted_as_unset(make_node) -> None:
    node_data = make_node(3, "linear", 2)
    next(a for a in node_data["attrs"] if a["name"] == "tid")["value"] = None
    tid = next(a for a in _convert_one(node_data).attr if a.name == "tid")
    assert not tid.HasField("int64_val")


@pytest.mark.parametrize(
    ("name", "cat", "expected"),
    [
        ("## process_group:init ##", None, ConverterNodeType.METADATA),
        ("gemm", "kernel", ConverterNodeType.GPU_OP),
        ("aten::linear", None, ConverterNodeType.CPU_OP),
    ],
)
def test_converter_node_type(name, cat, expected, make_node) -> None:
    node = ConverterNode("1.1.1-chakra.0.0.4", make_node(3, name, 2, cat=cat))
    assert node.op_type is expected
    assert node.is_cpu_op() is (expected is ConverterNodeType.CPU_OP)
    assert node.is_metadata_op() is (expected is ConverterNodeType.METADATA)


def test_trace_without_a_thread_root_gives_a_warning(make_trace_data, tmp_path, caplog) -> None:
    data = make_trace_data()
    data["nodes"][1]["name"] = "[pytorch|profiler|execution_graph|thread]"  # the PyTorch 2.0 name: not a thread root
    source = tmp_path / "host.json"
    source.write_text(json.dumps(data))
    with caplog.at_level(logging.WARNING):
        PyTorchConverter().convert(str(source), str(tmp_path / "out.et"), False)
    assert "no thread root" in caplog.text


def test_sendrecv_kernel_gets_the_process_group_of_its_record_param_comms_parent(make_trace_data, make_node) -> None:
    # nccl:send (3) → record_param_comms with pg_name "5" (4) → SendRecv kernel without a pg_name of its own (5).
    data = make_trace_data(
        nodes=[
            make_node(1, "[pytorch|profiler|execution_trace|process]", 1),
            make_node(2, "[pytorch|profiler|execution_trace|thread]", 1),
            make_node(3, "nccl:send", 2),
            make_node(4, "record_param_comms", 3),
            make_node(5, "ncclDevKernel_SendRecv(ncclDevComm*, unsigned long, ncclWork*)", 4, cat="kernel"),
        ]
    )
    data["nodes"][3]["attrs"].append({"name": "pg_name", "type": "string", "value": "5"})
    converter = PyTorchConverter()
    trace = HostTrace(data, node_class=ConverterNode)
    converter.register_converter_links(trace.nodes)
    protobuf_nodes: Dict = {}
    converter.convert_json_to_protobuf_nodes(trace.nodes, protobuf_nodes)
    kernel = protobuf_nodes[5]
    assert kernel.type == COMM_SEND_NODE
    assert [a.string_val for a in kernel.attr if a.name == "pg_name"] == ["5"]
    assert trace.nodes[5].pg_name == ""  # the parsed value is never overwritten
