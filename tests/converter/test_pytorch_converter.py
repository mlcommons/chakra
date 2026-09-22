import json
from copy import deepcopy
from typing import Dict
from unittest.mock import MagicMock, mock_open, patch

import pytest
from chakra.schema.protobuf.et_def_pb2 import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    BROADCAST,
    COMM_COLL_NODE,
    COMP_NODE,
    METADATA_NODE,
    REDUCE_SCATTER,
)
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode
from chakra.src.converter.pytorch_converter import PyTorchConverter
from chakra.src.converter.pytorch_node import PyTorchNode


@pytest.fixture
def stream_trace(sample_pytorch_data):
    trace = deepcopy(sample_pytorch_data)
    root, launcher = trace["nodes"]
    root["name"] = "[pytorch|profiler|execution_trace|thread]"
    for node_id, name, timestamp in [(40, "gemm", 10), (30, "ncclKernel_AllReduce", 20), (20, "consumer", 30)]:
        node = deepcopy(launcher)
        node.update(id=node_id, name=name, ctrl_deps=2, cat="kernel", device=0, stream=0, ts=timestamp)
        node["inputs"] = node["outputs"] = {"values": [], "shapes": [], "types": []}
        trace["nodes"].append(node)
    return trace


def converted_nodes(trace):
    converter = PyTorchConverter()
    with (
        patch.object(converter, "load_json_execution_traces", return_value=trace),
        patch.object(converter, "write_protobuf_execution_trace") as write,
    ):
        converter.convert("input.json", "output.et", False)
    return write.call_args.args[2]


def test_convert_preserves_stream_order_and_launcher_dependencies(stream_trace):
    nodes = converted_nodes(stream_trace)
    assert nodes[30].type == COMM_COLL_NODE
    assert set(nodes[30].data_deps) == {2, 40}
    assert set(nodes[20].data_deps) == {2, 30}
    assert list(nodes[40].data_deps) == [2]
    assert list(nodes[20].ctrl_deps) == [2]


@pytest.mark.parametrize("field,value", [("stream", 1), ("device", 1)])
def test_convert_keeps_device_streams_independent(stream_trace, field, value):
    stream_trace["nodes"][-2][field] = value
    nodes = converted_nodes(stream_trace)
    assert list(nodes[30].data_deps) == [2]
    assert set(nodes[20].data_deps) == {2, 40}


def test_convert_sorts_stream_timestamps_stably(stream_trace):
    stream_trace["nodes"][2:] = list(reversed(stream_trace["nodes"][2:]))
    nodes = converted_nodes(stream_trace)
    assert 40 in nodes[30].data_deps
    assert 30 in nodes[20].data_deps
    for node in stream_trace["nodes"][2:]:
        node["ts"] = 10
    nodes = converted_nodes(stream_trace)
    assert 20 in nodes[30].data_deps
    assert 30 in nodes[40].data_deps


@pytest.mark.parametrize("field", ["device", "stream", "ts"])
@pytest.mark.parametrize("explicit_null", [False, True])
def test_convert_skips_missing_stream_metadata(stream_trace, field, explicit_null):
    for node in stream_trace["nodes"][2:]:
        if explicit_null:
            node[field] = None
        else:
            del node[field]
    nodes = converted_nodes(stream_trace)
    assert all(list(nodes[node_id].data_deps) == [2] for node_id in (40, 30, 20))


def test_convert_does_not_duplicate_sync_and_stream_edges(stream_trace):
    stream_trace["nodes"][-1]["sync_dep"] = [30, 30]
    nodes = converted_nodes(stream_trace)
    assert list(nodes[20].data_deps).count(30) == 1


@pytest.mark.parametrize(
    "source_ts,target_ts,expected",
    [(10, 30, True), (30, 10, False), (10, 10, False), (None, 30, False), (10, None, False)],
)
@pytest.mark.parametrize("cpu_target", [False, True])
def test_convert_sync_uses_linked_timestamps(stream_trace, source_ts, target_ts, expected, cpu_target):
    stream_trace["nodes"][2]["ts"] = source_ts
    target = stream_trace["nodes"][-1]
    target.update(ts=target_ts, stream=1, sync_dep=[40, 999])
    if cpu_target:
        target.pop("cat")
        target["ctrl_deps"] = 1
    nodes = converted_nodes(stream_trace)
    assert (40 in nodes[20].data_deps) == expected
    assert 999 not in nodes[20].data_deps


def test_convert_preserves_copy_order(stream_trace):
    stream_trace["nodes"][3].update(name="Memcpy DtoD", cat="gpu_memcpy")
    nodes = converted_nodes(stream_trace)
    assert 40 in nodes[30].data_deps
    assert 30 in nodes[20].data_deps


def test_stream_dependencies_are_idempotent(stream_trace):
    converter = PyTorchConverter()
    _, json_nodes = converter.parse_json_trace(stream_trace)
    nodes = converted_nodes(stream_trace)
    before = {node_id: list(node.data_deps) for node_id, node in nodes.items()}
    converter.add_stream_dependencies(json_nodes, nodes)
    assert before == {node_id: list(node.data_deps) for node_id, node in nodes.items()}


def test_convert_rejects_cycles_after_adding_stream_edges(stream_trace):
    converter = PyTorchConverter()

    def add_conflicting_dependency(json_nodes, protobuf_nodes, root):
        protobuf_nodes[40].data_deps.append(20)

    with (
        patch.object(converter, "load_json_execution_traces", return_value=stream_trace),
        patch.object(converter, "convert_ctrl_dep_to_data_dep", side_effect=add_conflicting_dependency),
        patch.object(converter, "write_protobuf_execution_trace") as write,
        pytest.raises(Exception, match="Cyclic dependency"),
    ):
        converter.convert("input.json", "output.et", False)
    write.assert_not_called()


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


@patch("builtins.open", new_callable=mock_open)
def test_load_json_execution_traces(mock_file: MagicMock, sample_pytorch_data: Dict) -> None:
    mock_file.return_value.read.return_value = json.dumps(sample_pytorch_data)
    converter = PyTorchConverter()
    data = converter.load_json_execution_traces("input.json")
    assert data == sample_pytorch_data
    mock_file.assert_called_once_with("input.json", "r")


def test_parse_json_trace(sample_pytorch_data: Dict) -> None:
    converter = PyTorchConverter()
    json_metadata, json_node_map = converter.parse_json_trace(sample_pytorch_data)

    assert json_metadata["schema"] == "1.0.2-chakra.0.0.4"
    assert json_metadata["pid"] == 1234
    assert json_metadata["time"] == "2023-01-01 12:00:00"
    assert json_metadata["start_ts"] == 1000
    assert json_metadata["finish_ts"] == 2000
    assert len(json_node_map) == 2
    assert json_node_map[1].id == 1
    assert json_node_map[2].id == 2


def create_sample_graph(parent_id: int = 0, expected_child_id: int = 0) -> Dict[int, PyTorchNode]:
    node1_data = {
        "id": 1,
        "name": "node1",
        "ctrl_deps": None,
        "inputs": {"values": ["val1"], "shapes": ["shape1"], "types": ["type1"]},
        "outputs": {"values": ["val1"], "shapes": ["shape1"], "types": ["type1"]},
        "attrs": [],
    }
    node2_data = {
        "id": 2,
        "name": "node2",
        "ctrl_deps": parent_id,
        "inputs": {"values": ["val2"], "shapes": ["shape2"], "types": ["type2"]},
        "outputs": {"values": ["val2"], "shapes": ["shape2"], "types": ["type2"]},
        "attrs": [],
    }
    node1 = PyTorchNode("1.0.2-chakra.0.0.4", node1_data)
    node2 = PyTorchNode("1.0.2-chakra.0.0.4", node2_data)
    return {1: node1, 2: node2}


@pytest.mark.parametrize("parent_id, expected_child_id", [(1, 2), (None, None)])
def test_establish_parent_child_relationships(parent_id: int, expected_child_id: int) -> None:
    converter = PyTorchConverter()
    json_node_map = create_sample_graph(parent_id, expected_child_id)

    json_node_map = converter.establish_parent_child_relationships(json_node_map, [])

    if expected_child_id:
        assert json_node_map[parent_id].children[0].id == expected_child_id
    else:
        assert len(json_node_map[1].children) == 0


def test_convert_json_to_protobuf_nodes(sample_pytorch_data: Dict) -> None:
    converter = PyTorchConverter()
    json_metadata, json_node_map = converter.parse_json_trace(sample_pytorch_data)
    json_node_map = converter.establish_parent_child_relationships(json_node_map, [])
    chakra_nodes = {}
    converter.convert_json_to_protobuf_nodes(json_node_map, chakra_nodes)
    assert len(chakra_nodes) == 2
    assert chakra_nodes[1].id == 1
    assert chakra_nodes[2].id == 2


def test_convert_ctrl_dep_to_data_dep(sample_pytorch_data: Dict) -> None:
    converter = PyTorchConverter()
    json_metadata, json_node_map = converter.parse_json_trace(sample_pytorch_data)
    json_node_map = converter.establish_parent_child_relationships(json_node_map, [])
    chakra_nodes = {}
    converter.convert_json_to_protobuf_nodes(json_node_map, chakra_nodes)
    root_node = chakra_nodes[1]
    converter.convert_ctrl_dep_to_data_dep(json_node_map, chakra_nodes, root_node)
    assert root_node.data_deps == []


@patch("builtins.open", new_callable=mock_open)
def test_write_chakra_et(mock_file: MagicMock, sample_pytorch_data: Dict) -> None:
    converter = PyTorchConverter()
    json_metadata, json_node_map = converter.parse_json_trace(sample_pytorch_data)
    json_node_map = converter.establish_parent_child_relationships(json_node_map, [])
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
    # Create a mock PyTorchNode with the required attributes
    pytorch_node = MagicMock(spec=PyTorchNode)
    pytorch_node.name = pytorch_node_data["name"]
    pytorch_node.is_gpu_op = MagicMock(return_value=pytorch_node_data["is_gpu_op"])
    pytorch_node.is_metadata_op = MagicMock(return_value=pytorch_node_data["is_metadata_op"])

    # Create a mock json_node_map dictionary with actual PyTorchNode instances
    mock_pytorch_node_data = {
        "id": 0,
        "name": "mock_node",
        "ctrl_deps": None,
        "exclusive_dur": 0,
        "inputs": {"values": [], "shapes": [], "types": []},
        "outputs": {"values": [], "shapes": [], "types": []},
        "attrs": [],
    }
    mock_pytorch_node = PyTorchNode("1.0.2-chakra.0.0.4", mock_pytorch_node_data)
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
