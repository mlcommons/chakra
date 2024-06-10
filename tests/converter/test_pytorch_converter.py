import json
import logging
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
    REDUCE_SCATTER,
)
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode
from chakra.src.converter.pytorch_converter import PyTorchConverter
from chakra.src.converter.pytorch_node import PyTorchNode


@pytest.fixture
def mock_logger() -> logging.Logger:
    logger = logging.getLogger("PyTorchConverter")
    logger.setLevel(logging.DEBUG)
    return logger


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


def test_initialization(mock_logger: logging.Logger) -> None:
    converter = PyTorchConverter("input.json", "output.json", mock_logger)
    assert converter.input_filename == "input.json"
    assert converter.output_filename == "output.json"
    assert converter.logger == mock_logger


@patch("builtins.open", new_callable=mock_open)
def test_load_pytorch_execution_traces(
    mock_file: MagicMock, mock_logger: logging.Logger, sample_pytorch_data: Dict
) -> None:
    mock_file.return_value.read.return_value = json.dumps(sample_pytorch_data)
    converter = PyTorchConverter("input.json", "output.json", mock_logger)
    data = converter.load_pytorch_execution_traces()
    assert data == sample_pytorch_data
    mock_file.assert_called_once_with("input.json", "r")


def test_parse_and_instantiate_nodes(mock_logger: logging.Logger, sample_pytorch_data: Dict) -> None:
    converter = PyTorchConverter("input.json", "output.json", mock_logger)
    (
        pytorch_schema,
        pytorch_pid,
        pytorch_time,
        pytorch_start_ts,
        pytorch_finish_ts,
        pytorch_nodes,
    ) = converter._parse_and_instantiate_nodes(sample_pytorch_data)
    assert pytorch_schema == "1.0.2-chakra.0.0.4"
    assert pytorch_pid == 1234
    assert pytorch_time == "2023-01-01 12:00:00"
    assert pytorch_start_ts == 1000
    assert pytorch_finish_ts == 2000
    assert len(pytorch_nodes) == 2
    assert pytorch_nodes[1].id == 1
    assert pytorch_nodes[2].id == 2


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
def test_establish_parent_child_relationships(mock_logger: MagicMock, parent_id: int, expected_child_id: int) -> None:
    converter = PyTorchConverter("input.json", "output.json", mock_logger)
    pytorch_nodes = create_sample_graph(parent_id, expected_child_id)

    pytorch_nodes = converter._establish_parent_child_relationships(pytorch_nodes, [])

    if expected_child_id:
        assert pytorch_nodes[parent_id].children[0].id == expected_child_id
    else:
        assert len(pytorch_nodes[1].children) == 0


def test_convert_nodes(mock_logger: logging.Logger, sample_pytorch_data: Dict) -> None:
    converter = PyTorchConverter("input.json", "output.json", mock_logger)
    (
        pytorch_schema,
        pytorch_pid,
        pytorch_time,
        pytorch_start_ts,
        pytorch_finish_ts,
        pytorch_nodes,
    ) = converter._parse_and_instantiate_nodes(sample_pytorch_data)
    pytorch_nodes = converter._establish_parent_child_relationships(pytorch_nodes, [])
    chakra_nodes = {}
    converter.convert_nodes(pytorch_nodes, chakra_nodes)
    assert len(chakra_nodes) == 2
    assert chakra_nodes[1].id == 1
    assert chakra_nodes[2].id == 2


def test_convert_ctrl_dep_to_data_dep(mock_logger: logging.Logger, sample_pytorch_data: Dict) -> None:
    converter = PyTorchConverter("input.json", "output.json", mock_logger)
    (
        pytorch_schema,
        pytorch_pid,
        pytorch_time,
        pytorch_start_ts,
        pytorch_finish_ts,
        pytorch_nodes,
    ) = converter._parse_and_instantiate_nodes(sample_pytorch_data)
    pytorch_nodes = converter._establish_parent_child_relationships(pytorch_nodes, [])
    chakra_nodes = {}
    converter.convert_nodes(pytorch_nodes, chakra_nodes)
    root_node = chakra_nodes[1]
    converter.convert_ctrl_dep_to_data_dep(pytorch_nodes, chakra_nodes, root_node)
    assert root_node.data_deps == []


@patch("builtins.open", new_callable=mock_open)
def test_write_chakra_et(mock_file: MagicMock, mock_logger: logging.Logger, sample_pytorch_data: Dict) -> None:
    converter = PyTorchConverter("input.json", "output.json", mock_logger)
    converter.chakra_et = mock_file()
    (
        pytorch_schema,
        pytorch_pid,
        pytorch_time,
        pytorch_start_ts,
        pytorch_finish_ts,
        pytorch_nodes,
    ) = converter._parse_and_instantiate_nodes(sample_pytorch_data)
    pytorch_nodes = converter._establish_parent_child_relationships(pytorch_nodes, [])
    chakra_nodes = {}
    converter.convert_nodes(pytorch_nodes, chakra_nodes)
    converter.write_chakra_et(
        converter.chakra_et,
        pytorch_schema,
        pytorch_pid,
        pytorch_time,
        pytorch_start_ts,
        pytorch_finish_ts,
        chakra_nodes,
    )
    assert mock_file().write.called


@patch("builtins.open", new_callable=mock_open)
def test_close_chakra_execution_trace(mock_file: MagicMock, mock_logger: logging.Logger) -> None:
    converter = PyTorchConverter("input.json", "output.json", mock_logger)
    file_handle = mock_file()
    file_handle.closed = False  # Simulate an open file
    converter.chakra_et = file_handle
    converter.close_chakra_execution_trace(converter.chakra_et)
    file_handle.close.assert_called_once()


@pytest.mark.parametrize(
    "pytorch_node_data, expected_type",
    [
        ({"name": "ncclKernel", "is_gpu_op": True}, COMM_COLL_NODE),
        ({"name": "ncclDevKernel", "is_gpu_op": True}, COMM_COLL_NODE),
        ({"name": "c10d::all_reduce", "is_gpu_op": True}, COMM_COLL_NODE),
        ({"name": "other_op", "is_gpu_op": False}, COMP_NODE),
    ],
)
def test_get_chakra_node_type_from_pytorch_node(
    mock_logger: logging.Logger, pytorch_node_data: Dict, expected_type: int
) -> None:
    pytorch_node = MagicMock()
    pytorch_node.name = pytorch_node_data["name"]
    pytorch_node.is_gpu_op = MagicMock(return_value=pytorch_node_data["is_gpu_op"])

    converter = PyTorchConverter("input.json", "output.json", mock_logger)
    node_type = converter.get_chakra_node_type_from_pytorch_node(pytorch_node)
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
def test_get_collective_comm_type(mock_logger: logging.Logger, name: str, expected_comm_type: int) -> None:
    converter = PyTorchConverter("input.json", "output.json", mock_logger)
    comm_type = converter.get_collective_comm_type(name)
    assert comm_type == expected_comm_type
