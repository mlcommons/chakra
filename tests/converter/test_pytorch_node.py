import json
import tarfile
from pathlib import Path
from typing import Any, Dict

import pytest
from chakra.src.converter.pytorch_node import PyTorchNode


@pytest.fixture
def extract_tar_gz_file(tmp_path: Path) -> Path:
    """
    Fixture to extract a tar.gz file to a temporary directory.

    Args:
        tmp_path (Path): Temporary directory path provided by pytest.

    Returns:
        Path: Path to the extracted directory.
    """
    tar_gz_file = Path("tests/data/1.0.2-chakra.0.0.4.tgz")
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()

    with tarfile.open(tar_gz_file, "r:gz") as tar:
        tar.extractall(path=extracted_dir)

    return extracted_dir


def load_pytorch_execution_traces(file_path: str) -> Dict[str, Any]:
    """
    Loads PyTorch execution traces from a file.

    Args:
        file_path (str): Path to the PyTorch execution trace file.

    Returns:
        Dict[str, Any]: Parsed PyTorch execution trace data.
    """
    with open(file_path, "r") as pytorch_et:
        return json.load(pytorch_et)


def test_pytorch_node_parsing(extract_tar_gz_file: Path) -> None:
    """
    Test to check if PyTorchNode can parse nodes properly from the extracted data.

    Args:
        extract_tar_gz_file (Path): Path to the extracted directory containing
            the PyTorch execution trace file.
    """
    pytorch_trace_file = extract_tar_gz_file / "1.0.2-chakra.0.0.4.json"
    pytorch_et_data = load_pytorch_execution_traces(str(pytorch_trace_file))

    pytorch_schema = pytorch_et_data["schema"]
    pytorch_nodes = pytorch_et_data["nodes"]

    for node_data in pytorch_nodes:
        node = PyTorchNode(pytorch_schema, node_data)
        assert node is not None  # Check if node is instantiated properly


@pytest.fixture
def sample_node_data_1_0_2_chakra_0_0_4() -> Dict:
    return {
        "id": 1,
        "name": "node1",
        "ctrl_deps": None,
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
        "exclusive_dur": 50,
    }


@pytest.fixture
def sample_node_data_1_0_3_chakra_0_0_4() -> Dict:
    return {
        "id": 2,
        "name": "node2",
        "ctrl_deps": 1,
        "inputs": {"values": [], "shapes": [], "types": []},
        "outputs": {"values": [], "shapes": [], "types": []},
        "attrs": [
            {"name": "rf_id", "type": "uint64", "value": 2},
            {"name": "fw_parent", "type": "uint64", "value": 0},
            {"name": "seq_id", "type": "int64", "value": -1},
            {"name": "scope", "type": "uint64", "value": 7},
            {"name": "tid", "type": "uint64", "value": 1},
            {"name": "fw_tid", "type": "uint64", "value": 0},
            {"name": "op_schema", "type": "string", "value": ""},
        ],
        "exclusive_dur": 30,
    }


@pytest.fixture
def sample_node_data_unsupported_schema() -> Dict:
    return {
        "id": 4,
        "name": "## process_group:init ##",
        "ctrl_deps": 3,
        "inputs": {
            "values": [],
            "shapes": [[]],
            "types": ["String"],
        },
        "outputs": {"values": [], "shapes": [], "types": []},
        "attrs": [
            {"name": "rf_id", "type": "uint64", "value": 2},
            {"name": "fw_parent", "type": "uint64", "value": 0},
            {"name": "seq_id", "type": "int64", "value": -1},
            {"name": "scope", "type": "uint64", "value": 7},
            {"name": "tid", "type": "uint64", "value": 1},
            {"name": "fw_tid", "type": "uint64", "value": 0},
            {"name": "op_schema", "type": "string", "value": ""},
        ],
        "exclusive_dur": 40,
    }


def test_pytorch_node_parsing_1_0_2_chakra_0_0_4(sample_node_data_1_0_2_chakra_0_0_4) -> None:
    schema = "1.0.2-chakra.0.0.4"
    node = PyTorchNode(schema, sample_node_data_1_0_2_chakra_0_0_4)
    assert node is not None
    assert node.schema == schema
    assert isinstance(node.id, int)
    assert isinstance(node.name, str)
    assert node.exclusive_dur == 50


def test_pytorch_node_parsing_1_0_3_chakra_0_0_4(sample_node_data_1_0_3_chakra_0_0_4) -> None:
    schema = "1.0.3-chakra.0.0.4"
    node = PyTorchNode(schema, sample_node_data_1_0_3_chakra_0_0_4)
    assert node is not None
    assert node.schema == schema
    assert isinstance(node.id, int)
    assert isinstance(node.name, str)
    assert node.exclusive_dur == 30


def test_pytorch_node_unsupported_schema(sample_node_data_unsupported_schema) -> None:
    schema = "1.1.0-chakra.0.0.4"
    with pytest.raises(ValueError, match=f"Unsupported schema version '{schema}'"):
        PyTorchNode(schema, sample_node_data_unsupported_schema)
