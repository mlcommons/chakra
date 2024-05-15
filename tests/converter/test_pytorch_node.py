import json
import tarfile
from pathlib import Path
from typing import Any, Dict

import pytest

from src.converter.pytorch_node import PyTorchNode


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
