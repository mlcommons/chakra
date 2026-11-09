from unittest.mock import MagicMock

import pytest
from chakra.src.trace_link.chakra_host_trace_loader import ChakraHostTraceLoader
from et_replay.execution_trace import Node as PyTorchOperator


@pytest.fixture
def mock_trace():
    """Fixture to create a mock trace with a specific structure."""
    # Create a mock trace node structure
    root_node = MagicMock(spec=PyTorchOperator)
    child_node1 = MagicMock(spec=PyTorchOperator)
    child_node2 = MagicMock(spec=PyTorchOperator)

    # Setup mock hierarchy
    root_node.children = [child_node1, child_node2]
    root_node.id = 1
    child_node1.children = []
    child_node1.id = 2
    child_node2.children = []
    child_node2.id = 3

    mock_trace = MagicMock()
    mock_trace.get_nodes.return_value = [None, root_node]

    return mock_trace


@pytest.fixture
def loader():
    """Fixture to create a ChakraHostTraceLoader instance."""
    return ChakraHostTraceLoader()


def test_extract_chakra_host_ops(loader, mock_trace):
    """Test the extract_chakra_host_ops method."""
    root_node = mock_trace.get_nodes()[1]

    result = loader.extract_chakra_host_ops(root_node)

    assert len(result) == 3
    assert result[0].id == 1
    assert result[1].id == 2
    assert result[2].id == 3
