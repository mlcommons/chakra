import json
from unittest.mock import MagicMock, mock_open, patch

from chakra.src.host_trace.reader import read_trace_json


@patch("builtins.open", new_callable=mock_open)
def test_read_trace_json(mock_file: MagicMock) -> None:
    data = {"schema": "1.0.2-chakra.0.0.4", "nodes": []}
    mock_file.return_value.read.return_value = json.dumps(data)
    assert read_trace_json("input.json") == data
    mock_file.assert_called_once_with("input.json", "r")
