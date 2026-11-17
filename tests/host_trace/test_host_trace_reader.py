import json
from unittest.mock import call, mock_open, patch

from chakra.src.host_trace.reader import read_trace_json


def test_read_trace_json() -> None:
    data = {"schema": "1.0.2-chakra.0.0.4", "nodes": []}
    with patch("builtins.open", mock_open(read_data=json.dumps(data))) as mock_file:
        assert read_trace_json("input.json") == data
    # The first two bytes are checked for gzip, then the plain file is read as UTF-8 text.
    assert mock_file.call_args_list == [call("input.json", "rb"), call("input.json", "r", encoding="utf-8")]
