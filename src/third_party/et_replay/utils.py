import gzip
import json
from typing import Any, Dict

from .execution_trace import ExecutionTrace


def load_execution_trace_file(et_file_path: str) -> ExecutionTrace:
    """Loads Execution Trace from json file and parses it."""
    data = read_dictionary_from_json_file(et_file_path)
    return ExecutionTrace(data)


def read_dictionary_from_json_file(file_path: str) -> Dict[Any, Any]:
    """Read a json file and return it as a dictionary."""
    with (
        gzip.open(file_path, "rb") if file_path.endswith("gz") else open(file_path, "r")
    ) as f:
        return json.load(f)
