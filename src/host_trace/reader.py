import gzip
import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def read_trace_json(path: str) -> Dict[str, Any]:
    """
    Read a trace file (host, linked or Kineto device trace) as JSON.

    A path ending in "gz" is read as gzip-compressed JSON, any other path as plain JSON text.

    Args:
        path (str): Path of the trace file.

    Returns:
        Dict[str, Any]: The parsed JSON.
    """
    logger.debug(f"Reading trace JSON from {path}")
    with gzip.open(path, "rb") if str(path).endswith("gz") else open(path, "r") as f:
        return json.load(f)
