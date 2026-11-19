import gzip
import json
import logging
from typing import Any, Dict

from .errors import HostTraceError

logger = logging.getLogger(__name__)

GZIP_MAGIC = b"\x1f\x8b"


def named_gz(path: str) -> bool:
    return str(path).endswith("gz")


def read_trace_json(path: str) -> Dict[str, Any]:
    """
    Read a trace file (host, linked or Kineto device trace) as JSON.

    A path ending in "gz" is read as gzip-compressed JSON, any other path as plain JSON (UTF-8). The name decides, as
    in HTA, which also reads the device trace; the file's first two bytes are checked against it, so a mismatch is a
    clear error instead of a decoding failure.

    Args:
        path (str): Path of the trace file.

    Returns:
        Dict[str, Any]: The parsed JSON.

    Raises:
        HostTraceError: If the name and the content disagree about gzip.
    """
    logger.debug(f"Reading trace JSON from {path}")
    with open(path, "rb") as f:
        is_gzip = f.read(2) == GZIP_MAGIC
    gz_name = named_gz(path)
    if gz_name and not is_gzip:
        raise HostTraceError(f"{path}: the name ends in gz, but the file is not gzip-compressed")
    if is_gzip and not gz_name:
        raise HostTraceError(f"{path}: the file is gzip-compressed, but its name does not end in gz: name it *.gz")
    with gzip.open(path, "rb") if gz_name else open(path, "r", encoding="utf-8") as f:
        return json.load(f)
