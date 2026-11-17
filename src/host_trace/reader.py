import gzip
import json
import logging
from typing import Any, Dict

from .errors import HostTraceError

logger = logging.getLogger(__name__)

GZIP_MAGIC = b"\x1f\x8b"


def read_trace_json(path: str) -> Dict[str, Any]:
    """
    Read a trace file (host, linked or Kineto device trace) as JSON.

    A path ending in "gz" is read as gzip-compressed JSON, any other path as plain JSON (UTF-8). The name decides, as
    in HTA, which also reads the device trace; the file's first two bytes are checked against it, so a mismatch is a
    clear error instead of a decoding failure. The file is then streamed, so no extra copy of it is held in memory.

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
    named_gz = str(path).endswith("gz")
    if named_gz and not is_gzip:
        raise HostTraceError(f"{path}: the name ends in gz, but the file is not gzip-compressed")
    if is_gzip and not named_gz:
        raise HostTraceError(f"{path}: the file is gzip-compressed, but its name does not end in gz: name it *.gz")
    with gzip.open(path, "rb") if named_gz else open(path, "r", encoding="utf-8") as f:
        return json.load(f)
