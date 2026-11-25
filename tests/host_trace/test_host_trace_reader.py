import gzip
import json
from pathlib import Path

import pytest
from chakra.src.host_trace.errors import HostTraceError
from chakra.src.host_trace.reader import read_trace_json


def test_plain_file_is_read_as_utf8(tmp_path: Path) -> None:
    data = {"schema": "1.0.2-chakra.0.0.4", "nodes": [{"id": 1, "name": "aten::µ_op"}]}
    path = tmp_path / "host.json"
    path.write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    assert read_trace_json(str(path)) == data


def _write(tmp_path: Path, name: str, data: dict, compress: bool) -> Path:
    path = tmp_path / name
    raw = json.dumps(data).encode()
    path.write_bytes(gzip.compress(raw) if compress else raw)
    return path


def test_gzipped_file_named_gz_is_read_like_a_plain_one(tmp_path: Path) -> None:
    data = {"schema": "1.1.1-chakra.0.0.4", "nodes": [{"id": 1}]}
    assert read_trace_json(str(_write(tmp_path, "host.json.gz", data, True))) == data
    assert read_trace_json(str(_write(tmp_path, "host.json", data, False))) == data


def test_gzip_name_on_a_plain_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(HostTraceError, match="the name ends in gz, but the file is not gzip-compressed"):
        read_trace_json(str(_write(tmp_path, "host.json.gz", {}, False)))


def test_gzipped_file_without_a_gz_name_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(HostTraceError, match="the file is gzip-compressed, but its name does not end in gz"):
        read_trace_json(str(_write(tmp_path, "host.json", {}, True)))
