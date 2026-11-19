"""
Both tools on the committed fixtures produce exactly the reference outputs in tests/data/reference_outputs.tar.gz.

The references were recorded with the code before the shared host-trace parser (see the archive's README.txt), so these
tests check that valid output never changes. They are regenerated only on purpose:
    python tests/test_pipeline_outputs.py --regenerate
which prints what changed, for the pull request. Regenerated references must be approved by a reviewer in the pull
request that regenerates them.
"""

import json
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import patch

import pytest
from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
from chakra.src.converter.pytorch_converter import PyTorchConverter
from chakra.src.host_trace.trace import HostTrace
from chakra.src.third_party.utils.protolib import decodeMessage, openFileRd
from chakra.src.trace_link.trace_linker import TraceLinker
from google.protobuf.json_format import MessageToDict

DATA = Path(__file__).resolve().parent / "data"
REFERENCES = DATA / "reference_outputs.tar.gz"
SIMPLE = DATA / "simple_onerank_traces.tar.gz"
V1_0_2 = DATA / "1.0.2-chakra.0.0.4.tgz"


def _extract(archive: Path, member: str, dest: Path) -> Path:
    # Read members one by one by name, so no path in an archive can escape dest.
    with tarfile.open(archive) as tar:
        source = tar.extractfile(member)
        assert source is not None, f"{member} is not a file in {archive}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read())
    return dest


def decode_et(path: Path) -> Dict[str, Any]:
    """Return a .et file's metadata and nodes, in file order, as plain data."""
    et = openFileRd(str(path))
    try:
        metadata = GlobalMetadata()
        assert decodeMessage(et, metadata), f"{path}: no GlobalMetadata"
        nodes = []
        node = Node()
        while decodeMessage(et, node):
            nodes.append(MessageToDict(node, preserving_proto_field_name=True))
            node = Node()
    finally:
        et.close()
    return {"metadata": MessageToDict(metadata, preserving_proto_field_name=True), "nodes": nodes}


def first_differences(
    actual: Any, expected: Any, path: str = "$", out: Optional[List[str]] = None, limit: int = 5
) -> List[str]:
    """Return up to `limit` paths where two JSON values differ, with the values (types included)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if type(actual) is not type(expected):
        out.append(
            f"{path}: {type(actual).__name__} {actual!r:.60} != expected {type(expected).__name__} {expected!r:.60}"
        )
    elif isinstance(actual, dict):
        for key in sorted(set(actual) | set(expected), key=str):
            if key not in actual or key not in expected:
                out.append(f"{path}.{key}: {'missing' if key not in actual else 'unexpected'}")
            else:
                first_differences(actual[key], expected[key], f"{path}.{key}", out, limit)
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            out.append(f"{path}: length {len(actual)} != expected {len(expected)}")
        for i, (a, e) in enumerate(zip(actual, expected, strict=False)):
            first_differences(a, e, f"{path}[{i}]", out, limit)
    elif actual != expected:
        out.append(f"{path}: {actual!r:.60} != expected {expected!r:.60}")
    return out[:limit]


@pytest.fixture(scope="module")
def work() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _extract(SIMPLE, "gpu/host.0.json", root / "simple" / "host.0.json")
        _extract(SIMPLE, "gpu/device.0.json", root / "simple" / "device.0.json")
        _extract(V1_0_2, "1.0.2-chakra.0.0.4.json", root / "1.0.2" / "host.json")
        for member in (
            "simple_onerank/linked.0.json",
            "simple_onerank/linked.0.et",
            "simple_onerank/host.0.et",
            "1.0.2-chakra.0.0.4/host.et",
        ):
            _extract(REFERENCES, f"reference_outputs/{member}", root / "references" / member)
        yield root


@pytest.fixture(scope="module")
def linked(work: Path) -> Path:
    """Link the simple_onerank traces once for the module (HTA's analysis takes about a second)."""
    out = work / "out" / "linked.0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    TraceLinker().link(0, str(work / "simple" / "host.0.json"), str(work / "simple" / "device.0.json"), str(out))
    return out


def _convert(source: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    PyTorchConverter().convert(str(source), str(out), False)
    return out


def test_linked_trace_matches_the_reference(work: Path, linked: Path) -> None:
    with open(linked) as a, open(work / "references" / "simple_onerank" / "linked.0.json") as b:
        diff = first_differences(json.load(a), json.load(b))
    assert not diff, "chakra_trace_link output differs from the reference:\n" + "\n".join(diff)


def test_linked_trace_converts_to_the_reference_et(work: Path, linked: Path) -> None:
    out = _convert(linked, work / "out" / "linked.0.et")
    diff = first_differences(decode_et(out), decode_et(work / "references" / "simple_onerank" / "linked.0.et"))
    assert not diff, "chakra_converter output on the linked trace differs from the reference:\n" + "\n".join(diff)


@pytest.mark.parametrize(
    ("source", "reference"),
    [("simple/host.0.json", "simple_onerank/host.0.et"), ("1.0.2/host.json", "1.0.2-chakra.0.0.4/host.et")],
    ids=["simple_onerank", "schema_1_0_2"],
)
def test_raw_host_trace_converts_to_the_reference_et(work: Path, source: str, reference: str) -> None:
    out = _convert(work / source, work / "out" / f"{reference.replace('/', '_')}")
    diff = first_differences(decode_et(out), decode_et(work / "references" / reference))
    assert not diff, f"chakra_converter output on {source} differs from the reference:\n" + "\n".join(diff)


def test_linking_does_not_modify_the_parsed_host_trace(work: Path, tmp_path: Path) -> None:
    """The linker builds its output from new node dicts: HostTrace.raw stays exactly as read."""
    traces = []
    real_from_file = HostTrace.from_file

    def from_file(path, *args, **kwargs):
        trace = real_from_file(path, *args, **kwargs)
        traces.append((trace, json.dumps(trace.raw, sort_keys=True)))
        return trace

    with patch.object(HostTrace, "from_file", side_effect=from_file):
        TraceLinker().link(
            0,
            str(work / "simple" / "host.0.json"),
            str(work / "simple" / "device.0.json"),
            str(tmp_path / "linked.json"),
        )
    ((trace, before),) = traces
    assert json.dumps(trace.raw, sort_keys=True) == before


# ---------------------------------------------------------------------------------------------------------------------
# Regenerating the references: only for a deliberate change of output, with the change shown and explained in the PR.

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = (
    "simple_onerank/linked.0.json",
    "simple_onerank/linked.0.et",
    "simple_onerank/host.0.et",
    "1.0.2-chakra.0.0.4/host.et",
)


def _stale_modules() -> List[str]:
    """Return the Chakra modules whose imported file differs from this checkout's (empty: the checkout is tested)."""
    import importlib.util
    import tomllib

    package_dir = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["package-dir"]
    stale = []
    for package, source in package_dir.items():
        for path in sorted((REPO / source).rglob("*.py")):
            if "__pycache__" in path.parts or path.name.endswith(("_pb2.py", "_pb2_grpc.py")):
                continue
            parts = path.relative_to(REPO / source).with_suffix("").parts
            module = ".".join([package, *parts[:-1]] + ([] if parts[-1] == "__init__" else [parts[-1]]))
            spec = importlib.util.find_spec(module)
            if spec is None or not spec.origin or Path(spec.origin).read_bytes() != path.read_bytes():
                stale.append(module)
    return stale


def _flat(value: Any, path: str = "") -> Dict[str, Any]:
    """Flatten JSON into {path: value}. A .et node's attr list becomes attr.<name>, so changes name the attribute."""
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if key == "attr" and isinstance(item, list):
                item = {a.get("name", "?"): {k: v for k, v in a.items() if k != "name"} for a in item}
            out.update(_flat(item, f"{path}.{key}" if path else str(key)))
        return out
    if isinstance(value, list):
        out = {}
        for i, item in enumerate(value):
            out.update(_flat(item, f"{path}[{i}]"))
        return out
    return {path: value}


def _field(path: str) -> str:
    """Return the field a flat path belongs to: attr.<name> for .et attributes, else the first key."""
    parts = path.replace("[", ".[").split(".")
    return ".".join(parts[:2]) if parts[0] == "attr" else parts[0]


def _as_int(node_id: str) -> int:
    return int(node_id) if node_id.lstrip("-").isdigit() else 0


def _ids(ids: List[str]) -> str:
    return ", ".join(sorted(ids, key=_as_int)[:10]) + (" ..." if len(ids) > 10 else "")


def _summary(name: str, old: Path, new: Path) -> List[str]:
    """Return a readable comparison of one reference output, old -> new, for the pull request."""
    if name.endswith(".json"):
        with open(old) as a, open(new) as b:
            before, after = json.load(a), json.load(b)
    else:
        before, after = decode_et(old), decode_et(new)
    if before == after:
        return [f"{name}: unchanged"]
    lines = [f"{name}: CHANGED"]
    top = sorted(k for k in set(before) | set(after) if k != "nodes" and before.get(k) != after.get(k))
    if top:
        lines.append(f"  top level: {', '.join(top)}")
    old_nodes = {str(n.get("id")): _flat(n) for n in before.get("nodes", [])}
    new_nodes = {str(n.get("id")): _flat(n) for n in after.get("nodes", [])}
    added, removed = list(new_nodes.keys() - old_nodes.keys()), list(old_nodes.keys() - new_nodes.keys())
    changed = sorted((i for i in old_nodes.keys() & new_nodes.keys() if old_nodes[i] != new_nodes[i]), key=_as_int)
    lines.append(
        f"  nodes: {len(old_nodes)} -> {len(new_nodes)}; {len(added)} added, {len(removed)} removed, "
        f"{len(changed)} changed"
    )
    fields: Dict[str, int] = {}
    for i in changed:
        a, b = old_nodes[i], new_nodes[i]
        for field in {_field(k) for k in set(a) | set(b) if a.get(k) != b.get(k)}:
            fields[field] = fields.get(field, 0) + 1
    if fields:
        lines.append("  changed fields: " + ", ".join(f"{f} ({n} nodes)" for f, n in sorted(fields.items())))
    for i in changed[:3]:
        a, b = old_nodes[i], new_nodes[i]
        diffs = [
            f"{k}: {a.get(k, '<none>')!r} -> {b.get(k, '<none>')!r}"
            for k in sorted(set(a) | set(b))
            if a.get(k) != b.get(k)
        ]
        lines.append(f"  e.g. node {i}: " + "; ".join(diffs[:2]) + (" ..." if len(diffs) > 2 else ""))
    if added:
        lines.append(f"  added node ids: {_ids(added)}")
    if removed:
        lines.append(f"  removed node ids: {_ids(removed)}")
    return lines


def _git(*args: str) -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _readme(out: Path, previous: str) -> str:
    import datetime
    import hashlib
    import importlib.metadata as md

    versions = ", ".join(f"{d} {md.version(d)}" for d in ("mlc-chakra", "protobuf", "traceinsight"))
    dirty = " (with uncommitted changes)" if _git("status", "--porcelain", "--", "src", "schema") else ""
    sums = "\n".join(f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}" for n in OUTPUTS)
    return f"""Reference outputs for tests/test_pipeline_outputs.py.

Regenerated on {datetime.date.today()} with Chakra {_git("rev-parse", "--short", "HEAD")}{dirty}, Python
{sys.version.split()[0]}, {versions}. The pull request that regenerated them says why.

Previous provenance: {previous}

Inputs: tests/data/simple_onerank_traces.tar.gz (gpu/host.0.json, gpu/device.0.json) and
tests/data/1.0.2-chakra.0.0.4.tgz.

  simple_onerank/linked.0.json   chakra_trace_link --rank 0 on gpu/host.0.json and gpu/device.0.json
  simple_onerank/linked.0.et     chakra_converter PyTorch on linked.0.json
  simple_onerank/host.0.et       chakra_converter PyTorch on gpu/host.0.json
  1.0.2-chakra.0.0.4/host.et     chakra_converter PyTorch on 1.0.2-chakra.0.0.4.json

The tests compare the outputs of the current code with these. They are never regenerated to make a test pass: a
deliberate change of output regenerates them with
  python tests/test_pipeline_outputs.py --regenerate
which prints what changed, to paste into the pull request.
Regenerated references must be approved by a reviewer in the pull request that regenerates them.

SHA-256:
{sums}
"""


def _regenerate() -> None:
    """Rebuild the references from this checkout, print what changed, and rewrite the archive's README."""
    stale = _stale_modules()
    if stale:
        sys.exit("The installed Chakra is not this checkout (reinstall with pip install -e .): " + ", ".join(stale[:5]))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        host = _extract(SIMPLE, "gpu/host.0.json", root / "in" / "host.0.json")
        device = _extract(SIMPLE, "gpu/device.0.json", root / "in" / "device.0.json")
        v102 = _extract(V1_0_2, "1.0.2-chakra.0.0.4.json", root / "in" / "1.0.2.json")
        out = root / "reference_outputs"
        (out / "simple_onerank").mkdir(parents=True)
        TraceLinker().link(0, str(host), str(device), str(out / "simple_onerank" / "linked.0.json"))
        _convert(out / "simple_onerank" / "linked.0.json", out / "simple_onerank" / "linked.0.et")
        _convert(host, out / "simple_onerank" / "host.0.et")
        _convert(v102, out / "1.0.2-chakra.0.0.4" / "host.et")
        for name in OUTPUTS + ("README.txt",):
            _extract(REFERENCES, f"reference_outputs/{name}", root / "old" / name)
        paragraphs = (root / "old" / "README.txt").read_text().split("\n\n")
        starts = ("Recorded with", "Regenerated on")
        previous = next((" ".join(p.split()) for p in paragraphs if p.startswith(starts)), "unknown")
        report = [line for name in OUTPUTS for line in _summary(name, root / "old" / name, out / name)]
        (out / "README.txt").write_text(_readme(out, previous))
        with tarfile.open(REFERENCES, "w:gz") as tar:
            tar.add(out, arcname="reference_outputs")
    print("\n".join(["Changes to the reference outputs (paste into the pull request):", *report]))
    print(f"\nregenerated {REFERENCES}")


if __name__ == "__main__":
    if sys.argv[1:] != ["--regenerate"]:
        sys.exit(f"usage: python {sys.argv[0]} --regenerate   (only for a deliberate change of output)")
    _regenerate()
