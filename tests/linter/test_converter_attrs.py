"""MET001, MET002, ATT001 and ATT002 check converter metadata and attributes."""

from pathlib import Path

import linter
from chakra.schema.protobuf.et_def_pb2 import AttributeProto, GlobalMetadata, Node
from chakra.src.third_party.utils.protolib import encodeMessage

Attr = AttributeProto
META = {
    "schema": Attr(name="schema", string_val="1.0.2-chakra.0.0.4"),
    "pid": Attr(name="pid", uint64_val=1),
    "time": Attr(name="time", string_val="now"),
    "start_ts": Attr(name="start_ts", uint64_val=10),
    "finish_ts": Attr(name="finish_ts", uint64_val=20),
}


def cpu_attrs() -> list[Attr]:
    """Return the attributes the converter writes on a CPU node."""
    ints = ("rf_id", "fw_parent", "seq_id", "scope", "tid", "fw_tid")
    return [
        *(Attr(name=n, int64_val=1) for n in ints),
        Attr(name="op_schema", string_val="aten::add() -> Tensor"),
        Attr(name="is_cpu_op", bool_val=True),
    ]


def write(path: Path, meta: list[Attr], attrs: list[Attr]) -> Path:
    """Write a one-node trace with the given metadata and node attributes."""
    with path.open("wb") as handle:
        encodeMessage(handle, GlobalMetadata(attr=meta))
        node = Node(id=1, name="aten::add", type=4, duration_micros=5)
        node.attr.extend(attrs)
        encodeMessage(handle, node)
    return path


def rules(path: Path) -> list[str]:
    """Return the rule IDs of the gaps found in a trace."""
    return [g.rule for g in linter.lint_trace(path).gaps]


def test_complete_trace_triggers_nothing(tmp_path) -> None:
    path = write(tmp_path / "ok.et", list(META.values()), cpu_attrs())
    assert not {"MET001", "MET002", "ATT001", "ATT002"} & set(rules(path))


def test_met001_names_missing_metadata(tmp_path) -> None:
    meta = [a for n, a in META.items() if n != "pid"]
    result = linter.lint_trace(write(tmp_path / "m.et", meta, cpu_attrs()))
    gap = next(g for g in result.gaps if g.rule == "MET001")
    assert "pid" in gap.message and gap.message.total == 1


def test_met002_flags_reversed_timestamps(tmp_path) -> None:
    meta = [*META.values(), Attr(name="start_ts", uint64_val=99)]
    meta = [a for a in meta if a.name != "start_ts" or a.uint64_val == 99]
    assert "MET002" in rules(write(tmp_path / "t.et", meta, cpu_attrs()))


def test_att001_reports_one_gap_per_missing_attribute(tmp_path) -> None:
    attrs = [a for a in cpu_attrs() if a.name not in ("rf_id", "op_schema")]
    result = linter.lint_trace(write(tmp_path / "a.et", list(META.values()), attrs))
    found = [g.message for g in result.gaps if g.rule == "ATT001"]
    assert len(found) == 2
    assert any("rf_id" in m for m in found) and any("op_schema" in m for m in found)


def test_att001_exempts_device_nodes_but_not_is_cpu_op(tmp_path) -> None:
    device = [Attr(name="is_cpu_op", bool_val=False), Attr(name="tid", int64_val=1)]
    meta = list(META.values())
    assert "ATT001" not in rules(write(tmp_path / "d.et", meta, device))
    assert "ATT001" in rules(write(tmp_path / "n.et", meta, device[1:]))


def test_att001_and_met_skip_synthetic_traces(tmp_path) -> None:
    found = rules(write(tmp_path / "s.et", [], []))
    assert not {"MET001", "ATT001"} & set(found)


def test_att002_flags_wrong_type_and_repeated_names(tmp_path) -> None:
    meta = list(META.values())
    wrong = [a for a in cpu_attrs() if a.name != "tid"]
    wrong.append(Attr(name="tid", uint64_val=1))
    assert "ATT002" in rules(write(tmp_path / "w.et", meta, wrong))
    repeat = [*cpu_attrs(), Attr(name="tid", int64_val=2)]
    assert "ATT002" in rules(write(tmp_path / "r.et", meta, repeat))


def attrs_with(**values: int) -> list[Attr]:
    """Return CPU attributes with the named integer values replaced."""
    base = [a for a in cpu_attrs() if a.name not in values]
    return [*base, *(Attr(name=n, int64_val=v) for n, v in values.items())]


def test_att003_flags_rf_id_zero_but_exempts_thread_roots(tmp_path) -> None:
    meta = list(META.values())
    assert "ATT003" in rules(write(tmp_path / "z.et", meta, attrs_with(rf_id=0)))
    assert "ATT003" not in rules(write(tmp_path / "ok.et", meta, attrs_with(rf_id=2)))


def test_att004_flags_scope_outside_record_scope(tmp_path) -> None:
    meta = list(META.values())
    assert "ATT004" in rules(write(tmp_path / "s.et", meta, attrs_with(scope=10)))
    assert "ATT004" not in rules(write(tmp_path / "v.et", meta, attrs_with(scope=9)))


def test_att005_flags_backward_node_without_forward_parent(tmp_path) -> None:
    meta = list(META.values())
    bad = attrs_with(fw_tid=5, fw_parent=0)
    assert "ATT005" in rules(write(tmp_path / "b.et", meta, bad))
    assert "ATT005" not in rules(write(tmp_path / "f.et", meta, attrs_with(fw_tid=0)))
