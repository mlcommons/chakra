"""Replay checks target childless CPU operators, not GPU kernels."""

from pathlib import Path

import linter


def test_exempt_matcher_uses_shell_patterns() -> None:
    skip = linter.exempt_matcher(["nccl:*", "record_param_comms"])
    assert skip("nccl:all_reduce")
    assert skip("record_param_comms")
    assert not skip("aten::add")
    assert not linter.exempt_matcher([])("anything")


def test_shipped_exempt_list_covers_wrappers() -> None:
    skip = linter.exempt_matcher(linter.default_exempt())
    names = ("TorchDynamo Cache Lookup", "MmBackward0", "triton_poi_fused_0")
    for name in (*names, "detach", "_Linear", "GeLUFunction", "ProfilerStep#3"):
        assert skip(name), name
    for name in ("aten::add", "aten::detach", "my_custom_op"):
        assert not skip(name), name


def test_op_family_groups_generically() -> None:
    assert linter.op_family("void (anonymous namespace)::rope<float>(int)") == "rope"
    assert linter.op_family("r0_L1_attn") == "r#_L#_attn"
    assert linter.op_family("at::native::add") == "at"


def test_op_family_strips_tags_and_truncates() -> None:
    noisy = "batch_id:3; item_indices:[1, 2, 3]; worker: 10"
    assert linter.op_family(noisy) == "batch_id"
    assert linter.op_family("ProfilerStep#12") == "ProfilerStep"
    assert linter.op_family("Optimizer.step#FusedAdam.step") == "Optimizer.step"
    assert len(linter.op_family("x" * 200)) == linter.FAMILY_LABEL_LIMIT


def test_family_tally_entries_fold_the_tail() -> None:
    tally = linter.FamilyTally()
    for node_id in (7, 8):
        tally.add("detach", node_id)
    tally.add("enumerate(DataLoader)#x", 9)
    assert tally.entries() == [("detach", 2, 7), ("enumerate", 1, 9)]
    for index in range(linter.FAMILY_LIST_LIMIT + 2):
        tally.add(f"op{'x' * index}", 100 + index)
    last = tally.entries()[-1]
    assert last[0] == "4 other names" and last[1] == 4


def test_schema_ancestor_marks_subtree_of_a_unit() -> None:
    # 1 (no schema) > 2 (schema) > 3 (no schema) > 4 (no schema)
    ctrl = {1: (0,), 2: (1,), 3: (2,), 4: (3,)}
    cpu, schema, under = {1, 2, 3, 4}, {2}, {}

    def check(n):
        return linter.has_schema_ancestor(n, ctrl, cpu, schema, under)

    assert not check(1)  # a root wrapper
    assert not check(2)  # the unit itself
    assert check(3)
    assert check(4)


def test_collect_traces_recurses_only_on_request(tmp_path: Path) -> None:
    (tmp_path / "a.et").touch()
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    (tmp_path / "sub" / "deep" / "b.et").touch()
    (tmp_path / "sub" / "note.txt").touch()
    flat = linter.collect_traces([tmp_path])
    deep = linter.collect_traces([tmp_path], recursive=True)
    assert [p.name for p in flat] == ["a.et"]
    assert sorted(p.name for p in deep) == ["a.et", "b.et"]
