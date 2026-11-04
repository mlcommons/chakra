"""BL-15: the shipped rule file matches the code and the CLI."""

from pathlib import Path

import linter
import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "linter"


@pytest.fixture(scope="module")
def shipped_rules() -> dict[str, linter.Rule]:
    return linter.load_rules(linter.DEFAULT_RULES)


def test_rule_file_matches_code(shipped_rules) -> None:
    assert sorted(shipped_rules) == sorted(linter.RULE_PLACEHOLDERS)


def test_list_rules_shows_state(shipped_rules, capsys) -> None:
    argv = ["--list-rules", "--disable-rule", "DAG005"]
    assert linter.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(shipped_rules) + 1
    states = {ln.split()[0]: ln.split()[-1] for ln in lines[:-1]}
    assert states["DAG005"] == "disabled"
    assert states["INT001"] == "enabled"
    assert lines[-1] == f"{len(shipped_rules)} rules, 1 disabled"


def test_paths_required_without_list_rules() -> None:
    with pytest.raises(SystemExit) as exc:
        linter.main([])
    assert exc.value.code == 2


def test_kineto_no_stream_fires_only_without_stream_data() -> None:
    """Empty stream data reports the rule; any stream data does not."""
    samples = {"no_stream": [5]}
    none = linter.check_kineto_link(linter.Counter(), 0, {}, {}, samples)
    assert [i.rule for i in none] == ["KIN001"]
    assert none[0].ids == [5]
    some = linter.check_kineto_link(linter.Counter({7: 3}), 3, {}, {}, {})
    assert "KIN001" not in [i.rule for i in some]


def test_report_links_every_rule_to_a_description(shipped_rules) -> None:
    """Rule IDs in both report formats link to a heading in the Rules list."""
    trace = ROOT / "missing.et"
    result = linter.TraceResult()
    result.readable = False
    result.problems = ["cannot decode"]
    for as_html in (False, True):
        text = linter.render_report([(trace, result)], shipped_rules, as_html=as_html)
        assert "#### INT001" in text
        assert "#### SUI001" in text
        assert "(#INT001)" in text or 'href="#INT001"' in text


def test_selectors_accept_name_and_prefix(shipped_rules) -> None:
    by_name = linter.disabled_rules(shipped_rules, disable=["dag-cycle"])
    assert by_name == {"DAG005"}
    by_prefix = linter.disabled_rules(shipped_rules, disable=["DAG"])
    assert by_prefix == {f"DAG00{i}" for i in range(1, 6)}


def test_rule_ids_and_names_are_unique(shipped_rules) -> None:
    names = [rule.name for rule in shipped_rules.values()]
    assert len(names) == len(set(names))
