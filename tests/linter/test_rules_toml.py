"""The shipped rules.toml meets the criteria stated in its header comments."""

import ast
import json
import string
import subprocess
import tomllib
from pathlib import Path

import jsonschema
import linter
import pytest

RULES_FILE = linter.DEFAULT_RULES
SCHEMA_FILE = RULES_FILE.with_name("rules.schema.json")
EXEMPT_OWNER = "OPS001"


@pytest.fixture(scope="module")
def entries() -> list[dict]:
    """Return the raw [[rule]] tables, bypassing the loader's validation."""
    with RULES_FILE.open("rb") as handle:
        return tomllib.load(handle)["rule"]


def test_file_conforms_to_schema() -> None:
    schema = json.loads(SCHEMA_FILE.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    with RULES_FILE.open("rb") as handle:
        jsonschema.validate(tomllib.load(handle), schema)


def test_id_and_name_are_unique(entries) -> None:
    for key in ("id", "name"):
        values = [entry[key] for entry in entries]
        assert len(values) == len(set(values)), f"duplicate {key}"


def test_prefix_groups_are_contiguous_and_sequential(entries) -> None:
    """Rules sharing an ID prefix stay adjacent and numbered from 001."""
    seen: dict[str, int] = {}
    previous = None
    for entry in entries:
        prefix, number = entry["id"][:-3], int(entry["id"][-3:])
        if prefix != previous:
            assert prefix not in seen, f"{prefix} group is split"
        assert number == seen.get(prefix, 0) + 1, f"{entry['id']} breaks sequence"
        seen[prefix], previous = number, prefix


def test_exempt_belongs_to_one_rule_only(entries) -> None:
    for entry in entries:
        if "exempt" not in entry:
            continue
        assert entry["id"] == EXEMPT_OWNER


def test_ids_never_reused_for_a_different_rule(entries) -> None:
    """Every ID in the committed file keeps its name, or is retired."""
    proc = subprocess.run(
        ["git", "show", f"HEAD:{_repo_path(RULES_FILE)}"],
        capture_output=True,
        cwd=RULES_FILE.parent,
    )
    if proc.returncode != 0:
        pytest.skip("rules.toml has no committed version")
    old = {e["id"]: e["name"] for e in tomllib.loads(proc.stdout.decode())["rule"]}
    new = {e["id"]: e["name"] for e in entries}
    changed = {i for i in old.keys() & new.keys() if old[i] != new[i]}
    assert not changed, f"ID reused for a different rule: {sorted(changed)}"


def _repo_path(path: Path) -> str:
    """Return the repository-relative path that git show requires."""
    out = subprocess.run(
        ["git", "ls-files", "--full-name", str(path)],
        capture_output=True,
        text=True,
        cwd=path.parent,
    )
    return out.stdout.strip() or path.name


def _master() -> set[str]:
    """Return the placeholder names that the schema declares."""
    schema = json.loads(SCHEMA_FILE.read_text())
    return set(schema["x-placeholders"]) - {"$comment"}


def _used_in_code() -> set[str]:
    """Return keyword names that linter.py passes to render and issue."""
    tree = ast.parse(Path(linter.__file__).read_text())
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in {"render", "issue"}:
            used |= {kw.arg for kw in node.keywords if kw.arg}
        # The shared zero-duration helper passes its values as a dict
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "ratio":
            used |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    return used


def test_schema_describes_every_placeholder() -> None:
    schema = json.loads(SCHEMA_FILE.read_text())
    for name in _master():
        text = schema["x-placeholders"][name]
        assert isinstance(text, str) and text.endswith("."), f"{name}: no description"


def test_schema_placeholders_are_the_source_of_truth(entries) -> None:
    """Code and messages use only schema names; every name is in use."""
    master = _master()
    in_messages = {
        name
        for entry in entries
        for _, name, _, _ in string.Formatter().parse(entry["message"])
        if name is not None
    }
    used = _used_in_code()
    assert used <= master, "code supplies a placeholder missing from the schema"
    assert in_messages <= master, "rules.toml uses a placeholder missing from the schema"
    assert master <= used, "schema lists a placeholder that no check supplies"


def test_messages_render_with_their_checks_values(entries) -> None:
    """Every shipped message renders once given values for each name."""
    for entry in entries:
        entry["message"].format(**dict.fromkeys(_master(), "0"))
