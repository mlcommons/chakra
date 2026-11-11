"""Checks that the developer tooling files are correct and stay in step with each other."""

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# These tests read files that other branches also change: requirements-dev.txt, the workflows, the PARAM pin.
# `just test` and `just gate` set CHAKRA_TOOLING_TESTS=1, so they run these tests. The existing CI job does not set it,
# so it does not fail an open branch. Phase 3 makes CI call `just`, and then they run there too.
COUPLED = pytest.mark.skipif(
    os.environ.get("CHAKRA_TOOLING_TESTS") != "1",
    reason="set CHAKRA_TOOLING_TESTS=1 (just test does) to check files that other branches change",
)


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _recipes() -> dict[str, list[str]]:
    """Return the justfile recipes as {name: [dependencies]}."""
    recipes = {}
    for line in (ROOT / "justfile").read_text().splitlines():
        match = re.match(r"[A-Za-z][\w-]*", line)
        if match and ":" in line and ":=" not in line and not line.startswith("set "):
            recipes[match.group(0)] = line.rsplit(":", 1)[1].split()
    return recipes


def _recipe_body(name: str) -> str:
    """Return the command lines of one justfile recipe."""
    body: list[str] = []
    inside = False
    for line in (ROOT / "justfile").read_text().splitlines():
        if inside and (line.startswith("    ") or not line.strip()):
            body.append(line.strip())
        elif inside:
            break
        elif re.match(rf"{re.escape(name)}(\s[^:\n]*)?:", line):
            inside = True
    return "\n".join(body).strip()


def _workflow_texts() -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted((ROOT / ".github" / "workflows").glob("*.yml"))}


def _requirements_dev() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "requirements-dev.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, _, version = line.partition("==")
            pins[name.lower()] = version
    return pins


def test_setup_py_does_not_override_build_grpc() -> None:
    # setuptools-grpc adds build_grpc to build by itself. A second copy in setup.py breaks editable installs
    # ("command 'build_grpc' has no such option 'proto_files'").
    text = (ROOT / "setup.py").read_text()
    assert "cmdclass" not in text
    assert "sub_commands" not in text


# Slow: builds the package. It needs uv and network. Run it with CHAKRA_TEST_SLOW=1.
@pytest.mark.skipif(
    os.environ.get("CHAKRA_TEST_SLOW") != "1" or shutil.which("uv") is None,
    reason="set CHAKRA_TEST_SLOW=1 and install uv to run",
)
def test_editable_install_builds_and_imports(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    python = venv / "bin" / "python"
    subprocess.run(["uv", "venv", "-q", "--python", "3.11", str(venv)], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "--python", str(python), "-e", str(ROOT)], check=True)
    subprocess.run([str(python), "-c", "import chakra.schema.protobuf.et_def_pb2"], check=True)


@COUPLED
def test_dev_group_pins_equal_requirements_dev() -> None:
    # Until requirements-dev.txt is retired, the two lists must name the same tools at the same versions.
    group = _pyproject()["dependency-groups"]["dev"]
    pins = {}
    for item in group:
        name, _, version = item.partition("==")
        pins[name.lower()] = version
    assert pins == _requirements_dev()


@COUPLED
def test_uv_python_matches_ci_and_requires_python() -> None:
    # `just` runs uv with this Python. CI uses the same one.
    match = re.search(r'^export UV_PYTHON := "([0-9.]+)"', (ROOT / "justfile").read_text(), re.MULTILINE)
    assert match is not None
    version = match.group(1)
    ci_versions = {
        v for text in _workflow_texts().values() for v in re.findall(r"python-version:\s*['\"]?([0-9.]+)", text)
    }
    assert ci_versions == {version}
    floor = re.search(r">=\s*([0-9.]+)", _pyproject()["project"]["requires-python"])
    assert floor is not None
    assert tuple(map(int, floor.group(1).split("."))) <= tuple(map(int, version.split(".")))


def test_python_version_file_is_not_shipped_yet() -> None:
    # pyenv, mise and asdf read a .python-version file. It would reach people who do not use uv, and could stop
    # their `python` command. `just` pins Python with UV_PYTHON instead. Add the file when the pip setup is retired,
    # and delete this test then.
    assert not (ROOT / ".python-version").exists()


def test_uv_installs_the_dev_group_by_default() -> None:
    assert "dev" in _pyproject()["tool"]["uv"]["default-groups"]


def test_gitignore_covers_uv_environment_and_lock() -> None:
    # Phase 1 does not commit uv.lock (issues #237 and #254 still change the dependencies). Phase 2 removes this line.
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()}
    assert {".venv/", "uv.lock"} <= lines


@COUPLED
def test_param_pin_in_pyproject_equals_the_one_in_ci() -> None:
    # Temporary: PARAM (et_replay) is installed from git at one commit. Issue #237 vendors it and removes all of this.
    source = _pyproject()["tool"]["uv"]["sources"]["et-replay"]
    assert "et-replay" in "".join(_pyproject()["dependency-groups"]["param"])
    ci_commits = {c for text in _workflow_texts().values() for c in re.findall(r"git checkout ([0-9a-f]{40})", text)}
    assert ci_commits <= {source["rev"]}
    assert ci_commits, "CI no longer pins PARAM: delete the param group and this test"


REQUIRED_RECIPES = {
    "setup",
    "fmt",
    "lint",
    "types",
    "test",
    "gate",
    "cpp-fmt-check",
    "lint-scripts",
    "wheel-check",
    "ci-github-lint",
}


def test_justfile_has_the_required_recipes() -> None:
    assert set(_recipes()) >= REQUIRED_RECIPES


def test_gate_runs_every_check() -> None:
    assert {"lint", "lint-scripts", "types", "test", "cpp-fmt-check"} <= set(_recipes()["gate"])


def test_every_recipe_has_a_comment() -> None:
    lines = (ROOT / "justfile").read_text().splitlines()
    for index, line in enumerate(lines):
        if (
            re.match(r"[A-Za-z][\w-]*", line)
            and ":" in line
            and ":=" not in line
            and not line.startswith(("set ", "default"))
        ):
            assert index > 0
            above = index - 1
            while lines[above].startswith("["):  # an attribute such as [positional-arguments]
                above -= 1
            assert lines[above].startswith("#"), f"recipe has no comment: {line}"


def test_test_recipe_uses_python_dash_m_pytest() -> None:
    # Three test files import `src.*`. Plain pytest cannot find it: it needs the repo root on sys.path.
    assert "python -m pytest" in _recipe_body("test")


def test_lint_recipe_checks_formatting() -> None:
    # CI runs `ruff format .` without --check, so formatting never fails there. The justfile must check it.
    assert "ruff format --check" in _recipe_body("lint")


@COUPLED
def test_justfile_runs_the_commands_that_ci_runs() -> None:
    ci = "\n".join(_workflow_texts().values())
    recipe_for = {
        "ruff check .": "lint",
        "vulture src/ tests/": "lint",
        "pyright": "types",
        "python -m pytest -vv tests": "test",
    }
    for command, recipe in recipe_for.items():
        assert command in ci, f"CI no longer runs: {command}"
        assert command in _recipe_body(recipe), f"just {recipe} does not run: {command}"
    version = re.search(r"clangFormatVersion:\s*(\d+)", ci)
    assert version is not None
    assert f"clang-format=={version.group(1)}" in _recipe_body("cpp-fmt-check")


def test_types_recipe_matches_ci_and_skips_the_venv() -> None:
    # CI runs pyright where only the dev tools exist. With the project installed, pyright also reports issues that CI
    # never sees. A full-environment check belongs to the later hardening work, not to this gate.
    # Pyright must also not scan the project's .venv (thousands of files, over a minute): give it the tracked files.
    body = _recipe_body("types")
    assert "--only-group dev" in body
    assert "scripts/run_on_git_files.sh '*.py' --" in body


def test_actrc_maps_ubuntu_latest_and_stays_portable() -> None:
    text = (ROOT / ".actrc").read_text()
    assert "-P ubuntu-latest=catthehacker/ubuntu:act-24.04" in text
    # The CPU type and the Docker socket differ between engineers. scripts/act.sh finds them at run time.
    assert "--container-architecture" not in text
    assert "DOCKER_HOST" not in text


def test_act_recipe_default_workflow_exists() -> None:
    match = re.search(r'^act workflow="([^"]+)":', (ROOT / "justfile").read_text(), re.MULTILINE)
    assert match is not None
    assert (ROOT / ".github" / "workflows" / match.group(1)).is_file()


def test_gitignore_covers_act_secrets() -> None:
    # `act --secret-file .secrets` reads secrets from this file. It must never be committed.
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()}
    assert {".secrets", ".env", ".vars"} <= lines


def test_file_list_recipes_use_the_git_files_script() -> None:
    # The script includes new files that git does not track yet, and stops on an empty list (tests/test_scripts.py).
    for name in ("types", "cpp-fmt-check"):
        assert "scripts/run_on_git_files.sh" in _recipe_body(name)


def test_test_recipe_passes_arguments_with_spaces_through() -> None:
    justfile = (ROOT / "justfile").read_text()
    assert "[positional-arguments]" in justfile
    assert '"$@"' in _recipe_body("test")


def test_gate_includes_the_editable_install_and_protobuf_checks() -> None:
    # The gate checks both ways to install the package: editable (what `uv sync` does) and the built wheel (what
    # `pip install .` and CI use).
    assert {"test-slow", "proto-smoke", "wheel-check"} <= set(_recipes()["gate"])
    assert "test_editable_install_builds_and_imports" in _recipe_body("test-slow")
    assert "test_wheel_check_catches_a_file_that_is_missing_from_the_wheel" in _recipe_body("test-slow")
    assert _recipe_body("proto-smoke") == "scripts/proto_smoke.sh"
    assert _recipe_body("wheel-check") == "scripts/wheel_check.sh"


def test_lint_scripts_recipe_runs_shellcheck_on_every_script() -> None:
    assert "shellcheck-py==0.11.0.1 shellcheck scripts/*.sh" in _recipe_body("lint-scripts")


def test_act_recipe_calls_the_script() -> None:
    assert _recipe_body("act") == "scripts/act.sh {{quote(workflow)}}"  # quoted: no shell injection


def test_justfile_comments_point_only_to_files_that_exist() -> None:
    # A comment must not point to a file that is not in the repository (a doc that is not written yet, local notes).
    comments = "\n".join(line for line in (ROOT / "justfile").read_text().splitlines() if line.lstrip().startswith("#"))
    for path in re.findall(r"\b(?:docs?|tasks|scripts)/[\w./-]+", comments):
        assert (ROOT / path).exists(), f"justfile comment points to a missing file: {path}"


def test_uv_rebuilds_the_package_when_a_proto_file_changes() -> None:
    # The generated *_pb2.py files are built into the source tree. By default uv rebuilds the package only when
    # pyproject.toml, setup.py or setup.cfg change, so after an edit of a .proto file the generated code stays old.
    keys = [key.get("file") for key in _pyproject()["tool"]["uv"]["cache-keys"]]
    assert "schema/protobuf/*.proto" in keys


def test_test_recipe_enables_the_tooling_tests() -> None:
    assert "CHAKRA_TOOLING_TESTS=1" in _recipe_body("test")


def test_gitignore_covers_generated_protobuf_code() -> None:
    # The C++ feeder build runs protoc into schema/protobuf. The file lists include untracked files (as CI does), so
    # this generated code must be ignored, or `just cpp-fmt-check` checks files that CI never sees.
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()}
    assert {"*.pb.cc", "*.pb.h"} <= lines
