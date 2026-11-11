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


def test_uv_rebuilds_the_package_when_a_proto_file_changes() -> None:
    # The generated *_pb2.py files are built into the source tree. By default uv rebuilds the package only when
    # pyproject.toml, setup.py or setup.cfg change, so after an edit of a .proto file the generated code stays old.
    keys = [key.get("file") for key in _pyproject()["tool"]["uv"]["cache-keys"]]
    assert "schema/protobuf/*.proto" in keys
