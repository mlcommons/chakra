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


def _section(text: str, heading: str) -> str:
    """Return the text under a markdown heading, up to the next heading of the same or a higher level."""
    level = len(heading) - len(heading.lstrip("#"))
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == heading), None)
    assert start is not None, f"no section {heading!r}"
    body: list[str] = []
    in_code = False
    for line in lines[start + 1 :]:
        if line.startswith("```"):
            in_code = not in_code
        elif not in_code and re.match(rf"#{{1,{level}}} ", line):
            break
        body.append(line)
    return "\n".join(body) + "\n"


def _prose_sentences(text: str) -> list[str]:
    """Return the sentences of a markdown text, without code blocks, tables, diagrams, links and inline code."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`[^`]*`", "code", text)
    lines = [
        line.strip(" -*") for line in text.splitlines() if line.strip() and not line.lstrip().startswith(("|", "#"))
    ]
    return [s.strip() for s in re.split(r"(?<=[.!?:])\s+", " ".join(lines)) if s.strip()]


def test_readme_has_a_short_development_section() -> None:
    # The README stays short. It gives the quick start and points to the guide.
    readme = (ROOT / "README.md").read_text()
    section = _section(readme, "## Development")
    assert len(section.strip().splitlines()) <= 15
    assert "just setup" in section
    assert "just gate" in section
    assert "(doc/DEVELOPMENT.md)" in section
    assert "pip setup still works" in section  # nobody has to move to uv


def test_relative_links_in_the_docs_resolve() -> None:
    for name in ("README.md", "USER_GUIDE.md", "doc/DEVELOPMENT.md"):
        text = (ROOT / name).read_text()
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
            if not target.startswith(("http://", "https://", "mailto:")):
                assert (ROOT / Path(name).parent / target).exists(), f"{name} links to a missing file: {target}"


def test_docs_name_only_recipes_that_exist() -> None:
    recipes = set(_recipes())
    for name in ("README.md", "USER_GUIDE.md", "doc/DEVELOPMENT.md"):
        text = (ROOT / name).read_text()
        for recipe in re.findall(r"(?:^|`|\$ )just ([a-z][\w-]*)", text, re.MULTILINE):
            assert recipe in recipes, f"{name} names a recipe that does not exist: just {recipe}"


def test_new_docs_use_short_sentences() -> None:
    # A simple check for Simplified Technical English: at most 25 words in a sentence.
    readme = (ROOT / "README.md").read_text()
    user_guide = (ROOT / "USER_GUIDE.md").read_text()
    texts = {
        "doc/DEVELOPMENT.md": (ROOT / "doc/DEVELOPMENT.md").read_text(),
        "README.md Development": _section(readme, "## Development"),
        "USER_GUIDE.md uv": _section(user_guide, "### Alternative: install with uv"),
    }
    for name, text in texts.items():
        for sentence in _prose_sentences(text):
            assert len(sentence.split()) <= 25, f"{name}: sentence has {len(sentence.split())} words: {sentence}"


@COUPLED
def test_docs_use_the_same_param_commit_as_pyproject() -> None:
    commit = _pyproject()["tool"]["uv"]["sources"]["et-replay"]["rev"]
    user_guide = _section((ROOT / "USER_GUIDE.md").read_text(), "### Alternative: install with uv")
    for command in ("uv venv", "uv pip install mlc-chakra"):
        assert command in user_guide
    assert commit in user_guide
    assert commit in _section((ROOT / "doc" / "DEVELOPMENT.md").read_text(), "## The classic setup")


def test_uv_rebuilds_the_package_when_a_proto_file_changes() -> None:
    # The generated *_pb2.py files are built into the source tree. By default uv rebuilds the package only when
    # pyproject.toml, setup.py or setup.cfg change, so after an edit of a .proto file the generated code stays old.
    keys = [key.get("file") for key in _pyproject()["tool"]["uv"]["cache-keys"]]
    assert "schema/protobuf/*.proto" in keys


def test_test_recipe_enables_the_tooling_tests() -> None:
    assert "CHAKRA_TOOLING_TESTS=1" in _recipe_body("test")


def test_developer_guide_lists_every_recipe() -> None:
    guide = (ROOT / "doc" / "DEVELOPMENT.md").read_text()
    for recipe in set(_recipes()) - {"default"}:
        assert f"`just {recipe}`" in guide, f"doc/DEVELOPMENT.md does not list: just {recipe}"


def test_developer_guide_describes_the_shell_scripts() -> None:
    section = _section((ROOT / "doc" / "DEVELOPMENT.md").read_text(), "## Shell scripts")
    for text in ("scripts/", "just lint-scripts", "tests/test_scripts.py"):
        assert text in section


def test_gitignore_covers_generated_protobuf_code() -> None:
    # The C++ feeder build runs protoc into schema/protobuf. The file lists include untracked files (as CI does), so
    # this generated code must be ignored, or `just cpp-fmt-check` checks files that CI never sees.
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()}
    assert {"*.pb.cc", "*.pb.h"} <= lines


def test_section_ignores_comment_lines_in_code_blocks() -> None:
    text = "## A\n\n```bash\n# a comment\ncommand\n```\n\nafter the block\n\n## B\n\nother\n"
    section = _section(text, "## A")
    assert "after the block" in section
    assert "other" not in section


def test_classic_setup_installs_param_before_it_runs_the_tests() -> None:
    # The tests import et_replay (PARAM). Without it, pytest cannot collect them. The setup also uses Python 3.11 and
    # runs the lint commands that the Python Lint workflow runs.
    section = _section((ROOT / "doc" / "DEVELOPMENT.md").read_text(), "## The classic setup")
    assert section.index("facebookresearch/param") < section.index("python -m pytest")
    assert "python3.11 -m venv" in section
    for command in ("ruff check .", "pyright", "vulture src/ tests/"):
        assert command in section


def test_docs_do_not_promise_the_end_of_the_pip_setup() -> None:
    # The owner decides when the pip setup ends, and says so in the repository. A guide must not announce it.
    for name in ("README.md", "doc/DEVELOPMENT.md"):
        assert "retire" not in (ROOT / name).read_text().lower(), f"{name} announces the end of the pip setup"


def test_developer_guide_uses_one_word_for_one_meaning() -> None:
    # "workflow" is a file in .github/workflows/. The daily steps are not a workflow.
    guide = (ROOT / "doc" / "DEVELOPMENT.md").read_text()
    assert "## Daily steps" in guide
    assert "Daily workflow" not in guide
    assert "daily workflow" not in (ROOT / "README.md").read_text()
