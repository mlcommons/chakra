"""Behaviour tests for the shell scripts in scripts/.

Each script runs with fake programs (docker, act, uv, uname, pgrep) first on PATH. A fake program writes its
arguments to a log, so a test can check what the script did, in what order, without Docker or network.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def _clean_env() -> dict[str, str]:
    """Return the environment without the git settings of the caller. A git hook sets GIT_INDEX_FILE, for example."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})
    return env


def _run(
    script: str,
    args: list[str],
    tmp_path: Path,
    stubs: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    cwd: Path = ROOT,
    only_tools: tuple[str, ...] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run a script with fake programs first on PATH. Return the result and the log of the fake-program calls.

    With only_tools, PATH holds nothing but the fake programs and these real programs.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for tool in only_tools or ():
        (bin_dir / tool).symlink_to(shutil.which(tool) or tool)
    log = tmp_path / "calls.log"
    log.touch()
    for name, body in (stubs or {}).items():
        stub = bin_dir / name
        stub.write_text(f'#!/usr/bin/env bash\necho "{name} $*" >> "$STUB_LOG"\n{body}\n')
        stub.chmod(0o755)
    path = str(bin_dir) if only_tools is not None else f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
    run_env = {**_clean_env(), "PATH": path, "STUB_LOG": str(log)}
    run_env.pop("DOCKER_HOST", None)
    run_env.update(env or {})
    result = subprocess.run(
        [str(SCRIPTS / script), *args], capture_output=True, text=True, env=run_env, cwd=cwd, check=False
    )
    return result, log.read_text().splitlines()


def test_every_script_is_an_executable_bash_script() -> None:
    scripts = sorted(SCRIPTS.glob("*.sh"))
    assert scripts
    for script in scripts:
        text = script.read_text()
        assert text.startswith("#!/usr/bin/env bash\n"), f"{script.name}: no bash shebang"
        assert "set -euo pipefail" in text, f"{script.name}: no strict mode"
        assert os.access(script, os.X_OK), f"{script.name}: not executable"


UV_STUB = """case "$*" in
    *"${STUB_FAIL_ON:-NEVER_MATCHES}"*) exit 1 ;;
    *importlib.metadata*) echo 6.31.1 ;;
esac"""


def test_proto_smoke_imports_with_the_newest_and_then_the_lowest_protobuf(tmp_path: Path) -> None:
    result, calls = _run("proto_smoke.sh", [], tmp_path, {"uv": UV_STUB})
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    assert calls[0].startswith("uv run python -c import chakra.schema.protobuf.et_def_pb2")
    assert "importlib.metadata" in calls[1]
    assert calls[2].startswith("uv run --with protobuf==6.31.1 python -c import chakra.schema.protobuf.et_def_pb2")


def test_proto_smoke_fails_when_the_import_with_the_lowest_protobuf_fails(tmp_path: Path) -> None:
    result, _ = _run("proto_smoke.sh", [], tmp_path, {"uv": UV_STUB}, {"STUB_FAIL_ON": "--with"})
    assert result.returncode != 0


def test_proto_smoke_stops_at_the_first_failing_import(tmp_path: Path) -> None:
    result, calls = _run("proto_smoke.sh", [], tmp_path, {"uv": UV_STUB}, {"STUB_FAIL_ON": "import chakra"})
    assert result.returncode != 0
    assert len(calls) == 1


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=_clean_env())

    git("init", "-q")
    (repo / ".gitignore").write_text("ignored.py\n")
    for name in ("tracked.py", "gone.py", "with space.py", "-x.py", "other.txt"):
        (repo / name).write_text("x\n")
    git("add", "-A")  # the index is enough: `git ls-files --cached` reads it
    (repo / "gone.py").unlink()
    (repo / "new.py").write_text("x\n")
    (repo / "ignored.py").write_text("x\n")
    (repo / "somedir").mkdir()
    (repo / "link.py").symlink_to("somedir")  # a link to a directory is not a file to check
    return repo


def test_run_on_git_files_passes_tracked_and_new_files_only(tmp_path: Path) -> None:
    # New file: yes (CI checks it too). Ignored file: no. Deleted file: no. A link to a directory: no.
    # A name with a space stays one argument. The ./ in front keeps the name -x.py from looking like an option.
    repo = _git_repo(tmp_path)
    result, _ = _run("run_on_git_files.sh", ["*.py", "--", "printf", "%s\\n"], tmp_path, cwd=repo)
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == {"./tracked.py", "./new.py", "./with space.py", "./-x.py"}


def test_git_helpers_ignore_the_git_settings_of_the_caller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A git hook sets GIT_INDEX_FILE, and a global config can force commit signing. Neither may reach these tests.
    outer_index = tmp_path / "outer-index"
    config = tmp_path / "gitconfig"
    config.write_text("[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = false\n")
    monkeypatch.setenv("GIT_INDEX_FILE", str(outer_index))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    repo = _git_repo(tmp_path)
    result, _ = _run("run_on_git_files.sh", ["*.py", "--", "true"], tmp_path, cwd=repo)
    assert result.returncode == 0, result.stderr
    assert not outer_index.exists()


def test_run_on_git_files_stops_when_no_file_matches(tmp_path: Path) -> None:
    result, _ = _run("run_on_git_files.sh", ["*.rs", "--", "true"], tmp_path, cwd=_git_repo(tmp_path))
    assert result.returncode != 0
    assert "no files" in result.stderr


def test_run_on_git_files_returns_the_exit_code_of_the_command(tmp_path: Path) -> None:
    result, _ = _run("run_on_git_files.sh", ["*.py", "--", "false"], tmp_path, cwd=_git_repo(tmp_path))
    assert result.returncode != 0


@pytest.mark.parametrize("args", [[], ["*.py"], ["--", "true"], ["*.py", "--"]])
def test_run_on_git_files_needs_patterns_and_a_command(tmp_path: Path, args: list[str]) -> None:
    result, _ = _run("run_on_git_files.sh", args, tmp_path, cwd=tmp_path)
    assert result.returncode == 2
    assert "usage" in result.stderr


# Slow: builds the package. It needs uv and a network. Run it with CHAKRA_TEST_SLOW=1 (`just test-slow`).
@pytest.mark.skipif(
    os.environ.get("CHAKRA_TEST_SLOW") != "1" or shutil.which("uv") is None,
    reason="set CHAKRA_TEST_SLOW=1 and install uv to run",
)
def test_wheel_check_catches_a_file_that_is_missing_from_the_wheel(tmp_path: Path) -> None:
    # `uv sync` installs the project in editable mode, and an editable install finds a file that the wheel does not
    # hold. Here the .proto files are left out of the package data. Only a check of the built wheel sees it.
    copy = tmp_path / "copy"
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        env=_clean_env(),
    ).stdout
    for name in filter(None, listed.decode().split("\0")):
        if (ROOT / name).is_file():
            (copy / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, copy / name)
    subprocess.run(["git", "init", "-q"], cwd=copy, check=True, env=_clean_env())
    subprocess.run(["git", "add", "-A"], cwd=copy, check=True, env=_clean_env())
    pyproject = copy / "pyproject.toml"
    package_data = '"chakra.schema.protobuf" = ["et_def.proto", "storage.proto"]'
    assert package_data in pyproject.read_text()
    pyproject.write_text(pyproject.read_text().replace(package_data, '"chakra.schema.protobuf" = []'))
    result = subprocess.run(
        [str(copy / "scripts" / "wheel_check.sh")], capture_output=True, text=True, check=False, env=_clean_env()
    )
    assert result.returncode != 0
    assert "et_def.proto" in result.stdout + result.stderr
    # The wheel is built from a clean copy, as in CI. A stale build/ folder cannot leak in, and the repo stays clean.
    assert not (copy / "build").exists()


def test_wheel_check_stops_with_a_clear_message_outside_a_git_checkout(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SCRIPTS / "wheel_check.sh", tmp_path / "scripts" / "wheel_check.sh")
    result = subprocess.run(
        [str(tmp_path / "scripts" / "wheel_check.sh")], capture_output=True, text=True, check=False, env=_clean_env()
    )
    assert result.returncode != 0
    assert "git checkout" in result.stderr
