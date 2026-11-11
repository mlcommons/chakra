"""Checks that the developer tooling files are correct and stay in step with each other."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


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
