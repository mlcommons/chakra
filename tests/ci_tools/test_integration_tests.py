import subprocess
from unittest.mock import MagicMock, patch

import pytest

from ci_tools.integration_tests import extract_tgz, run_command, validate_log


def test_extract_tgz():
    """Test extracting a tgz file to ensure tarfile.open and extractall are called."""
    with patch("tarfile.open", MagicMock()) as mock_tar:
        mock_tar.return_value.__enter__.return_value.extractall = MagicMock()
        extract_tgz("path/to/test.tgz", "path/to/extract")
        mock_tar.assert_called_once_with("path/to/test.tgz", "r:gz")
        mock_tar.return_value.__enter__.return_value.extractall.assert_called_once_with(path="path/to/extract")


def test_run_command_success():
    """Test run_command with a command that succeeds without raising an error."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        run_command("echo 'Hello World'")
        mock_run.assert_called_once()


def test_run_command_failure():
    """Test run_command with a command that fails and should raise RuntimeError."""
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd", "Error message")):
        with pytest.raises(RuntimeError) as excinfo:
            run_command("exit 1")
        assert "Command failed: exit 1" in str(excinfo.value)


def test_validate_log_success(tmp_path):
    """Test validate_log to ensure it passes when the last operation completes within the expected time."""
    log_file = tmp_path / "log.txt"
    log_file.write_text("INFO [05/15/2024 08:32:04 PM] GPU Node ID 301123 completed at 1000000us")
    validate_log(str(log_file), 1000000, 0.05)


def test_validate_log_failure(tmp_path):
    """
    Test validate_log to ensure it raises a ValueError when the last operation is outside the acceptable time range.
    """
    log_file = tmp_path / "log.txt"
    log_file.write_text("INFO [05/15/2024 08:32:04 PM] GPU Node ID 301123 completed at 900000us")
    with pytest.raises(ValueError) as excinfo:
        validate_log(str(log_file), 1000000, 0.05)
    assert "expected between" in str(excinfo.value)
