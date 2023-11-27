import pytest
from chakra.et_jsonizer import main
from unittest.mock import patch, mock_open

def test_run_with_valid_input():
    test_args = ["et_jsonizer.py", "--input_filename", "valid_input.et", "--output_filename", "output.json"]
    with patch('sys.argv', test_args), \
         patch('chakra.et_jsonizer.open_file_rd') as mock_open_file_rd, \
         patch('builtins.open', mock_open()):
        main()  # No assertion needed; we're checking if it runs without error

def test_missing_arguments():
    with pytest.raises(SystemExit):
        with patch('sys.argv', ['et_jsonizer.py']):
            main()

def test_missing_file_path():
    test_args = ["et_jsonizer.py", "--input_filename", "nonexistent_input.et", "--output_filename", "output.json"]
    with patch('sys.argv', test_args):
        with pytest.raises(FileNotFoundError):
            main()
