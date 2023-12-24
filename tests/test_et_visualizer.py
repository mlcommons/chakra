import pytest
from chakra.et_visualizer import main
from unittest.mock import patch

# Assuming 'example_input.et' is the example input file you provided
example_input_file = 'example_input.et'

def test_run_with_pdf_output():
    test_args = ["et_visualizer.py", "--input_filename", example_input_file, "--output_filename", "output.pdf"]
    with patch('sys.argv', test_args):
        main()  # No assertion needed; we're checking if it runs without error

def test_run_with_dot_output():
    test_args = ["et_visualizer.py", "--input_filename", example_input_file, "--output_filename", "output.dot"]
    with patch('sys.argv', test_args):
        main()  # No assertion needed; we're checking if it runs without error

def test_run_with_graphml_output():
    test_args = ["et_visualizer.py", "--input_filename", example_input_file, "--output_filename", "output.graphml"]
    with patch('sys.argv', test_args):
        main()  # No assertion needed; we're checking if it runs without error

def test_incorrect_file_path():
    test_args = ["et_visualizer.py", "--input_filename", "nonexistent_input.et", "--output_filename", "output.pdf"]
    with patch('sys.argv', test_args):
        with pytest.raises(FileNotFoundError):
            main()

def test_unsupported_file_extension():
    test_args = ["et_visualizer.py", "--input_filename", example_input_file, "--output_filename", "output.xyz"]
    with patch('sys.argv', test_args):
        with pytest.raises(ValueError):
            main()
