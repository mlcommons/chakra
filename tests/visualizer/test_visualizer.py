import argparse
import tempfile
from unittest.mock import patch

from chakra.src.visualizer.visualizer import escape_label, main


def test_escape_label() -> None:
    """
    Tests the escape_label function.
    """
    assert escape_label("a{b}c") == "a\\{b\\}c"
    assert escape_label("a(b)c") == "a\\(b\\)c"
    assert escape_label("a<b>c") == "a\\<b\\>c"
    assert escape_label("a[b]c") == "a\\[b\\]c"
    assert escape_label("a|b&c-d") == "a\\|b\\&c\\-d"


@patch("chakra.src.visualizer.visualizer.open_file_rd")
@patch("chakra.src.visualizer.visualizer.decode_message")
@patch("chakra.src.visualizer.visualizer.graphviz.Digraph")
def test_main_pdf(mock_graphviz_digraph, mock_decode_message, mock_open_file_rd) -> None:
    """
    Tests the main function for PDF output.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_output:
        args = argparse.Namespace(input_filename="input_file", output_filename=temp_output.name)
        mock_node = mock_open_file_rd.return_value
        mock_global_metadata = mock_open_file_rd.return_value

        mock_decode_message.side_effect = [mock_global_metadata, mock_node, False]

        with patch("argparse.ArgumentParser.parse_args", return_value=args):
            main()

        mock_open_file_rd.assert_called_with("input_file")
        mock_decode_message.assert_called()
        mock_graphviz_digraph.return_value.render.assert_called()


@patch("chakra.src.visualizer.visualizer.open_file_rd")
@patch("chakra.src.visualizer.visualizer.decode_message")
@patch("chakra.src.visualizer.visualizer.nx.write_graphml")
def test_main_graphml(mock_write_graphml, mock_decode_message, mock_open_file_rd) -> None:
    """
    Tests the main function for GraphML output.
    """
    with tempfile.NamedTemporaryFile(suffix=".graphml") as temp_output:
        args = argparse.Namespace(input_filename="input_file", output_filename=temp_output.name)
        mock_node = mock_open_file_rd.return_value
        mock_global_metadata = mock_open_file_rd.return_value

        mock_decode_message.side_effect = [mock_global_metadata, mock_node, False]

        with patch("argparse.ArgumentParser.parse_args", return_value=args):
            main()

        mock_open_file_rd.assert_called_with("input_file")
        mock_decode_message.assert_called()
        mock_write_graphml.assert_called()
