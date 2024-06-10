import argparse
import tempfile
from unittest.mock import mock_open, patch

from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata
from chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode
from chakra.src.jsonizer.jsonizer import main
from google.protobuf.json_format import MessageToJson


@patch("chakra.src.jsonizer.jsonizer.open_file_rd")
@patch("chakra.src.jsonizer.jsonizer.decode_message")
@patch("builtins.open", new_callable=mock_open)
def test_main(mock_file_open, mock_decode_message, mock_open_file_rd) -> None:
    """
    Tests the main function for converting Chakra execution trace to JSON format.
    """
    with tempfile.NamedTemporaryFile(suffix=".json") as temp_output:
        args = argparse.Namespace(input_filename="input_file", output_filename=temp_output.name)
        mock_node = ChakraNode()
        mock_global_metadata = GlobalMetadata()

        mock_decode_message.side_effect = [mock_global_metadata, mock_node, False]

        with patch("argparse.ArgumentParser.parse_args", return_value=args):
            main()

        mock_open_file_rd.assert_called_with("input_file")
        mock_decode_message.assert_called()
        mock_file_open.assert_called_with(temp_output.name, "w")
        mock_file_open().write.assert_any_call(MessageToJson(mock_global_metadata))
        mock_file_open().write.assert_any_call(MessageToJson(mock_node))
