import json
import logging
from typing import Dict


def load_json_execution_traces(input_filename: str) -> Dict:
    """
    Load Chakra host + device execution traces in JSON format from a file.

    Args:
        input_filename (str): Input Chakra host + device execution trace in the JSON format.

    Returns:
        Dict: The loaded Chakra host + device execution trace data.
    """
    logging.debug(f"Loading Chakra host + device execution traces in JSON format from file: {input_filename}")
    with open(input_filename, "r") as json_file:
        return json.load(json_file)
