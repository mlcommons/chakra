import argparse
import logging

from .pytorch_converter import PyTorchConverter
from .text_converter import TextConverter


def setup_logging(log_filename: str) -> None:
    """Set up logging to file and stream handlers."""
    formatter = logging.Formatter("%(levelname)s [%(asctime)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")

    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])


def convert_text(args: argparse.Namespace) -> None:
    """Convert text input trace to Chakra execution trace."""
    converter = TextConverter(args.input, args.output, args.num_npus, args.num_passes)
    converter.convert()


def convert_pytorch(args: argparse.Namespace) -> None:
    """Convert PyTorch input trace to Chakra execution trace."""
    converter = PyTorchConverter()
    converter.convert(args.input, args.output, args.simulate)


def main() -> None:
    """Convert to Chakra execution trace in the protobuf format."""
    parser = argparse.ArgumentParser(
        description=(
            "Chakra execution trace converter for simulators. This converter is designed for any downstream "
            "simulators that take Chakra execution traces in the protobuf format. This converter takes an input file "
            "in another format and generates a Chakra execution trace output in the protobuf format."
        )
    )

    parser.add_argument("--log-filename", type=str, default="debug.log", help="Log filename")

    subparsers = parser.add_subparsers(title="subcommands", description="Valid subcommands", help="Input type")

    pytorch_parser = subparsers.add_parser(
        "PyTorch",
        help="Convert Chakra host + device execution trace in JSON to Chakra host + device execution trace in the "
        "Chakra schema with protobuf format",
    )
    pytorch_parser.add_argument(
        "--input", type=str, required=True, help="Input Chakra host + device traces in the JSON format"
    )
    pytorch_parser.add_argument(
        "--output", type=str, required=True, help="Output Chakra host + device traces in the protobuf format"
    )
    pytorch_parser.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Enable simulation of operators after the conversion for validation and debugging purposes. This option "
            "allows simulation of traces without running them through a simulator. Users can validate the converter "
            "or simulator against actual measured values using tools like chrome://tracing or https://perfetto.dev/. "
            "Read the duration of the timeline and compare the total execution time against the final simulation time "
            "of a trace. Disabled by default because it takes a long time."
        ),
    )
    pytorch_parser.set_defaults(func=convert_pytorch)

    text_parser = subparsers.add_parser(
        "Text", help="Convert text-based model description to Chakra schema-based traces in the protobuf format"
    )
    text_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Input file in the text format that describes a model. This follows the text format used in ASTRA-sim: "
            "https://github.com/astra-sim/astra-sim"
        ),
    )
    text_parser.add_argument(
        "--output", type=str, required=True, help="Output Chakra execution trace filename in the protobuf format"
    )
    text_parser.add_argument(
        "--num-npus",
        type=int,
        required=True,
        help="Number of NPUs in a system. Determines the number of traces the converter generates",
    )
    text_parser.add_argument(
        "--num-passes",
        type=int,
        required=True,
        help=(
            "Number of loops when generating traces based on the text input file. Increasing the number of passes "
            "increases the number of training iterations for a given text input."
        ),
    )
    text_parser.set_defaults(func=convert_text)

    args = parser.parse_args()

    if "func" in args:
        setup_logging(args.log_filename)
        args.func(args)
        logging.info(f"Conversion successful. Output file is available at {args.output}.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
