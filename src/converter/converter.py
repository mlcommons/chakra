import argparse
import logging
import sys
import traceback

from .pytorch_converter import PyTorchConverter
from .text_converter import TextConverter


def setup_logging(log_filename: str) -> None:
    formatter = logging.Formatter("%(levelname)s [%(asctime)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")

    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])


def main() -> None:
    parser = argparse.ArgumentParser(description="Execution Trace Converter")
    parser.add_argument("--input_type", type=str, default=None, required=True, help="Input execution trace type")
    parser.add_argument(
        "--input_filename", type=str, default=None, required=True, help="Input execution trace filename"
    )
    parser.add_argument(
        "--output_filename", type=str, default=None, required=True, help="Output Chakra execution trace filename"
    )
    parser.add_argument(
        "--num_npus", type=int, default=None, required="Text" in sys.argv, help="Number of NPUs in a system"
    )
    parser.add_argument(
        "--num_passes", type=int, default=None, required="Text" in sys.argv, help="Number of training passes"
    )
    parser.add_argument("--log_filename", type=str, default="debug.log", help="Log filename")
    args = parser.parse_args()

    setup_logging(args.log_filename)
    logging.debug(" ".join(sys.argv))

    try:
        if args.input_type == "Text":
            converter = TextConverter(args.input_filename, args.output_filename, args.num_npus, args.num_passes)
            converter.convert()
        elif args.input_type == "PyTorch":
            converter = PyTorchConverter(args.input_filename, args.output_filename)
            converter.convert()
        else:
            supported_types = ["Text", "PyTorch"]
            logging.error(
                f"The input type '{args.input_type}' is not supported. "
                f"Supported types are: {', '.join(supported_types)}."
            )
            sys.exit(1)
    except Exception:
        traceback.print_exc()
        logging.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
