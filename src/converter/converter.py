#!/usr/bin/env python3

import argparse
import logging
import sys
import traceback
from logging import FileHandler

from .pytorch_converter import PyTorchConverter
from .text_converter import TextConverter


def get_logger(log_filename: str) -> logging.Logger:
    formatter = logging.Formatter("%(levelname)s [%(asctime)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")

    file_handler = FileHandler(log_filename, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


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

    logger = get_logger(args.log_filename)
    logger.debug(" ".join(sys.argv))

    try:
        if args.input_type == "Text":
            converter = TextConverter(args.input_filename, args.output_filename, args.num_npus, args.num_passes, logger)
            converter.convert()
        elif args.input_type == "PyTorch":
            converter = PyTorchConverter(args.input_filename, args.output_filename, logger)
            converter.convert()
        else:
            logger.error(f"{args.input_type} unsupported")
            sys.exit(1)
    except Exception:
        traceback.print_exc()
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
