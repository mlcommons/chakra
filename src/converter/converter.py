#!/usr/bin/env python3

import argparse
import importlib.util
import logging
from pathlib import Path
from typing import Dict, Any
import glob
import sys 

# Try direct imports (when in same directory)
from .pytorch_converter import PyTorchConverter
from .text_converter import TextConverter
from .mpi_converter import MPIConverter

def get_converters() -> Dict[str, Any]:
    """
    Retrieves a dictionary of available converters.

    Returns:
        Dict[str, Any]: A dictionary mapping converter names to their respective classes.
    """
    converters = {
        "pytorch": PyTorchConverter,
        "text": TextConverter,
        "mpi": MPIConverter,
    }
    return converters


def main() -> None:
    """
    Main function to handle command-line arguments and initiate the conversion process.
    """
    parser = argparse.ArgumentParser(
        description="Chakra Execution Trace Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    converters = get_converters()
    converter_names = list(converters.keys())

    parser.add_argument(
        "--input_type",
        type=str,
        default=None,
        choices=converter_names,
        help=f"Type of input trace. Supported types: {', '.join(converter_names)}",
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        default=None,
        help="Input trace filename (for pytorch and text converters)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Output Chakra trace filename (for pytorch and text converters)",
    )
    parser.add_argument(
        "--log_filename",
        type=str,
        default="rank_0.log",
        help="Log filename (default: rank_0.log)",
    )
    parser.add_argument(
        "--converter",
        type=str,
        default=None,
        help="Path to a custom converter module",
    )
    
    # MPI-specific arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory for MPI traces (MPI converter only)",
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        help="Output directory for MPI traces (MPI converter only)",
    )
    parser.add_argument(
        "--num_npus",
        type=int,
        help="Number of NPUs to generate traces for (MPI converter only)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="File pattern for MPI traces (default: *.json, MPI converter only)",
    )
    parser.add_argument(
        "--unsupported_ops_log",
        type=str,
        help="Path to log file for unsupported operations (MPI converter only)",
    )
    parser.add_argument(
        "--node_debug_log",
        type=str,
        help="Path to log file for node debug information (MPI converter only)",
    )

    args = parser.parse_args()

    # Setup logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__)
    
    if args.log_filename:
        file_handler = logging.FileHandler(args.log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    # Handle custom converter if provided
    if args.converter:
        converter_path = Path(args.converter).resolve()
        if not converter_path.exists():
            parser.error(f"Converter module not found: {converter_path}")
        
        spec = importlib.util.spec_from_file_location("custom_converter", converter_path)
        if spec is None or spec.loader is None:
            parser.error(f"Failed to load converter module: {converter_path}")
        
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        
        if hasattr(custom_module, "CustomConverter"):
            converters["custom"] = custom_module.CustomConverter
            if args.input_type is None:
                args.input_type = "custom"
        else:
            parser.error("Custom converter module must contain a 'CustomConverter' class")

    # Check if input type is provided
    if args.input_type is None:
        parser.error("--input_type is required when not using a custom converter")

    # Validate converter type
    if args.input_type not in converters:
        parser.error(f"Unknown input type: {args.input_type}. Supported types: {', '.join(converter_names)}")

    # Initialize and run converter based on type
    if args.input_type == "mpi":
        # MPI converter uses directories instead of single files
        if not args.input_dir or not args.output_dir or not args.num_npus:
            parser.error(
                "MPI converter requires --input_dir, --output_dir, and --num_npus arguments"
            )
        
        # Find input files
        input_files = sorted(glob.glob(f"{args.input_dir}/{args.pattern}"))
        if not input_files:
            parser.error(f"No input files found matching {args.input_dir}/{args.pattern}")
        
        # Generate output filenames
        output_files = []
        for i in range(args.num_npus):
            output_files.append(f"{args.output_dir}/rank_{i}.et")
        
        # Create output directory if needed
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize MPI converter with its specific arguments
        converter_class = converters[args.input_type]
        converter = converter_class(
            input_filenames=input_files,
            output_filenames=output_files,
            num_npus=args.num_npus,
            unsupported_ops_log=args.unsupported_ops_log
        )
        
        logger.info(f"Starting MPI to Chakra conversion")
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Number of NPUs: {args.num_npus}")
        logger.info(f"Input files found: {len(input_files)}")
        
    else:
        # Other converters use single files
        if not args.input_filename or not args.output_filename:
            parser.error(
                f"{args.input_type} converter requires --input_filename and --output_filename arguments"
            )
        
        # Check if input file exists
        input_path = Path(args.input_filename)
        if not input_path.exists():
            parser.error(f"Input file not found: {args.input_filename}")
        
        # Initialize other converters with standard arguments
        converter_class = converters[args.input_type]
        converter = converter_class(args.input_filename, args.output_filename)
        
        logger.info(f"Starting {args.input_type} to Chakra conversion")
        logger.info(f"Input file: {args.input_filename}")
        logger.info(f"Output file: {args.output_filename}")

    # Run the converter
    try:
        converter.convert()
        logger.info("Conversion completed successfully")
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

