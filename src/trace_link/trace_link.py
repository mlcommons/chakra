import argparse
import logging

from .trace_linker import TraceLinker


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "This tool links Chakra host execution traces with Chakra device traces. Chakra host execution "
            "traces include host-side (CPU) operators only, missing GPU operators. While these traces show "
            "dependencies between operators, they lack operator duration. Chakra device traces include "
            "device-side (GPU) operators in an unstructured timeline without explicit dependencies. This tool "
            "adds duration information to CPU operators in Chakra host traces and encodes GPU operators into the "
            "final Chakra host + device trace in JSON format. The trace linker also identifies key dependencies, "
            "such as inter-thread and synchronization dependencies. For more information, see the guide at https://"
            "github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection-%E2%80%90-A-Comprehensive-Guide-on-"
            "Merging-PyTorch-and-Kineto-Traces"
        )
    )
    parser.add_argument(
        "--chakra-host-trace",
        type=str,
        required=True,
        help="Path to the Chakra host execution trace (formerly called PyTorch execution traces)",
    )
    parser.add_argument(
        "--chakra-device-trace",
        type=str,
        required=True,
        help="Path to the Chakra device execution trace (also known as Kineto traces)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path for the output Chakra host + device trace in the JSON format",
    )
    parser.add_argument("--log-level", default="INFO", type=str, help="Log output verbosity level")

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())

    linker = TraceLinker()
    linker.link(args.chakra_host_trace, args.chakra_device_trace, args.output_file)

    logging.info(f"Linking process successful. Output file is available at {args.output_file}.")
    logging.info("Please run the chakra_converter for further postprocessing.")

if __name__ == "__main__":
    main()
