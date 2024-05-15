import argparse

from .trace_linker import TraceLinker


def main() -> None:
    """
    Main function to execute the trace linking process.

    For more detailed steps on collecting traces and converting them to Chakra traces, visit the guide at:
    https://github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection-%E2%80%90-A-Comprehensive-Guide-on-Merging-PyTorch-and-Kineto-Traces
    """
    parser = argparse.ArgumentParser(
        description="Link PyTorch execution trace with Kineto trace to produce Chakra traces."
        "For more information, see the guide at https://github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection-%E2%80%90-A-Comprehensive-Guide-on-Merging-PyTorch-and-Kineto-Traces"
    )
    parser.add_argument(
        "--pytorch-et-file",
        type=str,
        required=True,
        help="Path to the PyTorch execution trace",
    )
    parser.add_argument("--kineto-file", type=str, required=True, help="Path to the Kineto trace")
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path for the output PyTorch execution trace plus file",
    )
    parser.add_argument("--log-level", default="INFO", type=str, help="Log output verbosity level")

    args = parser.parse_args()

    linker = TraceLinker(args.pytorch_et_file, args.kineto_file, args.log_level)
    linker.load_traces()
    linker.enforce_inter_thread_order()
    linker.link_traces()
    linker.dump_pytorch_execution_trace_plus(args.output_file)


if __name__ == "__main__":
    main()
