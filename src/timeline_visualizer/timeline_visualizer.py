import argparse
import json
import logging
import sys
from enum import IntEnum
from logging import FileHandler
from typing import Any, Dict, List, Tuple


class TID(IntEnum):
    """
    Enum representing the types of TID (Thread ID) used for classifying different nodes in a trace.

    Attributes
        LOCAL_MEMORY (int): Represents local memory nodes.
        REMOTE_MEMORY (int): Represents remote memory nodes.
        COMP (int): Represents compute nodes.
        COMM (int): Represents communication nodes.
    """

    LOCAL_MEMORY = 1
    REMOTE_MEMORY = 2
    COMP = 3
    COMM = 4


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


def is_local_mem_node(node_name: str) -> bool:
    return (
        ("MEM_LOAD_NODE" in node_name)
        and ("LOCAL_MEMORY" in node_name)
        or ("MEM_STORE_NODE" in node_name)
        and ("LOCAL_MEMORY" in node_name)
    )


def is_remote_mem_node(node_name: str) -> bool:
    return (
        ("MEM_LOAD_NODE" in node_name)
        and ("REMOTE_MEMORY" in node_name)
        or ("MEM_STORE_NODE" in node_name)
        and ("REMOTE_MEMORY" in node_name)
    )


def is_comp_node(node_name: str) -> bool:
    return "COMP_NODE" in node_name


def is_comm_node(node_name: str) -> bool:
    return ("COMM_SEND_NODE" in node_name) or ("COMM_RECV_NODE" in node_name) or ("COMM_COLL_NODE" in node_name)


def get_tid(node_name: str) -> TID:
    if is_local_mem_node(node_name):
        return TID.LOCAL_MEMORY
    elif is_remote_mem_node(node_name):
        return TID.REMOTE_MEMORY
    elif is_comp_node(node_name):
        return TID.COMP
    elif is_comm_node(node_name):
        return TID.COMM
    else:
        raise ValueError(f"Node type cannot be identified from {node_name}")


def parse_event(line: str) -> Tuple[str, int, int, int, str]:
    try:
        cols = line.strip().split(",")
        trace_type = cols[0]
        npu_id = int(cols[1].split("=")[1])
        curr_cycle = int(cols[2].split("=")[1])
        node_id = int(cols[3].split("=")[1])
        node_name = cols[4].split("=")[1]
        return (trace_type, npu_id, curr_cycle, node_id, node_name)
    except Exception as e:
        raise ValueError(f'Cannot parse the following event -- "{line}": {e}') from e


def get_trace_events(input_filename: str, num_npus: int, npu_frequency: int) -> List[Dict[str, Any]]:
    trace_dict = {i: {} for i in range(num_npus)}
    trace_events = []

    with open(input_filename, "r") as f:
        for line in f:
            if ("issue" in line) or ("callback" in line):
                (trace_type, npu_id, curr_cycle, node_id, node_name) = parse_event(line)

                if trace_type == "issue":
                    trace_dict[npu_id].update({node_id: [node_name, curr_cycle]})
                elif trace_type == "callback":
                    node_name = trace_dict[npu_id][node_id][0]
                    tid = get_tid(node_name)
                    issued_cycle = trace_dict[npu_id][node_id][1]
                    issued_ms = (issued_cycle / npu_frequency) / 1_000
                    duration_in_cycles = curr_cycle - issued_cycle
                    duration_in_ms = duration_in_cycles / (npu_frequency * 1_000)

                    trace_events.append(
                        {
                            "pid": npu_id,
                            "tid": tid,
                            "ts": issued_ms,
                            "dur": duration_in_ms,
                            "ph": "X",
                            "name": node_name,
                            "args": {"ms": duration_in_ms},
                        }
                    )

                    del trace_dict[npu_id][node_id]
                else:
                    raise ValueError(f"Unsupported trace_type, {trace_type}")

    return trace_events


def write_trace_events(output_filename: str, num_npus: int, trace_events: List[Dict[str, Any]]) -> None:
    output_dict = {"meta_user": "aras", "traceEvents": trace_events, "meta_cpu_count": num_npus}
    with open(output_filename, "w") as f:
        json.dump(output_dict, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Timeline Visualizer")
    parser.add_argument("--input_filename", type=str, default=None, required=True, help="Input timeline filename")
    parser.add_argument("--output_filename", type=str, default=None, required=True, help="Output trace filename")
    parser.add_argument("--num_npus", type=int, default=None, required=True, help="Number of NPUs in a system")
    parser.add_argument("--npu_frequency", type=int, default=None, required=True, help="NPU frequency in MHz")
    parser.add_argument("--log_filename", type=str, default="debug.log", help="Log filename")
    args = parser.parse_args()

    logger = get_logger(args.log_filename)
    logger.debug(" ".join(sys.argv))

    try:
        trace_events = get_trace_events(args.input_filename, args.num_npus, args.npu_frequency)
        write_trace_events(args.output_filename, args.num_npus, trace_events)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
