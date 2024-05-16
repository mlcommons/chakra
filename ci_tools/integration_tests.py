import argparse
import concurrent.futures
import os
import re
import subprocess
import tarfile


def extract_tgz(tgz_path: str, extract_to: str) -> None:
    """
    Extracts a .tgz file to the specified directory.

    Args:
        tgz_path (str): Path to the .tgz file.
        extract_to (str): Directory to extract the files to.
    """
    print(f"Extracting {tgz_path} to {extract_to}")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_to)


def run_command(command: str) -> None:
    """
    Executes a given shell command and checks for errors.

    Args:
        command (str): The shell command to execute.

    Raises:
        RuntimeError: If the command fails.
    """
    print(f"Running command: {command}")
    os.environ["PATH"] = "/Users/theo/venv/bin/:" + os.environ.get("PATH", "")
    try:
        subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {command}") from e


def run_commands_in_parallel(commands: list) -> None:
    """
    Executes multiple commands in parallel.

    Args:
        commands (list): A list of shell commands to execute.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_command, cmd) for cmd in commands]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def run_trace_link(data_path: str, num_ranks: int) -> None:
    """
    Prepares and runs chakra_trace_link commands in parallel for each file pair.

    Args:
        data_path (str): The directory where the data files are located.
        num_ranks (int): The number of file pairs to process.
    """
    commands = [
        f"chakra_trace_link --pytorch-et-file {data_path}/chakra_host_et_{i}.json "
        f"--kineto-file {data_path}/kineto_{i}.json "
        f"--output-file {data_path}/chakra_et_plus_{i}.json"
        for i in range(num_ranks)
    ]
    run_commands_in_parallel(commands)


def run_converter(data_path: str, num_ranks: int) -> None:
    """
    Prepares and runs chakra_converter commands in parallel for each output of chakra_trace_link.

    Args:
        data_path (str): The directory where the output files are located.
        num_ranks (int): The number of output files to process.
    """
    commands = [
        f"chakra_converter --input_filename {data_path}/chakra_et_plus_{i}.json "
        f"--output_filename {data_path}/chakra_final_{i}.chakra "
        f"--input_type PyTorch --log_filename /tmp/rank_{i}.log"
        for i in range(num_ranks)
    ]
    run_commands_in_parallel(commands)


def validate_log(filename: str, expected_time_us: int, tolerance: float) -> None:
    """
    Validates the log file to ensure the last operation completes within the expected time with an allowable error.

    Args:
        filename (str): Path to the log file.
        expected_time_us (int): Expected completion time in microseconds.
        tolerance (float): Acceptable error percentage as a decimal.

    Raises:
        ValueError: If the log does not contain the expected output or is outside the acceptable time range.
    """
    completion_pattern = re.compile(
        r"INFO \[\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} PM\] GPU Node ID \d+ completed at (\d+)us"
    )
    with open(filename, "r") as file:
        last_time = None
        for line in file:
            match = completion_pattern.search(line)
            if match:
                last_time = int(match.group(1))

        if last_time is None:
            raise ValueError(f"No completion time found in {filename}")

        lower_bound = expected_time_us * (1 - tolerance)
        upper_bound = expected_time_us * (1 + tolerance)

        if not lower_bound <= last_time <= upper_bound:
            raise ValueError(
                f"Completion time in {filename} is {last_time}us; expected between {lower_bound}us and {upper_bound}us."
            )
        print(f"Validation successful for {filename}: {last_time}us is within the acceptable range.")


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run integration tests for chakra_trace_link and chakra_converter.")
    parser.add_argument("--tgz_path", type=str, required=True, help="Path to the tgz file to extract.")
    parser.add_argument("--num_ranks", type=int, required=True, help="Number of ranks to process.")
    parser.add_argument("--tolerance", type=float, required=True, help="Acceptable error percentage as a decimal.")
    parser.add_argument(
        "--expected_times_ms", type=int, nargs="+", required=True, help="List of expected times in milliseconds."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to execute the integration test sequence.
    """
    args = parse_args()
    extract_dir = os.path.dirname(args.tgz_path)
    data_path = os.path.join(extract_dir, os.path.basename(args.tgz_path).replace(".tgz", ""))

    # Extracting files
    extract_tgz(args.tgz_path, extract_dir)

    expected_times_us = [time * 1000 for time in args.expected_times_ms]

    # Run trace link and converter processes
    run_trace_link(data_path, args.num_ranks)
    run_converter(data_path, args.num_ranks)

    # Validate output logs
    for i in range(args.num_ranks):
        log_file = f"/tmp/rank_{i}.log"
        validate_log(log_file, expected_times_us[i], args.tolerance)


if __name__ == "__main__":
    main()
