import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_memory.py <input1.csv/json> <input2.csv/json> ... <output_plot.pdf>")
        sys.exit(1)

    # All arguments except the last are input files
    input_files = sys.argv[1:-1]
    output_pdf = sys.argv[-1]

    plt.figure(figsize=(10, 4))

    for input_path in input_files:
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping.")
            continue

        # Read the file (supports CSV or JSON)
        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".json"):
            df = pd.read_json(input_path)
        else:
            print(f"Unsupported file format: {input_path}")
            continue

        # Determine timestamp column and convert to seconds if needed
        if "Timestamp (us)" in df.columns:
            df["Timestamp (s)"] = df["Timestamp (us)"] / 1_000_000
        elif "Timestamp (s)" not in df.columns:
            print(f"No timestamp column found in {input_path}, skipping.")
            continue

        # Offset timestamps relative to the first timestamp
        first_ts = df["Timestamp (s)"].iloc[0]
        df["Timestamp Offset (s)"] = df["Timestamp (s)"] - first_ts

        # Derive legend name from filename
        base_name = os.path.basename(input_path)
        if "memory_" in base_name and "." in base_name:
            legend_name = base_name.split("memory_")[-1].split(".")[0]
        else:
            legend_name = os.path.splitext(base_name)[0]

        # Plot “Total Allocated (MB)” if available
        if "Total Allocated (MB)" in df.columns:
            plt.plot(df["Timestamp Offset (s)"], df["Total Allocated (MB)"], label=legend_name)
        else:
            print(f"Column 'Total Allocated (MB)' not found in {input_path}, skipping.")
            continue
    plt.xlabel("Timeline (s)", fontsize=18)
    plt.ylabel("Memory (MB)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_pdf)
    print(f"Memory usage plot saved to {output_pdf}")

if __name__ == "__main__":
    main()
