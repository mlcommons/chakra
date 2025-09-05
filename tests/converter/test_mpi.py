#!/usr/bin/env python3
"""
Test script for MPI to Chakra converter
Tests conversion with various NPU counts including N-to-N mapping
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
import sys

def create_test_trace_1():
    """Create test trace with 1 compute PID and 4 GPU PIDs (5 total)"""
    return {
        "traceEvents": [
            # PID 11 - Compute rank
            {"name": "MPI_Init", "ph": "X", "pid": 11, "tid": 0, "ts": 1000, "dur": 100, "args": {}},
            {"name": "MPI_Comm_rank", "ph": "X", "pid": 11, "tid": 0, "ts": 1200, "dur": 50, "args": {}},
            {"name": "compute_kernel_1", "ph": "X", "pid": 11, "tid": 0, "ts": 1300, "dur": 1000, "args": {}},
            
            # PID 20 - GPU rank 0
            {"name": "MPI_Barrier", "ph": "X", "pid": 20, "tid": 0, "ts": 2000, "dur": 200, "args": {"comm": 4}},
            {"name": "MPI_Allreduce", "ph": "X", "pid": 20, "tid": 0, "ts": 2500, "dur": 500, "args": {"comm": 4, "count": 1024, "datatype": "0x0A"}},
            {"name": "local_gemm", "ph": "X", "pid": 20, "tid": 0, "ts": 3200, "dur": 800, "args": {}},
            {"name": "MPI_Isend", "ph": "X", "pid": 20, "tid": 0, "ts": 4100, "dur": 100, "args": {"dest": 21, "tag": 100, "count": 256, "datatype": "0x09"}},
            {"name": "MPI_Irecv", "ph": "X", "pid": 20, "tid": 0, "ts": 4300, "dur": 100, "args": {"source": 23, "tag": 101, "count": 256, "datatype": "0x09"}},
            {"name": "MPI_Wait", "ph": "X", "pid": 20, "tid": 0, "ts": 4500, "dur": 50, "args": {"request": "[56]"}},
            {"name": "MPI_Bcast", "ph": "X", "pid": 20, "tid": 0, "ts": 5000, "dur": 300, "args": {"comm": 4, "count": 512, "datatype": "0x0A"}},
            # Gap here - should insert dummy compute
            {"name": "MPI_Reduce", "ph": "X", "pid": 20, "tid": 0, "ts": 6000, "dur": 400, "args": {"comm": 4, "count": 256, "datatype": "0x0A"}},
            
            # PID 21 - GPU rank 1
            {"name": "MPI_Barrier", "ph": "X", "pid": 21, "tid": 0, "ts": 2000, "dur": 200, "args": {"comm": 4}},
            {"name": "MPI_Allreduce", "ph": "X", "pid": 21, "tid": 0, "ts": 2500, "dur": 500, "args": {"comm": 4, "count": 1024, "datatype": "0x0A"}},
            {"name": "local_gemm", "ph": "X", "pid": 21, "tid": 0, "ts": 3200, "dur": 800, "args": {}},
            {"name": "MPI_Irecv", "ph": "X", "pid": 21, "tid": 0, "ts": 4100, "dur": 100, "args": {"source": 20, "tag": 100, "count": 256, "datatype": "0x09"}},
            {"name": "MPI_Isend", "ph": "X", "pid": 21, "tid": 0, "ts": 4300, "dur": 100, "args": {"dest": 22, "tag": 102, "count": 256, "datatype": "0x09"}},
            {"name": "MPI_Wait", "ph": "X", "pid": 21, "tid": 0, "ts": 4500, "dur": 50, "args": {"request": "[57]"}},
            {"name": "MPI_Bcast", "ph": "X", "pid": 21, "tid": 0, "ts": 5000, "dur": 300, "args": {"comm": 4, "count": 512, "datatype": "0x0A"}},
            {"name": "MPI_Reduce", "ph": "X", "pid": 21, "tid": 0, "ts": 6000, "dur": 400, "args": {"comm": 4, "count": 256, "datatype": "0x0A"}},
            
            # PID 22 - GPU rank 2
            {"name": "MPI_Barrier", "ph": "X", "pid": 22, "tid": 0, "ts": 2000, "dur": 200, "args": {"comm": 4}},
            {"name": "MPI_Allreduce", "ph": "X", "pid": 22, "tid": 0, "ts": 2500, "dur": 500, "args": {"comm": 4, "count": 1024, "datatype": "0x0A"}},
            {"name": "local_gemm", "ph": "X", "pid": 22, "tid": 0, "ts": 3200, "dur": 800, "args": {}},
            {"name": "MPI_Irecv", "ph": "X", "pid": 22, "tid": 0, "ts": 4100, "dur": 100, "args": {"source": 21, "tag": 102, "count": 256, "datatype": "0x09"}},
            {"name": "MPI_Isend", "ph": "X", "pid": 22, "tid": 0, "ts": 4300, "dur": 100, "args": {"dest": 23, "tag": 103, "count": 256, "datatype": "0x09"}},
            {"name": "MPI_Wait", "ph": "X", "pid": 22, "tid": 0, "ts": 4500, "dur": 50, "args": {"request": "[58]"}},
            {"name": "MPI_Bcast", "ph": "X", "pid": 22, "tid": 0, "ts": 5000, "dur": 300, "args": {"comm": 4, "count": 512, "datatype": "0x0A"}},
            {"name": "MPI_Unknown_Op", "ph": "X", "pid": 22, "tid": 0, "ts": 5400, "dur": 100, "args": {}},  # Unsupported op
            {"name": "MPI_Reduce", "ph": "X", "pid": 22, "tid": 0, "ts": 6000, "dur": 400, "args": {"comm": 4, "count": 256, "datatype": "0x0A"}},
            
            # PID 23 - GPU rank 3  
            {"name": "MPI_Barrier", "ph": "X", "pid": 23, "tid": 0, "ts": 2000, "dur": 200, "args": {"comm": 4}},
            {"name": "MPI_Allreduce", "ph": "X", "pid": 23, "tid": 0, "ts": 2500, "dur": 500, "args": {"comm": 4, "count": 1024, "datatype": "0x0A"}},
            {"name": "local_gemm", "ph": "X", "pid": 23, "tid": 0, "ts": 3200, "dur": 800, "args": {}},
            {"name": "MPI_Irecv", "ph": "X", "pid": 23, "tid": 0, "ts": 4100, "dur": 100, "args": {"source": 22, "tag": 103, "count": 256, "datatype": "0x09"}},
            {"name": "MPI_Isend", "ph": "X", "pid": 23, "tid": 0, "ts": 4300, "dur": 100, "args": {"dest": 20, "tag": 101, "count": 256, "datatype": "0x09"}},
            {"name": "MPI_Wait", "ph": "X", "pid": 23, "tid": 0, "ts": 4500, "dur": 50, "args": {"request": "[59]"}},
            {"name": "MPI_Bcast", "ph": "X", "pid": 23, "tid": 0, "ts": 5000, "dur": 300, "args": {"comm": 4, "count": 512, "datatype": "0x0A"}},
            {"name": "MPI_Reduce", "ph": "X", "pid": 23, "tid": 0, "ts": 6000, "dur": 400, "args": {"comm": 4, "count": 256, "datatype": "0x0A"}},
        ]
    }

def create_test_trace_2():
    """Create test trace with 8 GPU PIDs only (for N-to-N mapping test)"""
    events = []
    
    # Create 8 GPU ranks (PIDs 0-7) with similar patterns
    for gpu_id in range(8):
        pid = gpu_id
        events.extend([
            {"name": "MPI_Init", "ph": "X", "pid": pid, "tid": 0, "ts": 1000, "dur": 100, "args": {}},
            {"name": "MPI_Barrier", "ph": "X", "pid": pid, "tid": 0, "ts": 2000, "dur": 200, "args": {"comm": 8}},
            {"name": "MPI_Allreduce", "ph": "X", "pid": pid, "tid": 0, "ts": 2500, "dur": 500, "args": {"comm": 8, "count": 2048, "datatype": "0x0A"}},
            {"name": "compute_kernel", "ph": "X", "pid": pid, "tid": 0, "ts": 3200, "dur": 800, "args": {}},
            {"name": "MPI_Alltoall", "ph": "X", "pid": pid, "tid": 0, "ts": 4200, "dur": 600, "args": {"comm": 8, "count": 512, "datatype": "0x09"}},
            {"name": "local_reduction", "ph": "X", "pid": pid, "tid": 0, "ts": 5000, "dur": 300, "args": {}},
            {"name": "MPI_Reduce_scatter", "ph": "X", "pid": pid, "tid": 0, "ts": 5500, "dur": 400, "args": {"comm": 8, "count": 1024, "datatype": "0x0A"}},
        ])
        
        # Add some send/recv pairs
        if gpu_id < 7:
            events.append({"name": "MPI_Send", "ph": "X", "pid": pid, "tid": 0, "ts": 6000, "dur": 100, 
                          "args": {"dest": pid + 1, "tag": 200 + pid, "count": 128, "datatype": "0x09"}})
        if gpu_id > 0:
            events.append({"name": "MPI_Recv", "ph": "X", "pid": pid, "tid": 0, "ts": 6100, "dur": 100,
                          "args": {"source": pid - 1, "tag": 199 + pid, "count": 128, "datatype": "0x09"}})
    
    return {"traceEvents": events}

def create_test_environment():
    """Create test environment with local output folders"""
    # Create directories in current working directory
    test_dir = Path("./mpi_converter_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    test_dir.mkdir()
    input_dir = test_dir / "input"
    input_dir.mkdir()
    
    # Create test trace files
    trace1_file = input_dir / "trace_5pids.json"
    with open(trace1_file, 'w') as f:
        json.dump(create_test_trace_1(), f, indent=2)
    
    trace2_file = input_dir / "trace_8gpus.json"
    with open(trace2_file, 'w') as f:
        json.dump(create_test_trace_2(), f, indent=2)
    
    return test_dir, input_dir

def run_converter_test(test_dir, input_file, test_name, num_npus_list):
    """Run converter test for different NPU counts"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    for num_npus in num_npus_list:
        output_dir = test_dir / f"output_{test_name}_{num_npus}npus"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nTesting with {num_npus} NPUs...")
        
        # Prepare command
        cmd = [
            "chakra_converter ",
            "--input_type", "mpi",
            "--input_dir", str(input_file.parent),
            "--output_dir", str(output_dir),
            "--num_npus", str(num_npus),
            "--unsupported_ops_log", str(output_dir / "unsupported_ops.txt"),
            "--node_debug_log", str(output_dir / "node_debug.txt"),
            "--pattern", input_file.name,
            "--log_filename", str(output_dir / "conversion.log")
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Run converter
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Conversion completed successfully")
                
                # Check results
                created_files = list(output_dir.glob("rank_*.et"))
                print(f"Created {len(created_files)} output files")
                
                # Check log for details
                log_file = output_dir / "conversion.log"
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        if "N-to-N mapping" in log_content:
                            print("Using N-to-N mapping strategy")
                        elif "Scaling from" in log_content:
                            print("Using scaling strategy")
                
                # Check unsupported ops log
                unsup_log = output_dir / "unsupported_ops.txt"
                if unsup_log.exists():
                    with open(unsup_log, 'r') as f:
                        content = f.read()
                        if "MPI_Unknown_Op" in content:
                            print("Correctly logged unsupported operation: MPI_Unknown_Op")
            else:
                print(f"X Conversion failed with return code: {result.returncode}")
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                
        except FileNotFoundError:
            print("X Error: chakra_converter command not found")
            print("  Make sure chakra is installed: pip install -e . from chakra root directory")
            print("  Trying with python -m instead...")
            
            # Try alternative method
            alt_cmd = [sys.executable, "-m", "chakra.src.converter.converter"] + cmd[1:]
            try:
                result = subprocess.run(alt_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("Conversion completed using python -m method")
                else:
                    print(f"X Alternative method also failed: {result.stderr}")
            except Exception as e:
                print(f"X Alternative method error: {str(e)}")
                
        except Exception as e:
            print(f"X Error running converter: {str(e)}")

def analyze_test_results(test_dir):
    """Analyze and display test results"""
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    # List all output directories
    output_dirs = sorted(test_dir.glob("output_*"))
    for output_dir in output_dirs:
        if output_dir.is_dir():
            files = list(output_dir.glob("rank_*.et"))
            print(f"\n{output_dir.name}:")
            print(f"  - Files created: {len(files)}")
            if files:
                print(f"  - File sizes: {files[0].stat().st_size} bytes (sample)")
            
            # Check for unsupported ops
            unsup_file = output_dir / "unsupported_ops.txt"
            if unsup_file.exists():
                with open(unsup_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("- "):
                            print(f"  - Unsupported op found: {line.strip()}")

def main():
    """Main test function"""
    print("MPI to Chakra Converter Test Suite")
    print("==================================")
    
    try:
        # Create test environment
        test_dir, input_dir = create_test_environment()
        print(f"Created test directory: {test_dir}")
        
        # Test 1: 5 PIDs (1 compute + 4 GPU) with various NPU counts
        print("\nTest 1: Trace with 1 compute PID + 4 GPU PIDs")
        run_converter_test(
            test_dir, 
            input_dir / "trace_5pids.json",
            "5pids",
            [4, 8, 16]  # Test scaling from 4 GPU PIDs to different NPU counts
        )
        
        # Test 2: 8 GPU PIDs with N-to-N mapping
        print("\nTest 2: Trace with 8 GPU PIDs (N-to-N mapping)")
        run_converter_test(
            test_dir,
            input_dir / "trace_8gpus.json", 
            "8gpus",
            [8, 16]  # Test both N-to-N (8) and scaling (16)
        )
        
        # Analyze all results
        analyze_test_results(test_dir)
        
        print(f"\n All test outputs saved in: {test_dir.absolute()}")
        print("  You can inspect the generated .et files and logs")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


