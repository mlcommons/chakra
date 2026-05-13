"""Simple script to collect post-execution traces with one rank only(=compute only)."""

import os

import torch
import torch.nn as nn
from torch.profiler import ExecutionTraceObserver, profile

from sample_model import SampleModel


def main() -> None:
    """Run 10 iterations of forward and backward passes."""
    batch_size = 64
    input_size = 1024
    output_size = 256
    output_path = "traces"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SampleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    model.to(device)

    # Initializing Host profiler and Kineto profiler
    rank = 0 # Single rank for compute-only
    wait_iters = 0
    warmup_iters = 5
    active_iters = 5
    total_steps = wait_iters + warmup_iters + active_iters

    os.makedirs(output_path, exist_ok=True)

    et = ExecutionTraceObserver()
    et.register_callback(f"{output_path}/host.{rank}.json")

    def device_trace_handler(prof):
        prof.export_chrome_trace(f"{output_path}/device.{rank}.json")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if "cuda" in device:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait_iters, warmup=warmup_iters, active=active_iters),
        on_trace_ready=device_trace_handler,
        record_shapes=True,
        execution_trace_observer=et,
    ) as prof:
        print("Starting training loop...")
        for step_id in range(total_steps):
            # Generate random input data
            x = torch.randn(batch_size, input_size, device=device)
            target = torch.randn(batch_size, output_size, device=device)

            # Forward pass
            output = model(x)

            # Compute loss
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if "cuda" in device:
                # Ensure all CUDA operations are complete before moving to the next step.
                torch.cuda.synchronize()

            print(f"Iteration {step_id + 1}/{total_steps}, Loss: {loss.item():.4f}")
            
            # Mark the end of a step
            prof.step()

    et.stop()
    et.unregister_callback()
    print("Training complete!")


if __name__ == "__main__":
    main()
