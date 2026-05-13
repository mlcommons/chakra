"""Simple script to collect post-execution traces with multiple ranks via torchrun."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ExecutionTraceObserver, profile

from sample_model import SampleModel


def main() -> None:
    """Run 10 iterations of forward and backward passes."""
    batch_size = 64
    input_size = 1024
    output_size = 256
    output_path = "traces"

    # torchrun provides these environment variables for each process.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        backend = "nccl"
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        backend = "gloo"
        device = "cpu"

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model = SampleModel()
    criterion = nn.MSELoss()
    model.to(device)
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Initializing Host profiler and Kineto profiler
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
        print(f"Starting training loop on rank {rank}/{world_size} using {device}...")
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

            print(f"Rank {rank} Iteration {step_id + 1}/{total_steps}, Loss: {loss.item():.4f}")
            
            # Mark the end of a step
            prof.step()

    et.stop()
    et.unregister_callback()
    dist.barrier()
    dist.destroy_process_group()
    print(f"Rank {rank} training complete!")


if __name__ == "__main__":
    main()
