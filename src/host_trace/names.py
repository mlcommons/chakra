"""Names PyTorch gives the root nodes of a host trace. The linker also gives them to the Kineto annotations it adds."""

# The root of the whole trace. It is its own parent.
PROCESS_ANNOTATION = "[pytorch|profiler|execution_trace|process]"

# The root of each thread's operators. Their parent is the process root.
THREAD_ANNOTATION = "[pytorch|profiler|execution_trace|thread]"
