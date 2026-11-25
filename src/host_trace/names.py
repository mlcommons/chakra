"""Names in a Chakra host trace that Chakra relies on. Every other module takes them from here."""

# The root of the whole trace. It is its own parent.
PROCESS_ANNOTATION = "[pytorch|profiler|execution_trace|process]"

# The root of each thread's operators. Their parent is the process root.
THREAD_ANNOTATION = "[pytorch|profiler|execution_trace|thread]"

# Node attributes every node has from schema 1.0.2 (PyTorch 2.2). HostNode converts each one into a field.
CORE_ATTRS = ("rf_id", "fw_parent", "seq_id", "scope", "tid", "fw_tid", "op_schema")

# Node attributes every node has from schema 1.1.0 (PyTorch 2.4). Not converted; available in HostNode.attrs.
KERNEL_ATTRS = ("kernel_backend", "kernel_file")

# Keys of a node's inputs and outputs from schema 1.0.2; STRIDES from schema 1.1.1 (PyTorch 2.5).
TENSOR_KEYS = ("values", "shapes", "types")
STRIDES = "strides"
