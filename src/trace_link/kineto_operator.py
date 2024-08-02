from typing import Any, Dict, Optional

from et_replay.execution_trace import Node as PyTorchOperator


class KinetoOperator:
    """
    Represents a single operator in a Kineto trace.

    Attributes
        id (Optional[int]): Identifier of the operator.
        category (str): Category of the operator.
        name (str): Name of the operator.
        phase (Optional[str]): Execution phase of the operator.
        inclusive_dur (int): Total duration of the operator, including its children.
        exclusive_dur (int): Duration of the operator execution alone. Corresponds to the self time field in
            chrome://tracing.
        timestamp (int): Start time of the operator in microseconds.
        external_id (int): An external identifier associated with the operator.
        ev_idx (int): Event index of the operator.
        tid (int): Thread identifier where the operator was executed.
        host_op (Optional[PyTorchOperator]): Corresponding PyTorch operator object.
        parent_host_op_id (Optional[int]): ID of the parent PyTorch operator.
        inter_thread_dep (Optional[int]): Identifier for inter-thread dependencies.
        stream (Optional[int]): CUDA stream identifier associated with the operator.
        rf_id (Optional[int]): Record function identifier.
        correlation (int): Identifier used to correlate CUDA runtime and GPU operations.
        pg_name (Optional[str]): Process Group name for the collective communication.
    """

    def __init__(self, kineto_op: Dict[str, Any]) -> None:
        """
        Initialize a new instance of the KinetoOperator class.

        Args:
            kineto_op (Dict[str, Any]): The dictionary representing the
                                        operator data.
        """
        self.id: Optional[int] = kineto_op.get("id")
        self.category: str = kineto_op.get("cat", "")
        self.name: str = kineto_op.get("name", "")
        self.phase: Optional[str] = kineto_op.get("ph")
        self.inclusive_dur: int = kineto_op.get("dur", 0)
        self.exclusive_dur: int = kineto_op.get("dur", 0)
        self.timestamp: int = kineto_op.get("ts", 0)
        self.external_id: int = int(kineto_op.get("args", {}).get("External id", -1))
        self.ev_idx: int = int(kineto_op.get("args", {}).get("Ev Idx", -1))
        self.tid: int = kineto_op.get("tid", 0)
        self.host_op: Optional[PyTorchOperator] = None
        self.parent_host_op_id: Optional[int] = None
        self.inter_thread_dep: Optional[int] = None
        self.stream: Optional[int] = kineto_op.get("args", {}).get("stream", None)
        self.rf_id: Optional[int] = kineto_op.get("args", {}).get("Record function id", None)
        self.correlation: int = kineto_op.get("args", {}).get("correlation", -1)
        self.pg_name: Optional[str] = kineto_op.get("args", {}).get("Process Group Name", None)

    def __repr__(self) -> str:
        """
        Represent the KinetoOperator as a string.

        Returns
            str: A string representation of the KinetoOperator.
        """
        return (
            f"KinetoOperator(id={self.id}, category={self.category}, name={self.name}, "
            f"phase={self.phase}, inclusive_dur={self.inclusive_dur}, "
            f"exclusive_dur={self.exclusive_dur}, timestamp={self.timestamp}, "
            f"external_id={self.external_id}, ev_idx={self.ev_idx}, tid={self.tid}, "
            f"parent_host_op_id={self.parent_host_op_id}, inter_thread_dep={self.inter_thread_dep}, "
            f"stream={self.stream}, rf_id={self.rf_id}, correlation={self.correlation})"
        )

    def is_cpu_op(self) -> bool:
        """
        Determine if the operator is simulatable based on its category and name.

        The categories 'cpu_op' and 'user_annotation' are considered CPU operators.
        Notably, 'user_annotation' operators often include the duration of CPU operator launch times.
        Ignoring the duration measured in 'user_annotation' can lead to inaccuracies in simulation.
        An exception to this is 'ProfilerStep', which should be completely ignored.
        Ideally, a more general rule should be developed to identify such exception nodes.

        Returns
            bool: True if the operator is simulatable, False otherwise.
        """
        simulatable_categories = {"cpu_op", "user_annotation"}
        name_exceptions = {"ProfilerStep"}
        if self.category in simulatable_categories and all(exc not in self.name for exc in name_exceptions):
            return True
        return False

    def is_cuda_runtime_op(self) -> bool:
        """
        Determine whether the operator is a CUDA runtime operator.

        Returns
            bool: True if it's a CUDA runtime operator, otherwise False.
        """
        return self.category == "cuda_runtime"

    def is_cuda_driver_op(self) -> bool:
        """
        Determine whether the operator is a CUDA driver operator.

        Returns
            bool: True if it's a CUDA driver operator, otherwise False.
        """
        return self.category == "cuda_driver"

    def is_ac2g_op(self) -> bool:
        """
        Check if the operator is categorized as 'ac2g', which stands for arrows from CPU to GPU.

        Excerpt from https://pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html
        ```
            Every kernel on the GPU occurs after being launched by code running on the CPU. The profiler can draw
            connections (i.e. "flows") between the GPU and CPU events to show which CPU event launched a GPU kernel.
            This is particularly helpful because, with a few exceptions, GPU kernels are launched asynchronously.

            To view a flow connection, click on a GPU kernel and click "ac2g".
        ````

        Returns
            bool: True if the operator is an 'ac2g' type, otherwise False.
        """
        return self.category == "ac2g"

    def is_kernel_launch_op(self) -> bool:
        """
        Determine whether the operator is a kernel-launching CUDA runtime operator.

        Returns
            bool: True if it's a launch operation, otherwise False.
        """
        cuda_launch_categories = self.is_cuda_runtime_op() or self.is_cuda_driver_op()
        cuda_launch_operations = {
            "cuLaunchKernel",
            "cuLaunchKernelEx",
            "cudaLaunchKernel",
            "cudaLaunchKernelExC",
            "cudaMemcpy",
            "cudaMemcpyAsync",
            "cudaMemcpyFromSymbol",
            "cudaMemcpyToSymbol",
            "cudaLaunchCooperativeKernel",
        }
        return cuda_launch_categories and self.name in cuda_launch_operations

    def is_gpu_op(self) -> bool:
        """
        Check if the operator is a GPU-side operator based on its category.

        Returns
            bool: True if it's a GPU-side operation, otherwise False.
        """
        gpu_categories = {"kernel", "gpu_memcpy"}
        return self.category in gpu_categories

    def is_inter_gpu_comms_op(self) -> bool:
        """
        Check if the operator is a inter-GPU communication operator based on its name.

        Both point-to-point send/receive primitives and collective communication primitives are considered.

        Returns
            bool: True if it's a inter-GPU communication, otherwise False.
        """
        return "ncclDevKernel" in self.name
