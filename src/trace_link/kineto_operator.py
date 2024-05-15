from typing import Any, Dict, Optional

from param_bench.train.compute.python.tools.execution_trace import Node as PyTorchOperator


class KinetoOperator:
    """
    Represents a single operator in a Kineto trace by default, with fields primarily sourced
    from the Kineto traces. In addition to the default fields from Kineto traces, additional
    fields have been introduced for postprocessing purposes. These additional fields facilitate
    the correlation of PyTorch operators and the enforcement of dependencies among them,
    enhancing trace analysis and utility.

    Attributes:
        op_dict (Dict[str, Any]): Dictionary containing the operator data.
        category (str): Category of the operator.
        name (str): Name of the operator.
        phase (Optional[str]): Phase of the operator.
        inclusive_dur (int): Inclusive duration of the operator in microseconds.
        exclusive_dur (int): Exclusive duration of the operator in microseconds.
        timestamp (int): Timestamp of the operator in microseconds.
        external_id (str): External ID associated with the operator.
        ev_idx (str): Event index associated with the operator.
        tid (int): Thread ID associated with the operator.
        pytorch_op (Optional[PyTorchOperator]): Associated PyTorch operator.
        parent_pytorch_op_id (Optional[int]): ID of the parent PyTorch operator.
        inter_thread_dep (Optional[int]): ID of the latest CPU node from other
            threads before the gap.
        stream (Optional[int]): Stream ID associated with the operator.
        rf_id (Optional[int]): Record function ID.
        correlation (int): Correlation ID used to link CUDA runtime operations
            with their GPU counterparts.
    """

    def __init__(self, kineto_op: Dict[str, Any]) -> None:
        """
        Initializes a new instance of the KinetoOperator class.

        Args:
            kineto_op (Dict[str, Any]): The dictionary representing the
                                        operator data.
        """
        self.op_dict: Dict[str, Any] = kineto_op
        self.category: str = kineto_op.get("cat", "")
        self.name: str = kineto_op.get("name", "")
        self.phase: Optional[str] = kineto_op.get("ph")
        self.inclusive_dur: int = kineto_op.get("dur", 0)
        self.exclusive_dur: int = kineto_op.get("dur", 0)
        self.timestamp: int = kineto_op.get("ts", 0)
        self.external_id: str = kineto_op.get("args", {}).get("External id", "")
        self.ev_idx: str = kineto_op.get("args", {}).get("Ev Idx", "")
        self.tid: int = kineto_op.get("tid", 0)
        self.pytorch_op: Optional[PyTorchOperator] = None
        self.parent_pytorch_op_id: Optional[int] = None
        self.inter_thread_dep: Optional[int] = None
        self.stream: Optional[int] = kineto_op.get("args", {}).get("stream")
        self.rf_id: Optional[int] = kineto_op.get("args", {}).get("Record function id")
        self.correlation: int = kineto_op.get("args", {}).get("correlation", -1)

    def __repr__(self) -> str:
        """
        Represent the KinetoOperator as a string.

        Returns:
            str: A string representation of the KinetoOperator.
        """
        return (
            f"KinetoOperator(category={self.category}, name={self.name}, phase={self.phase}, "
            f"inclusive_dur={self.inclusive_dur}, exclusive_dur={self.exclusive_dur}, "
            f"timestamp={self.timestamp}, external_id={self.external_id}, ev_idx={self.ev_idx}, "
            f"tid={self.tid}, parent_pytorch_op_id={self.parent_pytorch_op_id}, "
            f"inter_thread_dep={self.inter_thread_dep}, stream={self.stream}, rf_id={self.rf_id}, "
            f"correlation={self.correlation})"
        )

    def is_valid(
        self,
        category: str,
        name_exception: str = "ProfilerStep",
        phase: Optional[str] = None,
    ) -> bool:
        """
        Checks if the operator matches specified filtering criteria.

        Comment (TODO):
            This is legacy code from a previous implementation. Ideally, we should merge this logic
            into trace_linker.py. The purpose of is_valid is ambiguous, and it is unclear whether
            the function is essential. However, we keep it as it is to avoid breaking downstream
            tools. After properly setting up CI/CD pipelines and testing, we can consider removing it.

        Args:
            category (str): The category to check against.
            name_exception (str): A name to exclude in the check.
            phase (Optional[str]): The phase to check against, if any.

        Returns:
            bool: True if the operator matches the criteria, False otherwise.
        """
        return (
            self.category is not None
            and name_exception not in self.name
            and self.category == category
            and (phase is None or self.phase == phase)
        )
