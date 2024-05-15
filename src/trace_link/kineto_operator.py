from typing import Any, Dict, Optional

from param_bench.train.compute.python.tools.execution_trace import (
    Node as PyTorchOperator,
)


class KinetoOperator:
    """
    Represents a single operator extracted from the Kineto trace.

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
        self.op_dict = kineto_op
        self.category = kineto_op.get("cat", "")
        self.name = kineto_op.get("name", "")
        self.phase = kineto_op.get("ph")
        self.inclusive_dur = kineto_op.get("dur", 0)
        self.exclusive_dur = kineto_op.get("dur", 0)
        self.timestamp = kineto_op.get("ts", 0)
        self.external_id = ""
        self.ev_idx = ""
        self.tid = kineto_op.get("tid", 0)
        self.pytorch_op: Optional[PyTorchOperator] = None
        self.parent_pytorch_op_id = None
        self.inter_thread_dep: Optional[int] = None
        self.stream: Optional[int] = None
        self.rf_id: Optional[int] = None
        self.correlation: int = None

        if "args" in kineto_op:
            self.external_id = kineto_op["args"].get("External id")
            self.ev_idx = kineto_op["args"].get("Ev Idx", "")
            self.stream = kineto_op["args"].get("stream")
            if "Record function id" in kineto_op["args"]:
                self.rf_id = int(kineto_op["args"]["Record function id"])
            if "correlation" in kineto_op["args"]:
                self.correlation = int(kineto_op["args"]["correlation"])

    def is_valid(
        self,
        category: str,
        name_exception: str = "ProfilerStep",
        phase: Optional[str] = None,
    ) -> bool:
        """
        Checks if the operator matches specified filtering criteria.

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

    def __repr__(self) -> str:
        """
        Represent the KinetoOperator as a string.

        Returns:
            str: A string representation of the KinetoOperator.
        """
        return (
            f"KinetoOperator(category={self.category}, "
            f"name={self.name}, phase={self.phase}, "
            f"inclusive_dur={self.inclusive_dur}, "
            f"exclusive_dur={self.exclusive_dur}, "
            f"timestamp={self.timestamp}, external_id={self.external_id}, "
            f"ev_idx={self.ev_idx}, tid={self.tid}, "
            f"rf_id={self.rf_id}, "
            f"parent_pytorch_op_id={self.parent_pytorch_op_id})"
        )
