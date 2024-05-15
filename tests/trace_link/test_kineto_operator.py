import pytest

from src.trace_link.kineto_operator import KinetoOperator


@pytest.fixture
def sample_operator_data():
    """Provides sample Kineto trace data for testing."""
    return {
        "cat": "Kernel",
        "name": "cudaLaunchKernel",
        "ph": "X",
        "dur": 100,
        "ts": 1590000000,
        "tid": 1234,
        "args": {"External id": "ext123", "Ev Idx": "ev456", "stream": 7, "Record function id": 12, "correlation": 99},
    }


def test_init_kineto_operator(sample_operator_data):
    """Test the initialization and attribute assignment of KinetoOperator."""
    operator = KinetoOperator(sample_operator_data)
    assert operator.category == "Kernel"
    assert operator.name == "cudaLaunchKernel"
    assert operator.phase == "X"
    assert operator.inclusive_dur == 100
    assert operator.exclusive_dur == 100
    assert operator.timestamp == 1590000000
    assert operator.external_id == "ext123"
    assert operator.ev_idx == "ev456"
    assert operator.tid == 1234
    assert operator.stream == 7
    assert operator.rf_id == 12
    assert operator.correlation == 99
    assert operator.pytorch_op is None  # Ensure default None
    assert operator.parent_pytorch_op_id is None  # Ensure default None
    assert operator.inter_thread_dep is None  # Ensure default None


def test_repr_method(sample_operator_data):
    """Test the __repr__ method output."""
    operator = KinetoOperator(sample_operator_data)
    expected_repr = (
        "KinetoOperator(category=Kernel, name=cudaLaunchKernel, phase=X, "
        "inclusive_dur=100, exclusive_dur=100, timestamp=1590000000, external_id=ext123, ev_idx=ev456, "
        "tid=1234, parent_pytorch_op_id=None, inter_thread_dep=None, stream=7, rf_id=12, "
        "correlation=99)"
    )
    assert repr(operator) == expected_repr


def test_is_valid_method(sample_operator_data):
    """Test the is_valid method under various conditions."""
    operator = KinetoOperator(sample_operator_data)
    assert operator.is_valid(category="Kernel")  # Matching category
    assert not operator.is_valid(category="Memory")  # Non-matching category
    assert operator.is_valid(category="Kernel", name_exception="cudaMalloc")  # Matching category, name not excluded
    assert not operator.is_valid(category="Kernel", name_exception="cudaLaunchKernel")  # Name excluded
    assert operator.is_valid(category="Kernel", phase="X")  # Matching phase
    assert not operator.is_valid(category="Kernel", phase="B")  # Non-matching phase
