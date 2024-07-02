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
        "args": {"External id": "123", "Ev Idx": "456", "stream": 7, "Record function id": 12, "correlation": 99},
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
    assert operator.external_id == 123
    assert operator.ev_idx == 456
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
        "KinetoOperator(id=None, category=Kernel, name=cudaLaunchKernel, phase=X, "
        "inclusive_dur=100, exclusive_dur=100, timestamp=1590000000, external_id=123, ev_idx=456, "
        "tid=1234, parent_pytorch_op_id=None, inter_thread_dep=None, stream=7, rf_id=12, "
        "correlation=99)"
    )
    assert repr(operator) == expected_repr

@pytest.mark.parametrize("category, name, expected", [
    ("cuda_driver", "cuLaunchKernel", True),
    ("cuda_driver", "cuLaunchKernelEx", True),
    ("cuda_driver", "cudaLaunchKernel", True),
    ("cuda_driver", "cudaLaunchKernelExC", True),
    ("cuda_runtime", "cuLaunchKernel", True),
    ("cuda_runtime", "cuLaunchKernelEx", True),
    ("cuda_runtime", "cudaLaunchKernel", True),
    ("cuda_runtime", "cudaLaunchKernelExC", True),
    ("cuda_runtime", "cudaMemcpy", True),
    ("cuda_runtime", "cudaMemcpyAsync", True),
    ("cuda_runtime", "cudaMemcpyFromSymbol", True),
    ("cuda_runtime", "cudaMemcpyToSymbol", True),
    ("cpu_op", "cudaLaunchKernel", False),
    ("cuda_runtime", "someOtherOperation", False),
    ("some_other_category", "cudaLaunchKernel", False)
])
def test_is_cuda_launch_op(category, name, expected):
    """Test the is_cuda_launch_op method with various inputs."""
    operator_data = {
        "cat": category,
        "name": name,
        "ph": "X",
        "dur": 100,
        "ts": 1590000000,
        "tid": 1234,
        "args": {"External id": "123", "Ev Idx": "456", "stream": 7, "Record function id": 12, "correlation": 99},
    }
    operator = KinetoOperator(operator_data)
    assert operator.is_cuda_launch_op() == expected
