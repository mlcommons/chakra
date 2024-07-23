import pytest
from chakra.src.trace_link.chakra_device_trace_loader import ChakraDeviceTraceLoader
from chakra.src.trace_link.kineto_operator import KinetoOperator


@pytest.fixture
def trace_loader():
    return ChakraDeviceTraceLoader()


@pytest.mark.parametrize(
    "kineto_ops, expected_exclusive_durs",
    [
        (
            [
                {"ts": 100, "dur": 10, "inclusive_dur": 10},
                {"ts": 105, "dur": 3, "inclusive_dur": 3},
                {"ts": 108, "dur": 1, "inclusive_dur": 1},
            ],
            [6, 3, 1],  # Expected exclusive durations
        ),
        (
            [
                {"ts": 100, "dur": 20, "inclusive_dur": 20},
                {"ts": 105, "dur": 5, "inclusive_dur": 5},
                {"ts": 110, "dur": 5, "inclusive_dur": 5},
            ],
            [10, 5, 5],  # Expected exclusive durations
        ),
    ],
)
def test_calculate_exclusive_dur(trace_loader, kineto_ops, expected_exclusive_durs):
    kineto_tid_cpu_ops_map = {1: [KinetoOperator(op) for op in kineto_ops]}
    trace_loader.calculate_exclusive_dur(kineto_tid_cpu_ops_map)

    for i, op in enumerate(kineto_tid_cpu_ops_map[1]):
        assert op.exclusive_dur == expected_exclusive_durs[i]


@pytest.mark.parametrize(
    "intervals, expected_result",
    [
        ([(1, 3), (2, 6), (8, 10), (15, 18)], [(1, 6), (8, 10), (15, 18)]),
        ([(1, 4), (4, 5)], [(1, 5)]),
        ([], []),
        ([(1, 2), (2, 3), (3, 4)], [(1, 4)]),
        ([(1, 5), (2, 6), (6, 8), (7, 9)], [(1, 9)]),
    ],
)
def test_merge_overlapping_intervals(intervals, expected_result):
    result = ChakraDeviceTraceLoader.merge_overlapping_intervals(intervals)
    assert result == expected_result
