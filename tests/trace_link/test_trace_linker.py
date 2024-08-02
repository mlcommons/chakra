from unittest.mock import MagicMock, patch

import pytest
from chakra.src.trace_link.kineto_operator import KinetoOperator
from chakra.src.trace_link.trace_linker import TraceLinker
from chakra.src.trace_link.unique_id_assigner import UniqueIdAssigner
from et_replay.execution_trace import (
    EXECUTION_TRACE_PROCESS_ANNOTATION,
    EXECUTION_TRACE_THREAD_ANNOTATION,
)
from et_replay.execution_trace import Node as PyTorchOperator


@pytest.fixture
def trace_linker():
    return TraceLinker()


def test_initialization(trace_linker):
    assert isinstance(trace_linker.id_assigner, UniqueIdAssigner)


@patch("chakra.src.trace_link.trace_linker.TraceLinker.construct_et_plus_data")
@patch("chakra.src.trace_link.trace_linker.TraceLinker.add_thread_and_process_annotations")
@patch("chakra.src.trace_link.trace_linker.TraceLinker.map_host_to_device_ops")
def test_link_traces(mock_map_ops, mock_add_annotations, mock_construct_et_plus, trace_linker):
    mock_add_annotations.return_value = ([], [], [])
    mock_map_ops.return_value = ({}, {}, {}, {}, {})
    mock_construct_et_plus.return_value = {}

    host_ops = [MagicMock(spec=PyTorchOperator)]
    kineto_cpu_ops = [MagicMock(spec=KinetoOperator)]
    sorted_kineto_cpu_ops = [MagicMock(spec=KinetoOperator)]
    sorted_kineto_cpu_op_ts = [100]
    kineto_correlation_cuda_runtime_map = {1: MagicMock(spec=KinetoOperator)}
    kineto_rf_id_to_kineto_op_map = {1: MagicMock(spec=KinetoOperator)}
    kineto_gpu_ops = [MagicMock(spec=KinetoOperator)]
    kineto_thread_info = {1: (100, 200)}
    kineto_process_start_time = 50
    kineto_process_end_time = 300

    trace_linker.link_traces(
        "pytorch_et_file",
        host_ops,
        kineto_cpu_ops,
        sorted_kineto_cpu_ops,
        sorted_kineto_cpu_op_ts,
        kineto_correlation_cuda_runtime_map,
        kineto_rf_id_to_kineto_op_map,
        kineto_gpu_ops,
        kineto_thread_info,
        kineto_process_start_time,
        kineto_process_end_time,
    )

    mock_add_annotations.assert_called_once()
    mock_map_ops.assert_called_once()
    mock_construct_et_plus.assert_called_once()


@patch("chakra.src.trace_link.trace_linker.TraceLinker.find_last_cpu_node_before_timestamp")
def test_process_thread_inter_thread_order(mock_find_last, trace_linker):
    mock_op1 = MagicMock(spec=KinetoOperator)
    mock_op1.timestamp = 100
    mock_op1.inclusive_dur = 50
    mock_op1.tid = 1
    mock_op1.name = "op1"
    mock_op1.rf_id = 1

    mock_op2 = MagicMock(spec=KinetoOperator)
    mock_op2.timestamp = 200
    mock_op2.inclusive_dur = 50
    mock_op2.tid = 1
    mock_op2.name = "op2"
    mock_op2.rf_id = 2

    ops_by_tid = {1: [mock_op1, mock_op2]}
    trace_linker.process_thread_inter_thread_order(1, [mock_op1, mock_op2], ops_by_tid, 1000)

    mock_find_last.assert_called()


@patch("concurrent.futures.Future.result")
@patch("chakra.src.trace_link.trace_linker.TraceLinker.process_thread_inter_thread_order")
def test_enforce_inter_thread_order_exception(mock_process_thread, mock_future_result, trace_linker):
    mock_future_result.side_effect = Exception("Test Exception")
    kineto_tid_cpu_ops_map = {1: [MagicMock(spec=KinetoOperator)]}

    with pytest.raises(Exception, match="Test Exception"):
        trace_linker.enforce_inter_thread_order(kineto_tid_cpu_ops_map)


@pytest.mark.parametrize(
    "ops_by_tid, exclude_tid, timestamp, expected_result",
    [
        (
            {
                1: [MagicMock(spec=KinetoOperator, timestamp=100, category="cpu_op", rf_id=1)],
                2: [
                    MagicMock(spec=KinetoOperator, timestamp=150, category="cpu_op", rf_id=2),
                    MagicMock(spec=KinetoOperator, timestamp=200, category="cpu_op", rf_id=3),
                ],
            },
            1,
            175,
            2,
        ),
        (
            {
                1: [MagicMock(spec=KinetoOperator, timestamp=100, category="cpu_op", rf_id=1)],
                2: [
                    MagicMock(spec=KinetoOperator, timestamp=150, category="cpu_op", rf_id=2),
                    MagicMock(spec=KinetoOperator, timestamp=200, category="cpu_op", rf_id=3),
                ],
            },
            2,
            125,
            1,
        ),
        (
            {
                1: [MagicMock(spec=KinetoOperator, timestamp=100, category="cpu_op", rf_id=1)],
                2: [
                    MagicMock(spec=KinetoOperator, timestamp=150, category="cpu_op", rf_id=2),
                    MagicMock(spec=KinetoOperator, timestamp=200, category="cpu_op", rf_id=3),
                ],
            },
            2,
            50,
            None,
        ),
        (
            {
                1: [MagicMock(spec=KinetoOperator, timestamp=100, category="cpu_op", rf_id=1)],
                2: [
                    MagicMock(spec=KinetoOperator, timestamp=150, category="cpu_op", rf_id=2),
                    MagicMock(spec=KinetoOperator, timestamp=200, category="cpu_op", rf_id=3),
                ],
            },
            1,
            50,
            None,
        ),
    ],
)
def test_find_last_cpu_node_before_timestamp(ops_by_tid, exclude_tid, timestamp, expected_result, trace_linker):
    result = trace_linker.find_last_cpu_node_before_timestamp(ops_by_tid, exclude_tid, timestamp)
    assert result == expected_result


def test_add_thread_and_process_annotations(trace_linker):
    kineto_cpu_ops = []
    sorted_kineto_cpu_ops = []
    sorted_kineto_cpu_op_ts = []
    kineto_thread_info = {1: (100, 200), 2: (150, 250)}
    kineto_process_start_time = 50
    kineto_process_end_time = 300

    kineto_cpu_ops, sorted_kineto_cpu_ops, sorted_kineto_cpu_op_ts = trace_linker.add_thread_and_process_annotations(
        kineto_cpu_ops,
        sorted_kineto_cpu_ops,
        sorted_kineto_cpu_op_ts,
        kineto_thread_info,
        kineto_process_start_time,
        kineto_process_end_time,
    )

    assert len(kineto_cpu_ops) == 3
    assert kineto_cpu_ops[0].name == EXECUTION_TRACE_PROCESS_ANNOTATION
    assert kineto_cpu_ops[1].name == EXECUTION_TRACE_THREAD_ANNOTATION
    assert kineto_cpu_ops[2].name == EXECUTION_TRACE_THREAD_ANNOTATION

    assert len(sorted_kineto_cpu_ops) == 3
    assert sorted_kineto_cpu_ops[0].timestamp == kineto_cpu_ops[0].timestamp
    assert sorted_kineto_cpu_ops[1].timestamp == kineto_cpu_ops[1].timestamp
    assert sorted_kineto_cpu_ops[2].timestamp == kineto_cpu_ops[2].timestamp

    assert len(sorted_kineto_cpu_op_ts) == 3
    assert sorted_kineto_cpu_op_ts[0] == kineto_cpu_ops[0].timestamp
    assert sorted_kineto_cpu_op_ts[1] == kineto_cpu_ops[1].timestamp
    assert sorted_kineto_cpu_op_ts[2] == kineto_cpu_ops[2].timestamp


def test_group_gpu_ops_by_cpu_launchers(trace_linker):
    kineto_gpu_op1 = MagicMock(spec=KinetoOperator)
    kineto_gpu_op1.correlation = 123
    kineto_gpu_op1.name = "gpu_op1"
    kineto_gpu_op1.timestamp = 150
    kineto_gpu_op1.tid = 1

    kineto_gpu_op2 = MagicMock(spec=KinetoOperator)
    kineto_gpu_op2.correlation = 456
    kineto_gpu_op2.name = "gpu_op2"
    kineto_gpu_op2.timestamp = 250
    kineto_gpu_op2.tid = 2

    kineto_runtime_op1 = MagicMock(spec=KinetoOperator)
    kineto_runtime_op1.ev_idx = "cpu_op1"
    kineto_runtime_op1.timestamp = 100
    kineto_runtime_op1.tid = 1
    kineto_runtime_op1.name = "runtime_op1"
    kineto_runtime_op1.correlation = 123

    kineto_runtime_op2 = MagicMock(spec=KinetoOperator)
    kineto_runtime_op2.ev_idx = "cpu_op2"
    kineto_runtime_op2.timestamp = 200
    kineto_runtime_op2.tid = 2
    kineto_runtime_op2.name = "runtime_op2"
    kineto_runtime_op2.correlation = 456

    kineto_correlation_cuda_runtime_map = {123: kineto_runtime_op1, 456: kineto_runtime_op2}
    sorted_kineto_cpu_ops = [kineto_runtime_op1, kineto_runtime_op2]
    sorted_kineto_cpu_op_ts = [kineto_runtime_op1.timestamp, kineto_runtime_op2.timestamp]

    with patch.object(trace_linker, "find_parent_cpu_op", side_effect=[kineto_runtime_op1, kineto_runtime_op2]):
        result = trace_linker.group_gpu_ops_by_cpu_launchers(
            [kineto_gpu_op1, kineto_gpu_op2],
            kineto_correlation_cuda_runtime_map,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
        )

    assert result == {"cpu_op1": [kineto_gpu_op1], "cpu_op2": [kineto_gpu_op2]}


@patch("chakra.src.trace_link.trace_linker.TraceLinker.find_closest_op")
def test_find_parent_cpu_op(mock_find_closest_op, trace_linker):
    kineto_gpu_op = MagicMock(spec=KinetoOperator)
    kineto_gpu_op.correlation = 123
    kineto_gpu_op.name = "gpu_op"

    kineto_runtime_op = MagicMock(spec=KinetoOperator)
    kineto_runtime_op.timestamp = 100
    kineto_runtime_op.tid = 1
    kineto_runtime_op.name = "runtime_op"

    kineto_correlation_cuda_runtime_map = {123: kineto_runtime_op}

    mock_find_closest_op.return_value = kineto_runtime_op

    sorted_kineto_cpu_ops = [MagicMock(spec=KinetoOperator)]
    sorted_kineto_cpu_op_ts = [100]

    result = trace_linker.find_parent_cpu_op(
        kineto_gpu_op, kineto_correlation_cuda_runtime_map, sorted_kineto_cpu_ops, sorted_kineto_cpu_op_ts
    )

    assert result == kineto_runtime_op
    mock_find_closest_op.assert_called_once_with(
        kineto_gpu_op, sorted_kineto_cpu_ops, sorted_kineto_cpu_op_ts, kineto_runtime_op.timestamp
    )


@pytest.mark.parametrize(
    "host_op, kineto_op, expected_linked_gpu_ops, expected_inclusive_dur, expected_exclusive_dur, "
    "expected_timestamp, expected_inter_thread_dep",
    [
        (
            MagicMock(spec=PyTorchOperator, id=1),
            MagicMock(
                spec=KinetoOperator,
                ev_idx="1",
                inclusive_dur=100,
                exclusive_dur=50,
                timestamp=123456,
                host_op=None,
            ),
            [MagicMock(spec=KinetoOperator)],
            100,
            50,
            123456,
            None,
        ),
        (
            MagicMock(spec=PyTorchOperator, id=2),
            MagicMock(
                spec=KinetoOperator,
                ev_idx="2",
                inclusive_dur=200,
                exclusive_dur=150,
                timestamp=223456,
                host_op=None,
            ),
            [MagicMock(spec=KinetoOperator), MagicMock(spec=KinetoOperator)],
            200,
            150,
            223456,
            42,
        ),
    ],
)
@patch.object(TraceLinker, "get_inter_thread_dep")
@patch.object(TraceLinker, "link_gpu_ops")
def test_link_ops(
    mock_link_gpu_ops,
    mock_get_inter_thread_dep,
    trace_linker,
    host_op,
    kineto_op,
    expected_linked_gpu_ops,
    expected_inclusive_dur,
    expected_exclusive_dur,
    expected_timestamp,
    expected_inter_thread_dep,
):
    mock_get_inter_thread_dep.return_value = expected_inter_thread_dep

    cpu_ev_idx_to_gpu_ops_map = {kineto_op.ev_idx: expected_linked_gpu_ops}
    kineto_rf_id_to_kineto_op_map = {1: MagicMock(spec=KinetoOperator, host_op=MagicMock(id=42))}

    result = trace_linker.link_ops(
        host_op,
        kineto_op,
        cpu_ev_idx_to_gpu_ops_map,
        kineto_rf_id_to_kineto_op_map,
    )

    assert result == (
        expected_linked_gpu_ops,
        expected_inclusive_dur,
        expected_exclusive_dur,
        expected_timestamp,
        expected_inter_thread_dep,
    )
    mock_link_gpu_ops.assert_called_once_with(host_op, expected_linked_gpu_ops)


def test_link_ops_with_no_gpu_ops(trace_linker):
    host_op = MagicMock(spec=PyTorchOperator, id=1)
    kineto_op = MagicMock(
        spec=KinetoOperator,
        ev_idx="1",
        inclusive_dur=100,
        exclusive_dur=50,
        timestamp=123456,
        host_op=None,
        inter_thread_dep=None,
    )

    cpu_ev_idx_to_gpu_ops_map = {}
    kineto_rf_id_to_kineto_op_map = {}

    result = trace_linker.link_ops(
        host_op,
        kineto_op,
        cpu_ev_idx_to_gpu_ops_map,
        kineto_rf_id_to_kineto_op_map,
    )

    assert result == ([], 100, 50, 123456, None)


def test_get_inter_thread_dep(trace_linker):
    kineto_op = MagicMock(spec=KinetoOperator)
    kineto_op.inter_thread_dep = 1
    inter_thread_dep_kineto_op = MagicMock(spec=KinetoOperator)
    inter_thread_dep_kineto_op.host_op = MagicMock(id=42)

    kineto_rf_id_to_kineto_op_map = {1: inter_thread_dep_kineto_op}

    result = trace_linker.get_inter_thread_dep(kineto_op, kineto_rf_id_to_kineto_op_map)
    assert result == 42


def test_get_inter_thread_dep_none(trace_linker):
    kineto_op = MagicMock(spec=KinetoOperator)
    kineto_op.inter_thread_dep = None

    kineto_rf_id_to_kineto_op_map = {}

    result = trace_linker.get_inter_thread_dep(kineto_op, kineto_rf_id_to_kineto_op_map)
    assert result is None


def test_link_gpu_ops(trace_linker):
    # Create a mock PyTorch operator
    host_op = MagicMock(spec=PyTorchOperator)
    host_op.id = 123

    # Create mock Kineto GPU operators
    kineto_gpu_op1 = MagicMock(spec=KinetoOperator)
    kineto_gpu_op2 = MagicMock(spec=KinetoOperator)
    kineto_gpu_ops = [kineto_gpu_op1, kineto_gpu_op2]

    # Call the method
    trace_linker.link_gpu_ops(host_op, kineto_gpu_ops)


@patch("chakra.src.trace_link.trace_linker.TraceLinker.process_op_and_dependents")
@patch("builtins.open", new_callable=MagicMock)
@patch("json.load")
def test_construct_et_plus_data(mock_json_load, mock_open, mock_process_op_and_dependents, trace_linker):
    mock_json_load.return_value = {"nodes": [{"id": 1}, {"id": 2}]}
    mock_process_op_and_dependents.side_effect = lambda x, *args: [{"id": x["id"] + 2}]

    host_op_id_to_kineto_ops_map = {1: [], 2: []}
    host_op_id_to_inclusive_dur_map = {1: 100, 2: 200}
    host_op_id_to_exclusive_dur_map = {1: 50, 2: 150}
    host_op_id_to_timestamp_map = {1: 1000, 2: 2000}
    host_op_id_to_inter_thread_dep_map = {1: None, 2: None}

    pytorch_et_plus_data = trace_linker.construct_et_plus_data(
        "path/to/pytorch_et_file",
        host_op_id_to_kineto_ops_map,
        host_op_id_to_inclusive_dur_map,
        host_op_id_to_exclusive_dur_map,
        host_op_id_to_timestamp_map,
        host_op_id_to_inter_thread_dep_map,
    )

    assert pytorch_et_plus_data["nodes"] == [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]


@pytest.mark.parametrize(
    "orig_op_id, cpu_op, kineto_gpu_ops, expected_ids, expected_fields",
    [
        (
            1,
            {
                "id": 1,
                "inputs": ["input1", "input2"],
                "outputs": ["output1"],
            },
            [
                {
                    "timestamp": 200,
                    "category": "gpu_op",
                    "name": "gpu_op1",
                    "phase": "X",
                    "inclusive_dur": 100,
                    "exclusive_dur": 80,
                    "stream": 1,
                },
                {
                    "timestamp": 300,
                    "category": "gpu_op",
                    "name": "gpu_op2",
                    "phase": "X",
                    "inclusive_dur": 120,
                    "exclusive_dur": 100,
                    "stream": 2,
                },
            ],
            [100, 101],
            [
                {
                    "ctrl_deps": 1,
                    "inputs": ["input1", "input2"],
                    "outputs": ["output1"],
                },
                {
                    "ctrl_deps": 1,
                    "inputs": ["input1", "input2"],
                    "outputs": ["output1"],
                },
            ],
        ),
    ],
)
def test_process_dependent_gpu_ops(trace_linker, orig_op_id, cpu_op, kineto_gpu_ops, expected_ids, expected_fields):
    # Create mock dependent GPU operators
    kineto_gpu_op_objects = []
    for gpu_op_data in kineto_gpu_ops:
        gpu_op = MagicMock(spec=KinetoOperator)
        gpu_op.timestamp = gpu_op_data["timestamp"]
        gpu_op.category = gpu_op_data["category"]
        gpu_op.name = gpu_op_data["name"]
        gpu_op.phase = gpu_op_data["phase"]
        gpu_op.inclusive_dur = gpu_op_data["inclusive_dur"]
        gpu_op.exclusive_dur = gpu_op_data["exclusive_dur"]
        gpu_op.stream = gpu_op_data["stream"]
        gpu_op.pg_name = gpu_op_data.get("pg_name", None)
        kineto_gpu_op_objects.append(gpu_op)

    host_op_id_to_kineto_ops_map = {orig_op_id: kineto_gpu_op_objects}

    # Override the generate_new_id method to return the expected IDs
    original_generate_new_id = trace_linker.id_assigner.generate_new_id
    trace_linker.id_assigner.generate_new_id = MagicMock(side_effect=expected_ids)

    # Call the method
    updated_gpu_ops = trace_linker.process_dependent_gpu_ops(cpu_op, orig_op_id, host_op_id_to_kineto_ops_map)

    # Restore the original generate_new_id method
    trace_linker.id_assigner.generate_new_id = original_generate_new_id

    # Assert the new GPU operators have the updated IDs and fields
    assert len(updated_gpu_ops) == len(kineto_gpu_ops)
    for i, updated_gpu_op in enumerate(updated_gpu_ops):
        assert updated_gpu_op["id"] == expected_ids[i]
        assert updated_gpu_op["ctrl_deps"] == expected_fields[i]["ctrl_deps"]
        assert updated_gpu_op["inputs"] == expected_fields[i]["inputs"]
        assert updated_gpu_op["outputs"] == expected_fields[i]["outputs"]
        assert updated_gpu_op["cat"] == kineto_gpu_op_objects[i].category
        assert updated_gpu_op["name"] == kineto_gpu_op_objects[i].name
        assert updated_gpu_op["ph"] == kineto_gpu_op_objects[i].phase
        assert updated_gpu_op["inclusive_dur"] == kineto_gpu_op_objects[i].inclusive_dur
        assert updated_gpu_op["exclusive_dur"] == kineto_gpu_op_objects[i].exclusive_dur
        assert updated_gpu_op["ts"] == kineto_gpu_op_objects[i].timestamp
        assert updated_gpu_op["stream"] == kineto_gpu_op_objects[i].stream
        if kineto_gpu_op_objects[i].is_inter_gpu_comms_op() and kineto_gpu_op_objects[i].pg_name is not None:
            assert updated_gpu_op["pg_name"] == kineto_gpu_op_objects[i].pg_name


@patch("builtins.open", new_callable=MagicMock)
@patch("json.dump")
def test_dump_chakra_execution_trace_plus(mock_json_dump, mock_open, trace_linker):
    trace_linker.pytorch_et_plus_data = {"nodes": [{"id": 1}, {"id": 2}]}
    trace_linker.dump_chakra_execution_trace_plus(trace_linker.pytorch_et_plus_data, "output.json")

    mock_open.assert_called_once_with("output.json", "w")
    mock_open.return_value.__enter__.assert_called_once()
    mock_open.return_value.__exit__.assert_called_once()
    mock_json_dump.assert_called_once_with(
        {"nodes": [{"id": 1}, {"id": 2}]}, mock_open.return_value.__enter__(), indent=4
    )
