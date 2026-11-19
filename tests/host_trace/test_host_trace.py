import json
import logging

import pytest
from chakra.src.host_trace.errors import HostTraceError
from chakra.src.host_trace.names import PROCESS_ANNOTATION, THREAD_ANNOTATION
from chakra.src.host_trace.node import HostNode
from chakra.src.host_trace.trace import HostTrace


def create_sample_trace(parent_id: int = 0) -> HostTrace:
    nodes = [
        {
            "id": 1,
            "name": "node1",
            "ctrl_deps": None,
            "inputs": {"values": ["val1"], "shapes": ["shape1"], "types": ["type1"]},
            "outputs": {"values": ["val1"], "shapes": ["shape1"], "types": ["type1"]},
            "attrs": [],
        },
        {
            "id": 2,
            "name": "node2",
            "ctrl_deps": parent_id,
            "inputs": {"values": ["val2"], "shapes": ["shape2"], "types": ["type2"]},
            "outputs": {"values": ["val2"], "shapes": ["shape2"], "types": ["type2"]},
            "attrs": [],
        },
    ]
    return HostTrace({"schema": "1.0.2-chakra.0.0.4", "nodes": nodes})


@pytest.mark.parametrize("parent_id, expected_child_id", [(1, 2), (None, None)])
def test_children_are_attached(parent_id: int, expected_child_id: int) -> None:
    json_node_map = create_sample_trace(parent_id).nodes

    if expected_child_id:
        assert json_node_map[parent_id].children[0].id == expected_child_id
    else:
        assert len(json_node_map[1].children) == 0


def test_trace_without_nodes_is_an_error(make_trace_data) -> None:
    data = make_trace_data()
    del data["nodes"]
    with pytest.raises(HostTraceError, match="missing top-level key 'nodes'"):
        HostTrace(data)


def test_nodes_that_is_not_a_list_of_objects_is_an_error(make_trace_data) -> None:
    with pytest.raises(HostTraceError, match="'nodes' must be a list of node objects"):
        HostTrace(make_trace_data(nodes=[1, 2]))


def test_duplicate_node_id_is_an_error_naming_both_nodes(make_trace_data, make_node) -> None:
    data = make_trace_data()
    data["nodes"].append(make_node(4, "relu", 3))
    with pytest.raises(HostTraceError, match="duplicate node id 4 \\(addmm and relu\\)"):
        HostTrace(data)


def test_process_root_is_not_its_own_child_and_children_are_sorted(make_trace_data, make_node) -> None:
    data = make_trace_data()
    data["nodes"].insert(0, make_node(6, "relu", 2))  # before node 3 in the file, after it by id
    trace = HostTrace(data)
    assert [c.id for c in trace.nodes[1].children] == [2]
    assert [c.id for c in trace.nodes[2].children] == [3, 6]


def test_process_root_is_found_by_name_even_with_id_0(make_trace_data, make_node) -> None:
    # chakra_trace_link renumbers ids from 0: the root of a linked trace is id 0, and id 1 is a thread root.
    nodes = [make_node(0, PROCESS_ANNOTATION, 0), make_node(1, THREAD_ANNOTATION, 0), make_node(2, "linear", 1)]
    assert HostTrace(make_trace_data(nodes=nodes)).process_root().id == 0


def test_no_or_several_process_roots_is_an_error(make_trace_data, make_node) -> None:
    data = make_trace_data()
    data["nodes"] = [n for n in data["nodes"] if n["id"] != 1]
    with pytest.raises(HostTraceError, match="found none"):
        HostTrace(data).process_root()
    data = make_trace_data()
    data["nodes"].append(make_node(9, PROCESS_ANNOTATION, 9))
    with pytest.raises(HostTraceError, match="found 2 \\(ids 1, 9\\)"):
        HostTrace(data).process_root()


def test_orphans_and_parent_cycles_are_not_reachable_and_the_walk_ends(make_trace_data, make_node) -> None:
    data = make_trace_data()
    data["nodes"] += [make_node(7, "orphan", 999), make_node(8, "in_cycle", 9), make_node(9, "in_cycle", 8)]
    trace = HostTrace(data)
    root = trace.process_root()
    assert [n.id for n in trace.reachable_from(root)] == [1, 2, 3, 4, 5]


def test_nodes_are_created_with_the_given_node_class(make_trace_data) -> None:
    class Custom(HostNode):
        pass

    assert all(type(n) is Custom for n in HostTrace(make_trace_data(), node_class=Custom).nodes.values())


def test_parsing_does_not_modify_the_raw_trace(make_trace_data) -> None:
    data = make_trace_data()
    before = json.dumps(data, sort_keys=True)
    HostTrace(data)
    assert json.dumps(data, sort_keys=True) == before


def test_one_layout_warning_when_nodes_lack_keys_of_their_version(make_trace_data, caplog) -> None:
    data = make_trace_data()
    for node in data["nodes"][:3]:
        del node["inputs"]["strides"]  # schema 1.1.1 has strides
    with caplog.at_level(logging.WARNING):
        HostTrace(data)
    assert len(caplog.records) == 1
    assert "3 of 5 nodes do not have every key of schema 1.1.1-chakra.0.0.4" in caplog.text


def test_no_warning_for_a_trace_that_matches_its_version(make_trace_data, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        HostTrace(make_trace_data())
    assert not caplog.records
