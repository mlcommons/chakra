import pytest
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
