from typing import Dict

import pytest
from chakra.src.host_trace.node import HostNode
from chakra.src.host_trace.trace import attach_children


def create_sample_graph(parent_id: int = 0, expected_child_id: int = 0) -> Dict[int, HostNode]:
    node1_data = {
        "id": 1,
        "name": "node1",
        "ctrl_deps": None,
        "inputs": {"values": ["val1"], "shapes": ["shape1"], "types": ["type1"]},
        "outputs": {"values": ["val1"], "shapes": ["shape1"], "types": ["type1"]},
        "attrs": [],
    }
    node2_data = {
        "id": 2,
        "name": "node2",
        "ctrl_deps": parent_id,
        "inputs": {"values": ["val2"], "shapes": ["shape2"], "types": ["type2"]},
        "outputs": {"values": ["val2"], "shapes": ["shape2"], "types": ["type2"]},
        "attrs": [],
    }
    node1 = HostNode("1.0.2-chakra.0.0.4", node1_data)
    node2 = HostNode("1.0.2-chakra.0.0.4", node2_data)
    return {1: node1, 2: node2}


@pytest.mark.parametrize("parent_id, expected_child_id", [(1, 2), (None, None)])
def test_attach_children(parent_id: int, expected_child_id: int) -> None:
    json_node_map = create_sample_graph(parent_id, expected_child_id)

    attach_children(json_node_map)

    if expected_child_id:
        assert json_node_map[parent_id].children[0].id == expected_child_id
    else:
        assert len(json_node_map[1].children) == 0
