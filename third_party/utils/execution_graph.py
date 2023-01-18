import json

from enum import Enum
from typing import Any, Dict, List, Union

# OPERATOR: nodes actually does something
# LABEL: nodes used as markers
class NodeType(Enum):
    NodeType = 1
    OPERATOR = 2
    LABEL = 3

# Label markers
LABEL_MARKERS = [
    "##",
    "__",
    "module::",
    "DLRM ",
    "DistributedDataParallel",
    "Profiler",
    "[pytorch|",
    "forward",
    "backward",
    "Optimizer.zero_grad",
    "[param",
    "<forward op>",
    "reduce-grads",
    "multiply-grads",
    "clip-grads",
    "optimizer",
    "gans_torchscript_ops::",
    "split_with_sizes",
    "chunk",
    "All2All_Pooled_ReqBackward",
    "All2All_Pooled_Req",
    "All2All_Pooled_Wait",
    "c10d::",
]


"""
Node

Contains all the information about a non-tensor node in the PyTorch computation
graph.

A node has an unique ID. This ID is in the order of execution in the original
graph. Special nodes:
- A single label node __ROOT_PROCESS__ has node ID 1 and is the root of the execution
graph.
- Each thread has its __ROOT_THREAD__ node with an unique ID.

All the input tensors will have ID < node ID.
"""

class Node:
    def __init__(
        self,
        name: str,
        id: int,
        parent_id: int,
        tid: int,
        op_schema: str,
        inputs: List[Any],
        input_types: List[str],
        input_shapes: List[Any],
        outputs: List[Any],
    ) -> None:
        self.name: str = name
        self.parent_id: int = parent_id
        self.parent: Any = None
        self.children: List[Any] = []
        self.id: int = id
        self.tid: int = tid
        self.op_schema: str = op_schema
        self.type: NodeType = self.detect_type(name, inputs, outputs)
        self.inputs: List[Any] = inputs
        self.input_types: List[str] = input_types
        self.input_shapes: List[Any] = input_shapes
        self.outputs: List[Any] = outputs

    def set_parent(
        self,
        parent: Any
    ) -> None:
        assert parent.id == self.parent_id
        self.parent = parent

    def add_child(
        self,
        child: Any
    ) -> None:
        self.children.append(child)

    def detect_type(
        self,
        name: str,
        inputs: List[Any], outputs: List[Any]
    ) -> NodeType:
        if any(name.startswith(x) for x in LABEL_MARKERS):
            return NodeType.LABEL
        else:
            return NodeType.OPERATOR

    def sort_children(self) -> None:
        self.children.sort(key=lambda x: x.id)

class ExecutionGraph:
    def __init__(
        self,
        json: Any
    ) -> None:
        self.nodes = {}
        self.clean_nodes = {}  # w/o DataLoader ops
        nodes_list = json["nodes"]
        for x in nodes_list:
            id = x["id"]
            tid = x["tid"]
            self.nodes[id] = Node(
                x["name"],
                id,
                x["parent"],
                tid,
                x["op_schema"] if "op_schema" in x.keys() else "",
                x["inputs"],
                x["input_types"],
                x["input_shapes"],
                x["outputs"]
            )

        # populate parent and children nodes
        for n in self.nodes.values():
            # skip root node
            if n.id != 1:
                if n.parent_id in self.nodes:
                    self.nodes[n.parent_id].add_child(n)
                    n.set_parent(self.nodes[n.parent_id])

        # sort children nodes by id
        for n in self.nodes.values():
            n.sort_children()

        # remove all dataloader ops
        self.remove_dataloader_ops()

    def get_nodes(
        self,
        clean: bool = False
    ) -> List[Any]:
        if clean:
            return self.clean_nodes
        return self.nodes

    def remove_dataloader_ops(self) -> None:
        def check_parent(node: Node) -> bool:
            tmp = node
            while tmp and tmp.id != tmp.parent_id:  # while not the final root
                if "DataLoader" in tmp.name:
                    return True
                tmp = tmp.parent
            return False

        if len(self.clean_nodes.keys()) == 0:  # clean_nodes is empty
            for id, node in self.nodes.items():
                if not check_parent(node):  # if the op is not under dataloader
                    self.clean_nodes[id] = node
