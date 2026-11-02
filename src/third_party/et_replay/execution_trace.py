from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
)

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


FORMAT = "[%(asctime)s] %(filename)s:%(lineno)d [%(levelname)s]: %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

EXECUTION_TRACE_PROCESS_ANNOTATION = "[pytorch|profiler|execution_trace|process]"
EXECUTION_TRACE_THREAD_ANNOTATION = "[pytorch|profiler|execution_trace|thread]"


# OPERATOR: nodes actually does something
# LABEL: nodes used as markers
class NodeType(Enum):
    OPERATOR = 1
    LABEL = 2


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
    "TorchDynamo Cache Lookup",
    "CompiledFunction",
    "Torch-Compiled Region",
]


"""
TensorNode

Contains information about a tensor object (TensorImpl in PyTorch). Each has an
unique ID, element data type, source node IDs, sink nodes IDs, and all the
shapes associated with this tensor used in different contexts. Their producer
consumer relationship is:

source node --> tensor --> sink node

It's important to note that a tensor object maybe shared and used in different
ways in the PyTorch computation graph.
"""


class TensorNode:
    def __init__(self, id: int, dtype: str):
        self.id: int = id
        self.dtype = dtype
        self.sources: Set = set()
        self.sinks: Set = set()
        self.shapes: Set = set()

    def add_source(self, id: int):
        self.sources.add(id)

    def add_sink(self, id: int):
        self.sinks.add(id)

    def add_shape(self, shape: List[Any]):
        self.shapes.add(tuple(shape))


@dataclass
class _CommArgs:
    """Contains communication collective metadata"""

    collective_name: str
    dtype: str
    # .. TODO add more see https://github.com/pytorch/pytorch/issues/124674


"""
Node

Contains all the information about a non-tensor node in the PyTorch computation
graph.

A node has an unique ID. This ID is in the order of execution in the original
graph. Special nodes:
- A single label node __ROOT_PROCESS__ has node ID 1 and is the root of the execution
trace.
- Each thread has its __ROOT_THREAD__ node with an unique ID.

All the input tensors will have ID < node ID.
"""


class Node:
    def __init__(
        self,
        name: str,
        id: int,
        parent_id: int,
        fw_parent_id: int,
        seq_id: int,
        pid: int,
        tid: int,
        fw_tid: int,
        op_schema: str,
        scope: int,
        inputs: List[Any],
        input_types: List[str],
        input_shapes: List[Any],
        outputs: List[Any],
        output_types: List[str],
        output_shapes: List[Any],
        rf_id: Optional[int] = None,
        kernel_backend: Optional[str] = None,
        kernel_file: Optional[str] = None,
        comm_args: Optional[_CommArgs] = None,
        input_strides: Optional[List[Any]] = None,
        output_strides: Optional[List[Any]] = None,
    ):
        self.name: str = name
        self.parent_id: int = parent_id
        self.parent: Optional[Node] = None
        self.children: List[Node] = []
        self.id: int = id
        self.rf_id: Optional[int] = rf_id
        self.kernel_backend: Optional[str] = kernel_backend
        self.kernel_file: Optional[str] = kernel_file
        self.pid: int = pid
        self.tid: int = tid
        self.fw_tid: int = fw_tid
        self.op_schema: str = op_schema
        self.fw_parent_id: int = fw_parent_id
        self.seq_id: int = seq_id
        self.scope: int = scope
        self.type: NodeType = self.detect_type(name, inputs, outputs)
        # self.inputs: List[Any] = [tuple(i) if isinstance(i, list) else i for i in inputs]
        self.inputs: List[Any] = inputs
        self.input_types: List[str] = input_types
        self.input_shapes: List[Any] = input_shapes
        self.input_strides: Optional[List[Any]] = input_strides
        # self.outputs: List[Any] = [tuple(o) if isinstance(o, list) else o for o in outputs]
        self.outputs: List[Any] = outputs
        self.output_types: List[str] = output_types
        self.output_shapes: List[Any] = output_shapes
        self.output_strides: Optional[List[Any]] = output_strides
        self.commArgs: Optional[_CommArgs] = comm_args

    def get_inputs(self) -> Iterable:
        return zip(self.input_types, self.inputs, self.input_shapes)

    def get_outputs(self) -> Iterable:
        return zip(self.output_types, self.outputs, self.output_shapes)

    def set_parent(self, parent: Node):
        assert parent.id == self.parent_id
        self.parent = parent

    def add_child(self, child: Node):
        self.children.append(child)


    def detect_type(self, name: str, inputs: List[Any], outputs: List[Any]) -> NodeType:
        if (
            any(name.startswith(x) for x in LABEL_MARKERS)
            # and not outputs
        ):
            # if outputs:
            #     print(f"{name} has outputs, not expected.")
            return NodeType.LABEL
        else:
            return NodeType.OPERATOR

    def get_tensors(self, param_list: Iterable) -> List[tuple]:
        tensors = []
        for type, input, shape in param_list:
            if type.startswith("Tensor"):
                tensors.append((type, tuple(input), shape))
            # GenericList could have tensor elements
            if type.startswith("GenericList"):
                elem_type = type[12:-1].split(",")
                tensors.extend(self.get_tensors(zip(elem_type, input, shape)))
        return tensors

    def get_input_tensors(self) -> List[tuple]:
        return self.get_tensors(self.get_inputs())

    def get_output_tensors(self) -> List[tuple]:
        return self.get_tensors(self.get_outputs())

    def sort_children(self):
        self.children.sort(key=lambda x: x.id)


class ExecutionTrace:
    def __init__(self, json):
        self.nodes = {}
        self.clean_nodes = {}  # w/o DataLoader ops
        self.tensors = {}
        self.proc_group = {}
        # list of node ids that start an iteration
        self.iteration_ids = []
        self.schema: str = json["schema"]
        pid = json["pid"]
        self.proc_group = {pid: {}}
        nodes_list = json["nodes"]

        # Depending on schema, call the right method
        node_creation_func = {
            "1.0.1": ExecutionTrace._create_node_v1_0_1,
            "1.0.2-chakra.0.0.4": ExecutionTrace._create_node_v1_0_2_chakra_0_0_4,
            # 1.0.3 expands pg name to <pg_name, pg_desc> so it use the same parser as 1.0.2
            "1.0.3-chakra.0.0.4": ExecutionTrace._create_node_v1_0_2_chakra_0_0_4,
            # 1.0.4 adds PT2 kernel backend and kernel file
            "1.0.4-chakra.0.0.4": ExecutionTrace._create_node_v1_0_2_chakra_0_0_4,
            # 1.1.0 includes new comm args in record_param_comms
            "1.1.0-chakra.0.0.4": ExecutionTrace._create_node_v1_0_2_chakra_0_0_4,
            # 1.1.1 includes tensor strides
            "1.1.1-chakra.0.0.4": ExecutionTrace._create_node_v1_1_1_chakra_0_0_4,
            # Add future versions here
        }
        create_node = node_creation_func.get(self.schema, None)
        if create_node is None:
            raise ValueError(
                f"No corresponding node creation function found for schema version {self.schema}"
            )

        for x in nodes_list:
            id = x["id"]
            self.nodes[id] = create_node(pid, x)
            input_tensors = self.nodes[id].get_input_tensors()
            output_tensors = self.nodes[id].get_output_tensors()

            # track the various process and threads we have
            if x["name"] == "__ROOT_THREAD__":
                tid = self.nodes[id].tid
                self.proc_group[pid][tid] = id

            # build tensor reference table
            for t_type, t_id, shape in input_tensors:
                if type(t_id) != tuple:
                    t_id = tuple(t_id)
                if t_id not in self.tensors:
                    dtype = t_type[7:-1]
                    self.tensors[t_id] = TensorNode(t_id, dtype)
                self.tensors[t_id].add_sink(id)
                self.tensors[t_id].add_shape(shape)

            for t_type, t_id, shape in output_tensors:
                if type(t_id) != tuple:
                    t_id = tuple(t_id)
                if t_id not in self.tensors:
                    dtype = t_type[7:-1]
                    self.tensors[t_id] = TensorNode(t_id, dtype)
                self.tensors[t_id].add_source(id)
                self.tensors[t_id].add_shape(shape)

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


    ATTR_TYPES = {
        "fw_parent": int,
        "seq_id": int,
        "fw_tid": int,
        "op_schema": str,
        "rf_id": int,
        "scope": int,
        "tid": int,
        "kernel_backend": str,
        "kernel_file": str,
    }
    OPTIONAL_ATTR = ["kernel_backend", "kernel_file"]

    @classmethod
    def _read_attrs(cls, node: Dict[str, Any]) -> Tuple:
        attr_dict = {
            attr["name"]: cls.ATTR_TYPES[attr["name"]](attr["value"])
            for attr in node["attrs"]
            if attr["name"] in cls.ATTR_TYPES.keys()
        }
        for opt_key in cls.OPTIONAL_ATTR:
            if opt_key not in attr_dict:
                attr_dict[opt_key] = None

        return tuple(
            attr_dict[key] for key in cls.ATTR_TYPES.keys() if key in attr_dict.keys()
        )

    @staticmethod
    def _create_node_v1_0_1(pid, x: Dict[str, Any]) -> Node:
        return Node(
            x["name"],
            x["id"],
            x["parent"],
            x["fw_parent"],
            x["seq_id"],
            pid,
            x["tid"],
            x["fw_tid"],
            x.get("op_schema", ""),
            x["scope"],
            x["inputs"],
            x["input_types"],
            x["input_shapes"],
            x["outputs"],
            x["output_types"],
            x["output_shapes"],
            x.get("rf_id", None),
        )

    @staticmethod
    def _create_node_v1_0_2_chakra_0_0_4(pid, x: Dict[str, Any]) -> Node:
        (
            fw_parent,
            seq_id,
            fw_tid,
            op_schema,
            rf_id,
            scope,
            tid,
            kernel_backend,
            kernel_file,
        ) = ExecutionTrace._read_attrs(x)

        return Node(
            x["name"],
            x["id"],
            x["ctrl_deps"],
            fw_parent,
            seq_id,
            pid,
            tid,
            fw_tid,
            op_schema,
            scope,
            x["inputs"]["values"],
            x["inputs"]["types"],
            x["inputs"]["shapes"],
            x["outputs"]["values"],
            x["outputs"]["types"],
            x["outputs"]["shapes"],
            rf_id,
            kernel_backend,
            kernel_file,
        )

    @staticmethod
    def _create_node_v1_1_1_chakra_0_0_4(pid, x: Dict[str, Any]) -> Node:
        (
            fw_parent,
            seq_id,
            fw_tid,
            op_schema,
            rf_id,
            scope,
            tid,
            kernel_backend,
            kernel_file,
        ) = ExecutionTrace._read_attrs(x)

        return Node(
            x["name"],
            x["id"],
            x["ctrl_deps"],
            fw_parent,
            seq_id,
            pid,
            tid,
            fw_tid,
            op_schema,
            scope,
            x["inputs"]["values"],
            x["inputs"]["types"],
            x["inputs"]["shapes"],
            x["outputs"]["values"],
            x["outputs"]["types"],
            x["outputs"]["shapes"],
            rf_id,
            kernel_backend,
            kernel_file,
            x["inputs"]["strides"],
            x["outputs"]["strides"],
        )

    def get_nodes(self, clean: bool = False):
        if clean:
            return self.clean_nodes
        return self.nodes


    def remove_dataloader_ops(self):
        def check_parent(node):
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
