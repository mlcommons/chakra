from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
)

import argparse
import copy
import gzip
import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO, Tuple

import pydot


FORMAT = "[%(asctime)s] %(filename)s:%(lineno)d [%(levelname)s]: %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

PROFILER_STEP_ANNOTATION: str = "ProfilerStep"
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

    def is_leaf_tensor(self):
        return (
            not self.sources
        ) and self.sinks  # A tensor having no sources yet having some sinks is a leaf tensor


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

    # Self-type is op yet none of its parent in any hierarchy is an op.
    def is_op(self, detail: bool = False):
        if detail:
            return self.type == NodeType.OPERATOR
        else:
            has_parent_op = False
            tmp = self
            while 1:
                if (
                    tmp.parent is None or tmp.id == tmp.parent_id
                ):  # Reach the root process
                    break
                if tmp.parent.type == NodeType.OPERATOR:
                    has_parent_op = True
                    break
                tmp = tmp.parent

            return self.type == NodeType.OPERATOR and not has_parent_op

    def is_leaf_op(self) -> bool:
        return not self.children

    def _get_grandest_parent(self) -> Node:
        if (
            self.parent is None
            or self.parent.name == "## BENCHMARK ##"
            or self.parent.name == "__ROOT_THREAD__"
        ):
            return self
        return self.parent._get_grandest_parent()

    def get_grandest_parent(self) -> Node:
        return self._get_grandest_parent()

    def _get_base_op(self) -> Node:
        if self.parent is None or self.parent.type == NodeType.LABEL:
            return self
        return self.parent._get_base_op()

    def get_base_op(self) -> Node:
        return self._get_base_op()

    def _get_child_by_name(self, name) -> Optional[Node]:
        for c in self.children:
            if name in c.name:
                return c
            node = c._get_child_by_name(name)
            if node:
                return node
        return None

    def get_child_by_name(self, names) -> Optional[Node]:
        for name in names:
            node = self._get_child_by_name(name)
            if node is not None:
                return node
        return None

    def _get_parent_by_name(self, name) -> Optional[Node]:
        if self.parent:
            if name in self.parent.name:
                return self.parent
            node = self.parent._get_parent_by_name(name)
            if node:
                return node
        return None

    def get_parent_by_name(self, names) -> Optional[Node]:
        for name in names:
            node = self._get_parent_by_name(name)
            if node is not None:
                return node
        return None

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

    def _versiontuple(self, v: str) -> Tuple[int]:
        return tuple(map(int, (v.split("."))))

    def schema_pytorch(self) -> Tuple[int]:
        return self._versiontuple(self.schema.split("-")[0])

    def schema_chakra(self) -> Tuple[int]:
        if "-" not in self.schema:
            return (0, 0, 0)
        return self._versiontuple(self.schema.split("-")[1])

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

    def set_iterations(self, step_annotation=PROFILER_STEP_ANNOTATION) -> None:
        """Sets an array demarcating interations in the trace"""
        self.iteration_ids = [1]

        for id in sorted(self.nodes.keys()):
            if step_annotation in self.nodes[id].name:
                self.iteration_ids.append(id)
        self.iteration_ids = sorted(self.iteration_ids)
        logging.info(f"Iteration node ids list = {self.iteration_ids}")

    def iterations(self) -> Optional[int]:
        if len(self.iteration_ids) == 0:
            return None
        return len(self.iteration_ids) - 1

    def get_unique_ops(
        self, detail: bool = False, clean: bool = False, json_format: bool = False
    ):
        def get_param(value, type, shape):
            type = type.lower()
            SCALAR_TYPES = {"int", "long", "float", "double", "bool"}
            param = {"type": type}
            if type.startswith("genericlist"):
                param = {"type": "genericlist"}
                param["value"] = []
                type_list = type[12:-1].split(",")
                param_list = zip(value, type_list, shape)
                for v, t, s in param_list:
                    param["value"].append(get_param(v, t, s))
                param["size"] = len(value)
            elif type in SCALAR_TYPES or type == "device":
                param["value"] = value
            elif type.startswith("tensor"):
                param["type"] = "tensor"
                param["dtype"] = type[7:-1]
                param["shape"] = shape
            return param

        def convert_inputs(inputs, types, shapes):
            input_info = zip(inputs, types, shapes)
            params = []
            for value, type, shape in input_info:
                params.append(get_param(value, type, shape))
            return params

        ops = {}
        nodes_dict = self.clean_nodes if clean else self.nodes
        for n in nodes_dict.values():
            if n.is_op(detail):
                if n.name in ops:
                    ops[n.name]["count"] += 1
                    ops[n.name]["inputs"].append(
                        convert_inputs(n.inputs, n.input_types, n.input_shapes)
                    )
                else:
                    ops[n.name] = {"count": 1}
                    ops[n.name]["inputs"] = [
                        convert_inputs(n.inputs, n.input_types, n.input_shapes)
                    ]
        # Remove dupliates
        for attr in ops.values():
            # use json to serialize list of dict to string for set
            unique = {json.dumps(x, sort_keys=True) for x in attr["inputs"]}
            attr["inputs"] = list(map(json.loads, unique))

        return ops

    def print_op_stats(
        self, detail: bool = False, clean: bool = False, json_format: bool = False
    ):
        ops = self.get_unique_ops(detail, clean, json_format)

        if json_format:
            print(json.dumps(ops, indent=2, sort_keys=True))
        else:
            print("### OP STATS ###")
            for key, val in sorted(ops.items()):
                print(f"op: {key}")
                print(f"  count: {val['count']}")
                print("  unique inputs:")
                for i in val["inputs"]:
                    print(f"  input: {i}")

    def gen_graphviz(self, file_name):
        dot = pydot.Dot(graph_type="digraph")
        for id, n in self.nodes.items():
            dot.add_node(
                pydot.Node(
                    id,
                    label=f"{n.name} ({n.id})",
                    shape="box",
                    style="filled",
                    fillcolor="#fffbed",
                )
            )
        for id in self.tensors:
            dot.add_node(
                pydot.Node(id, label=f"T{id}", style="filled", fillcolor="#e8faff")
            )
        nodes = len(self.nodes) + len(self.tensors)
        edges = 0
        for id, n in self.nodes.items():
            dot.add_edge(pydot.Edge(n.parent_id, id, arrowhead="odiamond"))
            edges += 1
            for _, input, _ in n.get_input_tensors():
                dot.add_edge(pydot.Edge(input, id))
                edges += 1
            for _, output, _ in n.get_output_tensors():
                dot.add_edge(pydot.Edge(id, output))
                edges += 1
        dot.write_svg(file_name, prog="dot")
        logging.info(f"nodes: {nodes}")
        logging.info(f"edges: {edges}")

    def gen_graphml(self, file_name):
        graphml = GraphML(self)
        graphml.write("execution trace", file_name)

    def gen_graph(self, file_name, type=None):
        dot_max_nodes = 300
        if (
            len(self.nodes) < dot_max_nodes or type == "graphviz"
        ) and type != "graphml":
            out_name = f"{file_name}.svg"
            self.gen_graphviz(out_name)
        else:
            out_name = f"{file_name}.graphml"
            self.gen_graphml(out_name)

        print(f"Execution trace written to {out_name}")

    def print_tensors(self, detail: bool = False):
        print("### TENSORS ###")
        for id, t in self.tensors.items():
            if detail:
                print(f"ID {id}:")
                print("     type:", t.dtype)
                print("   shapes:", t.shapes)
                print("  sources:", t.sources)
                print("    sinks:", t.sinks)
            else:
                print(f"id = {id}:")
                print("     type:", t.dtype)
                print("   shapes:", t.shapes)

    def _print_tree_preorder(self, n, indent, pid, tid, detail: bool):
        if n.type == NodeType.OPERATOR:
            print(f"{indent}({n.parent_id}:{n.id}) {n.name}")
            inputs = list(n.get_inputs())
            print(f"{indent}    arg: {inputs}")
            outputs = list(n.get_outputs())
            print(f"{indent}    out: {outputs}")
            if not detail:
                return
        else:
            print(f"{indent}({n.id}) {n.name}")
        for c in n.children:
            self._print_tree_preorder(c, indent + "  ", pid, tid, detail)

    def print_tree(self, detail: bool = False):
        print("### Execution Tree ###")
        for pid, threads in self.proc_group.items():
            print(f"process: {pid}")
            for tid in sorted(threads):
                print(f"  thread: {tid}")
                thread_node = self.nodes[threads[tid]]
                self._print_tree_preorder(thread_node, "    ", pid, tid, detail)

    def node_depend(self, id: int):
        n = self.nodes[id]
        print(f"ID {id}: Operator")
        print("          name:", n.name)
        print("            id:", n.id)
        print("         rf_id:", n.rf_id)
        print("           tid:", n.tid)
        print("     parent_id:", n.parent_id)
        print("        fw_tid:", n.fw_tid)
        print("          type:", n.type)
        print("     op_schema:", n.op_schema)
        print("  fw_parent_id:", n.fw_parent_id)
        print("         scope:", n.scope)
        print("      children:", [child.id for child in n.children])
        print("        inputs:")
        for dtype, tensor_id, shape in n.get_input_tensors():
            prev_id = 0
            for s in self.tensors[tensor_id].sources:
                if s < id and s > prev_id:
                    prev_id = s
            if prev_id not in self.nodes:
                print(f"Missing source node for {prev_id}")
            elif prev_id:
                print(
                    f"{' '*16}{tensor_id}: {dtype} {shape} <-- {prev_id} ({self.nodes[prev_id].name})"
                )
            else:
                print(f"{' '*16}{tensor_id}: {dtype} {shape}")

        print("       outputs:")
        for dtype, tensor_id, shape in n.get_output_tensors():
            next_id = sys.maxsize
            for s in self.tensors[tensor_id].sinks:
                # We could have cycle (s == id), where an op read and write to
                # the same tensor.
                if s > id and s < next_id:
                    next_id = s
            if next_id != sys.maxsize:
                print(
                    f"{' '*16}{tensor_id}: {dtype} {shape} --> {next_id} ({self.nodes[next_id].name})"
                )
            else:
                print(f"{' '*16}{tensor_id}: {dtype} {shape}")

    def tensor_depend(self, id: int):
        t = self.tensors[id]
        print(f"ID {id}: Tensor")
        print("     type:", t.dtype)
        print("   shapes:", t.shapes)
        sources = {}
        for node_id in t.sources:
            sources[node_id] = self.nodes[node_id].name
        print("  sources:", sources)
        sinks = {}
        for node_id in t.sinks:
            sinks[node_id] = self.nodes[node_id].name
        print("    sinks:", sinks)

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

    def clone_one_iteration(self, n) -> ExecutionTrace:
        """Clone the entire Execution Trace but with only one iteration

        @args: n (int): specific iteration to copy, zero based index.
        """
        assert n >= 0, "Iteration too low"
        assert n < len(self.iteration_ids), "Iteration too high"

        start_id, end_id = self.iteration_ids[n], self.iteration_ids[n + 1]
        logging.info(
            f"Copying nodes for iter {n} for ids in the range [{start_id}, {end_id})"
        )

        clone = copy.deepcopy(self)
        trimmed_nodes = filter(
            lambda p: (p[1].id >= start_id and p[1].id < end_id)
            or p[1].parent_id == 1,  # process and thread nodes
            clone.nodes.items(),
        )
        clone.nodes = dict(trimmed_nodes)
        node_id_set = clone.nodes.keys()
        logging.debug(f"filtered node ID set = {node_id_set}")

        # There may be incomplete user annotations that are parents to events
        # in the execution trace. If so just fix up the parent to the corresponding thread parent
        # get all the top level thread nodes
        thread_nodes = {
            node.tid: node
            for node in clone.nodes.values()
            if node.parent_id == 1 and (EXECUTION_TRACE_THREAD_ANNOTATION in node.name)
        }
        assert len(thread_nodes) > 0

        for node in clone.nodes.values():
            if (
                node.parent is not None
                and node.parent_id != 1
                and (node.parent_id not in node_id_set)
            ):
                logging.info(
                    f"Fixing parent for node id = {node.id}, parent = {node.parent_id}"
                )

                thread_parent = thread_nodes[node.tid]
                node.parent_id = thread_parent.id
                node.set_parent(thread_parent)
                thread_parent.add_child(node)

        # Similarly fix the children relationship
        for node in clone.nodes.values():
            children = [child for child in node.children if child.id in node_id_set]
            node.children = children

        # remove all dataloader ops
        clone.clean_nodes = {}
        clone.remove_dataloader_ops()

        logging.info(f"Nodes trimmed ET = {len(clone.get_nodes())}")
        return clone


class GraphML:
    def __init__(self, execution_trace: ExecutionTrace):
        self.nodes: List = []
        self.edges: List = []
        # construct op nodes and edges
        for id, n in execution_trace.nodes.items():
            self._create_node(id, f"{n.name} ({n.id})", n.name)
        for tensor in execution_trace.tensors.values():
            self._create_tensor_node(tensor)
        for id, n in execution_trace.nodes.items():
            self._create_edge(n.parent_id, id)
            for _, input, _ in n.get_input_tensors():
                self._create_edge(input, id)
            for _, output, _ in n.get_output_tensors():
                self._create_edge(id, output)

        logging.info(f"nodes: {len(self.nodes)}")
        logging.info(f"edges: {len(self.edges)}")

    def _create_node(
        self,
        node_id: int,
        name: str = "",
        type: str = "",
        input: str = "",
        output: str = "",
        arg: str = "",
        device: str = "",
        engine: str = "",
        is_grad: str = "",
        info: str = "",
    ):
        self.nodes.append(
            {
                "id": node_id,
                "name": name,
                "type": type,
                "input": input,
                "output": output,
                "arg": arg,
                "is_grad": is_grad,
                "device": device,
                "engine": engine,
                "info": info,
            }
        )

    def _create_tensor_node(self, tensor):
        self._create_node(
            tensor.id,
            f"T{tensor.id}",
            type="Tensor",
            input=tensor.sources,
            output=tensor.sinks,
        )

    def _create_edge(self, source, target):
        self.edges.append({"source": source, "target": target})

    def write(self, name, file_name):
        """
        Given the graph information, write a GraphML file.
        Parameters:
        name (string): Name of the network.
        file_name: Output file name.
        """

        def write_header():
            out.write('<?xml version="1.0" encoding="UTF-8"?>\n')

        def graphml_begin():
            graphml = (
                '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" '
                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
                'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n'
                '<key id="name" attr.name="name" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="type" attr.name="type" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="input" attr.name="input" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="output" attr.name="output" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="arg" attr.name="arg" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="device" attr.name="device" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="engine" attr.name="engine" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="is_grad" attr.name="is_grad" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="info" attr.name="info" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
            )
            out.write(graphml)

        def graphml_end():
            out.write("</graphml>\n")

        def write_graph():
            out.write(f'<graph id="{name}" edgedefault="directed">\n')
            for node in self.nodes:
                write_node(node)
            for id, edge in enumerate(self.edges):
                write_edge(id, edge["source"], edge["target"])
            out.write("</graph>\n")

        def write_node(node):
            out.write(f'<node id="{node["id"]}">\n')
            for name, data in node.items():
                if name != "id" and data:
                    out.write(f'  <data key="{name}">{data}</data>\n')
            out.write("</node>\n")

        def write_edge(id, source, target):
            out.write(f'<edge id="e_{id}" source="{source}" target="{target}"/>\n')

        with open(file_name, "wt") as out:
            write_header()
            graphml_begin()
            write_graph()
            graphml_end()


def main():
    parser = argparse.ArgumentParser(
        description="Execution trace building and analysis"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="input execution trace json file."
    )
    parser.add_argument(
        "--step-annotation",
        type=str,
        default=PROFILER_STEP_ANNOTATION,
        help="Annotation in the trace that distinguishes trace iterations.",
    )
    parser.add_argument(
        "--graph",
        dest="graph",
        default=False,
        action="store_true",
        help="generate a graph output using either graphviz (dot) or graphml.",
    )
    parser.add_argument(
        "--graphviz",
        dest="graphviz",
        default=False,
        action="store_true",
        help="generate graph output in graphviz (dot) format.",
    )
    parser.add_argument(
        "--graphml",
        dest="graphml",
        default=False,
        action="store_true",
        help="generate graph output in graphml format.",
    )
    parser.add_argument(
        "--list-tensor",
        dest="list_tensor",
        default=False,
        action="store_true",
        help="list all tensors and its shape and type information.",
    )
    parser.add_argument(
        "--list-op",
        dest="list_op",
        default=False,
        action="store_true",
        help="list all the ops in the execution trace.",
    )
    parser.add_argument(
        "--node",
        type=int,
        dest="node",
        default=-1,
        action="store",
        help="query information about a node and its dependency given a node ID.",
    )
    parser.add_argument(
        "--detail",
        dest="detail",
        default=False,
        action="store_true",
        help="combined with some other options, will show more detailed information.",
    )
    parser.add_argument(
        "--clean",
        dest="clean",
        default=False,
        action="store_true",
        help="remove all dataloader ops.",
    )
    parser.add_argument(
        "--json",
        dest="json",
        default=False,
        action="store_true",
        help="for --list-op option, generate outputs in JSON format.",
    )
    parser.add_argument(
        "--tree",
        dest="tree",
        default=False,
        action="store_true",
        help="generate an execution tree by process, threads, and order of node execution.",
    )
    args = parser.parse_args()

    execution_json: str = args.input

    with (
        gzip.open(execution_json, "rb")
        if execution_json.endswith("gz")
        else open(execution_json, "r")
    ) as execution_data:
        execution_data: TextIO
        execution_trace: ExecutionTrace = ExecutionTrace(json.load(execution_data))
        execution_trace.set_iterations(args.step_annotation)
        # execution_trace = execution_trace.clone_one_iteration(2)

        if args.list_op:
            execution_trace.print_op_stats(args.detail, args.json)
        if args.list_tensor:
            execution_trace.print_tensors(args.detail)
        if args.tree:
            execution_trace.print_tree(args.detail)
        if args.node != -1:
            if args.node in execution_trace.nodes:
                execution_trace.node_depend(args.node)
            elif args.node in execution_trace.tensors:
                execution_trace.tensor_depend(args.node)
            else:
                logging.error(f"node {args.node} not found.")

        if args.graph or args.graphviz or args.graphml:
            out_file: str = "execution_trace"
            if args.graphviz:
                execution_trace.gen_graph(out_file, "graphviz")
            elif args.graphml:
                execution_trace.gen_graph(out_file, "graphml")
            else:
                execution_trace.gen_graph(out_file)


if __name__ == "__main__":
    main()