#!/usr/bin/env python3

import bisect
import copy
import json
import logging

from enum import Enum
from typing import Any, Dict, List

from chakra.third_party.utils.protolib import encodeMessage as encode_message
from chakra.et_def.et_def_pb2 import (
    GlobalMetadata,
    Node as ChakraNode,
    AttributeProto as ChakraAttr,
    INVALID_NODE,
    COMP_NODE,
    COMM_COLL_NODE,
    ALL_REDUCE,
    ALL_GATHER,
    BROADCAST,
    ALL_TO_ALL,
    REDUCE_SCATTER,
)


class PyTorchNodeType(Enum):
    CPU_OP = 1
    GPU_OP = 2
    LABEL = 3 # Non-operator nodes


class PyTorch2ChakraConverter:
    def __init__(
        self,
        input_filename: str,
        output_filename: str,
        num_dims: int,
        logger: logging.Logger
    ) -> None:
        try:
            self.pytorch_et = open(input_filename, "r")
        except IOError as e:
            raise Exception(f"Could not open file {input_filename}")
        pytorch_et_data = json.load(self.pytorch_et)
        self.pt_schema = pytorch_et_data["schema"]
        self.pt_pid = pytorch_et_data["pid"]
        self.pt_time = pytorch_et_data["time"]
        self.pt_start_ts = pytorch_et_data["start_ts"]
        self.pt_finish_ts = pytorch_et_data["finish_ts"]
        self.pt_nodes = pytorch_et_data["nodes"]

        try:
            self.chakra_et = open(output_filename, "wb")
        except IOError as e:
            raise Exception(f"Could not open file {output_filename}")

        self.num_dims = num_dims
        self.logger = logger

        # All PyTorch CPU operators are kept in pt_cpu_node_dict.
        # Mappings between PyTorch NIDs and PyTorch nodes.
        self.pt_cpu_node_dict = {}

        # All PyTorch GPU operators are kept in pt_gpu_node_dict.
        # Mappings between PyTorch CPU node IDs (parent) and PyTorch GPU nodes (children).
        self.pt_gpu_node_dict = {}

        # All record_param_comms nodes are tracked in pt_record_param_comms_node_dict.
        # Mappings between parent PyTorch NIDs and PyTorch record_param_comms nodes.
        self.pt_record_param_comms_node_dict = {}

        # All PyTorch NCCL nodes are kept in pt_nccl_node_dict.
        # Mappings between parent PyTorch NIDs and PyTorch NCCL nodes.
        self.pt_nccl_node_dict = {}

        # All Chakra nodes are maintained in ck_node_dict.
        # Mappings between Chakra NIDs and Chakra nodes.
        self.ck_node_dict = {}

        # A list of NIDs to enforce dependencies between phases.
        # Phase of training iteration may include forward-pass, back-prop, optimizer, etc.
        # We assume a phase ops cannot start until after all ops of previous phases are executed
        self.inter_phase_dependency = []

        # ---------------------------------------------------------------------
        # These four dictionaries are used for identifying data dependencies
        # between operators. Data dependencies can be discovered by identifying
        # tensor input-output relationships between operators.
        #
        # Tensors have two types of IDs: storage ID and tensor ID
        # A storage ID is considered as valid when it is larger than zero.
        # When a storage ID is valid, it should be used for identifying a tensor
        # Otherwise, a tensor ID should be utilized.
        # ---------------------------------------------------------------------
        # Mapping between storage_id and nid
        self.input_storage_id_nid_dict = {} # storage_id is an input of a node with nid
        self.output_storage_id_nid_dict = {} # storage_id is an output of a node with nid
        # Mapping between tensor_id and nid
        self.input_tensor_id_nid_dict = {} # tensor_id is an input of a node with nid
        self.output_tensor_id_nid_dict = {} # tensor_id is an output of a node with nid

    def __del__(self):
        if self.pytorch_et and not self.pytorch_et.closed:
            self.pytorch_et.close()
        if self.chakra_et and not self.chakra_et.closed:
            self.chakra_et.close()

    @staticmethod
    def is_valid_tensor(
        obj: Any
    ) -> bool:
        """
        Returns true if a given object is a valid tensor.

        An object is a valid tensor object when it is a list and the length of
        the list is six.
        """
        return isinstance(obj, list) and (len(obj) == 6)

    @staticmethod
    def get_storage_id_from_tensor(
        tensor: List[Any]
    ) -> int:
        """
        Returns the storage ID of a tensor.
        """
        if len(tensor) < 2:
            raise IndexError("Index out of bounds")
        return tensor[1]

    @staticmethod
    def get_tensor_id_from_tensor(
        tensor: List[Any]
    ) -> int:
        """
        Returns the tensor ID of a tensor.
        """
        if len(tensor) < 1:
            raise IndexError("Index out of bounds")
        return tensor[0]

    def has_valid_storage_id(
        self,
        tensor: List[Any]
    ) -> bool:
        """
        Returns true if a given tensor has a valid storage ID.

        A storage ID is considered valid if it is larger than zero.
        When a storage ID is valid, it should be used instead of a tensor ID.
        """
        storage_id = self.get_storage_id_from_tensor(tensor)
        return storage_id > 0

    @staticmethod
    def has_cat_field(
        node: Dict[str, Any]
    ) -> bool:
        """
        Returns true if a PyTorch node has a category field.
        """
        return "cat" in node.keys()

    @staticmethod
    def get_cat_field(
        node: Dict[str, Any]
    ) -> bool:
        """
        Returns the category field of a given PyTorch node.
        """
        return node["cat"]

    @staticmethod
    def has_dur(
        node: Dict[str, Any]
    ) -> bool:
        """
        Returns true if a PyTorch node has a duration field.
        """
        return "dur" in node.keys()

    def get_pytorch_node_type(
        self,
        node: Dict[str, Any]
    ) -> PyTorchNodeType:
        if self.is_gpu_op(node):
            return PyTorchNodeType.GPU_OP
        elif (node["op_schema"] or node["outputs"])\
                or ("c10d::" in node["name"] or ("nccl:" in node["name"])):
            return PyTorchNodeType.CPU_OP
        else:
            return PyTorchNodeType.LABEL

    @staticmethod
    def is_record_param_comms_node(
        node: Dict[str, Any]
    ) -> bool:
        """
        Returns true if a PyToch node has "record_param_comms" in its name.
        """
        return "record_param_comms" in node["name"]

    @staticmethod
    def is_nccl_node(
        node: Dict[str, Any]
    ) -> bool:
        """
        Returns true if a PyToch node is a NCCL node.
        """
        return "nccl:" in node["name"]

    def is_cpu_op_with_dur(
        self,
        node: Dict[str, Any]
    ) -> bool:
        """
        Returns true if a PyTorch node is a CPU operator and has a duration field.
        """
        return (self.get_pytorch_node_type(node) == PyTorchNodeType.CPU_OP)\
                and self.has_dur(node)

    def is_cpu_op(
        self,
        node: Dict[str, Any]
    ) -> bool:
        """
        Takes a PyTorch node and returns true if the node is a CPU operator.
        """
        return self.get_pytorch_node_type(node) == PyTorchNodeType.CPU_OP

    @staticmethod
    def get_collective_comm_type(
        node: Dict[str, Any]
    ) -> int:
        """
        Returns the collective communication type of a given PyTorch node.
        """
        if "all_reduce" in node["name"]:
            return ALL_REDUCE
        elif "all_to_all" in node["name"]:
            return ALL_TO_ALL
        elif "all_gather" in node["name"]:
            return ALL_GATHER
        elif "reduce_scatter" in node["name"]:
            return REDUCE_SCATTER
        elif "broadcast" in node["name"]:
            return BROADCAST
        else:
            node_name = node["name"]
            raise ValueError(f"{node_name} is not supported")
        return INVALID_COMM

    @staticmethod
    def get_data_type_size(
        data_type: str
    ) -> int:
        """
        Returns the data type size of a given data type in string.

        References
        * https://pytorch.org/docs/stable/tensors.html
        * https://github.com/pytorch/pytorch/blob/master/c10/util/Half.h
        """
        data_type_size_dict = {
                "Tensor(float32)": 4,
                "Tensor(float)": 4,
                "Tensor(float64)": 8,
                "Tensor(double)": 8,
                "Tensor(float16)": 2,
                "Tensor(half)": 2,
                "Tensor(bfloat16)": 2,
                "Tensor(complex64)": 8,
                "Tensor(complex128)": 16,
                "Tensor(uint8)": 1,
                "Tensor(int8)": 1,
                "Tensor(int16)": 2,
                "Tensor(short)": 2,
                "Tensor(int32)": 4,
                "Tensor(int)": 4,
                "Tensor(int64)": 8,
                "Tensor(long)": 8,
                "Tensor(c10::Half)": 2,
                "Tensor(unsigned char)": 1,
                "Tensor(long int)": 8,
        }
        try:
            data_type_size = data_type_size_dict[data_type]
            return data_type_size
        except:
            raise ValueError(f"{data_type} is unsupported")

    def get_chakra_node_type_from_pytorch_node(
        self,
        node: Dict[str, Any]
    ) -> int:
        if self.has_cat_field(node) and ("ncclKernel" in node["name"]):
            return COMM_COLL_NODE
        elif self.has_cat_field(node):
            return COMP_NODE
        elif ("c10d::" in node["name"]) or ("nccl:" in node["name"]):
            return COMM_COLL_NODE
        elif (node["op_schema"] != "") or node["outputs"]:
            return COMP_NODE
        return INVALID_NODE

    def has_gpu_op(
        self,
        nid: int
    ) -> bool:
        """
        Returns true if a Chakra node has any associated GPU operator.
        """
        return nid in self.pt_gpu_node_dict.keys()

    def get_comm_size(
        self,
        node: Dict[str, Any]
    ) -> int:
        """
        Calculates the communication size for a given input_type and input_shape.
        """
        comm_size = 1
        for input_types in node["input_types"]:
            comm_size *= self.get_data_type_size(input_types)
        for input_shape_outer in node["input_shapes"]:
            for input_shape_inner in input_shape_outer:
                comm_size = comm_size * input_shape_inner
        return comm_size

    def sort_pytorch_nodes_with_starting_time(
        self
    ) -> None:
        """
        Sorts PyTorch nodes with their starting time ("ts").

        Sorting helps executing nodes with earlier starting time first.
        """
        self.pt_nodes = sorted(self.pt_nodes, key=lambda kv: kv["ts"])

    def get_total_runtime_ms(
        self,
        pt_node_list: List[Any]
    ) -> int:
        """
        Returns the total runtime of PyTorch CPU operators with a duration field.
        """
        total_runtime_ms = 0
        for pt_node in pt_node_list:
            if self.is_cpu_op_with_dur(pt_node):
                total_runtime_ms += pt_node["dur"] # in milliseconds
        return total_runtime_ms

    def get_prev_inter_phase_dep_nid(
        self,
        node: ChakraNode,
    ) -> int:
        """
        Returns the NID of the latest operator of the previous phase.

        Finds the closest but smaller value from inter_phase_dependency compared to node.id.
        """
        index = bisect.bisect_left(self.inter_phase_dependency, node.id)

        if index == 0:
            # All elements in the list are greater than node.id; no element satisfies the condition.
            return -1
        else:
            # The element at index-1 will be the closest, smaller value compared to node.id.
            return self.inter_phase_dependency[index - 1]

    @staticmethod
    def find_root_nids(
        nodes: List[Any]
    ) -> int:
        """
        Finds a root node and return its NID.

        * Assumption: There could be multiple root node in a given execution trace.
        """
        root_nids = []
        for node in nodes:
            if "[pytorch|profiler|execution_graph|thread]" in node["name"]:
                root_nids.append(node["id"])
        if not root_nids:
            raise ValueError("Cannot find a root NID")
        return root_nids

    @staticmethod
    def is_label_node(
        node: Dict[str, Any]
    ) -> bool:
        """
        Returns true if a given PyTorch node is a label node.

        All label node names start with "## ".
        """
        return node["name"].startswith("## ")

    def is_phase_root_node(
        self,
        root_nids: List[int],
        node: Dict[str, Any]
    ) -> bool:
        return node["parent"] in root_nids

    def is_gpu_op(
        self,
        node: Dict[str, Any]
    ) -> bool:
        """
        Takes a PyTorch node and returns true if it is a GPU operator.

        All GPU operators have a category field.
        """
        return self.has_cat_field(node)

    def find_children_gpu_ops(
        self,
        root_cpu_nid: int,
        cpu_node: Dict[str, Any],
    ) -> None:
        """
        Discovers all GPU operators under a CPU operator.

        Once discovered, GPU operators are tracked in pt_gpu_node_dict.
        """
        cpu_nid = cpu_node["id"]
        for node in self.pt_nodes:
            if node["parent"] == cpu_nid:
                if self.is_gpu_op(node):
                    self.pt_gpu_node_dict.setdefault(root_cpu_nid, []).append(node)
                else:
                    # label or CPU operators
                    self.find_children_gpu_ops(root_cpu_nid, node)

    def dfs(
        self,
        node: Dict[str, Any],
        root_nid: int,
    ) -> int:
        """
        Discovers all PyTorch CPU operators under a given node while populating
        pt_cpu_node_dict, After that, returns the largest NID in the tree.
        """
        nid = node["id"]
        node_type = self.get_pytorch_node_type(node)
        if node_type == PyTorchNodeType.GPU_OP:
            return -1
        elif node_type == PyTorchNodeType.CPU_OP:
            self.pt_cpu_node_dict[nid] = node
            self.find_children_gpu_ops(node["id"], node)
            return nid
        elif node_type == PyTorchNodeType.LABEL:
            largest_nid = -1
            for child in self.pt_nodes:
                # We should not call dfs for the root node or phase root nodes
                # as they will be covered by other DFS calls.
                if child["parent"] == nid:
                    largest_nid = max(largest_nid, self.dfs(child, root_nid))
            return largest_nid
        else:
            raise ValueError(f"Invalid node type: {node_type}")
        return -1

    def discover_pytorch_cpu_ops(
        self
    ) -> None:
        """
        Discovers PyTorch CPU operators and populate pt_cpu_node_dict.

        Run DFS on a root node and phase root nodes as they may have CPU operators.
        DFS populates pt_cpu_node_dict and returns the largest NID within the phase.
        """
        root_nids = self.find_root_nids(self.pt_nodes)
        for node in self.pt_nodes:
            if self.is_phase_root_node(root_nids, node):
                largest_nid_within_phase = self.dfs(node, root_nids)
                if largest_nid_within_phase != -1:
                    self.inter_phase_dependency.append(largest_nid_within_phase)

        # Make sure that the NIDs in inter_phase_dependency are in the increasing order.
        self.inter_phase_dependency.sort()

    def assign_chakra_ids(
        self,
        total_assigned_ids: Dict[int,bool],
        assigned_ids: List[int],
        initial_id_to_assign: int
    ) -> int:
        """
        This function is used to assign unique ids to the ops. During the conversion, we may decompose an op into multiple
        ops. So it is important to re-assign unique ids to all ops and make sure the ops that should be executed first have
        smaller ids.
        """
        orig_id = initial_id_to_assign
        while True:
            if initial_id_to_assign in total_assigned_ids.keys():
                initial_id_to_assign += 1
            else:
                total_assigned_ids[initial_id_to_assign] = True
                if orig_id in assigned_ids.keys():
                    assigned_ids[orig_id].append(initial_id_to_assign)
                else:
                    assigned_ids[orig_id] = [initial_id_to_assign]
                return initial_id_to_assign

    def merge_gpu_ops_with_cpu_ops(
        self,
    ) -> Any:
        """
        This function decomposes the CPU ops that have GPU child ops into multiple sub_ops.
        This required to allow running GPU ops and CPU ops at the same time.
        """
        self.logger.info("Merge CPU ops with GPU ops")

        decomposed_nodes = []
        assigned_ids = {}
        total_assigned_ids = {}
        new_pt_gpu_node_dict = {}
        decomposed_nodes_dep = {}
        for nid, node in self.pt_cpu_node_dict.items():
            if self.has_gpu_op(nid):
                self.pt_gpu_node_dict[nid] = sorted(self.pt_gpu_node_dict[nid], key=lambda kv: kv["ts"])

                for gpu_node in self.pt_gpu_node_dict[nid]:
                    assert (node["ts"] + node["dur"]) > gpu_node["ts"]

                last_ts = node["ts"]
                for i in range(len(self.pt_gpu_node_dict[nid])+1):
                    copy_node = copy.deepcopy(node)
                    copy_node["id"] = self.assign_chakra_ids(total_assigned_ids, assigned_ids, nid)
                    copy_node["name"] = copy_node["name"]+"("+str(i)+")"
                    if i < len(self.pt_gpu_node_dict[nid]):
                        self.pt_gpu_node_dict[nid][i]["id"] =\
                                self.assign_chakra_ids(
                                        total_assigned_ids,
                                        assigned_ids,
                                        self.pt_gpu_node_dict[nid][i]["id"])
                        assert self.pt_gpu_node_dict[nid][i]["ts"] > copy_node["ts"]
                        copy_node["ts"] = last_ts
                        copy_node["dur"] = self.pt_gpu_node_dict[nid][i]["ts"]-last_ts
                        last_ts = self.pt_gpu_node_dict[nid][i]["ts"]
                        new_pt_gpu_node_dict.setdefault(copy_node["id"], []).append(self.pt_gpu_node_dict[nid][i])
                    else:
                        copy_node["dur"] = copy_node["dur"]-(last_ts-copy_node["ts"])
                        copy_node["ts"] = last_ts
                        last_ts = copy_node["ts"]+copy_node["dur"]

                    assert (copy_node["ts"] >= 0) and (copy_node["dur"] > 0)
                    if i > 0:
                        assert copy_node["ts"] > decomposed_nodes[-1]["ts"]
                        decomposed_nodes_dep[copy_node["id"]] = decomposed_nodes[-1]["id"]
                    decomposed_nodes.append(copy_node)
            else:
                node["id"] = self.assign_chakra_ids(total_assigned_ids, assigned_ids, nid)
                decomposed_nodes.append(node)

        merged_pt_cpu_node_dict = {
            decomposed_node["id"]: decomposed_node for decomposed_node in decomposed_nodes
        }

        self.pt_cpu_node_dict = merged_pt_cpu_node_dict
        self.pt_gpu_node_dict = new_pt_gpu_node_dict
        return assigned_ids, decomposed_nodes_dep

    def validate_pt_node_dict(
        self,
    ) -> None:
        """
        Raises an exception if any anomaly is detected in pt_cpu_node_dict or
        pt_gpu_node_dict.

        * NIDs of CPU nodes should be unique.
        * CPU operators can have at most one GPU operator.
        """
        seen_nids = set()
        for nid, node in self.pt_cpu_node_dict.items():
            assert nid == node["id"]
            if nid in seen_nids:
                self.logger.error(f"NID {nid} is duplicate")
                raise ValueError("Duplicate NID detected!")
            seen_nids.add(nid)
            if nid in self.pt_gpu_node_dict.keys():
                assert len(self.pt_gpu_node_dict[nid]) == 1

    def discover_pytorch_comm_ops(
        self,
        assigned_ids: List[int]
    ) -> None:
        """
        Discovers communication nodes and populate pt_record_param_comms_node_dict
        and pt_nccl_node_dict.
        """
        self.logger.info("Discover communication nodes")
        for node in self.pt_nodes:
            if self.is_record_param_comms_node(node):
                if node["parent"] in assigned_ids.keys():
                    nodes_to_assign = assigned_ids[node["parent"]]
                    for parent_id in nodes_to_assign:
                        self.pt_record_param_comms_node_dict.update({parent_id: node})
                else:
                    self.pt_record_param_comms_node_dict.update({node["parent"]: node})
            if self.is_nccl_node(node):
                if node["parent"] in assigned_ids.keys():
                        nodes_to_assign=assigned_ids[node["parent"]]
                        for parent_id in nodes_to_assign:
                            self.pt_nccl_node_dict.update({parent_id: node})
                else:
                    self.pt_nccl_node_dict.update({node["parent"]: node})

        for i in range(len(self.inter_phase_dependency)):
            # If an op is decomposed into multiple sub_ops, we want to point to the last subop [-1]
            self.inter_phase_dependency[i] = assigned_ids[self.inter_phase_dependency[i]][-1]
        self.inter_phase_dependency.sort()

    def update_input_tensor_dict(
        self,
        nid: int,
        inputs: str
    ) -> int:
        """
        Updates input_storage_id_nid_dict and input_tensor_id_nid_dict

        Each dictionary is populcated with mappings between storage ID
        (or tensor ID) and corresponding node IDs. If node 0 takes tensor 10 as
        an input, a new mapping will be created like this `10: [0]`
        """
        for i in inputs:
            if self.is_valid_tensor(i):
                if self.has_valid_storage_id(i):
                    storage_id = self.get_storage_id_from_tensor(i)
                    self.input_storage_id_nid_dict.setdefault(storage_id, []).append(nid)
                else:
                    tensor_id = self.get_tensor_id_from_tensor(i)
                    self.input_tensor_id_nid_dict.setdefault(tensor_id, []).append(nid)

    def update_output_tensor_dict(
        self,
        nid: int,
        outputs: str
    ) -> int:
        """
        Updates output_storage_id_nid_dict and output_tensor_id_nid_dict.

        Each dictionary is populcated with mappings between storage ID
        (or tensor ID) and corresponding node IDs. If node 0 produces tensor 10
        as an output, a new mapping will be created like this `10: [0]`.
        """
        for o in outputs:
            if self.is_valid_tensor(o):
                if self.has_valid_storage_id(o):
                    storage_id = self.get_storage_id_from_tensor(o)
                    self.output_storage_id_nid_dict.setdefault(storage_id, []).append(nid)
                else:
                    tensor_id = self.get_tensor_id_from_tensor(o)
                    self.output_tensor_id_nid_dict.setdefault(tensor_id, []).append(nid)

    def convert_pytorch_node_to_chakra_node(
        self,
        pt_node: Dict[str, Any]
    ) -> ChakraNode:
        """
        Converts a PyToch node to a Chakra node.
        """
        ck_node = ChakraNode()
        ck_node.id = pt_node["id"]
        ck_node.name = pt_node["name"]
        ck_node.type = self.get_chakra_node_type_from_pytorch_node(pt_node)
        ck_node.ctrl_deps.append(pt_node["parent"])
        if "dur" in pt_node.keys():
            ck_node.duration_micros = pt_node["dur"]
        else:
            ck_node.duration_micros = 0
        ck_node.inputs.values = str(pt_node["inputs"])
        ck_node.inputs.shapes = str(pt_node["input_shapes"])
        ck_node.inputs.types = str(pt_node["input_types"])
        ck_node.outputs.values = str(pt_node["outputs"])
        ck_node.outputs.shapes = str(pt_node["output_shapes"])
        ck_node.outputs.types = str(pt_node["output_types"])
        ck_node.attr.append(
                ChakraAttr(name="is_cpu_op",
                           bool_val=self.is_cpu_op(pt_node)))
        if "fw_parent" in pt_node.keys():
            ck_node.attr.append(
                    ChakraAttr(name="fw_parent",
                               int64_val=pt_node["fw_parent"]))
        if "fw_tid" in pt_node.keys():
            ck_node.attr.append(
                    ChakraAttr(name="fw_tid",
                               int64_val=pt_node["fw_tid"]))
        if "op_schema" in pt_node.keys():
            ck_node.attr.append(
                    ChakraAttr(name="op_schema",
                               string_val=pt_node["op_schema"]))
        if "seq_id" in pt_node.keys():
            ck_node.attr.append(
                    ChakraAttr(name="seq_id",
                               int64_val=pt_node["seq_id"]))
        if "rf_id" in pt_node.keys():
            ck_node.attr.append(
                    ChakraAttr(name="rf_id",
                               int64_val=pt_node["rf_id"]))
        if "scope" in pt_node.keys():
            ck_node.attr.append(
                    ChakraAttr(name="scope",
                               int64_val=pt_node["scope"]))
        if "tid" in pt_node.keys():
            ck_node.attr.append(
                    ChakraAttr(name="tid",
                               int64_val=pt_node["tid"]))
        return ck_node

    def get_nccl_node(
        self,
        nid: int
    ) -> Dict[str, Any]:
        """
        Returns a PyTorch NCCL node for a given Chakra NID.

        For communication nodes, finding a corresponding NCCL node is critical
        to identify the communication type and communication size.

        There are two cases:
          (1) Given node is a parent of a record_param_comms node
            * In this case, the corresponding NCCL node should be a child of
            the record_param_comms_pt node.
          (2) Given node is a parent of a NCCL node
        """
        pt_nccl_node = None
        if nid in self.pt_record_param_comms_node_dict.keys():
            pt_record_param_comms_node = self.pt_record_param_comms_node_dict[nid]
            rpcp_nid = pt_record_param_comms_node["id"]
            if rpcp_nid in self.pt_nccl_node_dict.keys():
                pt_nccl_node = self.pt_nccl_node_dict[rpcp_nid]
            else:
                raise ValueError(
                        f"NID {nid} has a pt_record_param_comms_node "
                        f"but it does not have a correspondin pt_nccl_node.")
        elif nid in self.pt_nccl_node_dict.keys():
            pt_nccl_node = self.pt_nccl_node_dict[nid]
        else:
            raise ValueError(
                f"NID {nid} does not have an entry in pt_record_param_comms_node_dict "
                f"nor pt_nccl_node_dict"
            )
        return pt_nccl_node

    def add_gpu_chakra_node(
        self,
        ck_cpu_node: ChakraNode
    ) -> None:
        """
        Converts a PyTorch GPU node to a Chakra node and add it to ck_node_dict.
        """
        assert ck_cpu_node.id in self.pt_gpu_node_dict.keys()
        pt_gpu_node = self.pt_gpu_node_dict[ck_cpu_node.id][0]
        if len(self.pt_gpu_node_dict[ck_cpu_node.id]) != 1:
            raise ValueError(f"Chakra node {ck_cpu_node.id} has more than one GPU operators")
        ck_gpu_node = self.convert_pytorch_node_to_chakra_node(pt_gpu_node)
        if ck_cpu_node.type == COMM_COLL_NODE:
            pt_nccl_node = self.get_nccl_node(ck_cpu_node.id)
            ck_gpu_node.attr.append(
                    ChakraAttr(name="comm_type",
                               int64_val=self.get_collective_comm_type(pt_nccl_node)))
            ck_gpu_node.attr.append(
                    ChakraAttr(name="comm_size",
                               int64_val=self.get_comm_size(pt_nccl_node)))
            attr = ChakraAttr(name="involved_dim")
            for _ in range(self.num_dims):
                attr.bool_list.values.append(True)
            ck_gpu_node.attr.append(attr)
        ck_gpu_node.data_deps.append(ck_cpu_node.id)
        self.ck_node_dict[ck_gpu_node.id] = ck_gpu_node

    def identify_data_dependency_with_storage_id(
        self
    ) -> None:
        """
        Identifies data dependency between operators with storage IDs.
        """
        self.logger.info("Identify data dependency with storage IDs")
        for input_storage_id, child_nids in self.input_storage_id_nid_dict.items():
            if input_storage_id in self.output_storage_id_nid_dict:
                parent_nids = self.output_storage_id_nid_dict[input_storage_id]
                for child_nid in child_nids:
                    for parent_nid in parent_nids:
                        child_node = self.ck_node_dict[child_nid]
                        if (parent_nid not in child_node.data_deps)\
                        and (parent_nid < child_nid):
                            child_node.data_deps.append(parent_nid)

    def identify_data_dependency_with_tensor_id(
        self
    ) -> None:
        """
        Identifies data dependency between operators with tensor IDs.
        """
        self.logger.info("Identify data dependency with tensor IDs")
        for input_tensor_id, child_nids in self.input_tensor_id_nid_dict.items():
            if input_tensor_id in self.output_tensor_id_nid_dict:
                parent_nids = self.output_tensor_id_nid_dict[input_tensor_id]
                for child_nid in child_nids:
                    for parent_nid in parent_nids:
                        child_node = self.ck_node_dict[child_nid]
                        if (parent_nid not in child_node.data_deps)\
                        and (parent_nid < child_nid):
                            child_node.data_deps.append(parent_nid)

    def identify_data_dependency(
        self
    ) -> None:
        """
        Identifies data dependency between operators using tensors.

        Dependencies between operators can be identified by their tensor input/
        output relationships. A tensor can be identified by either a storage ID
        or a tensor ID. Use the storage ID if it's valid; otherwise, use the
        tensor ID.
        """
        self.logger.info("Identify data dependency")
        self.identify_data_dependency_with_storage_id()
        self.identify_data_dependency_with_tensor_id()

    def write_chakra_et(
        self,
    ) -> None:
        self.logger.info("Write Chakra trace")

        self.logger.info("Encode global metadata")
        md = GlobalMetadata(
            attr=[
                ChakraAttr(name="schema", string_val=self.pt_schema),
                ChakraAttr(name="pid", uint64_val=self.pt_pid),
                ChakraAttr(name="time", string_val=self.pt_time),
                ChakraAttr(name="start_ts", uint64_val=self.pt_start_ts),
                ChakraAttr(name="finish_ts", uint64_val=self.pt_finish_ts)
            ]
        )
        encode_message(self.chakra_et, md)

        self.logger.info("Encode nodes (operators)")
        seen_nids = set()
        for nid in sorted(self.ck_node_dict.keys()):
            if nid in seen_nids:
                self.logger.error(f"NID {nid} is duplicate")
                raise ValueError("Duplicate NID detected!")
            seen_nids.add(nid)
            ck_node = self.ck_node_dict[nid]
            encode_message(self.chakra_et, ck_node)

        self.logger.info("All Chakra nodes are written to the output file")

    def convert(
        self
    ) -> None:
        self.sort_pytorch_nodes_with_starting_time()

        self.discover_pytorch_cpu_ops()

        total_runtime_ns = self.get_total_runtime_ms(list(self.pt_cpu_node_dict.values())) * 1000
        self.logger.info(f"Total runtime exluding children operators: {total_runtime_ns} ns")

        assigned_ids, decomposed_nodes_dep = self.merge_gpu_ops_with_cpu_ops()

        self.validate_pt_node_dict()

        self.discover_pytorch_comm_ops(assigned_ids)

        self.logger.info("Convert PyTorch nodes to Chakra nodes")
        for pt_nid, pt_node in self.pt_cpu_node_dict.items():
            self.update_input_tensor_dict(pt_node["id"], pt_node["inputs"])
            self.update_output_tensor_dict(pt_node["id"], pt_node["outputs"])
            if pt_nid in self.pt_gpu_node_dict.keys():
                for pt_gpu_node in self.pt_gpu_node_dict[pt_nid]:
                    # Assumption: same input / output as its parent CPU operator
                    self.update_input_tensor_dict(pt_gpu_node["id"], pt_gpu_node["inputs"])
                    # For now we ignore GPU->CPU dependencies since it creates unwanted dependencies.
                    # self.update_output_tensor_dict(pt_gpu_node["id"], pt_gpu_node["outputs"])

            ck_node = self.convert_pytorch_node_to_chakra_node(pt_node)
            self.ck_node_dict[ck_node.id] = ck_node
            if self.has_gpu_op(ck_node.id):
                self.add_gpu_chakra_node(ck_node)

            # Adding previous phase node dependency
            dep_nid = self.get_prev_inter_phase_dep_nid(ck_node)
            if (dep_nid != -1) and (dep_nid not in ck_node.data_deps):
                ck_node.data_deps.append(dep_nid)

            # Adding decomposed nodes dependency
            # When we decompose a CPU op into multiple sub_ops, these ops have linear dependeny with themselves
            # For example, the first sub_op should be finished before the second sub_op. Here, we capture these dependencies.
            if (pt_nid in decomposed_nodes_dep.keys())\
                    and (decomposed_nodes_dep[pt_nid] not in ck_node.data_deps):
                 ck_node.data_deps.append(decomposed_nodes_dep[pt_nid])

        self.identify_data_dependency()

        self.write_chakra_et()
