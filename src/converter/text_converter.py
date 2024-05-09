#!/usr/bin/env python3

import logging
from io import TextIOWrapper
from typing import Any, List

from ...schema.protobuf.et_def_pb2 import (
    ALL_GATHER,
    ALL_REDUCE,
    ALL_TO_ALL,
    COMM_COLL_NODE,
    COMP_NODE,
    REDUCE_SCATTER,
    GlobalMetadata,
    Node,
    NodeType,
    AttributeProto as ChakraAttr,
)
from ..third_party.utils.protolib import encodeMessage as encode_message


class Layer:
    def __init__(self, line: str) -> None:
        try:
            col = line.strip().split()
            self.name = col[0]

            # forward
            self.fwd_comp_time = int(col[2])
            self.fwd_comm_type = str(col[3])
            self.fwd_comm_size = int(col[4])
            self.fwd_comp_node = None
            self.fwd_comm_node = None

            # backward input gradient
            self.bwd_ig_comp_time = int(col[5])
            self.bwd_ig_comm_type = str(col[6])
            self.bwd_ig_comm_size = int(col[7])
            self.bwd_ig_comp_node = None
            self.bwd_ig_comm_node = None

            # backward weight gradient
            self.bwd_wg_comp_time = int(col[8])
            self.bwd_wg_comm_type = str(col[9])
            self.bwd_wg_comm_size = int(col[10])
            self.bwd_wg_update_time = str(col[11])
            self.bwd_wg_comp_node = None
            self.bwd_wg_comm_node = None
        except Exception:
            raise ValueError(f'Cannot parse the following layer -- "{line}"')


class TextConverter:
    def __init__(
        self, input_filename: str, output_filename: str, num_npus: int, num_passes: int, logger: logging.Logger
    ) -> None:
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_npus = num_npus
        self.num_passes = num_passes
        self.logger = logger
        self.next_node_id = 0

    def get_global_metadata(self):
        input_text = ""
        with open(self.input_filename, "r") as input_file:
            input_text = input_file.read()
        attr = [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="input_file", string_val=input_text),
        ]
        metadata = GlobalMetadata(attr=attr)
        return metadata

    def get_layers(self, f: TextIOWrapper, num_layers: int) -> List[Layer]:
        layers = []
        for line in f:
            layers.append(Layer(line))
        return layers

    def get_node(self, name: str, node_type: NodeType) -> Any:
        node = Node()
        node.id = self.next_node_id
        self.next_node_id += 1
        node.name = name
        node.type = node_type
        return node

    def get_comp_node(self, layer_name: str, phase: str, comp_time: int) -> Any:
        node = self.get_node("COMP_NODE_" + layer_name + "_" + phase, COMP_NODE)
        node.duration_micros = comp_time
        return node

    def get_comm_type(self, comm_type: str) -> int:
        if comm_type == "ALLREDUCE":
            return ALL_REDUCE
        elif comm_type == "ALLTOALL":
            return ALL_TO_ALL
        elif comm_type == "ALLGATHER":
            return ALL_GATHER
        elif comm_type == "REDUCESCATTER":
            return REDUCE_SCATTER
        return 0

    def get_comm_coll_node(self, layer_name: str, comm_type: str, comm_size: int) -> Any:
        node = self.get_node(f"COMM_COLL_NODE_{layer_name}_{comm_type}", COMM_COLL_NODE)
        node.attr.append(ChakraAttr(name="comm_type", int64_val=self.get_comm_type(comm_type)))
        node.attr.append(ChakraAttr(name="comm_size", uint64_val=comm_size))
        return node

    def add_parent(self, child_node: Any, parent_node: Any) -> None:
        child_node.data_deps.append(parent_node.id)

    def convert(self) -> None:
        with open(self.input_filename, "r") as f:
            first_line = f.readline().strip().split()
            parallelism_type = first_line[0]
            num_layers = int(f.readline().strip())

            if parallelism_type == "MICRO":
                self.convert_microbenchmark(f, num_layers)
            elif parallelism_type == "DATA":
                self.convert_data_parallel(f, num_layers)
            elif parallelism_type == "MODEL":
                self.convert_model_parallel(f, num_layers)
            elif parallelism_type == "HYBRID_DATA_MODEL":
                self.convert_hybrid_data_model(f, num_layers)
            elif parallelism_type == "HYBRID_MODEL_DATA":
                self.convert_hybrid_model_data(f, num_layers)
            elif (parallelism_type == "HYBRID_DLRM") or (parallelism_type == "HYBRID_DLRM_ENHANCED"):
                last_bottom_layer = int(first_line[1])
                self.convert_hybrid_dlrm(f, num_layers, last_bottom_layer)
            else:
                raise ValueError(f"Unsupported parallelism type, {parallelism_type}")

    def convert_microbenchmark(self, f: TextIOWrapper, num_layers: int) -> None:
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)
                for i in range(self.num_passes):
                    for layer in layers:
                        bwd_wg_comm_node = self.get_comm_coll_node(
                            layer.name, layer.bwd_wg_comm_type, layer.bwd_wg_comm_size
                        )
                        encode_message(g, bwd_wg_comm_node)

    def convert_data_parallel(self, f: TextIOWrapper, num_layers: int) -> None:
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)
                for i in range(self.num_passes):
                    fwd_comp_node = None

                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(layer.name, "FWD", layer.fwd_comp_time)
                        if idx != 0:
                            self.add_parent(fwd_comp_node, layers[idx - 1].fwd_comp_node)
                        if layer.bwd_wg_comm_node is not None:
                            self.add_parent(fwd_comp_node, layer.bwd_wg_comm_node)
                        layer.fwd_comp_node = fwd_comp_node
                        encode_message(g, fwd_comp_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_wg_comp_node = self.get_comp_node(layer.name, "BWD_WG", layer.bwd_wg_comp_time)
                        if idx == 0:
                            if fwd_comp_node is None:
                                raise ValueError("fwd_comp_node is None")
                            self.add_parent(bwd_wg_comp_node, fwd_comp_node)
                        else:
                            self.add_parent(bwd_wg_comp_node, layers[len(layers) - idx].bwd_ig_comp_node)
                        encode_message(g, bwd_wg_comp_node)

                        bwd_wg_comm_node = self.get_comm_coll_node(
                            layer.name, layer.bwd_wg_comm_type, layer.bwd_wg_comm_size
                        )

                        self.add_parent(bwd_wg_comm_node, bwd_wg_comp_node)
                        layer.bwd_wg_comm_node = bwd_wg_comm_node
                        encode_message(g, bwd_wg_comm_node)

                        if idx != (len(layers) - 1):
                            bwd_ig_comp_node = self.get_comp_node(layer.name, "BWD_IG", layer.bwd_ig_comp_time)
                            self.add_parent(bwd_ig_comp_node, bwd_wg_comp_node)
                            layer.bwd_ig_comp_node = bwd_ig_comp_node
                            encode_message(g, bwd_ig_comp_node)

                for layer in layers:
                    layer.bwd_wg_comm_node = None

    def convert_model_parallel(self, f: TextIOWrapper, num_layers: int) -> None:
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)
                for i in range(self.num_passes):
                    fwd_comm_node = None

                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(layer.name, "FWD", layer.fwd_comp_time)
                        if idx != 0:
                            self.add_parent(fwd_comp_node, layers[idx - 1].fwd_comm_node)
                        if layer.bwd_wg_comp_node is not None:
                            self.add_parent(fwd_comp_node, layer.bwd_wg_comp_node)
                        layer.fwd_comp_node = fwd_comp_node
                        encode_message(g, fwd_comp_node)

                        fwd_comm_node = self.get_comm_coll_node(layer.name, layer.fwd_comm_type, layer.fwd_comm_size)
                        layer.fwd_comm_node = fwd_comm_node
                        self.add_parent(fwd_comm_node, fwd_comp_node)
                        encode_message(g, fwd_comm_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_ig_comp_node = self.get_comp_node(layer.name, "BWD_IG", layer.bwd_ig_comp_time)
                        if idx == 0:
                            if fwd_comm_node is None:
                                raise ValueError("fwd_comm_node is None")
                            self.add_parent(bwd_ig_comp_node, fwd_comm_node)
                        else:
                            self.add_parent(bwd_ig_comp_node, layers[len(layers) - idx].bwd_wg_comp_node)
                            self.add_parent(bwd_ig_comp_node, layers[len(layers) - idx].bwd_ig_comm_node)
                        encode_message(g, bwd_ig_comp_node)

                        if idx != (num_layers - 1):
                            bwd_ig_comm_node = self.get_comm_coll_node(
                                layer.name, layer.bwd_ig_comm_type, layer.bwd_ig_comm_size
                            )
                            self.add_parent(bwd_ig_comm_node, bwd_ig_comp_node)
                            layer.bwd_ig_comm_node = bwd_ig_comm_node
                            encode_message(g, bwd_ig_comm_node)

                        bwd_wg_comp_node = self.get_comp_node(layer.name, "BWD_WG", layer.bwd_wg_comp_time)
                        self.add_parent(bwd_wg_comp_node, bwd_ig_comp_node)
                        layer.bwd_wg_comp_node = bwd_wg_comp_node
                        encode_message(g, bwd_wg_comp_node)

                for layer in layers:
                    layer.bwd_wg_comp_node = None

    def convert_hybrid_data_model(self, f: TextIOWrapper, num_layers: int) -> None:
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)
                for i in range(self.num_passes):
                    fwd_comm_node = None

                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(layer.name, "FWD", layer.fwd_comp_time)
                        if layer.bwd_wg_comm_node is not None:
                            self.add_parent(fwd_comp_node, layer.bwd_wg_comm_node)
                        if idx != 0:
                            self.add_parent(fwd_comp_node, layers[idx - 1].fwd_comm_node)
                        encode_message(g, fwd_comp_node)

                        fwd_comm_node = self.get_comm_coll_node(layer.name, layer.fwd_comm_type, layer.fwd_comm_size)
                        self.add_parent(fwd_comm_node, fwd_comp_node)
                        layer.fwd_comm_node = fwd_comm_node
                        encode_message(g, fwd_comm_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_ig_comp_node = self.get_comp_node(layer.name, "BWD_IG", layer.bwd_ig_comp_time)
                        if idx == 0:
                            if fwd_comm_node is None:
                                raise ValueError("fwd_comm_node is None")
                            self.add_parent(bwd_ig_comp_node, fwd_comm_node)
                        else:
                            self.add_parent(bwd_ig_comp_node, layers[len(layers) - idx].bwd_wg_comp_node)
                            self.add_parent(bwd_ig_comp_node, layers[len(layers) - idx].bwd_ig_comm_node)
                        encode_message(g, bwd_ig_comp_node)

                        if idx != num_layers - 1:
                            bwd_ig_comm_node = self.get_comm_coll_node(
                                layer.name + "_IG_COMM_", layer.bwd_ig_comm_type, layer.bwd_ig_comm_size
                            )
                            self.add_parent(bwd_ig_comm_node, bwd_ig_comp_node)
                            layer.bwd_ig_comm_node = bwd_ig_comm_node
                            encode_message(g, bwd_ig_comm_node)

                        bwd_wg_comp_node = self.get_comp_node(layer.name, "BWD_WG", layer.bwd_wg_comp_time)
                        self.add_parent(bwd_wg_comp_node, bwd_ig_comp_node)
                        layer.bwd_wg_comp_node = bwd_wg_comp_node
                        encode_message(g, bwd_wg_comp_node)

                        bwd_wg_comm_node = self.get_comm_coll_node(
                            layer.name, layer.bwd_wg_comm_type, layer.bwd_wg_comm_size
                        )
                        self.add_parent(bwd_wg_comm_node, bwd_wg_comp_node)
                        layer.bwd_wg_comm_node = bwd_wg_comm_node
                        encode_message(g, bwd_wg_comm_node)

                for layer in layers:
                    layer.bwd_wg_comm_node = None

    def convert_hybrid_model_data(self, f: TextIOWrapper, num_layers: int) -> None:
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)
                for i in range(self.num_passes):
                    fwd_comm_node = None

                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(layer.name, "FWD", layer.fwd_comp_time)
                        if layer.bwd_wg_comm_node is not None:
                            self.add_parent(fwd_comp_node, layer.bwd_wg_comm_node)
                        if idx != 0:
                            self.add_parent(fwd_comp_node, layers[idx - 1].fwd_comm_node)
                        encode_message(g, fwd_comp_node)

                        fwd_comm_node = self.get_comm_coll_node(layer.name, layer.fwd_comm_type, layer.fwd_comm_size)
                        self.add_parent(fwd_comm_node, fwd_comp_node)
                        layer.fwd_comm_node = fwd_comm_node
                        encode_message(g, fwd_comm_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_ig_comp_node = self.get_comp_node(layer.name, "BWD_IG", layer.bwd_ig_comp_time)
                        if idx == 0:
                            if fwd_comm_node is None:
                                raise ValueError("fwd_comm_node is None")
                            self.add_parent(bwd_ig_comp_node, fwd_comm_node)
                        else:
                            self.add_parent(bwd_ig_comp_node, layers[len(layers) - idx].bwd_wg_comp_node)
                            self.add_parent(bwd_ig_comp_node, layers[len(layers) - idx].bwd_ig_comm_node)
                        encode_message(g, bwd_ig_comp_node)

                        if idx != num_layers - 1:
                            bwd_ig_comm_node = self.get_comm_coll_node(
                                layer.name, layer.bwd_ig_comm_type, layer.bwd_ig_comm_size
                            )
                            self.add_parent(bwd_ig_comm_node, bwd_ig_comp_node)
                            layer.bwd_ig_comm_node = bwd_ig_comm_node
                            encode_message(g, bwd_ig_comm_node)

                        bwd_wg_comp_node = self.get_comp_node(layer.name, "BWD_WG", layer.bwd_wg_comp_time)
                        self.add_parent(bwd_wg_comp_node, bwd_ig_comp_node)
                        layer.bwd_wg_comp_node = bwd_wg_comp_node
                        encode_message(g, bwd_wg_comp_node)

                        bwd_wg_comm_node = self.get_comm_coll_node(
                            layer.name, layer.bwd_wg_comm_type, layer.bwd_wg_comm_size
                        )
                        self.add_parent(bwd_wg_comm_node, bwd_wg_comp_node)
                        layer.bwd_wg_comm_node = bwd_wg_comm_node
                        encode_message(g, bwd_wg_comm_node)

                for layer in layers:
                    layer.bwd_wg_comm_node = None

    def convert_hybrid_dlrm(self, f: TextIOWrapper, num_layers: int, last_bottom_layer: int) -> None:
        layers = self.get_layers(f, num_layers)
        for npu_id in range(self.num_npus):
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata()
                encode_message(g, global_metadata)
                for i in range(self.num_passes):
                    fwd_comp_node = None

                    # forward pass
                    for idx, layer in enumerate(layers):
                        fwd_comp_node = self.get_comp_node(layer.name, "FWD", layer.fwd_comp_time)
                        if layer.bwd_wg_comm_node is not None:
                            self.add_parent(fwd_comp_node, layer.bwd_wg_comm_node)
                        elif layer.bwd_wg_comp_node is not None:
                            self.add_parent(fwd_comp_node, layer.bwd_wg_comp_node)
                        if idx != 0:
                            self.add_parent(fwd_comp_node, layers[idx - 1].fwd_comp_node)
                        if idx == last_bottom_layer:
                            self.add_parent(fwd_comp_node, layers[0].fwd_comm_node)
                        layer.fwd_comp_node = fwd_comp_node
                        encode_message(g, fwd_comp_node)

                        if layer.fwd_comm_type == "ALLTOALL":
                            fwd_comm_node = self.get_comm_coll_node(
                                layer.name, layer.fwd_comm_type, layer.fwd_comm_size
                            )
                            self.add_parent(fwd_comm_node, fwd_comp_node)
                            layer.fwd_comm_node = fwd_comm_node
                            encode_message(g, fwd_comm_node)

                    # backward pass
                    for idx, layer in enumerate(reversed(layers)):
                        bwd_wg_comp_node = self.get_comp_node(layer.name, "BWD_WG", layer.bwd_wg_comp_time)
                        if idx == 0:
                            if fwd_comp_node is None:
                                raise ValueError("fwd_comp_node is None")
                            self.add_parent(bwd_wg_comp_node, fwd_comp_node)
                        else:
                            if layers[len(layers) - idx].bwd_ig_comp_node is not None:
                                self.add_parent(bwd_wg_comp_node, layers[len(layers) - idx].bwd_ig_comp_node)
                            if layers[len(layers) - idx - 1].bwd_ig_comm_node is not None:
                                self.add_parent(bwd_wg_comp_node, layers[len(layers) - idx - 1].bwd_ig_comm_node)
                        layer.bwd_wg_comp_node = bwd_wg_comp_node
                        encode_message(g, bwd_wg_comp_node)

                        if layer.bwd_wg_comm_type != "NONE":
                            bwd_wg_comm_node = self.get_comm_coll_node(
                                layer.name, layer.bwd_wg_comm_type, layer.bwd_wg_comm_size
                            )
                            self.add_parent(bwd_wg_comm_node, bwd_wg_comp_node)
                            layer.bwd_wg_comm_node = bwd_wg_comm_node
                            encode_message(g, bwd_wg_comm_node)

                        bwd_ig_comp_node = None
                        if idx != (len(layers) - 1):
                            bwd_ig_comp_node = self.get_comp_node(layer.name, "BWD_IG", layer.bwd_ig_comp_time)
                            self.add_parent(bwd_ig_comp_node, bwd_wg_comp_node)
                            layer.bwd_ig_comp_node = bwd_ig_comp_node
                            encode_message(g, bwd_ig_comp_node)

                        if (len(layers) - idx - 1) == (last_bottom_layer + 1):
                            bwd_ig_comm_node = self.get_comm_coll_node(
                                layers[0].name, layers[0].bwd_ig_comm_type, layers[0].bwd_ig_comm_size
                            )
                            if bwd_ig_comp_node is None:
                                raise ValueError("bwd_ig_comp_node is None")
                            self.add_parent(bwd_ig_comm_node, bwd_ig_comp_node)
                            layers[0].bwd_ig_comm_node = bwd_ig_comm_node
                            encode_message(g, bwd_ig_comm_node)

                for layer in layers:
                    layer.bwd_wg_comm_node = None
                    layer.bwd_wg_comp_node = None
                    layer.bwd_ig_comm_node = None
                    layer.bwd_ig_comp_node = None
