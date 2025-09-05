#!/usr/bin/env python3

import json
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from pathlib import Path
import glob

from ...schema.protobuf.et_def_pb2 import (
    COMP_NODE,
    COMM_COLL_NODE,
    COMM_SEND_NODE,
    COMM_RECV_NODE,
    ALL_REDUCE,
    ALL_TO_ALL,
    ALL_GATHER,
    REDUCE_SCATTER,
    BROADCAST,
    REDUCE,
    BARRIER,
)
from ...schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr
from ...schema.protobuf.et_def_pb2 import Node as ChakraNode
from ...schema.protobuf.et_def_pb2 import GlobalMetadata
from ..third_party.utils.protolib import encodeMessage as encode_message


# MPI Operation Mappings - Easy to extend by adding new operations here
MPI_COLLECTIVE_OPS = {
    # MPI Operation -> (chakra_comm_type_constant, chakra_node_type)
    'MPI_Allreduce': (ALL_REDUCE, COMM_COLL_NODE),
    'MPI_Alltoall': (ALL_TO_ALL, COMM_COLL_NODE),
    'MPI_Allgather': (ALL_GATHER, COMM_COLL_NODE),
    'MPI_Reduce_scatter': (REDUCE_SCATTER, COMM_COLL_NODE),
    'MPI_Bcast': (BROADCAST, COMM_COLL_NODE),
    'MPI_Reduce': (REDUCE, COMM_COLL_NODE),
    'MPI_Barrier': (BARRIER, COMM_COLL_NODE),
    'MPI_Wait': (BARRIER, COMM_COLL_NODE),
    'MPI_Waitall': (BARRIER, COMM_COLL_NODE),
    # Add more collective operations here as needed:
    # 'MPI_Scan': (BARRIER, COMM_COLL_NODE),  # Map to BARRIER until SCAN is available
    # 'MPI_Gather': (BARRIER, COMM_COLL_NODE),
    # 'MPI_Scatter': (BROADCAST, COMM_COLL_NODE),
}

MPI_P2P_SEND_OPS = {
    # Point-to-point send operations
    'MPI_Send': COMM_SEND_NODE,
    'MPI_Isend': COMM_SEND_NODE,
    'MPI_Ssend': COMM_SEND_NODE,
    'MPI_Bsend': COMM_SEND_NODE,
    'MPI_Rsend': COMM_SEND_NODE,
    # Add more send operations here
}

MPI_P2P_RECV_OPS = {
    # Point-to-point receive operations
    'MPI_Recv': COMM_RECV_NODE,
    'MPI_Irecv': COMM_RECV_NODE,
    # Add more receive operations here
}

# Operations that should be treated as compute nodes
MPI_COMPUTE_OPS = {
    'MPI_Init': COMP_NODE,
    'MPI_Finalize': COMP_NODE,
    'MPI_Comm_rank': COMP_NODE,
    'MPI_Comm_size': COMP_NODE,
    'MPI_Wtime': COMP_NODE,
    'MPI_Get_count': COMP_NODE,
    # Add more compute operations here
}

# MPI Datatype Conversion
MPI_DATATYPE_SIZES = {
    0x00: 0,      # OMPI_DATATYPE_MPI_EMPTY
    0x01: 8,      # OMPI_DATATYPE_MPI_INT8_T
    0x02: 8,      # OMPI_DATATYPE_MPI_UINT8_T
    0x03: 16,     # OMPI_DATATYPE_MPI_INT16_T
    0x04: 16,     # OMPI_DATATYPE_MPI_UINT16_T
    0x05: 32,     # OMPI_DATATYPE_MPI_INT32_T
    0x06: 32,     # OMPI_DATATYPE_MPI_UINT32_T
    0x07: 64,     # OMPI_DATATYPE_MPI_INT64_T
    0x08: 64,     # OMPI_DATATYPE_MPI_UINT64_T
    0x09: 32,     # OMPI_DATATYPE_MPI_FLOAT
    0x0A: 64,     # OMPI_DATATYPE_MPI_DOUBLE
    0x0B: 128,    # OMPI_DATATYPE_MPI_LONG_DOUBLE
    0x0C: 32,     # OMPI_DATATYPE_MPI_COMPLEX4
    0x0D: 64,     # OMPI_DATATYPE_MPI_COMPLEX8
    0x0E: 128,    # OMPI_DATATYPE_MPI_COMPLEX16
    0x0F: 256,    # OMPI_DATATYPE_MPI_COMPLEX32
    0x10: 16,     # OMPI_DATATYPE_MPI_WCHAR
    0x11: 16,
    0x12: 8,      # OMPI_DATATYPE_MPI_BOOL
    0x13: 32,     # OMPI_DATATYPE_MPI_LOGICAL
    0x14: 8,      # OMPI_DATATYPE_MPI_CHARACTER
    0x15: 32,     # OMPI_DATATYPE_MPI_INTEGER
    0x16: 32,     # OMPI_DATATYPE_MPI_REAL
    0x17: 64,     # OMPI_DATATYPE_MPI_DOUBLE_PRECISION
    0x18: 64,     # OMPI_DATATYPE_MPI_COMPLEX
    0x19: 128,    # OMPI_DATATYPE_MPI_DOUBLE_COMPLEX
    0x1A: 256,    # OMPI_DATATYPE_MPI_LONG_DOUBLE_COMPLEX
    0x1B: 64,     # OMPI_DATATYPE_MPI_2INT
    0x1C: 64,     # OMPI_DATATYPE_MPI_2INTEGER
    0x1D: 64,     # OMPI_DATATYPE_MPI_2REAL
    0x1E: 128,    # OMPI_DATATYPE_MPI_2DBLPREC
    0x1F: 128,    # OMPI_DATATYPE_MPI_2COMPLEX
    0x20: 256,    # OMPI_DATATYPE_MPI_2DOUBLE_COMPLEX
    0x21: 64,     # OMPI_DATATYPE_MPI_FLOAT_INT
    0x22: 128,    # OMPI_DATATYPE_MPI_DOUBLE_INT
    0x23: 256,    # OMPI_DATATYPE_MPI_LONG_DOUBLE_INT
    0x24: 32,     # OMPI_DATATYPE_MPI_LONG_INT
    0x25: 32,     # OMPI_DATATYPE_MPI_SHORT_INT
    0x26: 64,     # OMPI_DATATYPE_MPI_AINT
    0x27: 64,     # OMPI_DATATYPE_MPI_OFFSET
    0x28: 8,      # OMPI_DATATYPE_MPI_C_BOOL
    0x29: 64,     # OMPI_DATATYPE_MPI_C_COMPLEX
    0x2A: 64,     # OMPI_DATATYPE_MPI_C_FLOAT_COMPLEX
    0x2B: 128,    # OMPI_DATATYPE_MPI_C_DOUBLE_COMPLEX
    0x2C: 256,    # OMPI_DATATYPE_MPI_C_LONG_DOUBLE_COMPLEX
    0x2D: 0,      # OMPI_DATATYPE_MPI_LB
    0x2E: 0       # OMPI_DATATYPE_MPI_UB
}


@dataclass
class MPINode:
    """Represents a node from MPI trace"""
    name: str
    ph: str
    pid: int
    tid: int
    ts: float
    dur: float
    args: Dict = field(default_factory=dict)
    
    # Additional fields for MPI-specific data
    node_type: str = ""  # compute, collective, send, recv
    comm_type: int = 0  # Chakra communication type constant
    comm_size: int = 0
    comm_tag: int = 0
    rank: int = 0
    data_size: int = 0
    target_rank: int = -1  # for send/recv
    source_rank: int = -1  # for send/recv


class MPIConverter:
    """Converts MPI trace files to Chakra execution traces"""
    
    def __init__(self, input_filenames: List[str], output_filenames: List[str], 
                 num_npus: int, chakra_node_id: int = 0, unsupported_ops_log: str = None,
                 node_debug_log: str = None):
        """
        Initialize the MPI converter
        
        Args:
            input_filenames: List of input MPI trace JSON files (one per rank)
            output_filenames: List of output Chakra ET files (one per NPU)
            num_npus: Number of NPUs to generate (can be more than input traces)
            chakra_node_id: Starting node ID for Chakra nodes
            unsupported_ops_log: Path to log file for unsupported operations
            node_debug_log: Path to log file for node debug information
        """
        self.input_filenames = input_filenames
        self.output_filenames = output_filenames
        self.num_npus = num_npus
        self._chakra_node_id = chakra_node_id
        self._MPI_to_chakra_node_id = {}
        self.unsupported_ops_log = unsupported_ops_log
        self.node_debug_log = node_debug_log
        
        # Storage for parsed data
        self.MPI_nodes: List[List[MPINode]] = []  # List of nodes per rank
        self.pid_list: List[List[int]] = []  # Unique PIDs per rank
        
        # Track unsupported operations
        self.unsupported_ops: Set[str] = set()
        
        # Debug logging
        self.debug_nodes: List[Dict] = []
        
        self.logger = logging.getLogger(__name__)
        
    def convert(self) -> None:
        """Main conversion method"""
        self.logger.info("Starting MPI to Chakra conversion")
        
        # Parse all input files
        num_input_files = len(self.input_filenames)
        for i in range(num_input_files):
            mpi_nodes, pid_list = self._parse_MPI_nodes(
                self.input_filenames[i], 
                self.num_npus
            )
            self.MPI_nodes.append(mpi_nodes)
            self.pid_list.append(pid_list)
        
        # Analyze PIDs to determine compute vs GPU ranks
        all_pids = sorted(set(pid for pids in self.pid_list for pid in pids))
        
        # Separate compute and GPU PIDs
        # Assumption: Lower PIDs are compute ranks, higher PIDs are GPU ranks
        # Or we can use a simple heuristic: if there's only one low PID and multiple high PIDs
        compute_pids = []
        gpu_pids = []
        
        # Find clusters of PIDs
        if len(all_pids) > 1:
            # Simple heuristic: if there's a large gap, lower PIDs are compute
            pid_gaps = [(all_pids[i+1] - all_pids[i], i) for i in range(len(all_pids)-1)]
            max_gap = max(pid_gaps, key=lambda x: x[0]) if pid_gaps else (0, 0)
            
            if max_gap[0] > 5:  # Gap larger than 5 suggests compute vs GPU separation
                compute_pids = all_pids[:max_gap[1]+1]
                gpu_pids = all_pids[max_gap[1]+1:]
            else:
                # No clear separation, treat all as GPU PIDs
                gpu_pids = all_pids
        else:
            gpu_pids = all_pids
        
        self.logger.info(f"Detected compute PIDs: {compute_pids}")
        self.logger.info(f"Detected GPU PIDs: {gpu_pids}")
        
        # Determine mapping strategy
        num_gpu_pids = len(gpu_pids)
        if num_gpu_pids == 0:
            self.logger.warning("No GPU PIDs found, using all PIDs")
            gpu_pids = all_pids
            num_gpu_pids = len(gpu_pids)
        
        # Create NPU traces
        if self.num_npus == num_gpu_pids:
            # N-to-N mapping: direct correspondence
            self.logger.info(f"Using N-to-N mapping: {num_gpu_pids} GPU PIDs to {self.num_npus} NPUs")
            self._create_npu_traces_direct(gpu_pids, compute_pids)
        else:
            # Scaling: replicate pattern
            self.logger.info(f"Scaling from {num_gpu_pids} GPU PIDs to {self.num_npus} NPUs")
            self._create_npu_traces_scaled(gpu_pids, compute_pids)
            
        # Write unsupported operations log
        if self.unsupported_ops and self.unsupported_ops_log:
            self._write_unsupported_ops_log()
            
        # Write node debug log
        if self.node_debug_log and self.debug_nodes:
            self._write_node_debug_log()
            
        self.logger.info("Conversion completed successfully")
    
    def _create_npu_traces_direct(self, gpu_pids: List[int], compute_pids: List[int]) -> None:
        """Create NPU traces with direct N-to-N mapping"""
        for npu_id, gpu_pid in enumerate(gpu_pids):
            self._create_single_npu_trace(npu_id, gpu_pid, compute_pids)
    
    def _create_npu_traces_scaled(self, gpu_pids: List[int], compute_pids: List[int]) -> None:
        """Create NPU traces with scaling/replication"""
        for npu_id in range(self.num_npus):
            # Use modulo to cycle through available GPU PIDs
            gpu_pid = gpu_pids[npu_id % len(gpu_pids)]
            self._create_single_npu_trace(npu_id, gpu_pid, compute_pids)
    
    def _create_single_npu_trace(self, npu_id: int, gpu_pid: int, compute_pids: List[int]) -> None:
        """Create a single NPU trace file"""
        # Collect all nodes for this GPU PID
        gpu_nodes = [
            obj for obj in self.MPI_nodes[0] 
            if int(obj.pid) == int(gpu_pid)
        ]
        
        # Collect all compute nodes
        compute_nodes = []
        for compute_pid in compute_pids:
            compute_nodes.extend([
                obj for obj in self.MPI_nodes[0]
                if int(obj.pid) == int(compute_pid)
            ])
        
        # Merge compute and GPU nodes
        all_nodes = gpu_nodes + compute_nodes
        
        # Sort by timestamp to interleave properly
        all_nodes.sort(key=lambda x: x.ts)
        
        # Separate send/recv operations for later matching
        send_recv_types = list(MPI_P2P_SEND_OPS.keys()) + list(MPI_P2P_RECV_OPS.keys())
        send_recv_nodes = [node for node in all_nodes if node.name in send_recv_types]
        filtered_MPIs = [node for node in all_nodes if node.name not in send_recv_types]
        
        # Insert dummy compute nodes for timeline continuity
        self._insert_dummy_compute(filtered_MPIs)
        
        # Match and insert send/recv pairs
        self._match_and_insert_send_recv(filtered_MPIs, send_recv_nodes)
        
        # Sort again after insertions
        filtered_MPIs.sort(key=lambda x: x.ts)
        
        # Convert nodes to Chakra format
        prior_node = None
        output_filename = self._get_output_filename(npu_id)
        clean_id = -1
        
        with open(output_filename, "wb") as et:
            global_metadata = self._create_global_metadata()
            encode_message(et, global_metadata)
            
            for MPI_node in filtered_MPIs:
                MPI_node_id = clean_id
                if prior_node is not None:
                    MPI_node_parents = [prior_node]
                else:
                    MPI_node_parents = []
                
                # Create Chakra node
                chakra_node = self._convert_MPI_node_to_chakra_node(
                    MPI_node, 
                    MPI_node_id, 
                    MPI_node_parents
                )
                encode_message(et, chakra_node)
                prior_node = self._chakra_node_id - 1
                clean_id += 1
        
        self.logger.info(f"Created NPU {npu_id} trace: {output_filename}")
    
    def _parse_MPI_nodes(self, input_filename: str, num_npus: int) -> Tuple[List[MPINode], List[int]]:
        """Parse MPI trace file and extract nodes"""
        with open(input_filename, 'r') as f:
            data = json.load(f)
        
        MPI_nodes = []
        pid_set = set()
        
        # Extract trace events
        if 'traceEvents' in data:
            events = data['traceEvents']
        else:
            events = data
        
        for event in events:
            if not isinstance(event, dict):
                continue
            
            # Skip metadata events
            if 'ph' not in event or event['ph'] not in ['X']:
                continue
            
            # Skip if name is not in event
            if 'name' not in event:
                continue
            
            mpi_node = MPINode(
                name=event.get('name', ''),
                ph=event.get('ph', ''),
                pid=event.get('pid', 0),
                tid=event.get('tid', 0),
                ts=event.get('ts', 0),
                dur=event.get('dur', 0),
                args=event.get('args', {})
            )
            
            # Categorize node
            self._categorize_node(mpi_node)
            
            MPI_nodes.append(mpi_node)
            pid_set.add(mpi_node.pid)
        
        # Sort by timestamp
        MPI_nodes = sorted(MPI_nodes, key=lambda x: x.ts)
        
        return MPI_nodes, sorted(list(pid_set))
    
    def _categorize_node(self, node: MPINode) -> None:
        """Categorize MPI node based on its name and arguments"""
        name = node.name
        
        # Check if it's a collective operation
        if name in MPI_COLLECTIVE_OPS:
            node.node_type = 'collective'
            node.comm_type = MPI_COLLECTIVE_OPS[name][0]  # This is the Chakra constant directly
            # Extract comm_size from args
            if 'comm' in node.args:
                node.comm_size = self._calculate_data_size(node.args)
            
        # Check if it's a send operation
        elif name in MPI_P2P_SEND_OPS:
            node.node_type = 'send'
            if 'dest' in node.args:
                node.target_rank = node.args['dest']
            if 'tag' in node.args:
                node.comm_tag = node.args['tag']
            if 'count' in node.args and 'datatype' in node.args:
                node.data_size = self._calculate_data_size(node.args)
        
        # Check if it's a recv operation  
        elif name in MPI_P2P_RECV_OPS:
            node.node_type = 'recv'
            if 'source' in node.args:
                node.source_rank = node.args['source']
            if 'tag' in node.args:
                node.comm_tag = node.args['tag']
            if 'count' in node.args and 'datatype' in node.args:
                node.data_size = self._calculate_data_size(node.args)
        
        # Check if it's explicitly defined as compute
        elif name in MPI_COMPUTE_OPS:
            node.node_type = 'compute'
            
        # Unknown operation - default to compute and log it
        else:
            node.node_type = 'compute'
            # Track unsupported operations
            if name and not name.startswith('MPI_Gap_Fill_Compute'):  # Ignore dummy nodes
                self.unsupported_ops.add(name)
    
    def _calculate_data_size(self, args: Dict) -> int:
        """Calculate data size from MPI arguments"""
        count = args.get('count', 1)
        datatype = args.get('datatype', 0)
        
        # Handle hex string datatypes
        if isinstance(datatype, str):
            if datatype.startswith('0x'):
                datatype_val = int(datatype, 16)
            else:
                datatype_val = int(datatype)
        else:
            datatype_val = datatype
            
        size_in_bytes = MPI_DATATYPE_SIZES.get(datatype_val, 4)
        return count * size_in_bytes
    
    def _insert_dummy_compute(self, filtered_MPIs: List[MPINode]) -> None:
        """Insert dummy compute nodes to fill gaps in timeline"""
        i = 0
        while i < len(filtered_MPIs) - 1:
            step_1 = filtered_MPIs[i]
            step_2 = filtered_MPIs[i+1]
            
            # Check for gap
            if (step_1.ts + step_1.dur < step_2.ts):
                # Create dummy compute node with clear name
                dummy_node = MPINode(
                    name="MPI_Gap_Fill_Compute",
                    ph="X",
                    pid=step_1.pid,
                    tid=step_1.tid,
                    ts=step_1.ts + step_1.dur,
                    dur=step_2.ts - (step_1.ts + step_1.dur),
                    args={"info": "Dummy compute to fill timeline gap"}
                )
                dummy_node.node_type = 'compute'
                
                filtered_MPIs.insert(i+1, dummy_node)
                
            i += 1
    
    def _match_and_insert_send_recv(self, timeline_nodes: List[MPINode], send_recv_nodes: List[MPINode]) -> None:
        """Match send/recv pairs and insert them into the timeline"""
        # Group sends and recvs by tag and rank
        sends = {}
        recvs = {}
        
        for node in send_recv_nodes:
            if node.node_type == 'send':
                key = (node.pid, node.target_rank, node.comm_tag)
                if key not in sends:
                    sends[key] = []
                sends[key].append(node)
            elif node.node_type == 'recv':
                key = (node.source_rank, node.pid, node.comm_tag)
                if key not in recvs:
                    recvs[key] = []
                recvs[key].append(node)
        
        # Match pairs and add to timeline
        matched_nodes = []
        for key, send_list in sends.items():
            if key in recvs:
                recv_list = recvs[key]
                # Match by timestamp proximity
                for send_node in send_list:
                    best_recv = None
                    best_diff = float('inf')
                    for recv_node in recv_list:
                        time_diff = abs(recv_node.ts - send_node.ts)
                        if time_diff < best_diff:
                            best_diff = time_diff
                            best_recv = recv_node
                    
                    if best_recv:
                        # Add both to timeline
                        matched_nodes.append(send_node)
                        matched_nodes.append(best_recv)
                        recv_list.remove(best_recv)
        
        # Add unmatched sends/recvs as well (they might match with other ranks)
        for send_list in sends.values():
            for node in send_list:
                if node not in matched_nodes:
                    matched_nodes.append(node)
        
        for recv_list in recvs.values():
            for node in recv_list:
                if node not in matched_nodes:
                    matched_nodes.append(node)
        
        # Insert matched nodes into timeline
        timeline_nodes.extend(matched_nodes)
    
    def _get_output_filename(self, npu_id: int) -> str:
        """Get output filename for given NPU ID"""
        if npu_id < len(self.output_filenames):
            return self.output_filenames[npu_id]
        else:
            # Generate filename for scaled NPUs
            base_path = Path(self.output_filenames[0])
            return str(base_path.parent / f"rank_{npu_id}.et")
    
    def _create_global_metadata(self) -> Any:
        """Create global metadata for Chakra ET"""
        input_text = ""
        attr = [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
        ]
        metadata = GlobalMetadata(attr=attr)
        return metadata
    
    def _create_chakra_node(self) -> ChakraNode:
        """Create empty Chakra node"""
        ck_node = ChakraNode()
        ck_node.id = self._chakra_node_id
        self._chakra_node_id += 1
        return ck_node
    
    def _convert_MPI_node_to_chakra_node(self, mpi_node: MPINode, 
                                         mpi_id: int, 
                                         mpi_node_parents: List[int]) -> ChakraNode:
        """Convert MPI node to Chakra node"""
        ck_node = self._create_chakra_node()
        
        # Clean node name
        try:
            ck_node.name = mpi_node.name.split(",")[0]
        except:
            ck_node.name = mpi_node.name
        
        # Set node type based on MPI node type
        if mpi_node.node_type == 'compute':
            # Set compute node parameters
            ck_node.type = COMP_NODE
            ck_node.attr.append(self._create_attr("is_cpu_op", bool_val=False))
            ck_node.duration_micros = int(mpi_node.dur * 1000)  # Convert to microseconds
            
        elif mpi_node.node_type == 'collective':
            # Set collective node parameters
            ck_node.type = COMM_COLL_NODE
            ck_node.attr.append(self._create_attr("comm_type", 
                int64_val=mpi_node.comm_type))  # Using the Chakra constant directly
            
            # Handle comm_size based on operation type
            if mpi_node.name in ['MPI_Barrier', 'MPI_Wait', 'MPI_Waitall']:
                # For synchronization ops, use duration as a proxy for comm_size
                # or use the comm field if available
                if 'comm' in mpi_node.args:
                    comm_size = mpi_node.args.get('comm', 1) * 8  # Assume 8 bytes per rank
                else:
                    # Use duration in microseconds as comm_size for sync operations
                    comm_size = int(mpi_node.dur * 1000)
            else:
                # For data operations, calculate actual data size
                comm_size = mpi_node.comm_size
                
            ck_node.attr.append(self._create_attr("comm_size", uint64_val=comm_size))
            
            # Add comm_group for collective operations that have root
            if 'root' in mpi_node.args:
                ck_node.attr.append(self._create_attr("comm_group", 
                    int64_val=mpi_node.args.get('root', 0)))
            elif 'comm' in mpi_node.args:
                # Use comm field to indicate group participation
                ck_node.attr.append(self._create_attr("comm_group", 
                    int64_val=mpi_node.args.get('comm', 0)))
                
        elif mpi_node.node_type == 'send':
            # Set send node parameters
            ck_node.type = COMM_SEND_NODE
            ck_node.attr.append(self._create_attr("comm_size", 
                uint64_val=mpi_node.data_size))
            ck_node.attr.append(self._create_attr("comm_src", 
                int64_val=mpi_node.pid))
            # Use actual destination from the trace
            ck_node.attr.append(self._create_attr("comm_dst", 
                int64_val=mpi_node.target_rank))
            ck_node.attr.append(self._create_attr("comm_tag", 
                uint64_val=mpi_node.comm_tag))
                
        elif mpi_node.node_type == 'recv':
            # Set recv node parameters
            ck_node.type = COMM_RECV_NODE
            ck_node.attr.append(self._create_attr("comm_size", 
                uint64_val=mpi_node.data_size))
            ck_node.attr.append(self._create_attr("comm_dst", 
                int64_val=mpi_node.pid))
            # Use actual source from the trace
            ck_node.attr.append(self._create_attr("comm_src", 
                int64_val=mpi_node.source_rank))
            ck_node.attr.append(self._create_attr("comm_tag", 
                uint64_val=mpi_node.comm_tag))
        
        # Add involved NPUs extension
        ck_node.attr.extend([
            self._create_attr("comm_group", int64_val=0),
            self._create_attr("is_cpu_op", bool_val=False)
        ])
        
        # Set parents (dependencies)
        ck_node.data_deps.extend(mpi_node_parents)
        
        # Log debug information
        if self.node_debug_log:
            debug_info = {
                "chakra_id": ck_node.id,
                "name": ck_node.name,
                "node_type": self._get_node_type_name(ck_node.type),
                "mpi_type": mpi_node.node_type,
                "original_pid": mpi_node.pid,
                "timestamp": mpi_node.ts,
                "duration": mpi_node.dur,
                "dependencies": list(ck_node.data_deps),
                "attributes": {}
            }
            
            # Extract all attributes
            for attr in ck_node.attr:
                if attr.HasField("int64_val"):
                    debug_info["attributes"][attr.name] = attr.int64_val
                elif attr.HasField("uint64_val"):
                    debug_info["attributes"][attr.name] = attr.uint64_val
                elif attr.HasField("bool_val"):
                    debug_info["attributes"][attr.name] = attr.bool_val
                elif attr.HasField("string_val"):
                    debug_info["attributes"][attr.name] = attr.string_val
                    
            # Add duration for compute nodes
            if ck_node.type == COMP_NODE:
                debug_info["duration_micros"] = ck_node.duration_micros
                
            self.debug_nodes.append(debug_info)
        
        return ck_node
    
    def _get_node_type_name(self, node_type: int) -> str:
        """Get human-readable node type name"""
        type_map = {
            COMP_NODE: "COMP_NODE",
            COMM_COLL_NODE: "COMM_COLL_NODE", 
            COMM_SEND_NODE: "COMM_SEND_NODE",
            COMM_RECV_NODE: "COMM_RECV_NODE"
        }
        return type_map.get(node_type, f"UNKNOWN({node_type})")
    
    def _create_attr(self, name: str, **kwargs) -> ChakraAttr:
        """Create Chakra attribute"""
        attr = ChakraAttr()
        attr.name = name
        
        for key, value in kwargs.items():
            if key == 'int64_val':
                attr.int64_val = value
            elif key == 'uint64_val':
                attr.uint64_val = value
            elif key == 'bool_val':
                attr.bool_val = value
            elif key == 'string_val':
                attr.string_val = value
                
        return attr
    
    def _write_node_debug_log(self) -> None:
        """Write debug log with all node attributes"""
        with open(self.node_debug_log, 'w') as f:
            f.write("Chakra Node Debug Information\n")
            f.write("="*80 + "\n\n")
            
            # Sort by chakra ID for easier reading
            sorted_nodes = sorted(self.debug_nodes, key=lambda x: x['chakra_id'])
            
            for node in sorted_nodes:
                f.write(f"Node ID: {node['chakra_id']}\n")
                f.write(f"  Name: {node['name']}\n")
                f.write(f"  Type: {node['node_type']} (MPI: {node['mpi_type']})\n")
                f.write(f"  Original PID: {node['original_pid']}\n")
                f.write(f"  Timestamp: {node['timestamp']}\n")
                f.write(f"  Duration: {node['duration']}\n")
                
                if 'duration_micros' in node:
                    f.write(f"  Duration (micros): {node['duration_micros']}\n")
                    
                f.write(f"  Dependencies: {node['dependencies']}\n")
                f.write(f"  Attributes:\n")
                
                for attr_name, attr_value in node['attributes'].items():
                    f.write(f"    {attr_name}: {attr_value}\n")
                    
                f.write("\n")
                
        self.logger.info(f"Wrote debug information for {len(self.debug_nodes)} nodes to {self.node_debug_log}")
    
    def _write_unsupported_ops_log(self) -> None:
        """Write log file of unsupported operations encountered"""
        with open(self.unsupported_ops_log, 'w') as f:
            f.write("Unsupported MPI Operations Encountered\n")
            f.write("="*50 + "\n\n")
            f.write("The following MPI operations were not recognized and were converted to compute nodes:\n\n")
            
            for op in sorted(self.unsupported_ops):
                f.write(f"- {op}\n")
                
            f.write(f"\nTotal unsupported operations: {len(self.unsupported_ops)}\n")
            f.write("\nTo add support for these operations, update one of the following dictionaries:\n")
            f.write("- MPI_COLLECTIVE_OPS: for collective communication operations\n")
            f.write("- MPI_P2P_SEND_OPS: for point-to-point send operations\n")
            f.write("- MPI_P2P_RECV_OPS: for point-to-point receive operations\n")
            f.write("- MPI_COMPUTE_OPS: for operations that should be compute nodes\n")
            
        self.logger.warning(f"Found {len(self.unsupported_ops)} unsupported operations. See {self.unsupported_ops_log} for details.")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert MPI traces to Chakra format")
    parser.add_argument("--input-dir", "-i", required=True, 
                       help="Directory containing input MPI trace files")
    parser.add_argument("--output-dir", "-o", required=True, 
                       help="Directory for output Chakra ET files")
    parser.add_argument("--num-npus", "-n", type=int, required=True, 
                       help="Number of NPUs to generate traces for")
    parser.add_argument("--pattern", "-p", default="*.json", 
                       help="Pattern for input files (default: *.json)")
    parser.add_argument("--unsupported-ops-log", "-u", 
                       help="Path to log file for unsupported operations (optional)")
    parser.add_argument("--node-debug-log", "-d",
                       help="Path to log file for node debug information (optional)")
    parser.add_argument("--log-level", default="INFO", 
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Find input files
    input_files = sorted(glob.glob(f"{args.input_dir}/{args.pattern}"))
    if not input_files:
        raise ValueError(f"No input files found matching {args.input_dir}/{args.pattern}")
    
    # Generate output filenames
    output_files = []
    for i in range(args.num_npus):
        output_files.append(f"{args.output_dir}/rank_{i}.et")
    
    # Create output directory if needed
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run converter
    converter = MPIConverter(
        input_filenames=input_files,
        output_filenames=output_files,
        num_npus=args.num_npus,
        unsupported_ops_log=args.unsupported_ops_log,
        node_debug_log=args.node_debug_log
    )
    
    converter.convert()


if __name__ == "__main__":
    main()

