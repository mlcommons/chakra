import storage_pb2 as _storage_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NodeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INVALID_NODE: _ClassVar[NodeType]
    METADATA_NODE: _ClassVar[NodeType]
    MEM_LOAD_NODE: _ClassVar[NodeType]
    MEM_STORE_NODE: _ClassVar[NodeType]
    COMP_NODE: _ClassVar[NodeType]
    COMM_SEND_NODE: _ClassVar[NodeType]
    COMM_RECV_NODE: _ClassVar[NodeType]
    COMM_COLL_NODE: _ClassVar[NodeType]
    STORAGE_NODE: _ClassVar[NodeType]
    DATA_OP_NODE: _ClassVar[NodeType]

class DataOpType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DATA_OP_UNKNOWN: _ClassVar[DataOpType]
    DATA_OP_DATALOADER: _ClassVar[DataOpType]
    DATA_OP_CHECKPOINT: _ClassVar[DataOpType]
    DATA_OP_KV_TRANSFER: _ClassVar[DataOpType]

class CollectiveCommType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ALL_REDUCE: _ClassVar[CollectiveCommType]
    REDUCE: _ClassVar[CollectiveCommType]
    ALL_GATHER: _ClassVar[CollectiveCommType]
    GATHER: _ClassVar[CollectiveCommType]
    SCATTER: _ClassVar[CollectiveCommType]
    BROADCAST: _ClassVar[CollectiveCommType]
    ALL_TO_ALL: _ClassVar[CollectiveCommType]
    REDUCE_SCATTER: _ClassVar[CollectiveCommType]
    REDUCE_SCATTER_BLOCK: _ClassVar[CollectiveCommType]
    BARRIER: _ClassVar[CollectiveCommType]
INVALID_NODE: NodeType
METADATA_NODE: NodeType
MEM_LOAD_NODE: NodeType
MEM_STORE_NODE: NodeType
COMP_NODE: NodeType
COMM_SEND_NODE: NodeType
COMM_RECV_NODE: NodeType
COMM_COLL_NODE: NodeType
STORAGE_NODE: NodeType
DATA_OP_NODE: NodeType
DATA_OP_UNKNOWN: DataOpType
DATA_OP_DATALOADER: DataOpType
DATA_OP_CHECKPOINT: DataOpType
DATA_OP_KV_TRANSFER: DataOpType
ALL_REDUCE: CollectiveCommType
REDUCE: CollectiveCommType
ALL_GATHER: CollectiveCommType
GATHER: CollectiveCommType
SCATTER: CollectiveCommType
BROADCAST: CollectiveCommType
ALL_TO_ALL: CollectiveCommType
REDUCE_SCATTER: CollectiveCommType
REDUCE_SCATTER_BLOCK: CollectiveCommType
BARRIER: CollectiveCommType

class AttributeProto(_message.Message):
    __slots__ = ("name", "doc_string", "double_val", "double_list", "float_val", "float_list", "int32_val", "int32_list", "int64_val", "int64_list", "uint32_val", "uint32_list", "uint64_val", "uint64_list", "sint32_val", "sint32_list", "sint64_val", "sint64_list", "fixed32_val", "fixed32_list", "fixed64_val", "fixed64_list", "sfixed32_val", "sfixed32_list", "sfixed64_val", "sfixed64_list", "bool_val", "bool_list", "string_val", "string_list", "bytes_val", "bytes_list")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DOC_STRING_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_VAL_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_LIST_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VAL_FIELD_NUMBER: _ClassVar[int]
    FLOAT_LIST_FIELD_NUMBER: _ClassVar[int]
    INT32_VAL_FIELD_NUMBER: _ClassVar[int]
    INT32_LIST_FIELD_NUMBER: _ClassVar[int]
    INT64_VAL_FIELD_NUMBER: _ClassVar[int]
    INT64_LIST_FIELD_NUMBER: _ClassVar[int]
    UINT32_VAL_FIELD_NUMBER: _ClassVar[int]
    UINT32_LIST_FIELD_NUMBER: _ClassVar[int]
    UINT64_VAL_FIELD_NUMBER: _ClassVar[int]
    UINT64_LIST_FIELD_NUMBER: _ClassVar[int]
    SINT32_VAL_FIELD_NUMBER: _ClassVar[int]
    SINT32_LIST_FIELD_NUMBER: _ClassVar[int]
    SINT64_VAL_FIELD_NUMBER: _ClassVar[int]
    SINT64_LIST_FIELD_NUMBER: _ClassVar[int]
    FIXED32_VAL_FIELD_NUMBER: _ClassVar[int]
    FIXED32_LIST_FIELD_NUMBER: _ClassVar[int]
    FIXED64_VAL_FIELD_NUMBER: _ClassVar[int]
    FIXED64_LIST_FIELD_NUMBER: _ClassVar[int]
    SFIXED32_VAL_FIELD_NUMBER: _ClassVar[int]
    SFIXED32_LIST_FIELD_NUMBER: _ClassVar[int]
    SFIXED64_VAL_FIELD_NUMBER: _ClassVar[int]
    SFIXED64_LIST_FIELD_NUMBER: _ClassVar[int]
    BOOL_VAL_FIELD_NUMBER: _ClassVar[int]
    BOOL_LIST_FIELD_NUMBER: _ClassVar[int]
    STRING_VAL_FIELD_NUMBER: _ClassVar[int]
    STRING_LIST_FIELD_NUMBER: _ClassVar[int]
    BYTES_VAL_FIELD_NUMBER: _ClassVar[int]
    BYTES_LIST_FIELD_NUMBER: _ClassVar[int]
    name: str
    doc_string: str
    double_val: float
    double_list: DoubleList
    float_val: float
    float_list: FloatList
    int32_val: int
    int32_list: Int32List
    int64_val: int
    int64_list: Int64List
    uint32_val: int
    uint32_list: Uint32List
    uint64_val: int
    uint64_list: Uint64List
    sint32_val: int
    sint32_list: Sint32List
    sint64_val: int
    sint64_list: Sint64List
    fixed32_val: int
    fixed32_list: Fixed32List
    fixed64_val: int
    fixed64_list: Fixed64List
    sfixed32_val: int
    sfixed32_list: Sfixed32List
    sfixed64_val: int
    sfixed64_list: Sfixed64List
    bool_val: bool
    bool_list: BoolList
    string_val: str
    string_list: StringList
    bytes_val: bytes
    bytes_list: BytesList
    def __init__(self, name: _Optional[str] = ..., doc_string: _Optional[str] = ..., double_val: _Optional[float] = ..., double_list: _Optional[_Union[DoubleList, _Mapping]] = ..., float_val: _Optional[float] = ..., float_list: _Optional[_Union[FloatList, _Mapping]] = ..., int32_val: _Optional[int] = ..., int32_list: _Optional[_Union[Int32List, _Mapping]] = ..., int64_val: _Optional[int] = ..., int64_list: _Optional[_Union[Int64List, _Mapping]] = ..., uint32_val: _Optional[int] = ..., uint32_list: _Optional[_Union[Uint32List, _Mapping]] = ..., uint64_val: _Optional[int] = ..., uint64_list: _Optional[_Union[Uint64List, _Mapping]] = ..., sint32_val: _Optional[int] = ..., sint32_list: _Optional[_Union[Sint32List, _Mapping]] = ..., sint64_val: _Optional[int] = ..., sint64_list: _Optional[_Union[Sint64List, _Mapping]] = ..., fixed32_val: _Optional[int] = ..., fixed32_list: _Optional[_Union[Fixed32List, _Mapping]] = ..., fixed64_val: _Optional[int] = ..., fixed64_list: _Optional[_Union[Fixed64List, _Mapping]] = ..., sfixed32_val: _Optional[int] = ..., sfixed32_list: _Optional[_Union[Sfixed32List, _Mapping]] = ..., sfixed64_val: _Optional[int] = ..., sfixed64_list: _Optional[_Union[Sfixed64List, _Mapping]] = ..., bool_val: bool = ..., bool_list: _Optional[_Union[BoolList, _Mapping]] = ..., string_val: _Optional[str] = ..., string_list: _Optional[_Union[StringList, _Mapping]] = ..., bytes_val: _Optional[bytes] = ..., bytes_list: _Optional[_Union[BytesList, _Mapping]] = ...) -> None: ...

class DoubleList(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class FloatList(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class Int32List(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class Int64List(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class Uint32List(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class Uint64List(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class Sint32List(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class Sint64List(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class Fixed32List(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class Fixed64List(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class Sfixed32List(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class Sfixed64List(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class BoolList(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[bool]
    def __init__(self, values: _Optional[_Iterable[bool]] = ...) -> None: ...

class StringList(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, values: _Optional[_Iterable[str]] = ...) -> None: ...

class BytesList(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, values: _Optional[_Iterable[bytes]] = ...) -> None: ...

class GlobalMetadata(_message.Message):
    __slots__ = ("version", "attr")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    ATTR_FIELD_NUMBER: _ClassVar[int]
    version: str
    attr: _containers.RepeatedCompositeFieldContainer[AttributeProto]
    def __init__(self, version: _Optional[str] = ..., attr: _Optional[_Iterable[_Union[AttributeProto, _Mapping]]] = ...) -> None: ...

class Node(_message.Message):
    __slots__ = ("id", "name", "type", "ctrl_deps", "data_deps", "start_time_micros", "duration_micros", "inputs", "outputs", "attr", "storage", "data_op")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CTRL_DEPS_FIELD_NUMBER: _ClassVar[int]
    DATA_DEPS_FIELD_NUMBER: _ClassVar[int]
    START_TIME_MICROS_FIELD_NUMBER: _ClassVar[int]
    DURATION_MICROS_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    ATTR_FIELD_NUMBER: _ClassVar[int]
    STORAGE_FIELD_NUMBER: _ClassVar[int]
    DATA_OP_FIELD_NUMBER: _ClassVar[int]
    id: int
    name: str
    type: NodeType
    ctrl_deps: _containers.RepeatedScalarFieldContainer[int]
    data_deps: _containers.RepeatedScalarFieldContainer[int]
    start_time_micros: int
    duration_micros: int
    inputs: IOInfo
    outputs: IOInfo
    attr: _containers.RepeatedCompositeFieldContainer[AttributeProto]
    storage: _storage_pb2.StorageInfo
    data_op: DataOpType
    def __init__(self, id: _Optional[int] = ..., name: _Optional[str] = ..., type: _Optional[_Union[NodeType, str]] = ..., ctrl_deps: _Optional[_Iterable[int]] = ..., data_deps: _Optional[_Iterable[int]] = ..., start_time_micros: _Optional[int] = ..., duration_micros: _Optional[int] = ..., inputs: _Optional[_Union[IOInfo, _Mapping]] = ..., outputs: _Optional[_Union[IOInfo, _Mapping]] = ..., attr: _Optional[_Iterable[_Union[AttributeProto, _Mapping]]] = ..., storage: _Optional[_Union[_storage_pb2.StorageInfo, _Mapping]] = ..., data_op: _Optional[_Union[DataOpType, str]] = ...) -> None: ...

class IOInfo(_message.Message):
    __slots__ = ("values", "shapes", "types")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    SHAPES_FIELD_NUMBER: _ClassVar[int]
    TYPES_FIELD_NUMBER: _ClassVar[int]
    values: str
    shapes: str
    types: str
    def __init__(self, values: _Optional[str] = ..., shapes: _Optional[str] = ..., types: _Optional[str] = ...) -> None: ...

class Tensor(_message.Message):
    __slots__ = ("tensor_id", "storage_id", "offset", "num_elem", "elem_bytes", "device")
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    STORAGE_ID_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    NUM_ELEM_FIELD_NUMBER: _ClassVar[int]
    ELEM_BYTES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    tensor_id: int
    storage_id: int
    offset: int
    num_elem: int
    elem_bytes: int
    device: str
    def __init__(self, tensor_id: _Optional[int] = ..., storage_id: _Optional[int] = ..., offset: _Optional[int] = ..., num_elem: _Optional[int] = ..., elem_bytes: _Optional[int] = ..., device: _Optional[str] = ...) -> None: ...
