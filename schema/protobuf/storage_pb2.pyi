from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StorageLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STORAGE_LEVEL_UNKNOWN: _ClassVar[StorageLevel]
    STORAGE_LEVEL_FS: _ClassVar[StorageLevel]
    STORAGE_LEVEL_PAGE: _ClassVar[StorageLevel]
    STORAGE_LEVEL_BLOCK: _ClassVar[StorageLevel]
    STORAGE_LEVEL_DEVICE: _ClassVar[StorageLevel]
STORAGE_LEVEL_UNKNOWN: StorageLevel
STORAGE_LEVEL_FS: StorageLevel
STORAGE_LEVEL_PAGE: StorageLevel
STORAGE_LEVEL_BLOCK: StorageLevel
STORAGE_LEVEL_DEVICE: StorageLevel

class StorageInfo(_message.Message):
    __slots__ = ("level", "fs", "page", "block")
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    FS_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    BLOCK_FIELD_NUMBER: _ClassVar[int]
    level: StorageLevel
    fs: FsOp
    page: PageOp
    block: BlockOp
    def __init__(self, level: _Optional[_Union[StorageLevel, str]] = ..., fs: _Optional[_Union[FsOp, _Mapping]] = ..., page: _Optional[_Union[PageOp, _Mapping]] = ..., block: _Optional[_Union[BlockOp, _Mapping]] = ...) -> None: ...

class FsOp(_message.Message):
    __slots__ = ("op", "pathname", "fd", "offset", "size_bytes", "flags")
    class OpType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        FS_OP_UNKNOWN: _ClassVar[FsOp.OpType]
        FS_OP_OPEN: _ClassVar[FsOp.OpType]
        FS_OP_CLOSE: _ClassVar[FsOp.OpType]
        FS_OP_READ: _ClassVar[FsOp.OpType]
        FS_OP_WRITE: _ClassVar[FsOp.OpType]
        FS_OP_PREAD: _ClassVar[FsOp.OpType]
        FS_OP_PWRITE: _ClassVar[FsOp.OpType]
        FS_OP_FSYNC: _ClassVar[FsOp.OpType]
        FS_OP_FDATASYNC: _ClassVar[FsOp.OpType]
        FS_OP_SYNC_FILE_RANGE: _ClassVar[FsOp.OpType]
    FS_OP_UNKNOWN: FsOp.OpType
    FS_OP_OPEN: FsOp.OpType
    FS_OP_CLOSE: FsOp.OpType
    FS_OP_READ: FsOp.OpType
    FS_OP_WRITE: FsOp.OpType
    FS_OP_PREAD: FsOp.OpType
    FS_OP_PWRITE: FsOp.OpType
    FS_OP_FSYNC: FsOp.OpType
    FS_OP_FDATASYNC: FsOp.OpType
    FS_OP_SYNC_FILE_RANGE: FsOp.OpType
    OP_FIELD_NUMBER: _ClassVar[int]
    PATHNAME_FIELD_NUMBER: _ClassVar[int]
    FD_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    op: FsOp.OpType
    pathname: str
    fd: int
    offset: int
    size_bytes: int
    flags: int
    def __init__(self, op: _Optional[_Union[FsOp.OpType, str]] = ..., pathname: _Optional[str] = ..., fd: _Optional[int] = ..., offset: _Optional[int] = ..., size_bytes: _Optional[int] = ..., flags: _Optional[int] = ...) -> None: ...

class PageOp(_message.Message):
    __slots__ = ("op", "inode", "page_index", "nr_pages")
    class OpType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        PAGE_OP_UNKNOWN: _ClassVar[PageOp.OpType]
        PAGE_OP_ADD_TO_CACHE: _ClassVar[PageOp.OpType]
        PAGE_OP_DIRTY: _ClassVar[PageOp.OpType]
        PAGE_OP_READAHEAD_SYNC: _ClassVar[PageOp.OpType]
        PAGE_OP_READAHEAD_ASYNC: _ClassVar[PageOp.OpType]
    PAGE_OP_UNKNOWN: PageOp.OpType
    PAGE_OP_ADD_TO_CACHE: PageOp.OpType
    PAGE_OP_DIRTY: PageOp.OpType
    PAGE_OP_READAHEAD_SYNC: PageOp.OpType
    PAGE_OP_READAHEAD_ASYNC: PageOp.OpType
    OP_FIELD_NUMBER: _ClassVar[int]
    INODE_FIELD_NUMBER: _ClassVar[int]
    PAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    NR_PAGES_FIELD_NUMBER: _ClassVar[int]
    op: PageOp.OpType
    inode: int
    page_index: int
    nr_pages: int
    def __init__(self, op: _Optional[_Union[PageOp.OpType, str]] = ..., inode: _Optional[int] = ..., page_index: _Optional[int] = ..., nr_pages: _Optional[int] = ...) -> None: ...

class BlockOp(_message.Message):
    __slots__ = ("op", "queue_event", "device", "lba", "nr_sectors", "req_op_flags")
    class OpType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        BLOCK_OP_UNKNOWN: _ClassVar[BlockOp.OpType]
        BLOCK_OP_READ: _ClassVar[BlockOp.OpType]
        BLOCK_OP_WRITE: _ClassVar[BlockOp.OpType]
        BLOCK_OP_FLUSH: _ClassVar[BlockOp.OpType]
    BLOCK_OP_UNKNOWN: BlockOp.OpType
    BLOCK_OP_READ: BlockOp.OpType
    BLOCK_OP_WRITE: BlockOp.OpType
    BLOCK_OP_FLUSH: BlockOp.OpType
    class QueueEvent(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        BLOCK_EVENT_UNKNOWN: _ClassVar[BlockOp.QueueEvent]
        BLOCK_EVENT_QUEUE: _ClassVar[BlockOp.QueueEvent]
        BLOCK_EVENT_GET: _ClassVar[BlockOp.QueueEvent]
        BLOCK_EVENT_INSERT: _ClassVar[BlockOp.QueueEvent]
        BLOCK_EVENT_DISPATCH: _ClassVar[BlockOp.QueueEvent]
        BLOCK_EVENT_COMPLETE: _ClassVar[BlockOp.QueueEvent]
    BLOCK_EVENT_UNKNOWN: BlockOp.QueueEvent
    BLOCK_EVENT_QUEUE: BlockOp.QueueEvent
    BLOCK_EVENT_GET: BlockOp.QueueEvent
    BLOCK_EVENT_INSERT: BlockOp.QueueEvent
    BLOCK_EVENT_DISPATCH: BlockOp.QueueEvent
    BLOCK_EVENT_COMPLETE: BlockOp.QueueEvent
    OP_FIELD_NUMBER: _ClassVar[int]
    QUEUE_EVENT_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    LBA_FIELD_NUMBER: _ClassVar[int]
    NR_SECTORS_FIELD_NUMBER: _ClassVar[int]
    REQ_OP_FLAGS_FIELD_NUMBER: _ClassVar[int]
    op: BlockOp.OpType
    queue_event: BlockOp.QueueEvent
    device: str
    lba: int
    nr_sectors: int
    req_op_flags: int
    def __init__(self, op: _Optional[_Union[BlockOp.OpType, str]] = ..., queue_event: _Optional[_Union[BlockOp.QueueEvent, str]] = ..., device: _Optional[str] = ..., lba: _Optional[int] = ..., nr_sectors: _Optional[int] = ..., req_op_flags: _Optional[int] = ...) -> None: ...
