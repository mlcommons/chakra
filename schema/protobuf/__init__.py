"""
Chakra execution trace schema: the .proto files and the *_pb2 modules that scripts/gen_proto.sh generates from them.

The .proto files import each other by their package path (``import "chakra/schema/protobuf/storage.proto"``), so the
generated modules import each other as ``chakra.schema.protobuf.*_pb2``.
"""
