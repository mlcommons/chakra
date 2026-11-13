"""
Make the generated ``*_pb2`` modules importable as a package.

``protoc``'s ``--python_out`` emits flat, absolute imports between generated
modules (e.g. ``et_def_pb2`` contains ``import storage_pb2``). Those resolve
against ``sys.path``, not against this package's own directory, so importing
``chakra.schema.protobuf.et_def_pb2`` would otherwise raise
``ModuleNotFoundError: No module named 'storage_pb2'``. Putting this directory
on ``sys.path`` lets those cross-module imports resolve.

See https://github.com/protocolbuffers/protobuf/issues/1491.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
