import re

from third_party.utils.execution_graph import Node as PyTorchNode

def has_backward_parent(op: PyTorchNode) -> bool:
    if not op.parent or op.parent.id == op.id:
        return False
    if is_backward_parent(op):
        return True
    return has_backward_parent(op.parent)

def is_backward_parent(op: PyTorchNode) -> bool:
    return (
        "autograd::engine::evaluate_function: " in op.name
        or "Optimizer.step" in op.name
    )

def is_backward_aten(op: PyTorchNode) -> bool:
    return op.name.startswith("aten::") and has_backward_parent(op)

def is_fbgemm_forward(op: PyTorchNode) -> bool:
    return "fbgemm::split_embedding_codegen_lookup_" in op.name

def is_fbgemm_backward(op: PyTorchNode) -> bool:
    return "CppNode<SplitLookupFunction_" in op.name and not is_backward_parent(op)

def skip_op(op: PyTorchNode) -> bool:
    return (
        not is_fbgemm_forward(op)
        and op.parent is not None
        and (
            "embedding_lookup" in op.parent.name
            or "param|SplitTableBatchedEmbeddingBagsCodegen" in op.parent.name
            or (
                "fbgemm::" in op.name
                and "fbgemm::split_embedding_codegen_lookup_" not in op.name
            )
        )
        or ("fused" in op.name)
        or (
            op.name
            in [
                "aten::empty",
                "aten::to",
                "aten::lift",
                "aten::detach_",
                "aten::set_",
                "aten::pin_memory",
            ]
            and "thread" in op.parent.name
            and op.tid == 2
        )
        or (op.name == "record_param_comms" and op.inputs[3] == "init")
        or (op.name == "aten::view" and "aten::view.dtype" in op.op_schema)
    )
