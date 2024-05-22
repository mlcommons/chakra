from src.trace_link.trace_linker import TraceLinker


def test_trace_linker_initialization():
    pytorch_et_file = "path/to/pytorch_et.json"
    kineto_file = "path/to/kineto.json"
    TraceLinker(pytorch_et_file, kineto_file)
