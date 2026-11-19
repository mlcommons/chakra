import json
import tarfile
from pathlib import Path
from typing import Any, Dict

import pytest
from chakra.src.host_trace.errors import HostTraceError
from chakra.src.host_trace.node import HostNode
from chakra.src.host_trace.trace import HostTrace


@pytest.fixture
def extract_tar_gz_file(tmp_path: Path) -> Path:
    """
    Fixture to extract a tar.gz file to a temporary directory.

    Args:
        tmp_path (Path): Temporary directory path provided by pytest.

    Returns:
        Path: Path to the extracted directory.
    """
    tar_gz_file = Path("tests/data/1.0.2-chakra.0.0.4.tgz")
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()

    # Read the one file the tests use, by name, so no path stored in the archive can write outside extracted_dir.
    with tarfile.open(tar_gz_file, "r:gz") as tar:
        member = tar.extractfile("1.0.2-chakra.0.0.4.json")
        assert member is not None, f"1.0.2-chakra.0.0.4.json is not a file in {tar_gz_file}"
        (extracted_dir / "1.0.2-chakra.0.0.4.json").write_bytes(member.read())

    return extracted_dir


def load_pytorch_execution_traces(file_path: str) -> Dict[str, Any]:
    """
    Loads PyTorch execution traces from a file.

    Args:
        file_path (str): Path to the PyTorch execution trace file.

    Returns:
        Dict[str, Any]: Parsed PyTorch execution trace data.
    """
    with open(file_path, "r") as pytorch_et:
        return json.load(pytorch_et)


def test_host_node_parsing(extract_tar_gz_file: Path) -> None:
    """
    Test to check if HostNode can parse nodes properly from the extracted data.

    Args:
        extract_tar_gz_file (Path): Path to the extracted directory containing
            the PyTorch execution trace file.
    """
    pytorch_trace_file = extract_tar_gz_file / "1.0.2-chakra.0.0.4.json"
    pytorch_et_data = load_pytorch_execution_traces(str(pytorch_trace_file))

    pytorch_schema = pytorch_et_data["schema"]
    pytorch_nodes = pytorch_et_data["nodes"]

    for node_data in pytorch_nodes:
        node = HostNode(pytorch_schema, node_data)
        assert node is not None  # Check if node is instantiated properly


@pytest.fixture
def sample_node_data_1_0_2_chakra_0_0_4() -> Dict:
    return {
        "id": 1,
        "name": "node1",
        "ctrl_deps": None,
        "inputs": {"values": "values", "shapes": "shapes", "types": "types"},
        "outputs": {"values": "values", "shapes": "shapes", "types": "types"},
        "attrs": [
            {"name": "rf_id", "type": "uint64", "value": 0},
            {"name": "fw_parent", "type": "uint64", "value": 0},
            {"name": "seq_id", "type": "int64", "value": -1},
            {"name": "scope", "type": "uint64", "value": 7},
            {"name": "tid", "type": "uint64", "value": 1},
            {"name": "fw_tid", "type": "uint64", "value": 0},
            {"name": "op_schema", "type": "string", "value": ""},
        ],
        "exclusive_dur": 50,
    }


@pytest.fixture
def sample_node_data_1_0_3_chakra_0_0_4() -> Dict:
    return {
        "id": 2,
        "name": "node2",
        "ctrl_deps": 1,
        "inputs": {"values": [], "shapes": [], "types": []},
        "outputs": {"values": [], "shapes": [], "types": []},
        "attrs": [
            {"name": "rf_id", "type": "uint64", "value": 2},
            {"name": "fw_parent", "type": "uint64", "value": 0},
            {"name": "seq_id", "type": "int64", "value": -1},
            {"name": "scope", "type": "uint64", "value": 7},
            {"name": "tid", "type": "uint64", "value": 1},
            {"name": "fw_tid", "type": "uint64", "value": 0},
            {"name": "op_schema", "type": "string", "value": ""},
        ],
        "exclusive_dur": 30,
    }


@pytest.fixture
def sample_node_data_unsupported_schema() -> Dict:
    return {
        "id": 4,
        "name": "## process_group:init ##",
        "ctrl_deps": 3,
        "inputs": {
            "values": [],
            "shapes": [[]],
            "types": ["String"],
        },
        "outputs": {"values": [], "shapes": [], "types": []},
        "attrs": [
            {"name": "rf_id", "type": "uint64", "value": 2},
            {"name": "fw_parent", "type": "uint64", "value": 0},
            {"name": "seq_id", "type": "int64", "value": -1},
            {"name": "scope", "type": "uint64", "value": 7},
            {"name": "tid", "type": "uint64", "value": 1},
            {"name": "fw_tid", "type": "uint64", "value": 0},
            {"name": "op_schema", "type": "string", "value": ""},
        ],
        "exclusive_dur": 40,
    }


def test_host_node_parsing_1_0_2_chakra_0_0_4(sample_node_data_1_0_2_chakra_0_0_4) -> None:
    schema = "1.0.2-chakra.0.0.4"
    node = HostNode(schema, sample_node_data_1_0_2_chakra_0_0_4)
    assert node is not None
    assert node.schema == schema
    assert isinstance(node.id, int)
    assert isinstance(node.name, str)
    assert node.exclusive_dur == 50


def test_host_node_parsing_1_0_3_chakra_0_0_4(sample_node_data_1_0_3_chakra_0_0_4) -> None:
    schema = "1.0.3-chakra.0.0.4"
    node = HostNode(schema, sample_node_data_1_0_3_chakra_0_0_4)
    assert node is not None
    assert node.schema == schema
    assert isinstance(node.id, int)
    assert isinstance(node.name, str)
    assert node.exclusive_dur == 30


def test_unknown_schema_with_a_valid_layout_is_parsed_with_a_warning(
    sample_node_data_unsupported_schema, caplog: pytest.LogCaptureFixture
) -> None:
    # The schema is checked once per trace, against the released format versions. A newer one is read with the newest
    # known rules, after a warning.
    schema = "9999.9999.9999-chakra.0.0.4"
    trace = HostTrace({"schema": schema, "nodes": [sample_node_data_unsupported_schema]})
    assert trace.nodes[4].name == sample_node_data_unsupported_schema["name"]
    assert trace.schema.text == schema
    assert f"schema {schema} is newer than the newest known format" in caplog.text


def test_unrecognised_layout_is_rejected(sample_node_data_unsupported_schema) -> None:
    node_data = dict(sample_node_data_unsupported_schema)
    node_data["parent"] = node_data.pop("ctrl_deps")  # the PyTorch 2.0/2.1 key
    with pytest.raises(ValueError, match="node 4: missing key"):
        HostNode("1.1.1-chakra.0.0.4", node_data)


def test_host_node_has_a_field_for_every_core_attribute(sample_node_data_1_0_2_chakra_0_0_4) -> None:
    # names.CORE_ATTRS is the one list of the attributes Chakra reads; HostNode sets each as an explicit field.
    from chakra.src.host_trace.names import CORE_ATTRS

    node = HostNode("1.0.2-chakra.0.0.4", sample_node_data_1_0_2_chakra_0_0_4)
    for name in CORE_ATTRS:
        assert hasattr(node, name), f"HostNode has no field for {name}"
        assert getattr(node, name) is not None, f"{name} was not read from attrs"


@pytest.mark.parametrize("key", ["id", "name", "ctrl_deps", "inputs", "outputs"])
def test_missing_required_key_is_an_error_naming_it(key: str, make_node) -> None:
    node_data = make_node(3, "linear", 2)
    del node_data[key]
    with pytest.raises(HostTraceError, match=f"missing key\\(s\\) {key}"):
        HostNode("1.1.1-chakra.0.0.4", node_data)


def test_attrs_that_is_not_a_list_is_an_error(make_node) -> None:
    node_data = make_node(3, "linear", 2, attrs={"rf_id": 3})
    with pytest.raises(HostTraceError, match="attrs is a dict, not a list"):
        HostNode("1.1.1-chakra.0.0.4", node_data)


@pytest.mark.parametrize("missing", ["name", "value"])
def test_attrs_entry_without_name_or_value_is_an_error_naming_the_entry(missing: str, make_node) -> None:
    node_data = make_node(3, "linear", 2)
    del node_data["attrs"][1][missing]
    with pytest.raises(HostTraceError, match="node 3 \\(linear\\): attrs entry 1 is not a name/value entry"):
        HostNode("1.1.1-chakra.0.0.4", node_data)


def _with_attr(node_data: Dict[str, Any], name: str, value: Any) -> Dict[str, Any]:
    for attr in node_data["attrs"]:
        if attr["name"] == name:
            attr["value"] = value
    return node_data


def test_core_attributes_are_converted(make_node) -> None:
    node = HostNode("1.1.1-chakra.0.0.4", _with_attr(make_node(3, "linear", 2), "rf_id", "7"))
    assert node.rf_id == 7
    assert node.op_schema == "aten::linear"
    assert node.attrs["rf_id"] == "7"  # attrs keeps the value as read


def test_null_attribute_value_is_none(make_node) -> None:
    assert HostNode("1.1.1-chakra.0.0.4", _with_attr(make_node(3, "linear", 2), "tid", None)).tid is None


def test_unconvertible_attribute_value_is_an_error_naming_it(make_node) -> None:
    with pytest.raises(HostTraceError, match="node 3 \\(linear\\): attribute 'rf_id' has the value 'x'"):
        HostNode("1.1.1-chakra.0.0.4", _with_attr(make_node(3, "linear", 2), "rf_id", "x"))


@pytest.mark.parametrize(
    ("top", "attr", "expected"),
    [("top", "attr", "attr"), ("top", None, "top"), (None, "attr", "attr"), (None, None, "")],
    ids=["both-attrs-wins", "top-level-only", "attrs-only", "neither"],
)
def test_pg_name_precedence(top, attr, expected, make_node) -> None:
    node_data = make_node(3, "record_param_comms", 2)
    if top is not None:
        node_data["pg_name"] = top
    if attr is not None:
        node_data["attrs"].append({"name": "pg_name", "type": "string", "value": attr})
    assert HostNode("1.1.1-chakra.0.0.4", node_data).pg_name == expected


def test_a_gpu_node_is_one_with_a_category(make_node) -> None:
    assert HostNode("1.1.1-chakra.0.0.4", make_node(5, "kernel", 4, cat="kernel")).is_gpu_op()
    assert not HostNode("1.1.1-chakra.0.0.4", make_node(3, "linear", 2)).is_gpu_op()
