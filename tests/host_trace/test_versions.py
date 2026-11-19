import logging

import pytest
from chakra.src.host_trace import names, versions
from chakra.src.host_trace.errors import HostTraceError


def test_schema_is_split_into_format_and_chakra_suffix() -> None:
    schema = versions.parse_schema("1.1.1-chakra.0.0.4")
    assert (schema.text, schema.format, schema.suffix) == ("1.1.1-chakra.0.0.4", (1, 1, 1), (0, 0, 4))


def test_schema_without_suffix_is_parsed() -> None:
    assert versions.parse_schema("1.0.1").suffix is None


@pytest.mark.parametrize("text", ["abc", "1.1", "1.1.1-chakra", ""])
def test_unrecognised_schema_is_rejected(text: str) -> None:
    with pytest.raises(HostTraceError, match="unrecognised schema"):
        versions.parse_schema(text)


def test_schema_older_than_1_0_2_is_rejected() -> None:
    with pytest.raises(HostTraceError, match="older than 1.0.2.*PyTorch 2.2"):
        versions.check_supported(versions.parse_schema("1.0.1"))


@pytest.mark.parametrize(
    "text", ["1.0.2-chakra.0.0.4", "1.0.4-chakra.0.0.4", "1.1.0-chakra.0.0.4", "1.1.1-chakra.1.0.0"]
)
def test_supported_schemas_are_accepted_without_a_warning(text: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        versions.check_supported(versions.parse_schema(text))
    assert not caplog.records


def test_schema_newer_than_the_newest_known_gives_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        versions.check_supported(versions.parse_schema("9.9.9"))
    assert len(caplog.records) == 1
    assert "newer than the newest known format 1.1.1" in caplog.text


@pytest.mark.parametrize(
    ("text", "attrs", "tensor_keys"),
    [
        ("1.0.2-chakra.0.0.4", names.CORE_ATTRS, names.TENSOR_KEYS),
        ("1.0.4-chakra.0.0.4", names.CORE_ATTRS, names.TENSOR_KEYS),
        ("1.1.0-chakra.0.0.4", names.CORE_ATTRS + names.KERNEL_ATTRS, names.TENSOR_KEYS),
        ("1.1.1-chakra.0.0.4", names.CORE_ATTRS + names.KERNEL_ATTRS, names.TENSOR_KEYS + (names.STRIDES,)),
        ("9.9.9", names.CORE_ATTRS + names.KERNEL_ATTRS, names.TENSOR_KEYS + (names.STRIDES,)),
    ],
)
def test_expected_keys_follow_the_released_format_changes(text: str, attrs: tuple, tensor_keys: tuple) -> None:
    assert versions.expected_keys(versions.parse_schema(text)) == (attrs, tensor_keys)
