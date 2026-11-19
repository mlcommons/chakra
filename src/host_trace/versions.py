"""
Host trace schema versions: which PyTorch releases changed the node format, and what each version requires.

PyTorch writes the schema as "<format>-chakra.<suffix>", for example "1.1.1-chakra.0.0.4". Only the format version
decides how a trace is read; the suffix is kept as information. PyTorch has always written "chakra.0.0.4"; the
"chakra.1.0.0" suffix, read the same way, came from Chakra's own version bump (34f95f9), not from PyTorch. Versions are
compared as numbers, so a version between two listed ones (the nightly-only 1.0.3 and 1.0.4) gets the rules of the
older one, and a version newer than the newest listed one is read with the newest rules, after one warning.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .errors import HostTraceError
from .names import CORE_ATTRS, KERNEL_ATTRS, STRIDES, TENSOR_KEYS

logger = logging.getLogger(__name__)

Version = Tuple[int, int, int]


@dataclass(frozen=True)
class FormatChange:
    """A released PyTorch version that changed the node format, and the keys every node has from then on."""

    version: Version
    pytorch: str
    summary: str
    attrs: Tuple[str, ...] = ()
    tensor_keys: Tuple[str, ...] = ()


# Oldest first. Add an entry only when a PyTorch release changes the node format. Every change so far only added keys,
# and the table can express only that: a release that renames or removes a key needs a per-version rule here.
FORMAT_CHANGES = (
    FormatChange(
        (1, 0, 2),
        "2.2",
        "the current layout: ctrl_deps, an attrs list, inputs and outputs as dictionaries",
        attrs=CORE_ATTRS,
        tensor_keys=TENSOR_KEYS,
    ),
    FormatChange((1, 1, 0), "2.4", "kernel attributes", attrs=KERNEL_ATTRS),
    FormatChange((1, 1, 1), "2.5", "strides in inputs and outputs", tensor_keys=(STRIDES,)),
)
MIN_SUPPORTED = FORMAT_CHANGES[0].version
NEWEST_KNOWN = FORMAT_CHANGES[-1].version

_SCHEMA = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:-chakra\.(\d+)\.(\d+)\.(\d+))?")


@dataclass(frozen=True)
class Schema:
    """A parsed schema string. text is kept verbatim: it is written unchanged into linked traces and .et files."""

    text: str
    format: Version
    suffix: Optional[Version]


def _dotted(version: Version) -> str:
    return ".".join(map(str, version))


def parse_schema(text: str, source: str = "") -> Schema:
    """Split a schema string into its format version and Chakra suffix. Raise HostTraceError if it has neither."""
    match = _SCHEMA.fullmatch(str(text))
    if not match:
        raise HostTraceError(
            f"{source}: unrecognised schema {text!r}; a host trace schema looks like '1.1.1-chakra.0.0.4'"
        )
    numbers = [int(g) if g is not None else None for g in match.groups()]
    suffix = tuple(numbers[3:]) if numbers[3] is not None else None
    return Schema(str(text), (numbers[0], numbers[1], numbers[2]), suffix)  # type: ignore[arg-type]


def check_supported(schema: Schema, source: str = "") -> None:
    """Reject formats older than the first supported one; warn once about formats newer than the newest known one."""
    if schema.format < MIN_SUPPORTED:
        first = FORMAT_CHANGES[0]
        raise HostTraceError(
            f"{source}: schema {schema.text} is older than {_dotted(first.version)}, the first supported format "
            f"(PyTorch {first.pytorch}). PyTorch 2.0 and 2.1 wrote an older layout that is not supported: collect "
            f"the trace with PyTorch {first.pytorch} or later."
        )
    if schema.format > NEWEST_KNOWN:
        logger.warning(
            f"{source}: schema {schema.text} is newer than the newest known format {_dotted(NEWEST_KNOWN)}; "
            "reading it with the rules of the newest known format"
        )


def expected_keys(schema: Schema) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return the attributes and the inputs/outputs keys every node of this format version has."""
    attrs: Tuple[str, ...] = ()
    tensor_keys: Tuple[str, ...] = ()
    for change in FORMAT_CHANGES:
        if change.version <= schema.format:
            attrs += change.attrs
            tensor_keys += change.tensor_keys
    return attrs, tensor_keys
