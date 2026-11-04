"""
Chakra trace linter.

Copyright (C) Keysight Technologies, Inc - All Rights Reserved.
"""

import argparse
import ast
import fnmatch
import html
import json
import re
import string
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import jsonschema
from chakra.schema.protobuf.et_def_pb2 import (
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    GlobalMetadata,
    Node,
    NodeType,
)
from chakra.src.third_party.utils.protolib import decodeMessage, openFileRd

# Share of zero-duration nodes above which timing is absent for a trace
ZERO_WARN_RATIO = 0.5

# Share above which timing is partial; healthy samples stay below 5%
ZERO_PARTIAL_RATIO = 0.05

# Node types whose durations must not all be zero
CHECKED_TYPES = {
    COMP_NODE: "compute",
    COMM_COLL_NODE: "collective",
    COMM_SEND_NODE: "send",
    COMM_RECV_NODE: "receive",
}

# Stream that host (CPU) operators report; inferred from sample traces
DEFAULT_STREAM = 0

# Attributes only PyTorch-derived traces carry; synthetic traces lack them
PYTORCH_ATTRS = ("rf_id", "tid")

# Attributes the PyTorch converter writes on every CPU node it converts
# Older converters wrote only is_cpu_op and tid on device nodes
CONVERTER_ATTRS = frozenset(
    {
        "rf_id",
        "fw_parent",
        "seq_id",
        "scope",
        "tid",
        "fw_tid",
        "op_schema",
        "is_cpu_op",
    }
)

# Value field each known node attribute must use; readers assume it
ATTR_KINDS = {
    **dict.fromkeys(("rf_id", "fw_parent", "seq_id", "scope", "tid", "fw_tid"), "int64_val"),
    **dict.fromkeys(("stream", "comm_type", "comm_size"), "int64_val"),
    **dict.fromkeys(("op_schema", "pg_name"), "string_val"),
    "is_cpu_op": "bool_val",
}

# RecordScope values 0 to 9; NUM_SCOPES (10) is a sentinel, not a scope
MAX_SCOPE = 9

# Thread root nodes the observer adds; they carry rf_id 0 by design
ROOT_NAME_PREFIX = "[pytorch|"

# Global metadata attributes the PyTorch converter writes
METADATA_ATTRS = ("schema", "pid", "time", "start_ts", "finish_ts")

# Attributes every collective node must carry for replay
COLL_ATTRS = ("comm_type", "comm_size", "pg_name")

# Point-to-point nodes must declare a nonzero transfer size and both peers
P2P_TYPES = (COMM_SEND_NODE, COMM_RECV_NODE)
P2P_ATTRS = ("comm_size", "comm_src", "comm_dst")
COMM_TYPES = (COMM_COLL_NODE, *P2P_TYPES)

# Rule ID per missing communication attribute
MISSING_ATTR_RULES = {
    "comm_type": "COM001",
    "comm_size": "COM002",
    "pg_name": "COM003",
    "comm_src": "COM004",
    "comm_dst": "COM005",
}


# Consumers a trace may serve; each reports its own usability level
REPLAY, SIMULATOR, ANALYSIS = "replay", "simulator", "analysis"
CONSUMERS = (REPLAY, SIMULATOR, ANALYSIS)
CONSUMER_LABELS = [c.capitalize() for c in CONSUMERS]

# LIMITED: known gap lowers fidelity; REVIEW: linter cannot decide
READY, LIMITED, REVIEW, NOT_SUITABLE = "READY", "LIMITED", "REVIEW", "NOT SUITABLE"
SEVERITY = {READY: 0, LIMITED: 1, REVIEW: 2, NOT_SUITABLE: 3}

# Impact of a gap per consumer; consumers absent are unaffected
NO_SCHEMA = {REPLAY: NOT_SUITABLE}  # replay calls PyTorch via op_schema
PART_SCHEMA = {REPLAY: REVIEW}  # custom operators may need other handling
NO_CPU_OPS = {REPLAY: NOT_SUITABLE}  # replay runs CPU operators; none exist
UNKNOWN_CPU_OPS = {REPLAY: REVIEW}  # synthetic trace; origin unknown
NO_COST = {SIMULATOR: NOT_SUITABLE}  # no I/O and no duration to model
NO_TIMING = {SIMULATOR: REVIEW, ANALYSIS: NOT_SUITABLE}  # replay ignores timing
PART_TIMING = {ANALYSIS: LIMITED}
PART_LINK = {SIMULATOR: LIMITED, ANALYSIS: LIMITED}
COMM_GAP = {REPLAY: NOT_SUITABLE, SIMULATOR: NOT_SUITABLE, ANALYSIS: REVIEW}
UNVERIFIED = {ANALYSIS: REVIEW}
IO_MALFORMED = {REPLAY: NOT_SUITABLE}  # replay cannot build the arguments
IO_UNVERIFIED = {REPLAY: REVIEW}  # recorded I/O may not fit op_schema
META_GAP = {ANALYSIS: LIMITED}  # metadata describes the capture only
ATTR_MISSING = {REPLAY: REVIEW, ANALYSIS: REVIEW}  # readers see defaults
ATTR_TYPE = {REPLAY: REVIEW, SIMULATOR: REVIEW, ANALYSIS: REVIEW}
ATTR_VALUE = {REPLAY: REVIEW, ANALYSIS: REVIEW}  # sentinel or invalid value

# Evidence groups with levels ordered best to worst
TIMING, KINETO, COMM, OP_SCHEMA = "TIMING", "KINETO", "COMM", "OP_SCHEMA"
IO, META, ATTR = "IO", "META", "ATTR"
EVIDENCE = {
    TIMING: ("COMPLETE", "PARTIAL", "ABSENT"),
    KINETO: ("LINKED", "PARTIAL", "UNLINKED"),
    COMM: ("COMPLETE", "INCOMPLETE"),
    OP_SCHEMA: ("FULL", "PARTIAL", "NO CPU OPS", "NONE"),
    IO: ("CONSISTENT", "UNVERIFIED", "MALFORMED"),
    META: ("COMPLETE", "INCOMPLETE"),
    ATTR: ("COMPLETE", "INCOMPLETE"),
}

# Argument list of an op_schema, such as "aten::add.Tensor(...) -> Tensor"
SCHEMA_ARGS = re.compile(r"^[^(]*\((.*)\)\s*->")

SKIPPED, NOT_APPLICABLE, NO_CPU = "SKIPPED", "N/A", "NO CPU OPS"

# Integrity checks; any failure makes the trace fail integrity
DECODE, CONTENT, SCHEMA, DAG = "DECODE", "CONTENT", "SCHEMA", "DAG"
PASS, FAIL = "PASS", "FAIL"

# Node IDs kept per defect so engineers can inspect the trace
SAMPLE_LIMIT = 3

# Rule file shipped beside the linter, and the type of each entry field
DEFAULT_RULES = Path(__file__).with_name("rules.toml")
RULES_SCHEMA = Path(__file__).with_name("rules.schema.json")
RULE_CODE = re.compile(r"[A-Z]{2,3}[0-9]{3}")
RULE_FIELDS = {
    "id": str,
    "name": str,
    "description": str,
    "enabled": bool,
    "message": str,
}

# Placeholders each rule's message template may name; the check supplies them
COUNT, COUNT_TOTAL = frozenset({"count"}), frozenset({"count", "total"})
RULE_PLACEHOLDERS: dict[str, frozenset[str]] = {
    "INT001": frozenset({"error"}),
    "INT002": frozenset(),
    "INT003": frozenset({"type", "count", "version"}),
    "DAG001": COUNT,
    "DAG002": COUNT,
    "DAG003": COUNT,
    "DAG004": COUNT,
    "DAG005": frozenset({"count", "field"}),
    "TIM001": COUNT_TOTAL,
    "TIM002": COUNT_TOTAL,
    "TIM003": COUNT_TOTAL,
    "KIN001": frozenset(),
    "KIN002": frozenset(),
    "KIN003": COUNT,
    "KIN004": COUNT,
    "COM001": COUNT,
    "COM002": COUNT,
    "COM003": COUNT,
    "COM004": COUNT,
    "COM005": COUNT,
    "COM006": COUNT,
    "COM007": frozenset({"count", "stream"}),
    "OPS001": frozenset({"count", "total", "family"}),
    "OPS002": frozenset({"count", "total", "family"}),
    "OPS003": frozenset({"total", "schema"}),
    "OPS004": COUNT,
    "IO001": COUNT_TOTAL,
    "IO002": COUNT_TOTAL,
    "IO003": COUNT_TOTAL,
    "MET001": frozenset({"count", "names"}),
    "MET002": frozenset({"start", "finish"}),
    "ATT001": frozenset({"count", "name"}),
    "ATT002": COUNT,
    "ATT003": COUNT,
    "ATT004": COUNT,
    "ATT005": COUNT,
}

# Families listed in a message; the rest fold into "N more"
FAMILY_LIST_LIMIT = 10

# Kernel name noise: leading return type, anonymous namespace, arguments
# The name stops at the first template, argument, tag or annotation marker
KERNEL_BASE = re.compile(r"^(?:void\s+)?(?:\(anonymous namespace\)::)?((?:[^<(#:;\[]|::)*)")

# Longest family label shown; longer names truncate with an ellipsis
FAMILY_LABEL_LIMIT = 48

# Message template per rule ID, as loaded from the rule file
Messages = dict[str, str]

# Unreadable or empty traces cannot be graded, so these rules stay on
FATAL_RULES = ("INT001", "INT002")

# Next step per failed integrity check or (evidence group, level)
REMEDIATION: dict[str | tuple[str, str], str] = {
    DECODE: "Confirm the file is a Chakra `.et` trace and is not truncated.",
    CONTENT: "Re-capture the run; the trace holds no nodes.",
    SCHEMA: "Align the `mlc-chakra` schema version of the trace producer and "
    "consumers; the trace uses node types the installed schema lacks.",
    DAG: "Inspect the sample node IDs in the `DAG` row and re-run the Chakra " "converter on the source trace.",
    (TIMING, "PARTIAL"): "Re-capture with the PyTorch profiler enabled so " "that more nodes record durations.",
    (TIMING, "ABSENT"): "Re-capture with the PyTorch profiler enabled; " "analysis needs node durations.",
    (KINETO, "PARTIAL"): "Re-run `chakra_trace_link` and inspect the device " "nodes listed in the `KINETO` row.",
    (KINETO, "UNLINKED"): "Link the host trace to the Kineto device trace "
    "with `chakra_trace_link` before conversion.",
    (COMM, "INCOMPLETE"): "Supply the communication attributes listed in the "
    "`COMM` row; replay and simulation need them to map peers and groups.",
    (OP_SCHEMA, "PARTIAL"): "Confirm the replay tool handles the CPU operators "
    "without `op_schema` listed in the `OP_SCHEMA` row; add annotation names that "
    "replay skips to `exempt` in `rules.toml`.",
    (OP_SCHEMA, "NONE"): "Obtain a trace that records `op_schema` on CPU "
    "operators; replay cannot map any operator to a PyTorch call.",
    (IO, "MALFORMED"): "Re-run the Chakra converter; CPU operators in the "
    "`IO` row carry unreadable or misaligned input and output fields.",
    (META, "INCOMPLETE"): "Re-run the Chakra converter; the global metadata "
    "lacks attributes or holds inconsistent timestamps.",
    (ATTR, "INCOMPLETE"): "Re-run the Chakra converter; nodes lack attributes "
    "or carry them with a type that readers do not expect.",
    (IO, "UNVERIFIED"): "Compare the CPU operators in the `IO` row with "
    "their `op_schema`; replay may not build their arguments.",
}
REMEDIATION[(OP_SCHEMA, NO_CPU)] = (
    "Capture a host execution trace; replay runs CPU operators and the trace " "holds none."
)


@dataclass
class Issue:
    """An integrity defect or warning with a rule ID and sample node IDs."""

    rule: str
    message: str
    ids: list[int] = field(default_factory=list)

    def text(self) -> str:
        """Return the message with its sample node IDs."""
        return sample_text(self.message, self.ids)


@dataclass
class Gap:
    """A capability gap and the consumers it affects."""

    message: str
    impact: dict[str, str]
    node_type: int | None = None
    group: str | None = None  # evidence group the gap lowers, if any
    level: str = ""  # level the gap sets for its group
    ids: list[int] = field(default_factory=list)  # sample affected node IDs
    fix: str = ""  # next step; defaults to the group's remediation
    rule: str = ""  # stable rule ID for CLI output

    def remediation(self) -> str:
        """Return the next step that closes this gap."""
        if self.fix or self.group is None:
            return self.fix
        return REMEDIATION.get((self.group, self.level), "")


@dataclass
class TraceResult:
    """Outcome of linting one trace."""

    problems: list[str] = field(default_factory=list)
    total: int = 0
    elapsed: float = 0.0
    type_counts: Counter = field(default_factory=Counter)
    zero_by_type: Counter = field(default_factory=Counter)
    gaps: list[Gap] = field(default_factory=list)
    notes: list[Issue] = field(default_factory=list)  # informational only
    schema_problems: dict[int, Issue] = field(default_factory=dict)
    dag_problems: list[Issue] | None = None  # None when not checked
    dag_warnings: list[Issue] = field(default_factory=list)
    readable: bool = True
    synthetic: bool = False  # generated, not captured; no device merge
    pytorch_nodes: int = 0  # nodes carrying PyTorch-only attributes
    replay_ops: int = 0  # replay units; the schema and IO checks cover them

    def invalid(self) -> bool:
        """Return True on an integrity (tier 0) defect; gaps never invalidate a trace."""
        return bool(self.problems or self.dag_problems or self.schema_problems)

    def integrity(self) -> str:
        """Return PASS when every integrity check passes, else FAIL."""
        return FAIL if self.invalid() else PASS

    def failed_checks(self) -> list[str]:
        """Name the integrity checks that fail."""
        checks = [DECODE if not self.readable else CONTENT] if self.problems else []
        checks += [SCHEMA] if self.schema_problems else []
        return checks + ([DAG] if self.dag_problems else [])

    def next_steps(self) -> list[str]:
        """Return distinct next steps, integrity first, then by gap severity."""
        steps = [REMEDIATION[check] for check in self.failed_checks()]
        gaps = sorted(self.gaps, key=lambda g: -max(map(SEVERITY.__getitem__, g.impact.values())))
        steps += [g.remediation() for g in gaps]
        return list(dict.fromkeys(s for s in steps if s))

    def evidence(self, group: str) -> str:
        """Return the level of an evidence group: the worst gap sets it."""
        if self.synthetic and group in (TIMING, KINETO):
            return SKIPPED
        if group == COMM and not any(t in self.type_counts for t in COMM_TYPES):
            return NOT_APPLICABLE
        if group == IO and not self.replay_ops:
            return NOT_APPLICABLE
        order = EVIDENCE[group]
        levels = [g.level for g in self.gaps if g.group == group]
        return max(levels, key=order.index, default=order[0])

    def status(self, consumer: str) -> tuple[str, list[str]]:
        """Return a consumer's usability level and the reasons for it."""
        failed = ", ".join(self.failed_checks())
        hits = [(NOT_SUITABLE, f"integrity defect: {failed}")] if failed else []
        hits += [(g.impact[consumer], prefix(g)) for g in self.gaps if consumer in g.impact]
        if not hits:
            return READY, []
        worst = max((status for status, _ in hits), key=SEVERITY.__getitem__)
        return worst, [message for status, message in hits if status == worst]

    def causes(self, consumer: str) -> list[str]:
        """Return the distinct rule IDs that set a consumer's level."""
        level = self.status(consumer)[0]
        if level == READY:
            return []
        rules = []
        if self.problems:
            rules.append("INT001" if not self.readable else "INT002")
        rules += [i.rule for i in self.schema_problems.values()]
        rules += [i.rule for i in self.dag_problems or []]
        # Integrity defects set NOT SUITABLE, so gaps only add at that level
        rules = rules if level == NOT_SUITABLE else []
        rules += [g.rule for g in self.gaps if g.impact.get(consumer) == level]
        return list(dict.fromkeys(rules))


def lint_trace(  # noqa: C901
    path: Path,
    disabled: frozenset[str] = frozenset(),
    messages: Messages | None = None,
    exempt: Sequence[str] | None = None,
) -> TraceResult:
    """
    Scan one trace; integrity defects invalidate it, gaps qualify its uses.

    The trace origin (PyTorch capture or synthetic) is inferred from content.
    Findings of rules in `disabled` are dropped before grading.
    `messages` holds the report text per rule; the shipped file by default.
    `exempt` lists name patterns of childless CPU annotations needing no schema.
    """
    messages = messages or default_messages()
    skip = exempt_matcher(default_exempt() if exempt is None else exempt)
    result = TraceResult()
    try:
        handle = openFileRd(str(path))
        metadata = GlobalMetadata()
        decodeMessage(handle, metadata)
    except Exception as exc:  # noqa: BLE001 - any decode error is a defect
        result.problems.append(render(messages, "INT001", error=exc))
        result.readable = False
        return result

    zero = duplicates = zero_size = dup_edges = 0
    zero_by_type = result.zero_by_type
    samples: dict[object, list[int]] = {}
    missing_attrs: Counter = Counter()
    p2p_missing: dict[int, Counter] = {t: Counter() for t in P2P_TYPES}
    p2p_zero: Counter = Counter()
    deps: dict[int, tuple[int, ...]] = {}
    ctrl: dict[int, tuple[int, ...]] = {}
    data: dict[int, tuple[int, ...]] = {}
    streams: Counter = Counter()
    timed_streams = coll_streamed = coll_default = pytorch_nodes = 0
    comp_idle = gpu_comp = gpu_no_schema = comp_schema = 0
    io_hits: Counter = Counter()  # CPU operator I/O defects, keyed by rule
    cpu_ids: list[int] = []  # CPU compute nodes, in file order
    cpu_parents: set[int] = set()  # CPU nodes that parent a CPU node
    no_schema: dict[int, str] = {}  # CPU nodes lacking op_schema, by name
    schema_ids: set[int] = set()  # CPU nodes with an op_schema
    cpu_io: dict[int, str] = {}  # CPU nodes breaking an I/O rule
    kernel_families = FamilyTally()  # GPU kernels lacking op_schema
    gpu_parents: dict[int, tuple[int, ...]] = {}
    attr_missing: Counter = Counter()  # converter attributes absent, by name
    attr_bad = 0  # nodes with a wrongly typed or repeated attribute
    value_hits: Counter = Counter()  # ATT003 to ATT005 defects, by rule
    node = Node()
    start = time.perf_counter()
    while decodeMessage(handle, node):
        result.type_counts[node.type] += 1
        add_sample(samples, ("type", node.type), node.id)
        if node.duration_micros == 0:
            zero += 1
            zero_by_type[node.type] += 1
            add_sample(samples, "zero", node.id)
            add_sample(samples, ("zero", node.type), node.id)
        if node.type not in NodeType.DESCRIPTOR.values_by_number:
            add_sample(samples, node.type, node.id)
        if node.id in deps:
            duplicates += 1
            add_sample(samples, "duplicate", node.id)
        ctrl[node.id] = tuple(node.ctrl_deps)
        data[node.id] = tuple(node.data_deps)
        deps[node.id] = (*ctrl[node.id], *data[node.id])
        # Same ID in both fields is valid; repeats within one field are not
        if any(len(set(field)) != len(field) for field in (node.ctrl_deps, node.data_deps)):
            dup_edges += 1
            add_sample(samples, "dup_edge", node.id)
        attrs = {a.name: a for a in node.attr}
        pytorch_nodes += any(name in attrs for name in PYTORCH_ATTRS)
        cpu_attr = attrs.get("is_cpu_op")
        required = CONVERTER_ATTRS if cpu_attr and attr_value(cpu_attr) else ()
        for name in {"is_cpu_op", *required} - attrs.keys():
            attr_missing[name] += 1
            add_sample(samples, ("attr", name), node.id)
        bad_kind = len(attrs) != len(node.attr) or any(
            ATTR_KINDS.get(name, kind) != kind for name, kind in ((n, a.WhichOneof("value")) for n, a in attrs.items())
        )
        if bad_kind:
            attr_bad += 1
            add_sample(samples, "attr_bad", node.id)
        for rule in attr_value_rules(node, attrs):
            value_hits[rule] += 1
            add_sample(samples, rule, node.id)
        stream_attr = attrs.get("stream")
        if stream_attr is None:
            add_sample(samples, "no_stream", node.id)
        else:
            stream = stream_attr.int64_val
            streams[stream] += 1
            timed_streams += node.duration_micros > 0
            if node.duration_micros == 0:
                add_sample(samples, "stream_zero", node.id)
            if stream != DEFAULT_STREAM:
                gpu_parents[node.id] = tuple(node.ctrl_deps)
            if node.type == COMM_COLL_NODE:
                coll_streamed += 1
                coll_default += stream == DEFAULT_STREAM
                if stream == DEFAULT_STREAM:
                    add_sample(samples, "coll_default", node.id)
        if node.type == COMP_NODE:
            cpu_flag = attrs.get("is_cpu_op")
            schema = attrs.get("op_schema")
            op_text = attr_value(schema) if schema else None
            comp_schema += bool(op_text)
            if cpu_flag is not None and attr_value(cpu_flag):
                # Replay runs CPU operators; leaf status is known after the scan
                cpu_ids.append(node.id)
                cpu_parents.update(node.ctrl_deps)
                if op_text:
                    schema_ids.add(node.id)
                else:
                    no_schema[node.id] = node.name
                io_rule = check_io(node, op_text if isinstance(op_text, str) else "")
                if io_rule:
                    cpu_io[node.id] = io_rule
            elif cpu_flag is not None:
                # GPU kernels need no schema; the notes list them by family
                gpu_comp += 1
                if not op_text:
                    gpu_no_schema += 1
                    kernel_families.add(node.name, node.id)
                    add_sample(samples, "kernel_schema", node.id)
            idle = not (node.inputs.values or node.outputs.values)
            if idle and node.duration_micros == 0:
                comp_idle += 1
                add_sample(samples, "idle", node.id)
        if node.type == COMM_COLL_NODE:
            missing = [n for n in COLL_ATTRS if n not in attrs]
            missing_attrs.update(missing)
            size = attrs.get("comm_size")
            if size is not None and attr_value(size) == 0:
                zero_size += 1
                missing.append("zero_size")
        elif node.type in P2P_TYPES:
            missing = [n for n in P2P_ATTRS if n not in attrs]
            p2p_missing[node.type].update(missing)
            size = attrs.get("comm_size")
            if size is not None and attr_value(size) == 0:
                p2p_zero[node.type] += 1
                missing.append("zero_size")
        else:
            missing = []
        for name in missing:
            add_sample(samples, (node.type, name), node.id)
    result.elapsed = time.perf_counter() - start
    result.pytorch_nodes = pytorch_nodes
    replayable, missing_ops, op_families = 0, 0, FamilyTally()
    cpu_set = set(cpu_ids)
    under: dict[int, bool] = {}  # node has an op_schema ancestor
    for op_id in cpu_ids:
        if has_schema_ancestor(op_id, ctrl, cpu_set, schema_ids, under):
            continue  # an ancestor replays this node as part of its subtree
        if op_id in schema_ids:
            # A replay unit: the outermost CPU node that carries an op_schema
            replayable += 1
            if op_id in cpu_io:
                io_hits[cpu_io[op_id]] += 1
                add_sample(samples, cpu_io[op_id], op_id)
        elif op_id not in cpu_parents:
            # No schema and no children: replay finds nothing to call here
            name = no_schema[op_id]
            if not skip(name):
                replayable += 1
                missing_ops += 1
                op_families.add(name, op_id)
                add_sample(samples, "op_schema", op_id)
    result.replay_ops = replayable
    result.total = result.type_counts.total()

    if result.total == 0:
        result.problems.append(render(messages, "INT002"))
        return result

    # No capture markers and no device data: generated, so no merge applies
    result.synthetic = not (metadata.attr or pytorch_nodes or streams)

    gaps = result.gaps
    zero_flagged = False
    for node_type, count in result.type_counts.items():
        if node_type not in NodeType.DESCRIPTOR.values_by_number:
            result.schema_problems[node_type] = Issue(
                "INT003",
                render(
                    messages,
                    "INT003",
                    type=node_type,
                    count=count,
                    version=package_version("mlc-chakra"),
                ),
                samples.get(node_type, []),
            )
        # A trace with no durations at all reports once, at trace level
        if (
            not result.synthetic
            and node_type in CHECKED_TYPES
            and zero_by_type[node_type] == count
            and zero != result.total
        ):
            gaps.append(
                Gap(
                    render(messages, "TIM001", count=count, total=count),
                    NO_TIMING,
                    node_type,
                    TIMING,
                    "ABSENT",
                    ids=samples.get(("zero", node_type), []),
                    rule="TIM001",
                )
            )
            zero_flagged = True
    for node_type, missing in (
        (COMM_COLL_NODE, missing_attrs),
        *((t, p2p_missing[t]) for t in P2P_TYPES),
    ):
        gaps.extend(
            Gap(
                render(messages, MISSING_ATTR_RULES[name], count=n),
                COMM_GAP,
                node_type,
                COMM,
                "INCOMPLETE",
                ids=samples.get((node_type, name), []),
                rule=MISSING_ATTR_RULES[name],
            )
            for name, n in missing.items()
        )
        zeros = zero_size if node_type == COMM_COLL_NODE else p2p_zero[node_type]
        if zeros:
            gaps.append(
                Gap(
                    render(messages, "COM006", count=zeros),
                    COMM_GAP,
                    node_type,
                    COMM,
                    "INCOMPLETE",
                    ids=samples.get((node_type, "zero_size"), []),
                    rule="COM006",
                )
            )
        if node_type == COMM_COLL_NODE and coll_streamed == coll_default > 0:
            message = render(
                messages,
                "COM007",
                count=coll_streamed,
                stream=DEFAULT_STREAM,
            )
            gaps.append(
                Gap(
                    message,
                    UNVERIFIED,
                    COMM_COLL_NODE,
                    KINETO,
                    "PARTIAL",
                    ids=samples.get("coll_default", []),
                    rule="COM007",
                )
            )
    # GPU kernels without op_schema are informational; replay runs CPU operators
    for family, count, first in kernel_families.entries():
        text = render(
            messages,
            "OPS002",
            count=count,
            total=gpu_comp,
            family=family,
        )
        result.notes.append(Issue("OPS002", text, [first]))
    # A replay unit has an op_schema; its subtree needs none
    if missing_ops:
        partial = missing_ops < replayable
        # One gap per operator name, so each reads as its own finding
        for family, count, first in op_families.entries():
            message = render(
                messages,
                "OPS001",
                count=count,
                total=replayable,
                family=family,
            )
            gaps.append(
                Gap(
                    message,
                    PART_SCHEMA if partial else NO_SCHEMA,
                    COMP_NODE,
                    OP_SCHEMA,
                    "PARTIAL" if partial else "NONE",
                    ids=[first],
                    rule="OPS001",
                )
            )
    elif not replayable:
        message = render(
            messages,
            "OPS003",
            total=result.type_counts[COMP_NODE],
            schema=comp_schema,
        )
        # Sample the compute nodes, or any node when none exist
        first_type = next(iter(result.type_counts))
        ids = samples.get(("type", COMP_NODE)) or samples.get(("type", first_type))
        gaps.append(
            Gap(
                message,
                UNKNOWN_CPU_OPS if result.synthetic else NO_CPU_OPS,
                COMP_NODE,
                OP_SCHEMA,
                NO_CPU,
                ids=ids or [],
                rule="OPS003",
            )
        )
    io_gaps = (
        ("IO001", IO_MALFORMED, "MALFORMED"),
        ("IO002", IO_UNVERIFIED, "UNVERIFIED"),
        ("IO003", IO_UNVERIFIED, "UNVERIFIED"),
    )
    for rule, impact, level in io_gaps:
        if io_hits[rule]:
            gaps.append(
                Gap(
                    render(messages, rule, count=io_hits[rule], total=replayable),
                    impact,
                    COMP_NODE,
                    IO,
                    level,
                    ids=samples.get(rule, []),
                    rule=rule,
                )
            )
    if pytorch_nodes:
        meta = {a.name: attr_value(a) for a in metadata.attr}
        lacking = [n for n in METADATA_ATTRS if n not in meta]
        if lacking:
            gaps.append(
                Gap(
                    render(messages, "MET001", count=len(lacking), names=", ".join(lacking)),
                    META_GAP,
                    group=META,
                    level="INCOMPLETE",
                    rule="MET001",
                )
            )
        start, finish = meta.get("start_ts"), meta.get("finish_ts")
        if isinstance(start, int) and isinstance(finish, int) and finish < start:
            gaps.append(
                Gap(
                    render(messages, "MET002", start=start, finish=finish),
                    META_GAP,
                    group=META,
                    level="INCOMPLETE",
                    rule="MET002",
                )
            )
        # Only traces the PyTorch converter wrote carry its attributes
        for name in sorted(attr_missing):
            gaps.append(
                Gap(
                    render(messages, "ATT001", count=attr_missing[name], name=name),
                    ATTR_MISSING,
                    group=ATTR,
                    level="INCOMPLETE",
                    ids=samples.get(("attr", name), []),
                    rule="ATT001",
                )
            )
    if pytorch_nodes:
        for rule in ("ATT003", "ATT004", "ATT005"):
            if value_hits[rule]:
                gaps.append(
                    Gap(
                        render(messages, rule, count=value_hits[rule]),
                        ATTR_VALUE,
                        group=ATTR,
                        level="INCOMPLETE",
                        ids=samples.get(rule, []),
                        rule=rule,
                    )
                )
    if attr_bad:
        gaps.append(
            Gap(
                render(messages, "ATT002", count=attr_bad),
                ATTR_TYPE,
                group=ATTR,
                level="INCOMPLETE",
                ids=samples.get("attr_bad", []),
                rule="ATT002",
            )
        )
    # Modeled synthetic durations may be zero, so only captured traces count
    if comp_idle and not result.synthetic:
        gaps.append(
            Gap(
                render(messages, "OPS004", count=comp_idle),
                NO_COST,
                COMP_NODE,
                ids=samples.get("idle", []),
                fix="Re-capture so that compute nodes record tensor I/O or "
                "durations; the simulator has no cost to model.",
                rule="OPS004",
            )
        )

    if not result.synthetic:
        problems = check_kineto_link(streams, timed_streams, gpu_parents, deps, samples, messages)
        # Without stream data or durations nothing links; else only some do
        unlinked = not streams or not timed_streams
        impact = NO_TIMING if unlinked else PART_LINK
        level = "UNLINKED" if unlinked else "PARTIAL"
        gaps.extend(Gap(p.message, impact, group=KINETO, level=level, ids=p.ids, rule=p.rule) for p in problems)

    result.dag_problems, result.dag_warnings = check_graph(deps, ctrl, data, duplicates, dup_edges, samples, messages)
    zero_ids = samples.get("zero", [])
    ratio = {"count": zero, "total": result.total}
    if result.synthetic:
        if zero == result.total:
            rule = "TIM003"
            message = render(messages, rule, **ratio)
            fix = "Confirm that the trace generator models durations."
            gaps.append(Gap(message, {ANALYSIS: REVIEW}, fix=fix, ids=zero_ids, rule=rule))
    elif zero == result.total or (not zero_flagged and zero / result.total > ZERO_WARN_RATIO):
        gaps.append(
            Gap(
                render(messages, "TIM001", **ratio),
                NO_TIMING,
                group=TIMING,
                level="ABSENT",
                ids=zero_ids,
                rule="TIM001",
            )
        )
    elif not zero_flagged and zero / result.total > ZERO_PARTIAL_RATIO:
        gaps.append(
            Gap(
                render(messages, "TIM002", **ratio),
                PART_TIMING,
                group=TIMING,
                level="PARTIAL",
                ids=zero_ids,
                rule="TIM002",
            )
        )
    return drop_disabled(result, disabled)


def drop_disabled(result: TraceResult, disabled: frozenset[str]) -> TraceResult:
    """Remove findings of disabled rules so they set no level."""
    if not disabled:
        return result
    result.schema_problems = {t: issue for t, issue in result.schema_problems.items() if issue.rule not in disabled}
    if result.dag_problems is not None:
        result.dag_problems = [p for p in result.dag_problems if p.rule not in disabled]
    result.dag_warnings = [w for w in result.dag_warnings if w.rule not in disabled]
    result.gaps = [g for g in result.gaps if g.rule not in disabled]
    result.notes = [n for n in result.notes if n.rule not in disabled]
    return result


class RuleConfigError(Exception):
    """The rule configuration file is missing or invalid."""


@dataclass
class Rule:
    """One lint rule as defined in the configuration file."""

    id: str
    name: str
    description: str
    enabled: bool
    message: str
    exempt: list[str] = field(default_factory=list)  # name patterns, if any


def _validate_schema(document: dict, path: Path) -> None:
    """Validate the parsed rule file against `rules.schema.json`."""
    try:
        schema = json.loads(RULES_SCHEMA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuleConfigError(f"cannot load schema {RULES_SCHEMA}: {exc}") from exc
    try:
        jsonschema.validate(document, schema)
    except jsonschema.ValidationError as exc:
        where = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise RuleConfigError(f"{path}: {where}: {exc.message}") from exc


def load_rules(path: Path = DEFAULT_RULES) -> dict[str, Rule]:  # noqa: C901
    """Load and validate the rule file; return the rules keyed by ID."""
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError as exc:
        raise RuleConfigError(f"cannot read rules file {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RuleConfigError(f"malformed rules file {path}: {exc}") from exc
    _validate_schema(document, path)
    entries = document.get("rule", [])
    rules: dict[str, Rule] = {}
    for position, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise RuleConfigError(f"{path}: rule {position} is not a table")
        missing = [k for k in RULE_FIELDS if k not in entry]
        if missing:
            raise RuleConfigError(f"{path}: rule {position} lacks {', '.join(missing)}")
        for key, kind in RULE_FIELDS.items():
            if not isinstance(entry[key], kind):
                raise RuleConfigError(f"{path}: rule {position} field {key} must be {kind.__name__}")
        if not RULE_CODE.fullmatch(entry["id"]):
            raise RuleConfigError(f"{path}: rule ID {entry['id']} must be letters plus three digits")
        if entry["id"] in rules:
            raise RuleConfigError(f"{path}: duplicate rule ID {entry['id']}")
        if any(entry["name"] == r.name for r in rules.values()):
            raise RuleConfigError(f"{path}: duplicate rule name {entry['name']}")
        check_template(path, entry["id"], entry["message"])
        exempt = entry.get("exempt", [])
        if not isinstance(exempt, list) or not all(isinstance(x, str) for x in exempt):
            raise RuleConfigError(f"{path}: rule {position} exempt must be a list")
        rules[entry["id"]] = Rule(**{k: entry[k] for k in RULE_FIELDS}, exempt=exempt)
    if not rules:
        raise RuleConfigError(f"{path}: no rules defined")
    fatal = [r for r in FATAL_RULES if r in rules and not rules[r].enabled]
    if fatal:
        raise RuleConfigError(f"{path}: rule cannot be disabled: {', '.join(fatal)}")
    return rules


def check_template(path: Path, rule_id: str, template: str) -> None:
    """Reject a template that names a value its check does not supply."""
    allowed = RULE_PLACEHOLDERS.get(rule_id, frozenset())
    names = ", ".join(f"{{{n}}}" for n in sorted(allowed)) or "none"
    where = f"{path}: rule {rule_id} message"
    try:
        fields = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise RuleConfigError(f"{where} is malformed: {exc}") from exc
    unknown = [name for _, name, _, _ in fields if name not in (None, *allowed)]
    if unknown:
        raise RuleConfigError(f"{where} uses {{{unknown[0]}}}; allowed placeholders: {names}")
    # Trial render catches nested fields and bad format specs
    try:
        template.format(**dict.fromkeys(allowed, "0"))
    except (ValueError, KeyError, IndexError, AttributeError) as exc:
        raise RuleConfigError(f"{where} is malformed: {exc!r}") from exc


def rule_messages(rules: dict[str, Rule]) -> Messages:
    """Return the template per rule; the shipped file fills omitted rules."""
    if any(rule_id not in rules for rule_id in RULE_PLACEHOLDERS):
        rules = {**load_rules(DEFAULT_RULES), **rules}
    missing = sorted(set(RULE_PLACEHOLDERS) - set(rules))
    if missing:
        raise RuleConfigError(f"{DEFAULT_RULES}: no message for {', '.join(missing)}")
    return {rule_id: rule.message for rule_id, rule in rules.items()}


def rule_exempt(rules: dict[str, Rule]) -> list[str]:
    """Return the CPU annotation patterns of `OPS001`; shipped if absent."""
    if "OPS001" not in rules:
        rules = load_rules(DEFAULT_RULES)
    return rules["OPS001"].exempt


@cache
def default_exempt() -> list[str]:
    """Return the annotation patterns of the shipped rule file, loaded once."""
    return rule_exempt(load_rules(DEFAULT_RULES))


@cache
def default_messages() -> Messages:
    """Return the templates of the shipped rule file, loaded once."""
    return rule_messages(load_rules(DEFAULT_RULES))


class Message(str):
    """Rule text that remembers how many items it covers."""

    total: int | None
    noun: tuple[str, str]

    def __new__(
        cls,
        text: str,
        total: int | None = None,
        noun: tuple[str, str] = ("node", "nodes"),
    ) -> "Message":
        message = super().__new__(cls, text)
        message.total, message.noun = total, noun
        return message


# Rules whose count is not a node count: singular and plural noun
COUNT_NOUNS = {
    "DAG002": ("dependency", "dependencies"),
    "MET001": ("attribute", "attributes"),
}


def render(messages: Messages, rule_id: str, **values: object) -> Message:
    """Fill a rule's message template; keep the count for the CLI suffix."""
    text = {
        key: f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else str(value)
        for key, value in values.items()
    }
    count = values.get("count")
    noun = COUNT_NOUNS.get(rule_id, ("node", "nodes"))
    return Message(
        messages[rule_id].format(**text),
        count if isinstance(count, int) and not isinstance(count, bool) else None,
        noun,
    )


def expand_selectors(rules: dict[str, Rule], selectors: list[str]) -> list[str]:
    """Resolve codes, names and code prefixes (such as DAG) to rule IDs."""
    by_name = {r.name: r.id for r in rules.values()}
    resolved, unknown = [], []
    for selector in selectors:
        if selector in rules:
            resolved.append(selector)
        elif selector in by_name:
            resolved.append(by_name[selector])
        elif matches := [r for r in rules if r.rstrip("0123456789") == selector]:
            resolved.extend(matches)
        else:
            unknown.append(selector)
    if unknown:
        raise RuleConfigError(f"unknown rule ID: {', '.join(sorted(unknown))}")
    return resolved


def disabled_rules(
    rules: dict[str, Rule],
    enable: list[str] | None = None,
    disable: list[str] | None = None,
) -> frozenset[str]:
    """Return rule IDs to skip: file settings, then command-line overrides."""
    enable = expand_selectors(rules, enable or [])
    disable = expand_selectors(rules, disable or [])
    both = sorted(set(enable) & set(disable))
    if both:
        raise RuleConfigError(f"rule both enabled and disabled: {', '.join(both)}")
    fatal = [r for r in disable if r in FATAL_RULES]
    if fatal:
        raise RuleConfigError(f"rule cannot be disabled: {', '.join(fatal)}")
    off = {r.id for r in rules.values() if not r.enabled}
    return frozenset((off | set(disable)) - set(enable))


def keep_sample(ids: list[int], node_id: int) -> None:
    """Keep the first two node IDs, then the latest as the last."""
    if len(ids) < SAMPLE_LIMIT:
        ids.append(node_id)
    else:
        ids[-1] = node_id


def add_sample(samples: dict[object, list[int]], key: object, node_id: int) -> None:
    """Keep the first few node IDs that show a defect."""
    ids = samples.setdefault(key, [])
    keep_sample(ids, node_id)


def tally(hits: Iterable[tuple[int, int]]) -> tuple[int, list[int]]:
    """Sum per-node hit counts; keep the first few node IDs with a hit."""
    total, ids = 0, []
    for node_id, count in hits:
        total += count
        if count:
            keep_sample(ids, node_id)
    return total, ids


def sample_text(message: str, ids: list[int] | None) -> str:
    """Append the count and sample node IDs, as (17 nodes, 9, 41...65)."""
    count = getattr(message, "total", None)
    ids = ids or []
    parts = []
    if count:
        noun = getattr(message, "noun", ("node", "nodes"))[count != 1]
        parts.append(f"{count:,} {noun}")
    if count and count > len(ids) and len(ids) == SAMPLE_LIMIT:
        parts.append(f"{ids[0]}, {ids[1]}...{ids[2]}")
    elif ids:
        parts.append(", ".join(map(str, ids)))
    return f"{message} ({', '.join(parts)})" if parts else str(message)


def check_kineto_link(
    streams: Counter,
    timed_streams: int,
    gpu_parents: dict[int, tuple[int, ...]],
    deps: dict[int, tuple[int, ...]],
    samples: dict[object, list[int]],
    messages: Messages | None = None,
) -> list[Issue]:
    """Look for evidence that Kineto device data was linked in."""
    messages = messages or default_messages()

    def issue(rule: str, ids: list[int], **values: object) -> Issue:
        return Issue(rule, render(messages, rule, **values), ids)

    if not streams:
        return [issue("KIN001", samples.get("no_stream", []))]
    problems: list[Issue] = []
    if not timed_streams:
        problems.append(issue("KIN002", samples.get("stream_zero", [])))
    orphans, ids = tally((n, not any(p in deps for p in ps)) for n, ps in gpu_parents.items())
    if orphans:
        problems.append(issue("KIN003", ids, count=orphans))
    multi, ids = tally((n, sum(p in deps for p in ps) > 1) for n, ps in gpu_parents.items())
    if multi:
        problems.append(issue("KIN004", ids, count=multi))
    return problems


def attr_value(attr) -> object:
    """Return the populated value of a node attribute."""
    kind = attr.WhichOneof("value")
    return getattr(attr, kind) if kind else None


def attr_value_rules(node: Node, attrs: dict) -> list[str]:
    """Return the value rules (ATT003 to ATT005) that a node breaks."""

    def num(name: str) -> int | None:
        attr = attrs.get(name)
        return attr.int64_val if attr and attr.WhichOneof("value") == "int64_val" else None

    hits = []
    rf_id = num("rf_id")
    is_root = node.name.startswith(ROOT_NAME_PREFIX)
    if node.id == 0 or (rf_id == 0 and not is_root):
        hits.append("ATT003")
    scope = num("scope")
    if scope is not None and not 0 <= scope <= MAX_SCOPE:
        hits.append("ATT004")
    if num("fw_tid") and num("fw_parent") == 0:
        hits.append("ATT005")
    return hits


def parse_io(text: str) -> list | None:
    """Parse one I/O field; an empty string is an empty list, else None."""
    if not text:
        return []
    # Fields are Python literals; the C JSON parser is 3x faster when it fits
    try:
        value = json.loads(text.replace("'", '"'))
    except ValueError:
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    return value if isinstance(value, list) else None


def io_values(info) -> list | None:
    """Return the I/O values; None when a field fails or lengths differ."""
    values, shapes = parse_io(info.values), parse_io(info.shapes)
    types = parse_io(info.types)
    if values is None or shapes is None or types is None:
        return None
    return values if len(values) == len(shapes) == len(types) else None


def op_family(name: str) -> str:
    """
    Name the group of a kernel or operator: the part before any tag.

    A namespaced name groups by namespace, a plain one by its first words;
    digits fold to #, and long labels truncate.
    """
    match = KERNEL_BASE.match(name.strip())
    base = (match.group(1).strip() if match else "") or name.strip()
    parts = base.split("::")
    family = parts[0] if len(parts) > 1 else "_".join(base.split("_")[:3])
    family = re.sub(r"\d+", "#", family) or "unnamed"
    if len(family) > FAMILY_LABEL_LIMIT:
        family = family[: FAMILY_LABEL_LIMIT - 1] + "\u2026"
    return family


@dataclass
class FamilyTally:
    """Node counts per family, with the first node ID of each as an example."""

    counts: Counter = field(default_factory=Counter)
    first: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, node_id: int) -> None:
        """Count a node under its family."""
        family = op_family(name)
        self.counts[family] += 1
        self.first.setdefault(family, node_id)

    def entries(self) -> list[tuple[str, int, int]]:
        """
        Return (family, count, example node) for the largest families.

        Families beyond the limit fold into one closing "other names" entry.
        """
        top = self.counts.most_common(FAMILY_LIST_LIMIT)
        entries = [(name, count, self.first[name]) for name, count in top]
        rest = self.counts.most_common()[FAMILY_LIST_LIMIT:]
        if rest:
            label = f"{plural(len(rest), 'other name')}"
            entries.append((label, sum(c for _, c in rest), self.first[rest[0][0]]))
        return entries


def has_schema_ancestor(
    node_id: int,
    ctrl: dict[int, tuple[int, ...]],
    cpu_set: set[int],
    schema_ids: set[int],
    under: dict[int, bool],
) -> bool:
    """
    Tell whether a CPU ancestor of the node carries an op_schema.

    Follows the first CPU parent upward; `under` memoizes the answers.
    """
    path = [node_id]
    while True:
        parent = next((p for p in ctrl.get(path[-1], ()) if p in cpu_set), None)
        if parent is None:
            found = False
        elif parent in schema_ids:
            found = True
        elif parent in under:
            found = under[parent]
        else:
            path.append(parent)
            continue
        break
    under.update(dict.fromkeys(path, found))
    return found


def exempt_matcher(patterns: Sequence[str]) -> Callable[[str], bool]:
    """Return a test for names that match any shell-style pattern."""
    if not patterns:
        return lambda name: False
    regex = re.compile("|".join(fnmatch.translate(p) for p in patterns))
    return lambda name: regex.match(name) is not None


def schema_arity(op_schema: str) -> int | None:
    """Count op_schema arguments, keyword-only included; None if unparseable."""
    match = SCHEMA_ARGS.match(op_schema)
    if not match:
        return None
    depth, count, token = 0, 0, ""
    for char in match.group(1) + ",":
        depth += (char in "([") - (char in ")]")
        if char == "," and depth == 0:
            # The bare * marks keyword-only arguments, not an argument
            count += token.strip() not in ("", "*")
            token = ""
        else:
            token += char
    return count


def check_io(node: Node, op_schema: str) -> str:
    """Return the I/O rule a CPU operator breaks, or an empty string."""
    inputs, outputs = io_values(node.inputs), io_values(node.outputs)
    if inputs is None or outputs is None:
        return "IO001"
    if not op_schema:
        return ""  # OPS001 covers the node
    expected = schema_arity(op_schema)
    if not (inputs or outputs):
        # An operator without arguments may record no I/O
        return "" if expected == 0 else "IO003"
    if expected is not None and expected != len(inputs):
        return "IO002"
    return ""


def count_blocked(graph: dict[int, tuple[int, ...]]) -> tuple[int, list[int]]:
    """
    Count nodes in or behind a cycle using Kahn's algorithm.

    Returns (count, sample node IDs).
    """
    pending = {n: sum(p in graph and p != n for p in ps) for n, ps in graph.items()}
    children: dict[int, list[int]] = {}
    for node_id, parents in graph.items():
        for parent in parents:
            if parent in graph and parent != node_id:
                children.setdefault(parent, []).append(node_id)
    ready = [n for n, count in pending.items() if count == 0]
    while ready:
        for child in children.get(ready.pop(), ()):
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    return tally((n, count > 0) for n, count in pending.items())


def check_graph(
    deps: dict[int, tuple[int, ...]],
    ctrl: dict[int, tuple[int, ...]],
    data: dict[int, tuple[int, ...]],
    duplicates: int,
    dup_edges: int = 0,
    samples: dict[object, list[int]] | None = None,
    messages: Messages | None = None,
) -> tuple[list[Issue], list[Issue]]:
    """
    Report violations of the valid-DAG rule.

    Returns (problems, warnings).
    """
    samples = samples or {}
    messages = messages or default_messages()

    def issue(rule: str, ids: list[int], **values: object) -> Issue:
        return Issue(rule, render(messages, rule, **values), ids)

    problems: list[Issue] = []
    warnings: list[Issue] = []
    if duplicates:
        ids = samples.get("duplicate", [])
        problems.append(issue("DAG001", ids, count=duplicates))
    # Dependency 0 marks a root node with no parent, not a real node
    dangling, ids = tally((n, sum(bool(p) and p not in deps for p in ps)) for n, ps in deps.items())
    if dangling:
        problems.append(issue("DAG002", ids, count=dangling))
    self_refs, ids = tally((n, n in ps) for n, ps in deps.items())
    if self_refs:
        problems.append(issue("DAG003", ids, count=self_refs))
    if dup_edges:
        ids = samples.get("dup_edge", [])
        problems.append(issue("DAG004", ids, count=dup_edges))

    # Each field forms its own graph; a merged cycle is not a valid defect
    for name, graph in (("ctrl_deps", ctrl), ("data_deps", data)):
        stuck, ids = count_blocked(graph)
        if stuck:
            problems.append(issue("DAG005", ids, count=stuck, field=name))
    return problems, warnings


def type_name(node_type: int) -> str:
    """Name a node type, tolerating values missing from the schema."""
    try:
        return NodeType.Name(node_type)
    except ValueError:
        return f"UNKNOWN({node_type})"


def package_version(name: str) -> str:
    """Return an installed package version, or 'not installed'."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def collect_traces(paths: list[Path], recursive: bool = False) -> list[Path]:
    """Expand directories into their .et files, at any depth when recursive."""
    pattern = "**/*.et" if recursive else "*.et"
    traces: list[Path] = []
    for path in paths:
        traces.extend(sorted(path.glob(pattern)) if path.is_dir() else [path])
    return traces


# Levels that need no reader attention are plain; the rest are bold
PLAIN_LEVELS = {PASS, READY, NOT_APPLICABLE}


def prefix(gap: Gap) -> str:
    """Name the node type a gap concerns, when it has one."""
    text = sample_text(gap.message, gap.ids)
    if gap.node_type is None:
        return text
    return f"{type_name(gap.node_type)}: {text}"


def cell(text: str) -> str:
    """Escape text for a markdown table cell."""
    return text.replace("|", "\\|").replace("\n", "<br>")


def level_cell(level: str) -> str:
    """Bold any level that needs reader attention."""
    if not level or level in PLAIN_LEVELS:
        return level
    return f"**{level}**"


def table(headers: list[str], body: list[list[str]]) -> list[str]:
    """Render a markdown table."""
    lines = [f"| {' | '.join(headers)} |", f"|{'|'.join('---' for _ in headers)}|"]
    lines += [f"| {' | '.join(cell(c) for c in row)} |" for row in body]
    return lines


def trace_anchors(results: list[tuple[Path, TraceResult]]) -> list[str]:
    """Return the heading anchor of each trace section, in order."""
    seen: Counter = Counter()
    anchors = []
    for trace, _ in results:
        slug = re.sub(r"[^\w\- ]", "", trace.name.lower()).replace(" ", "-")
        anchors.append(f"{slug}-{seen[slug]}" if seen[slug] else slug)
        seen[slug] += 1
    return anchors


def primary_reason(result: TraceResult) -> str:
    """Name the reason behind the worst use level, with a count of the rest."""
    statuses = [result.status(c) for c in CONSUMERS]
    worst = max(statuses, key=lambda s: SEVERITY[s[0]])
    if not worst[1]:
        return "-"
    others = {r for _, reasons in statuses for r in reasons} - {worst[1][0]}
    return worst[1][0] + (f" (+{len(others)} more)" if others else "")


def summary_lines(results: list[tuple[Path, TraceResult]]) -> list[str]:
    """Render one overview row per trace with a level per consumer."""
    body = []
    for (trace, result), anchor in zip(results, trace_anchors(results), strict=False):
        levels = [result.status(c)[0] for c in CONSUMERS]
        body.append(
            [
                f"[`{trace.name}`](#{anchor})",
                level_cell(result.integrity()),
                f"{result.total:,}",
            ]
            + [level_cell(level) for level in levels]
            + [primary_reason(result)]
        )
    headers = ["Trace", "Integrity", "Nodes", *CONSUMER_LABELS, "Primary reason"]
    return [
        "## Summary",
        "",
        *table(headers, body),
        "",
        "Level definitions appear in [Levels](#levels).",
        "",
    ]


# Plain-language overview shown before the summary
OVERVIEW = (
    "## Overview",
    "",
    "This report checks Chakra execution traces and states whether ",
    "users can rely on them for their intended work.",
    "",
    "- **Integrity:** the linter confirms each file is readable, not empty,",
    "  matches the Chakra schema, and forms a valid dependency graph.",
    "  A trace that fails any of these is unusable for every purpose.",
    "- **Usability:** the linter grades each trace for three uses:",
    "  - `Replay`: re-run the recorded operations on hardware.",
    "  - `Simulator`: predict execution time from the graph.",
    "  - `Analysis`: study the timing and structure of the captured run.",
    "- **How to read it:** the Summary lists one row per trace. Each trace",
    "  section repeats the CLI output: node ID, severity, message, and rule.",
    "  Every rule ID links to its description in [Rules](#rules).",
    "",
)

# Level meanings shown before the rule descriptions
LEVEL_LEGEND = (
    "## Levels",
    "",
    "- **Integrity**",
    "  - `PASS`: every integrity check passes; the uses may still have gaps.",
    "  - `FAIL`: integrity defect; every use is `NOT SUITABLE`.",
    "- **Usability** (the worst applicable gap sets the level)",
    "  - `READY`: fully usable; no gap affects the use.",
    "  - `LIMITED`: a known gap lowers fidelity; the use still works.",
    "  - `REVIEW`: the linter cannot decide; a reviewer decides.",
    "  - `NOT SUITABLE`: a gap rules out the use.",
    "- **Severity** (trace rows)",
    "  - `error`: integrity defect, or a gap that makes any use `NOT SUITABLE`.",
    "  - `warning`: any other gap.",
    "  - `pass`: no gap affects the use.",
    f"- **Sample IDs:** up to {SAMPLE_LIMIT} affected node IDs to inspect.",
    "",
)

# Rule text for the suitability rows, which rules.toml does not define
SUITABILITY_RULES = {
    "SUI001": "Suitability (suitability-replay) for Replay: re-executing the "
    "recorded operators on hardware. The level is the worst gap that affects "
    "Replay; the causes name the rules at that level.",
    "SUI002": "Suitability (suitability-simulator) for Simulator: modeling "
    "execution time from the graph. The level is the worst gap that affects "
    "Simulator; the causes name the rules at that level.",
    "SUI003": "Suitability (suitability-analysis) for Analysis: studying the "
    "timing and structure of a captured run. The level is the worst gap that "
    "affects Analysis; the causes name the rules at that level.",
}


def rule_descriptions(rules: dict[str, Rule], disabled: frozenset[str]) -> list[str]:
    """Render one heading per rule; each ID link in a trace block lands here."""
    lines = ["## Rules", ""]
    for rule in rules.values():
        state = "disabled" if rule.id in disabled else "enabled"
        lines += [f"#### {rule.id}", "", f"- **State:** {state}"]
        lines += [f"- {rule.description}", ""]
    for rule_id, text in SUITABILITY_RULES.items():
        lines += [f"#### {rule_id}", "", f"- {text}", ""]
    return lines


def block_rows(result: TraceResult) -> list[tuple[str, str, str]]:
    """Return the CLI rows of one trace: findings, then suitability."""
    rows = stylish_rows(result)
    for consumer, name in zip(CONSUMERS, CONSUMER_LABELS, strict=False):
        level = result.status(consumer)[0]
        text = suitability_text(level, name, result.causes(consumer))
        rows.append((SUITABILITY_IDS[consumer], LEVEL_SEVERITY[level][0], text))
    return rows


def markdown_block(trace: Path, result: TraceResult) -> list[str]:
    """Render one trace as a table of CLI rows with linked rule IDs."""
    body = []
    for rule, severity, message in block_rows(result):
        label = f"**{severity}**" if severity == ERROR else severity
        link = f"[`{rule}`](#{rule})" if rule else ""
        body.append([link, label, message])
    return [*table(["Code", "State", "Description"], body), ""]


def html_block(trace: Path, result: TraceResult) -> list[str]:
    """Render one trace as a preformatted block laid out like the CLI."""
    css = {ERROR: "err", WARNING: "warn"}
    indent = " " * (2 + CODE_WIDTH + 2 + SEVERITY_WIDTH + 2)
    lines = [f'<span class="path">{html.escape(str(trace))}</span>']
    for rule, severity, message in block_rows(result):
        head, *more = message.split("\n")
        link = f'<a href="#{rule}">{rule}</a>' if rule else ""
        pad = " " * (CODE_WIDTH - len(rule))
        lines.append(
            f"  {link}{pad}  "
            f'<span class="{css.get(severity, "ok")}">'
            f"{severity.ljust(SEVERITY_WIDTH)}</span>  {html.escape(head)}"
        )
        lines += [indent + html.escape(line) for line in more]
    return ['<pre class="cli">' + "\n".join(lines) + "</pre>", ""]


def render_report(
    results: list[tuple[Path, TraceResult]],
    rules: dict[str, Rule],
    disabled: frozenset[str] = frozenset(),
    as_html: bool = False,
) -> str:
    """Render the markdown report; as_html swaps in CLI-style trace blocks."""
    skipped = ", ".join(f"`{r}`" for r in sorted(disabled)) or "none"
    block = html_block if as_html else markdown_block
    lines = [
        "# Chakra Linter Report",
        "",
        f"- **Date:** {date.today().isoformat()}",
        f"- **mlc-chakra:** {package_version('mlc-chakra')}",
        f"- **protobuf:** {package_version('protobuf')}",
        f"- **Traces:** {len(results)}",
        f"- **Disabled rules:** {skipped}",
        "",
        *OVERVIEW,
        *summary_lines(results),
        "## Traces",
        "",
    ]
    for index, (trace, result) in enumerate(results):
        # Horizontal rule separates consecutive traces
        lines += ["---", ""] if index else []
        lines += [
            f"### `{trace.name}`",
            "",
            f"- **Integrity:** {result.integrity()}",
            f"- **Nodes:** {result.total:,}",
            f"- **Scan time:** {result.elapsed:.2f}s",
            "",
        ]
        steps = result.next_steps()
        if steps:
            lines += ["#### Next steps", "", *(f"- {s}" for s in steps), ""]
        lines += [*block(trace, result), "[Back to summary](#summary)", ""]
    lines += [*LEVEL_LEGEND, *rule_descriptions(rules, disabled)]
    return "\n".join(lines)


# CLI severities, ordered worst first, as in ESLint's stylish format
ERROR, WARNING = "error", "warning"


@dataclass
class Finding:
    """One CLI finding; node_ids is empty for a trace-level finding."""

    severity: str
    message: str
    rule: str
    node_ids: list[int] = field(default_factory=list)

    def text(self) -> str:
        """Return the message with its node count and sample node IDs."""
        return sample_text(self.message, self.node_ids)


def gap_severity(gap: Gap) -> str:
    """Rate a gap an error when it rules out any use, else a warning."""
    if NOT_SUITABLE not in gap.impact.values():
        return WARNING
    return ERROR


def issue_findings(issue: Issue, severity: str, label: str = "") -> Finding:
    """Turn an issue into a finding, prefixed by its node type label."""
    message = issue.message
    if label:
        message = Message(f"{label}: {message}", message.total, message.noun)
    return Finding(severity, message, issue.rule, issue.ids)


def trace_findings(result: TraceResult) -> list[Finding]:
    """List every finding for one trace, errors first, in check order."""
    if not result.readable:
        return [Finding(ERROR, result.problems[0], "INT001")]
    if result.problems:
        return [Finding(ERROR, result.problems[0], "INT002")]
    # The message names the unknown type, so no type label is prefixed
    findings = [issue_findings(i, ERROR) for i in result.schema_problems.values()]
    findings += [issue_findings(p, ERROR) for p in result.dag_problems or []]
    findings += [issue_findings(w, WARNING) for w in result.dag_warnings]
    for gap in result.gaps:
        label = type_name(gap.node_type) if gap.node_type is not None else ""
        issue = Issue(gap.rule, gap.message, gap.ids)
        findings.append(issue_findings(issue, gap_severity(gap), label))
    return sorted(findings, key=lambda f: f.severity != ERROR)


def paint(text: str, code: str, color: bool) -> str:
    """Wrap text in an ANSI style when color is on."""
    return f"\033[{code}m{text}\033[0m" if color else text


def plural(count: int, noun: str) -> str:
    """Format a count with its noun, adding an s unless the count is one."""
    return f"{count:,} {noun}{'' if count == 1 else 's'}"


def stylish_rows(result: TraceResult) -> list[tuple[str, str, str]]:
    """Return the code, state, and description cells for one trace."""
    return [(f.rule, f.severity, f.text()) for f in trace_findings(result)]


# Severity-column label for a trace with no findings
PASS_LABEL = "pass"

# Fixed column widths keep every trace aligned while output streams
CODE_WIDTH, SEVERITY_WIDTH = 6, len(WARNING)


# ANSI style per usability level: green, yellow, or red
LEVEL_STYLES = {READY: "32", LIMITED: "33", REVIEW: "33", NOT_SUITABLE: "31"}


# Severity cell and style per usability level on the suitability rows
LEVEL_SEVERITY = {
    READY: (PASS_LABEL, "32"),
    LIMITED: (WARNING, "33"),
    REVIEW: (WARNING, "33"),
    NOT_SUITABLE: (ERROR, "31"),
}

# Stable code per consumer suitability row: SUI001 Replay, SUI002 Simulator,
# SUI003 Analysis; the code is never reused
SUITABILITY_IDS = {c: f"SUI{i:03d}" for i, c in enumerate(CONSUMERS, 1)}


def suitability_text(level: str, consumer: str, rules: list[str]) -> str:
    """Phrase a consumer's level, naming the rules that cause it."""
    if level == READY:
        return f"suitable for {consumer} use"
    verdict = {
        NOT_SUITABLE: "not suitable",
        REVIEW: "review needed",
        LIMITED: "limited",
    }[level]
    return f"{verdict} for {consumer} use due to {', '.join(rules)}"


def row_line(code: str, state: str, text: str, style: str, color: bool) -> list[str]:
    """Render one CLI row: code, state, description; extra lines align."""
    head, *more = text.split("\n")
    code_cell = paint(code.ljust(CODE_WIDTH), "2", color)
    state_cell = paint(state.ljust(SEVERITY_WIDTH), style, color)
    indent = " " * (2 + CODE_WIDTH + 2 + SEVERITY_WIDTH + 2)
    first = f"  {code_cell}  {state_cell}  {head}"
    return [first, *(f"{indent}{line}" for line in more)]


def suitability_lines(result: TraceResult, color: bool) -> list[str]:
    """Render one row per consumer with its level, causes, and ID."""
    lines = []
    for consumer, name in zip(CONSUMERS, CONSUMER_LABELS, strict=False):
        level = result.status(consumer)[0]
        label, style = LEVEL_SEVERITY[level]
        text = suitability_text(level, name, result.causes(consumer))
        lines += row_line(SUITABILITY_IDS[consumer], label, text, style, color)
    return lines


def stylish_block(trace: Path, result: TraceResult, color: bool) -> list[str]:
    """Render one trace's findings or pass line, then its suitability."""
    lines = [paint(str(trace), "4", color)]
    for code, severity, text in stylish_rows(result):
        style = "31" if severity == ERROR else "33"
        lines += row_line(code, severity, text, style, color)
    return [*lines, *suitability_lines(result, color), ""]


def stylish_summary(results: list[tuple[Path, TraceResult]], color: bool) -> str:
    """Total the problems, errors, and warnings across all traces."""
    counts = Counter(f.severity for _, r in results for f in trace_findings(r))
    total, traces = counts.total(), plural(len(results), "trace")
    if not total:
        return paint(f"No problems in {traces}", "32", color)
    summary = (
        f"\u2716 {plural(total, 'problem')} ({plural(counts[ERROR], ERROR)}, "
        f"{plural(counts[WARNING], WARNING)}) in {traces}"
    )
    return paint(summary, "1;31" if counts[ERROR] else "1;33", color)


def render_stylish(results: list[tuple[Path, TraceResult]], color: bool) -> str:
    """Render findings grouped by trace, in ESLint's stylish format."""
    lines = [line for t, r in results for line in stylish_block(t, r, color)]
    return "\n".join([*lines, stylish_summary(results, color)])


def render_json(results: list[tuple[Path, TraceResult]]) -> str:
    """Render findings per trace, following ESLint's json formatter layout."""
    entries = []
    for trace, result in results:
        findings = trace_findings(result)
        messages = [
            {
                "nodeId": f.node_ids[0] if f.node_ids else None,
                "sampleNodeIds": f.node_ids,
                "severity": f.severity,
                "message": str(f.message),
                "count": getattr(f.message, "total", None),
                "ruleId": f.rule,
            }
            for f in findings
        ]
        entries.append(
            {
                "filePath": str(trace),
                "messages": messages,
                "errorCount": sum(f.severity == ERROR for f in findings),
                "warningCount": sum(f.severity == WARNING for f in findings),
                "suitability": {c: result.status(c)[0] for c in CONSUMERS},
            }
        )
    return json.dumps(entries, indent=2)


# SARIF 2.1.0 identifiers that code-scanning dashboards validate against
SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
TOOL_NAME = "chakra-linter"


def artifact_uri(trace: Path) -> str:
    """Return a SARIF URI: relative paths as given, absolute as file URIs."""
    return trace.resolve().as_uri() if trace.is_absolute() else trace.as_posix()


def sarif_rule(rule: Rule, disabled: frozenset[str]) -> dict:
    """Describe one rule for the SARIF tool driver."""
    short = rule.description.split(". ")[0].rstrip(".") + "."
    return {
        "id": rule.id,
        "name": rule.name,
        "shortDescription": {"text": short},
        "fullDescription": {"text": rule.description},
        "defaultConfiguration": {"enabled": rule.id not in disabled},
    }


def sarif_result(finding: Finding, uri: str, index: dict[str, int]) -> dict:
    """Turn one finding into a SARIF result anchored on the trace file."""
    location: dict = {"physicalLocation": {"artifactLocation": {"uri": uri}}}
    if finding.node_ids:
        # Traces have no lines; the node ID is the logical location
        node = {"name": f"node {finding.node_ids[0]}", "kind": "element"}
        location["logicalLocations"] = [node]
    result = {
        "ruleId": finding.rule,
        "level": finding.severity,
        "message": {"text": finding.text()},
        "locations": [location],
        "properties": {"sampleNodeIds": finding.node_ids},
    }
    if finding.rule in index:
        result["ruleIndex"] = index[finding.rule]
    return result


def render_sarif(
    results: list[tuple[Path, TraceResult]],
    rules: dict[str, Rule],
    disabled: frozenset[str],
) -> str:
    """Render findings as one SARIF 2.1.0 run for code-scanning dashboards."""
    index = {rule_id: i for i, rule_id in enumerate(rules)}
    artifacts, sarif_results = [], []
    for trace, result in results:
        uri = artifact_uri(trace)
        suitability = {c: result.status(c)[0] for c in CONSUMERS}
        artifacts.append({"location": {"uri": uri}, "properties": {"suitability": suitability}})
        sarif_results += [sarif_result(f, uri, index) for f in trace_findings(result)]
    driver = {
        "name": TOOL_NAME,
        "rules": [sarif_rule(rule, disabled) for rule in rules.values()],
    }
    run = {"tool": {"driver": driver}, "artifacts": artifacts, "results": sarif_results}
    log = {"$schema": SARIF_SCHEMA, "version": SARIF_VERSION, "runs": [run]}
    return json.dumps(log, indent=2)


# CLI output formats a --cli-output file accepts; each takes the results,
# the loaded rules, and the disabled rule IDs
CLI_FORMATS = {
    "text": lambda results, rules, disabled: render_stylish(results, color=False),
    "json": lambda results, rules, disabled: render_json(results),
    "sarif": render_sarif,
}

# Stylesheet for the HTML report; follows the viewer's light or dark theme
HTML_STYLE = """
:root { color-scheme: light dark; --fg: #1f2328; --bg: #fff;
  --muted: #59636e; --line: #d1d9e0; --head: #f6f8fa; --link: #0969da;
  --err: #cf222e; --warn: #9a6700; --ok: #1a7f37; }
@media (prefers-color-scheme: dark) { :root { --fg: #f0f6fc; --bg: #0d1117;
  --muted: #9198a1; --line: #3d444d; --head: #151b23; --link: #4493f8;
  --err: #ff7b72; --warn: #d29922; --ok: #3fb950; } }
body { margin: 0 auto; max-width: 72rem; padding: 2rem 1rem;
  font: 15px/1.5 system-ui, sans-serif; color: var(--fg); background: var(--bg); }
a { color: var(--link); }
h1, h2 { border-bottom: 1px solid var(--line); padding-bottom: .3em; }
code { font: 0.9em ui-monospace, monospace; background: var(--head);
  padding: .1em .3em; border-radius: 4px; }
table { border-collapse: collapse; display: block; overflow-x: auto;
  margin: 0 0 1rem; }
th, td { border: 1px solid var(--line); padding: .35em .7em;
  text-align: left; vertical-align: top; }
th { background: var(--head); }
pre.cli { font: 13px/1.45 ui-monospace, monospace; background: var(--head);
  border: 1px solid var(--line); border-radius: 6px; padding: .8em 1em;
  overflow-x: auto; }
pre.cli .path { text-decoration: underline; }
pre.cli .err { color: var(--err); }
pre.cli .warn { color: var(--warn); }
pre.cli .ok { color: var(--ok); }
pre.cli a { color: var(--muted); }
hr { border: 0; border-top: 1px solid var(--line); margin: 2rem 0; }
"""


def render_html(report: str) -> str:
    """Convert the markdown report into a standalone HTML page."""
    try:
        import markdown
    except ImportError:
        sys.exit("HTML reports need the markdown package: pip install markdown")
    body = markdown.markdown(report, extensions=["tables", "toc"])
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Chakra Linter Report</title>\n<style>{HTML_STYLE}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )


def rule_inventory(rules: dict[str, Rule], disabled: frozenset[str]) -> str:
    """List each rule ID with its enabled state, then the rule count."""
    width = max(len(rule_id) for rule_id in rules)
    name_width = max(len(rule.name) for rule in rules.values())
    lines = [
        f"{rule_id:<{width}}  {rule.name:<{name_width}}  " f"{'disabled' if rule_id in disabled else 'enabled'}"
        for rule_id, rule in rules.items()
    ]
    off = sum(rule_id in disabled for rule_id in rules)
    lines.append(f"{plural(len(rules), 'rule')}, {off} disabled")
    return "\n".join(lines)


REPORT_FORMATS = {
    "markdown": lambda results, rules, disabled: render_report(results, rules, disabled),
    "html": lambda results, rules, disabled: render_html(render_report(results, rules, disabled, as_html=True)),
}


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description="Chakra trace linter.")
    parser.add_argument("paths", nargs="*", type=Path, help="traces or dirs")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="descend into subdirectories to find .et traces",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        metavar="FILE",
        help="write the report to FILE, or - for stdout; omit for no report",
    )
    parser.add_argument(
        "--report-format",
        choices=tuple(REPORT_FORMATS),
        default="markdown",
        help="format of the report (default: %(default)s)",
    )
    parser.add_argument(
        "--cli-color",
        choices=("auto", "always", "never"),
        default="auto",
        help="color CLI output (default: %(default)s, on for a terminal)",
    )
    parser.add_argument(
        "--cli-output",
        type=Path,
        metavar="FILE",
        help="also save the CLI output, uncolored, to FILE",
    )
    parser.add_argument(
        "--cli-format",
        choices=tuple(CLI_FORMATS),
        default="text",
        help="format of the --cli-output file (default: %(default)s)",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=DEFAULT_RULES,
        metavar="FILE",
        help="rule configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--enable-rule",
        action="append",
        default=[],
        metavar="ID",
        help="run rule ID, name or prefix (e.g. DAG) even if disabled; repeatable",
    )
    parser.add_argument(
        "--disable-rule",
        action="append",
        default=[],
        metavar="ID",
        help="skip rule ID, name or prefix (e.g. DAG); repeatable",
    )
    parser.add_argument(
        "--list-rules",
        action="store_true",
        help="list every rule with its enabled state, then exit",
    )
    args = parser.parse_args(argv)
    if not args.paths and not args.list_rules:
        parser.error("the following arguments are required: paths")

    # Fail before any scan when the rule file or overrides are unusable
    try:
        rules = load_rules(args.rules)
        messages = rule_messages(rules)
        exempt = rule_exempt(rules)
        disabled = disabled_rules(rules, args.enable_rule, args.disable_rule)
    except RuleConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.list_rules:
        print(rule_inventory(rules, disabled))
        return 0

    # The report on stdout replaces the CLI output
    show_cli = str(args.report_output) != "-"
    color = args.cli_color == "always" or (args.cli_color == "auto" and sys.stdout.isatty())
    traces = collect_traces(args.paths, args.recursive)
    # A full scan takes minutes; show progress on a terminal only
    progress = sys.stderr.isatty()
    results = []
    for index, trace in enumerate(traces, 1):
        if progress:
            sys.stderr.write(f"\r\033[KLinting {index}/{len(traces)}: {trace}")
            sys.stderr.flush()
        results.append((trace, lint_trace(trace, disabled, messages, exempt)))
        if progress:
            sys.stderr.write("\r\033[K")
        # Print each trace as soon as it is evaluated
        if show_cli:
            print(*stylish_block(trace, results[-1][1], color), sep="\n", flush=True)
    if args.cli_output:
        render = CLI_FORMATS[args.cli_format]
        args.cli_output.write_text(render(results, rules, disabled) + "\n")
    output = args.report_output
    if output:
        report = REPORT_FORMATS[args.report_format](results, rules, disabled)
        if str(output) == "-":
            print(report)
        else:
            output.write_text(report + "\n")
    if show_cli:
        print(stylish_summary(results, color))
    # Nonzero exit on any error, as linters do, so CI can gate on it
    return 1 if any(f.severity == ERROR for _, r in results for f in trace_findings(r)) else 0


if __name__ == "__main__":
    sys.exit(main())
