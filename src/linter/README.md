# Chakra Linter

## Purpose

- Measure how usable each Chakra execution trace (`.et` file) is before teams rely on it.
- Grade usability per consumer, because a trace serves some uses fully and others partially or not at all.
- Detect defects automatically, without manual inspection in a UI.
- Give engineers a usability level per consumer with a specific reason for each gap.

## Problem Statement

- Teams share Chakra traces without a standard check of what each trace supports.
- Trace quality varies with the producer, the conversion pipeline, and the capture method.
- Defects surface late, often as misleading results in a downstream tool.
- Consumers have no fast way to determine whether a trace fits their purpose.
- Trace authors receive no structured feedback on what to correct.

## Goals

- Provide a quick, standalone linter for Chakra traces.
- Scan a set of traces and report how usable each one is per consumer.
- Detect structural, semantic, and completeness defects.
- Provide specific evidence that helps trace authors and consumers resolve gaps.
- Serve as a quality gate for tools that require well-formed traces, such as replay.

## Position in the Toolchain

- The linter runs last, on the converted `.et` files, before any consumer reads them.
- Pipeline: trace, link, convert, lint.

| Stage   | Tool                | Input                          | Output                    |
|---------|---------------------|--------------------------------|---------------------------|
| Trace   | PyTorch profilers   | Workload run                   | Host trace, device trace  |
| Link    | `chakra_trace_link` | Host trace, device trace       | Merged host-device JSON   |
| Convert | `chakra_converter`  | Merged host-device JSON        | Chakra `.et` files        |
| Lint    | `chakra_linter`     | `.et` files or directories     | Usability report          |

Example:

```bash
# 1. Trace: capture host and device traces per rank (PyTorch profilers)

# 2. Link: merge host and device traces
chakra_trace_link \
    --chakra-host-trace host_trace_0.json \
    --chakra-device-trace device_trace_0.json \
    --rank 0 \
    --output-file linked_0.json

# 3. Convert: encode the merged trace as protobuf
chakra_converter PyTorch --input linked_0.json --output traces/chakra.0.et

# 4. Lint: grade every trace in the directory per consumer
chakra_linter traces/ --report-output report.md
```

- Run the lint stage after every conversion to detect defects at the stage that introduced them.
- Use the report as a gate before replay, simulation, or analysis starts.
- Share the report with the trace so consumers see the supported uses.

## Methodology

- Usability is a degree, not a yes or no answer. Every trace receives integrity levels (tier 0), evidence levels per check group, and a usability level per consumer (tier 1).
- Tier 0 covers decode failure, empty traces, schema violations, and DAG validity. A tier 0 defect makes the trace unusable for every consumer.
- Evidence levels grade how complete each check group is. Each gap sets a level for its group; the worst level wins.
- Tier 1 gaps reduce usability for specific consumers and never invalidate a trace. Each gap lists the consumers it affects.
- Consumers:
  - **Replay:** re-executes operators; requires collective and point-to-point attributes, and replay units: the outermost CPU nodes (`is_cpu_op` true) that carry a non-empty `op_schema`, with I/O that fits it. Replay runs a unit with its whole subtree, so descendants are not graded. GPU kernels need no schema; replay runs the CPU operator that launches them. Durations may be zero, because replay runs the operator rather than modeling its time.
  - **Simulator:** models timing from the graph; requires communication attributes and a duration or I/O for each compute node.
  - **Analysis:** reads timing and structure; requires nonzero durations and a device link.
- Synthetic traces skip the timing and Kineto checks, so they favor simulator suitability. No separate flag applies.
- Example: a replay unit lacks `op_schema` but has a duration above zero. Replay cannot call it; simulator and analysis are `READY`.
- A replay unit with an `op_schema` and readable I/O that fits the schema is replayable, whatever its duration.

## Rule Configuration (`rules.toml`)

- `rules.toml` declares every check the linter runs: ID, name, description, enabled state, and report message.
- `rules.schema.json` defines the syntax of that file.
- `linter.py` implements the check logic that each rule ID selects.
- The three files form one contract. No file is complete without the other two.

### Schema

- One `[[rule]]` table per check. The file requires at least one rule.
- Unknown keys are rejected.

| Key           | Type            | Required | Constraint                                                          |
|---------------|-----------------|----------|---------------------------------------------------------------------|
| `id`          | string          | Yes      | Two or three capital letters plus three digits, such as `DAG005`.   |
| `name`        | string          | Yes      | Kebab-case label, such as `dag-cycle`.                              |
| `description` | string          | Yes      | What the rule checks and why it matters. Ends with a period.        |
| `enabled`     | boolean         | Yes      | `true` runs the rule; `false` drops its findings before grading.    |
| `message`     | string          | Yes      | Report text. `{placeholders}` name values that the check supplies.  |
| `exempt`      | array of string | No       | Shell-style name patterns exempt from the check. `OPS001` only.     |

Example:

```toml
[[rule]]
id = "DAG003"
name = "dag-self-dep"
description = "A node lists its own ID as a dependency. The node can never become ready, so replay and simulation stall."
enabled = true
message = "self-dependency"
```

- The ID prefix groups rules by area: `INT`, `DAG`, `TIM`, `KIN`, `COM`, `OPS`, `IO`, `MET`, `ATT`.
- IDs stay stable and are never reused for a different rule.
- Placeholders are limited per rule. The loader rejects a message that names a value its check does not supply.

### Source of Truth

- `rules.toml` owns the rule catalog: which rules exist, their wording, and their default state.
- `linter.py` owns the behavior: how each check detects a defect and which consumers a finding affects.
- A rule requires both halves. A TOML entry without a check never fires; a check without an entry fails at startup.
- `RULE_PLACEHOLDERS` in `linter.py` lists the values that each message template may use.
- The loader falls back to the shipped file for any rule that a custom file omits.

| Concern                   | Defined in                      |
|---------------------------|---------------------------------|
| Rule ID, name, wording    | `rules.toml`                    |
| Enabled by default        | `rules.toml`                    |
| Exempt annotation names   | `rules.toml` (`OPS001`)         |
| File syntax               | `rules.schema.json`             |
| Detection logic           | `linter.py`                     |
| Allowed placeholders      | `linter.py` (`RULE_PLACEHOLDERS`) |
| Consumer impact per gap   | `linter.py`                     |

### Validation

- The loader fails fast, before any trace scan, on a missing, malformed, or invalid rule file.
- Checks at load time:
  - TOML syntax.
  - Conformance to `rules.schema.json`.
  - Unique rule IDs and names.
  - Valid message placeholders.
  - `INT001` and `INT002` stay enabled, because an unreadable or empty trace cannot be graded.
- `tests/linter/test_rules_toml.py` enforces the criteria that the schema cannot express:
  - IDs and names are unique.
  - Prefix groups are contiguous and numbered from 001.
  - Placeholders match what each check supplies.
  - `exempt` belongs to one rule only.
  - A committed ID keeps its name, or retires.
- `tests/linter/test_rule_inventory.py` confirms that the rule file and the code agree.

### Usage

- `--rules FILE` loads a custom rule file in place of the shipped one.
- `--list-rules` prints every rule with its enabled state, then exits.
- `--enable-rule` and `--disable-rule` override the file for one run. Each accepts a rule ID, name, or prefix, such as `DAG`, and repeats.
- Example: disable `KIN001` for a genuine CPU-only capture.

```bash
chakra_linter traces/ --disable-rule KIN001
```

### Changing a Rule

- Edit wording, default state, or `exempt` patterns in `rules.toml` alone. No code change applies.
- Add a rule by adding a `[[rule]]` entry, a check in `linter.py`, and a `RULE_PLACEHOLDERS` entry.
- Retire a rule by leaving its ID unused. Do not reassign the ID.
- Run the linter tests after every change to the file.
