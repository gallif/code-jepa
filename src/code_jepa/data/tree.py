from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

LEVELS = ("step", "unit", "approach", "intent")


@dataclass
class Node:
    node_id: str
    level: str
    parent_id: str | None
    span: tuple[int, int]
    text: str
    kind: str | None = None
    nested: bool = False
    children_ids: list[str] = field(default_factory=list)

    @property
    def is_leaf_step(self) -> bool:
        return self.level == "step" and not self.children_ids


def parse_span(span: Any) -> tuple[int, int]:
    if span is None:
        raise ValueError("missing span")
    if isinstance(span, dict):
        start = int(span.get("start", span.get("begin")))
        end = int(span.get("end"))
    elif isinstance(span, (list, tuple)) and len(span) == 2:
        start, end = int(span[0]), int(span[1])
    else:
        raise ValueError(f"unrecognised span: {span!r}")
    if end < start:
        start, end = end, start
    if start == 0 and end >= 0:
        start, end = 1, max(end, 1)
    return start, end


def _unit_text(unit: dict) -> str:
    resp = (unit.get("responsibility") or "").strip()
    contract = (unit.get("contract") or "").strip()
    return "\n".join(part for part in (resp, contract) if part)


def _approach_text(approach: dict | str | None) -> str:
    if approach is None:
        return ""
    if isinstance(approach, str):
        return approach.strip()
    strategy = (approach.get("strategy") or "").strip()
    structures = approach.get("key_data_structures") or []
    if isinstance(structures, list):
        struct_s = ", ".join(str(s) for s in structures)
    else:
        struct_s = str(structures)
    complexity = (approach.get("complexity") or "").strip()
    parts = [strategy]
    if struct_s:
        parts.append(f"data structures: {struct_s}")
    if complexity:
        parts.append(f"complexity: {complexity}")
    return "\n".join(parts)


def _iter_steps(
    steps: Iterable[dict],
    parent_id: str,
    depth: int,
    max_depth: int,
) -> Iterable[Node]:
    for step in steps or []:
        node_id = str(step.get("id") or f"{parent_id}.b")
        children = step.get("children") or []
        if depth >= max_depth and children:
            children = []
        node = Node(
            node_id=node_id,
            level="step",
            parent_id=parent_id,
            span=parse_span(step["span"]),
            text=(step.get("text") or "").strip(),
            kind=step.get("kind"),
            children_ids=[str(c.get("id")) for c in children if c.get("id")],
        )
        yield node
        if children:
            yield from _iter_steps(children, node_id, depth + 1, max_depth)


def _iter_units(
    units: Iterable[dict],
    parent_id: str | None,
    nested: bool,
    max_step_depth: int,
) -> Iterable[Node]:
    for unit in units or []:
        node_id = str(unit.get("id") or unit.get("name") or "unit")
        node = Node(
            node_id=node_id,
            level="unit",
            parent_id=parent_id,
            span=parse_span(unit["span"]),
            text=_unit_text(unit),
            kind=unit.get("kind") or unit.get("name"),
            nested=nested,
        )
        methods = unit.get("methods") or []
        steps = unit.get("steps") or []
        node.children_ids = [str(m.get("id")) for m in methods if m.get("id")] + [
            str(s.get("id")) for s in steps if s.get("id")
        ]
        yield node
        if methods:
            yield from _iter_units(methods, node_id, nested=True, max_step_depth=max_step_depth)
        if steps:
            yield from _iter_steps(steps, node_id, depth=1, max_depth=max_step_depth)


def n_code_lines(code: str) -> int:
    if not code:
        return 0
    return code.count("\n") + (0 if code.endswith("\n") else 1)


def flatten_record(record: dict, step_depth: int = 2) -> list[Node]:
    code = record.get("L4_code") or record.get("code") or ""
    n_lines = max(n_code_lines(code), 1)
    whole = (1, n_lines)
    nodes: list[Node] = []
    units = record.get("L2_units") or []
    nodes.extend(_iter_units(units, parent_id=None, nested=False, max_step_depth=step_depth))
    top_unit_ids = [n.node_id for n in nodes if n.level == "unit" and not n.nested]
    nodes.append(
        Node(
            node_id="approach",
            level="approach",
            parent_id=None,
            span=whole,
            text=_approach_text(record.get("L1_approach")),
            children_ids=list(top_unit_ids),
        )
    )
    nodes.append(
        Node(
            node_id="intent",
            level="intent",
            parent_id=None,
            span=whole,
            text=(record.get("L0_intent") or "").strip(),
            children_ids=list(top_unit_ids),
        )
    )
    return nodes


_BLANK_OR_COMMENT = re.compile(r"^\s*(#.*)?$")
_HEADERISH = re.compile(
    r"^\s*(def |class |async def |import |from |@|\"\"\"|'''|$)",
    re.I,
)


def lines_in_span(span: tuple[int, int]) -> set[int]:
    return set(range(span[0], span[1] + 1))


def validate_tree(code: str, nodes: list[Node]) -> list[str]:
    """Return warnings (empty if the tree looks consistent)."""
    warnings: list[str] = []
    lines = code.splitlines()
    n_lines = len(lines)
    by_id = {n.node_id: n for n in nodes}
    units = [n for n in nodes if n.level == "unit"]
    steps = [n for n in nodes if n.level == "step"]

    for node in nodes:
        if node.span[0] < 1 or node.span[1] > max(n_lines, 1):
            warnings.append(f"{node.node_id} span {node.span} outside 1..{n_lines}")

    for unit in units:
        nested = [
            by_id[cid]
            for cid in unit.children_ids
            if cid in by_id and by_id[cid].level == "unit"
        ]
        nested_lines: set[int] = set()
        for child in nested:
            nested_lines |= lines_in_span(child.span)
        own_lines = lines_in_span(unit.span) - nested_lines
        unit_steps = [
            by_id[cid]
            for cid in unit.children_ids
            if cid in by_id and by_id[cid].level == "step"
        ]
        covered: set[int] = set()
        for step in unit_steps:
            sl = lines_in_span(step.span)
            if sl & covered:
                warnings.append(f"overlapping steps in {unit.node_id}: {step.node_id}")
            if not sl <= (own_lines | nested_lines):
                warnings.append(f"step {step.node_id} not inside unit {unit.node_id}")
            covered |= sl
        uncovered = own_lines - covered
        for line_no in sorted(uncovered):
            text = lines[line_no - 1] if 0 < line_no <= n_lines else ""
            if _BLANK_OR_COMMENT.match(text) or _HEADERISH.match(text):
                continue
            warnings.append(f"uncovered executable line {line_no} in {unit.node_id}: {text!r}")
    return warnings
