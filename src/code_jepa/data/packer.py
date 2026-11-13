from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerBase

from code_jepa.config import SPECIAL_NODE_TOKENS, SPECIAL_PRED_TOKENS, ExperimentConfig
from code_jepa.data.tree import Node, flatten_record, lines_in_span, n_code_lines, validate_tree


@dataclass
class PackedExample:
    record_id: str
    nodes: list[Node]
    code: str
    warnings: list[str]
    view_input_ids: list[int]
    view_position_ids: list[int]
    view_mask: torch.Tensor
    node_seq_pos: list[int]
    pred_seq_pos: list[int]
    node_levels: list[str]
    lm_input_ids: list[int]
    lm_labels: list[int]
    dropped: bool = False
    drop_reason: str = ""
    extra: dict = field(default_factory=dict)


def _line_char_ranges(code: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    lines = code.splitlines(keepends=True)
    if not lines and code:
        lines = [code]
    for line in lines:
        ranges.append((start, start + len(line)))
        start += len(line)
    return ranges


def token_lines(
    tokenizer: PreTrainedTokenizerBase,
    code: str,
) -> tuple[list[int], list[int]]:
    try:
        encoded = tokenizer(
            code,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
    except Exception:
        encoded = None
    if encoded is not None and encoded.get("offset_mapping"):
        ids = list(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        ranges = _line_char_ranges(code)
        line_of: list[int] = []
        for start, _end in offsets:
            line = 1
            for i, (lo, hi) in enumerate(ranges, start=1):
                if start < hi:
                    line = i
                    break
                line = i
            line_of.append(line)
        return ids, line_of
    ids: list[int] = []
    line_of = []
    lines = code.splitlines(keepends=True)
    for i, line in enumerate(lines, start=1):
        piece = tokenizer(line, add_special_tokens=False)["input_ids"]
        ids.extend(piece)
        line_of.extend([i] * len(piece))
    return ids, line_of


def _node_token_id(tokenizer: PreTrainedTokenizerBase, level: str) -> int:
    token = SPECIAL_NODE_TOKENS[level]
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None or tid == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer missing special token {token}")
    return int(tid)


def _pred_token_id(tokenizer: PreTrainedTokenizerBase, level: str) -> int:
    token = SPECIAL_PRED_TOKENS[level]
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None or tid == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer missing special token {token}")
    return int(tid)


def _header_lines(unit: Node, nodes_by_id: dict[str, Node]) -> set[int]:
    own = lines_in_span(unit.span)
    for cid in unit.children_ids:
        child = nodes_by_id.get(cid)
        if child is None:
            continue
        own -= lines_in_span(child.span)
    return own


def _parent_step_header_lines(step: Node, nodes_by_id: dict[str, Node]) -> set[int]:
    own = lines_in_span(step.span)
    for cid in step.children_ids:
        child = nodes_by_id.get(cid)
        if child is not None:
            own -= lines_in_span(child.span)
    return own


def _allowed_code_lines(node: Node, nodes_by_id: dict[str, Node]) -> set[int]:
    if node.level == "step" and not node.children_ids:
        return lines_in_span(node.span)
    if node.level == "step":
        return _parent_step_header_lines(node, nodes_by_id)
    if node.level == "unit":
        return _header_lines(node, nodes_by_id)
    return set()


def _allowed_node_ids(node: Node) -> list[str]:
    if node.level in ("approach", "intent"):
        return list(node.children_ids)
    if node.level == "unit":
        return list(node.children_ids)
    if node.level == "step":
        return list(node.children_ids)
    return []


def _lm_prompt(record: dict, nodes: list[Node], fields: list[str]) -> str:
    by_level = {n.level: n for n in nodes if n.level in ("intent", "approach")}
    parts = []
    for name in fields:
        if name == "intent":
            parts.append(by_level["intent"].text)
        elif name == "approach":
            parts.append(by_level["approach"].text)
        elif name == "units":
            texts = [n.text for n in nodes if n.level == "unit"]
            parts.append("\n".join(texts))
        elif name == "steps":
            texts = [n.text for n in nodes if n.level == "step"]
            parts.append("\n".join(texts))
    return "\n\n".join(p for p in parts if p)


def pack_record(
    record: dict,
    tokenizer: PreTrainedTokenizerBase,
    cfg: ExperimentConfig,
    text_override: dict[str, str] | None = None,
) -> PackedExample:
    """``text_override`` (node_id -> text) replaces description texts in the view row only; the LM prompt
    keeps the record's true intent/approach (used by the shuffled-description control)."""
    code = record.get("L4_code") or record.get("code") or ""
    nodes_all = flatten_record(record, step_depth=cfg.hierarchy.step_depth)
    warnings = validate_tree(code, nodes_all)
    allowed_levels = set(cfg.hierarchy.node_levels)
    nodes = [n for n in nodes_all if n.level in allowed_levels] if allowed_levels else nodes_all
    prompt_nodes = nodes
    if text_override:
        from dataclasses import replace as _replace

        nodes = [_replace(n, text=text_override.get(n.node_id, n.text)) for n in nodes]
    ground_program = cfg.hierarchy.ground == "program"
    if cfg.hierarchy.program_tokens not in ("tree", "all_code", "units_and_code"):
        raise ValueError(f"hierarchy.program_tokens must be tree | all_code | units_and_code, got {cfg.hierarchy.program_tokens!r}")
    if cfg.hierarchy.ground not in ("span", "program"):
        raise ValueError(f"hierarchy.ground must be span | program, got {cfg.hierarchy.ground!r}")
    record_id = str(record.get("id") or record.get("problem_id") or "")

    if len(nodes) > cfg.hierarchy.max_nodes:
        return PackedExample(
            record_id=record_id,
            nodes=nodes,
            code=code,
            warnings=warnings,
            view_input_ids=[],
            view_position_ids=[],
            view_mask=torch.zeros(0, 0),
            node_seq_pos=[],
            pred_seq_pos=[],
            node_levels=[],
            lm_input_ids=[],
            lm_labels=[],
            dropped=True,
            drop_reason=f"too many nodes: {len(nodes)}",
        )

    code_ids, line_of = token_lines(tokenizer, code)
    n_code = len(code_ids)
    if n_code == 0:
        return PackedExample(
            record_id=record_id,
            nodes=nodes,
            code=code,
            warnings=warnings + ["empty code"],
            view_input_ids=[],
            view_position_ids=[],
            view_mask=torch.zeros(0, 0),
            node_seq_pos=[],
            pred_seq_pos=[],
            node_levels=[],
            lm_input_ids=[],
            lm_labels=[],
            dropped=True,
            drop_reason="empty code",
        )

    last_pos_of_line: dict[int, int] = {}
    tokens_on_line: dict[int, list[int]] = {i: [] for i in range(1, n_code_lines(code) + 1)}
    for idx, line in enumerate(line_of):
        tokens_on_line.setdefault(line, []).append(idx)
        last_pos_of_line[line] = idx

    nodes_by_id = {n.node_id: n for n in nodes}
    k_pred = cfg.predictor.tokens_per_level

    view_ids = list(code_ids)
    position_ids = list(range(n_code))
    node_seq_pos: list[int] = []
    node_levels: list[str] = []

    for node in nodes:
        node_levels.append(node.level)
        if ground_program and node.level in ("step", "unit"):
            node_seq_pos.append(-1)  # filled with the program token's position below
            continue
        pos = len(view_ids)
        view_ids.append(_node_token_id(tokenizer, node.level))
        span_last_line = min(node.span[1], max(last_pos_of_line, default=1))
        position_ids.append(last_pos_of_line.get(span_last_line, n_code - 1))
        node_seq_pos.append(pos)
    if ground_program:
        prog = [p for n, p in zip(nodes, node_seq_pos) if n.level == "approach"] or [
            p for n, p in zip(nodes, node_seq_pos) if n.level == "intent"
        ]
        if not prog:
            raise ValueError("hierarchy.ground=program needs an approach or intent node")
        node_seq_pos = [prog[0] if p == -1 else p for p in node_seq_pos]

    id_to_seq = {n.node_id: node_seq_pos[i] for i, n in enumerate(nodes)}

    pred_seq_pos: list[int] = []
    desc_spans: list[tuple[int, int]] = []
    for node in nodes:
        text_ids = tokenizer(node.text, add_special_tokens=False)["input_ids"] if node.text else []
        pred_ids = [_pred_token_id(tokenizer, node.level)] * max(k_pred, 0)
        if k_pred == 0:
            pred_ids = []
        start = len(view_ids)
        view_ids.extend(text_ids + pred_ids)
        n_seg = len(text_ids) + len(pred_ids)
        position_ids.extend(range(n_seg) if n_seg else [])
        end = len(view_ids)
        desc_spans.append((start, end))
        if k_pred == 0:
            pred_seq_pos.append(end - 1 if end > start else start)
        else:
            pred_seq_pos.append(end - 1)

    if len(view_ids) > cfg.hierarchy.max_row_tokens:
        return PackedExample(
            record_id=record_id,
            nodes=nodes,
            code=code,
            warnings=warnings,
            view_input_ids=view_ids,
            view_position_ids=position_ids,
            view_mask=torch.zeros(0, 0),
            node_seq_pos=node_seq_pos,
            pred_seq_pos=pred_seq_pos,
            node_levels=node_levels,
            lm_input_ids=[],
            lm_labels=[],
            dropped=True,
            drop_reason=f"view row {len(view_ids)} > max_row_tokens",
        )

    t = len(view_ids)
    mask = torch.full((t, t), float("-inf"))
    for i in range(n_code):
        mask[i, : i + 1] = 0.0

    for node, seq_i in zip(nodes, node_seq_pos):
        if ground_program and node.level in ("step", "unit"):
            continue  # no token of its own
        mask[seq_i, seq_i] = 0.0
        child_ids = [nid for nid in _allowed_node_ids(node) if nid in id_to_seq]
        if node.level in ("approach", "intent") and (
            not child_ids or cfg.hierarchy.program_tokens == "all_code" or ground_program
        ):
            mask[seq_i, :n_code] = 0.0
        else:
            allowed_lines = _allowed_code_lines(node, nodes_by_id)
            for line in allowed_lines:
                for tok_i in tokens_on_line.get(line, []):
                    mask[seq_i, tok_i] = 0.0
            for nid in child_ids:
                mask[seq_i, id_to_seq[nid]] = 0.0
        if node.level in ("approach", "intent") and cfg.hierarchy.program_tokens == "units_and_code":
            mask[seq_i, :n_code] = 0.0  # hybrid: unit tokens (above) and every code token

    for start, end in desc_spans:
        for i in range(start, end):
            mask[i, start : i + 1] = 0.0

    prompt = _lm_prompt(record, prompt_nodes, cfg.objectives.lm.prompt)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"] if prompt else []
    lm_ids = prompt_ids + code_ids
    lm_labels = [-100] * len(prompt_ids) + list(code_ids)

    return PackedExample(
        record_id=record_id,
        nodes=nodes,
        code=code,
        warnings=warnings,
        view_input_ids=view_ids,
        view_position_ids=position_ids,
        view_mask=mask,
        node_seq_pos=node_seq_pos,
        pred_seq_pos=pred_seq_pos,
        node_levels=node_levels,
        lm_input_ids=lm_ids,
        lm_labels=lm_labels,
        # Unit count from the full tree, so single/multi-unit splits survive node_levels filtering (E2).
        extra={"n_code": n_code, "n_view": t, "n_units": sum(n.level == "unit" for n in nodes_all)},
    )
