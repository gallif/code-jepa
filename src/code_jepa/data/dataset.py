from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from code_jepa.config import ExperimentConfig
from code_jepa.data.packer import PackedExample, pack_record


class HierarchicalJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        cfg: ExperimentConfig,
        split: str | None = None,
        max_samples: int | None = None,
        skip_dropped: bool = True,
    ):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.examples: list[PackedExample] = []
        self.dropped = 0
        path = Path(jsonl)
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if split and record.get("split") not in (None, split):
                    continue
                packed = pack_record(record, tokenizer, cfg)
                if packed.dropped:
                    self.dropped += 1
                    if skip_dropped:
                        continue
                self.examples.append(packed)
                if max_samples is not None and len(self.examples) >= max_samples:
                    break
        if not self.examples:
            raise ValueError(f"no usable records in {path} (dropped={self.dropped})")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PackedExample:
        return self.examples[idx]


def collate_packed(batch: list[PackedExample], pad_id: int) -> dict[str, torch.Tensor | list]:
    def pad_1d(seqs: list[list[int]], value: int) -> torch.Tensor:
        width = max(len(s) for s in seqs)
        out = torch.full((len(seqs), width), value, dtype=torch.long)
        for i, seq in enumerate(seqs):
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return out

    view_lens = [len(ex.view_input_ids) for ex in batch]
    max_t = max(view_lens)
    bsz = len(batch)
    view_mask = torch.full((bsz, 1, max_t, max_t), float("-inf"))
    for i, ex in enumerate(batch):
        t = len(ex.view_input_ids)
        view_mask[i, 0, :t, :t] = ex.view_mask
        for p in range(t, max_t):
            view_mask[i, 0, p, p] = 0.0

    max_nodes = max(len(ex.nodes) for ex in batch)
    node_pos = torch.full((bsz, max_nodes), -1, dtype=torch.long)
    pred_pos = torch.full((bsz, max_nodes), -1, dtype=torch.long)
    level_ids = torch.full((bsz, max_nodes), -1, dtype=torch.long)
    valid = torch.zeros((bsz, max_nodes), dtype=torch.bool)
    level_to_id = {"step": 0, "unit": 1, "approach": 2, "intent": 3}
    for i, ex in enumerate(batch):
        n = len(ex.nodes)
        node_pos[i, :n] = torch.tensor(ex.node_seq_pos, dtype=torch.long)
        pred_pos[i, :n] = torch.tensor(ex.pred_seq_pos, dtype=torch.long)
        level_ids[i, :n] = torch.tensor([level_to_id[lv] for lv in ex.node_levels], dtype=torch.long)
        valid[i, :n] = True

    lm_ids = pad_1d([ex.lm_input_ids for ex in batch], pad_id)
    lm_labels = pad_1d([ex.lm_labels for ex in batch], -100)
    lm_attn = (lm_ids != pad_id).long()

    return {
        "view_input_ids": pad_1d([ex.view_input_ids for ex in batch], pad_id),
        "view_position_ids": pad_1d([ex.view_position_ids for ex in batch], 0),
        "view_attention_mask": view_mask,
        "node_pos": node_pos,
        "pred_pos": pred_pos,
        "level_ids": level_ids,
        "valid_nodes": valid,
        "lm_input_ids": lm_ids,
        "lm_labels": lm_labels,
        "lm_attention_mask": lm_attn,
        "record_ids": [ex.record_id for ex in batch],
        "warnings": [ex.warnings for ex in batch],
    }
