"""Uniform encoders for the Code-JEPA benchmarks.

Every model exposes
  encode_code(records)        -> list (per record) of list of nodes {node_id, level, span, vecs: {readout: np.ndarray}}
  encode_text(texts, level)   -> {readout: np.ndarray [N, D]}
Readouts:
  JEPA models (tree / free / E1): node_raw (node token, last layer), node_head (256-d projection),
                                 pool_last / pool_mid (mean over the node's own code tokens; text: over text tokens)
  Plain backbone (E0):           pool_last / pool_mid
Text side of node_raw / node_head is the level's predictor token after the text (the description view never
attends to code, so any query text can be encoded this way).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/storage/users/glifshitz/code-jepa/repo")
sys.path.insert(0, str(REPO / "src"))

from code_jepa.config import SPECIAL_PRED_TOKENS  # noqa: E402
from code_jepa.data.dataset import collate_packed  # noqa: E402
from code_jepa.data.packer import pack_record, token_lines  # noqa: E402
from code_jepa.data.tree import flatten_record  # noqa: E402
from code_jepa.encode import CodeJEPAEncoder  # noqa: E402
from code_jepa.model import LEVEL_NAMES  # noqa: E402

ROOT = Path("/mnt/storage/users/glifshitz/code-jepa")
MODELS = {
    "free": ROOT / "release/code-jepa-0.5b-free-s82",
    "tree": ROOT / "release/code-jepa-0.5b-tree-s82",
    "E1": ROOT / "bench/models/e1_s82",
    "E0": ROOT / "models/hf/hub/models--Qwen--Qwen2.5-Coder-0.5B",
}


def _span_token_idx(line_of: list[int], span) -> list[int]:
    s, e = span
    return [i for i, ln in enumerate(line_of) if s <= ln <= e]


def _batches_by_budget(lengths: list[int], max_tokens: int, max_bs: int):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batch, width = [], 0
    for i in order:
        w = max(width, lengths[i])
        # budget: total attention-matrix size of the padded batch <= that of one max_tokens-long row
        if batch and (w * w * (len(batch) + 1) > max_tokens * max_tokens or len(batch) >= max_bs):
            yield batch
            batch, w = [], lengths[i]
        batch.append(i)
        width = w
    if batch:
        yield batch


class JEPAModel:
    has_nodes = True

    def __init__(self, name: str, folder: Path, device="cuda"):
        self.name = name
        self.enc = CodeJEPAEncoder.from_pretrained(folder, device=device)
        self.model, self.tok, self.cfg, self.dev = self.enc.model, self.enc.tokenizer, self.enc.cfg, self.enc.device
        self.n_layers = self.model.backbone.config.num_hidden_layers
        self.mid = self.n_layers // 2

    @torch.no_grad()
    def encode_code(self, records: list[dict], max_tokens: int = 6000) -> list[list[dict] | None]:
        out: list[list[dict] | None] = [None] * len(records)
        packed, idx = [], []
        for i, rec in enumerate(records):
            override = {n.node_id: "" for n in flatten_record(rec, step_depth=self.cfg.hierarchy.step_depth)}
            ex = pack_record(rec, self.tok, self.cfg, text_override=override)
            if ex.dropped:
                continue
            packed.append(ex)
            idx.append(i)
        lengths = [len(ex.view_input_ids) for ex in packed]
        for b in _batches_by_budget(lengths, max_tokens, 16):
            chunk = [packed[j] for j in b]
            batch = collate_packed(chunk, self.tok.pad_token_id)
            batch = {k: (v.to(self.dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            o = self.model.backbone(input_ids=batch["view_input_ids"],
                                    attention_mask={"full_attention": batch["view_attention_mask"]},
                                    position_ids=batch["view_position_ids"], use_cache=False, output_hidden_states=True)
            hs_last, hs_mid = o.hidden_states[-1], o.hidden_states[self.mid]
            raw = self.model._gather(hs_last, batch["node_pos"])
            head = self.model._project(raw, batch["level_ids"], batch["valid_nodes"])
            for jj, ex in enumerate(chunk):
                _, line_of = token_lines(self.tok, ex.code)
                nodes = []
                for k, node in enumerate(ex.nodes):
                    ti = _span_token_idx(line_of, node.span) or list(range(len(line_of)))
                    ti_t = torch.tensor(ti, device=self.dev)
                    nodes.append({"node_id": node.node_id, "level": node.level, "span": list(node.span),
                                  "vecs": {"node_raw": raw[jj, k].float().cpu().numpy(),
                                           "node_head": head[jj, k].float().cpu().numpy(),
                                           "pool_last": hs_last[jj, ti_t].float().mean(0).cpu().numpy(),
                                           "pool_mid": hs_mid[jj, ti_t].float().mean(0).cpu().numpy()}})
                out[idx[b[jj]]] = nodes
        return out

    @torch.no_grad()
    def encode_text(self, texts: list[str], level: str, max_len: int = 1024, bs: int = 32) -> dict[str, np.ndarray]:
        pred = int(self.tok.convert_tokens_to_ids(SPECIAL_PRED_TOKENS[level]))
        li = LEVEL_NAMES.index(level)
        seqs = [self.tok(t or "", add_special_tokens=False)["input_ids"][: max_len - 1] + [pred] for t in texts]
        res = {k: [None] * len(texts) for k in ("node_raw", "node_head", "pool_last", "pool_mid")}
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        for s in range(0, len(order), bs):
            b = order[s : s + bs]
            w = max(len(seqs[i]) for i in b)
            ids = torch.full((len(b), w), self.tok.pad_token_id, dtype=torch.long)
            am = torch.zeros((len(b), w), dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, : len(seqs[i])] = torch.tensor(seqs[i])
                am[r, : len(seqs[i])] = 1
            ids, am = ids.to(self.dev), am.to(self.dev)
            pos = torch.arange(w, device=self.dev).unsqueeze(0).expand(len(b), -1)
            o = self.model.backbone(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=False,
                                    output_hidden_states=True)
            hl, hm = o.hidden_states[-1], o.hidden_states[self.mid]
            last = am.sum(1) - 1
            raw = hl[torch.arange(len(b)), last]
            lv = torch.full((len(b), 1), li, device=self.dev)
            head = self.model._project(raw.unsqueeze(1), lv, torch.ones_like(lv, dtype=torch.bool),
                                       heads=self.model.text_heads)[:, 0]
            tm = am.clone().float()
            tm[torch.arange(len(b)), last] = 0  # exclude the predictor token from pooling
            tm[tm.sum(1) == 0, 0] = 1
            pl = (hl.float() * tm[..., None]).sum(1) / tm.sum(1, keepdim=True)
            pm = (hm.float() * tm[..., None]).sum(1) / tm.sum(1, keepdim=True)
            for r, i in enumerate(b):
                res["node_raw"][i] = raw[r].float().cpu().numpy()
                res["node_head"][i] = head[r].float().cpu().numpy()
                res["pool_last"][i] = pl[r].cpu().numpy()
                res["pool_mid"][i] = pm[r].cpu().numpy()
        return {k: np.stack(v) for k, v in res.items()}


class PlainModel:
    """Pretrained Qwen (E0): mean-pooled hidden states over each node's own code lines."""

    has_nodes = False

    def __init__(self, name: str, path: str, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        snap = sorted((Path(path) / "snapshots").glob("*"))[-1] if (Path(path) / "snapshots").exists() else Path(path)
        self.tok = AutoTokenizer.from_pretrained(snap)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).to(device).eval()
        self.dev = torch.device(device)
        self.n_layers = self.model.config.num_hidden_layers
        self.mid = self.n_layers // 2
        self.step_depth = 2

    @torch.no_grad()
    def _hidden(self, seqs: list[list[int]], bs_tokens: int = 16000):
        """Yield (index, last [T,D], mid [T,D]) for each sequence."""
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        s = 0
        while s < len(order):
            w = len(seqs[order[s]])
            e = s
            while e < len(order) and len(seqs[order[e]]) * (e - s + 1) <= bs_tokens and e - s < 32:
                w = len(seqs[order[e]])
                e += 1
            e = max(e, s + 1)
            b = order[s:e]
            w = max(len(seqs[i]) for i in b)
            ids = torch.full((len(b), w), self.tok.pad_token_id, dtype=torch.long)
            am = torch.zeros((len(b), w), dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, : len(seqs[i])] = torch.tensor(seqs[i])
                am[r, : len(seqs[i])] = 1
            o = self.model(input_ids=ids.to(self.dev), attention_mask=am.to(self.dev), use_cache=False,
                           output_hidden_states=True)
            for r, i in enumerate(b):
                n = len(seqs[i])
                yield i, o.hidden_states[-1][r, :n].float(), o.hidden_states[self.mid][r, :n].float()
            s = e

    def encode_code(self, records: list[dict], max_len: int = 8192) -> list[list[dict] | None]:
        lines, seqs = [], []
        for rec in records:
            ids, line_of = token_lines(self.tok, rec["L4_code"])
            seqs.append(ids[:max_len])
            lines.append(line_of[:max_len])
        out: list = [None] * len(records)
        for i, hl, hm in self._hidden(seqs):
            if not seqs[i]:
                continue
            nodes = []
            for node in flatten_record(records[i], step_depth=self.step_depth):
                ti = _span_token_idx(lines[i], node.span) or list(range(len(lines[i])))
                t = torch.tensor(ti, device=self.dev)
                nodes.append({"node_id": node.node_id, "level": node.level, "span": list(node.span),
                              "vecs": {"pool_last": hl[t].mean(0).cpu().numpy(), "pool_mid": hm[t].mean(0).cpu().numpy()}})
            out[i] = nodes
        return out

    def encode_text(self, texts: list[str], level: str = "", max_len: int = 1024) -> dict[str, np.ndarray]:
        seqs = [self.tok(t or " ", add_special_tokens=False)["input_ids"][:max_len] or [self.tok.pad_token_id] for t in texts]
        res = {"pool_last": [None] * len(texts), "pool_mid": [None] * len(texts)}
        for i, hl, hm in self._hidden(seqs):
            res["pool_last"][i] = hl.mean(0).cpu().numpy()
            res["pool_mid"][i] = hm.mean(0).cpu().numpy()
        return {k: np.stack(v) for k, v in res.items()}


def load(name: str, device="cuda"):
    if name == "E0":
        return PlainModel(name, str(MODELS["E0"]), device)
    return JEPAModel(name, MODELS[name], device)


def nodes_by_level(encoded: list[list[dict] | None], level: str, readout: str):
    """Flatten to (record_index, node, vec) for one level."""
    rows = []
    for ri, nodes in enumerate(encoded):
        if not nodes:
            continue
        for n in nodes:
            if n["level"] == level and readout in n["vecs"]:
                rows.append((ri, n, n["vecs"][readout]))
    return rows
