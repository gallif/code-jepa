"""Export a trained Code-JEPA checkpoint as a self-contained folder, and encode programs with it.

A release folder holds:
  backbone/      fine-tuned Qwen2.5-Coder + tokenizer (with the Code-JEPA special tokens), HF format
  heads.pt       per-level projection heads (code side; text side too if they were trained separately)
  config.yaml    the training config (backbone.name is rewritten to the local backbone/ folder on load)
  metadata.json  variant, recipe, seed, source checkpoint, final validation metrics

Usage:
    from code_jepa.encode import CodeJEPAEncoder
    enc = CodeJEPAEncoder.from_pretrained("release/code-jepa-0.5b-free-s82", device="cuda")
    feats = enc.encode([record, ...])   # records in the hierarchy-JSONL schema (L4_code + L2_units spans)

Each element of ``feats`` is a dict with ``record_id`` and ``nodes``: one entry per tree node with
``node_id``, ``level`` (step | unit | approach | intent), ``span`` (1-based line range), ``parent_id``,
``raw`` (last-layer hidden state at the node token, float32[hidden]) and ``head`` (the level's projection,
float32[256]). Descriptions are not needed: the code view never attends to them. With
``with_text=True`` each node also gets ``text_raw`` / ``text_head`` from its description (if present).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from code_jepa.config import ExperimentConfig
from code_jepa.data.dataset import collate_packed
from code_jepa.data.packer import pack_record
from code_jepa.model import LEVEL_NAMES, HierarchicalCodeJEPA
from code_jepa.tokenize_utils import load_tokenizer


def export_model(config_path: str | Path, checkpoint: str | Path, out_dir: str | Path, metadata: dict | None = None) -> Path:
    cfg = ExperimentConfig.load(config_path)
    cfg.backbone.grad_checkpointing = False
    tok = load_tokenizer(cfg.backbone)
    model = HierarchicalCodeJEPA(cfg, tok)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    out = Path(out_dir)
    (out / "backbone").mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(out / "backbone", safe_serialization=True)
    tok.save_pretrained(out / "backbone")
    heads = {"heads": model.heads.state_dict()}
    if model.text_heads is not None:
        heads["text_heads"] = model.text_heads.state_dict()
    torch.save(heads, out / "heads.pt")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    meta = {"source_config": str(config_path), "source_checkpoint": str(checkpoint),
            "step": state.get("step") if isinstance(state, dict) else None,
            "program_tokens": cfg.hierarchy.program_tokens, "seed": cfg.train.seed}
    meta.update(metadata or {})
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return out


class CodeJEPAEncoder:
    def __init__(self, model: HierarchicalCodeJEPA, tokenizer, cfg: ExperimentConfig, device: torch.device):
        self.model, self.tokenizer, self.cfg, self.device = model, tokenizer, cfg, device

    @classmethod
    def from_pretrained(cls, folder: str | Path, device: str = "cuda", dtype: str | None = None,
                        max_row_tokens: int = 8192, max_nodes: int = 512) -> "CodeJEPAEncoder":
        folder = Path(folder)
        cfg = ExperimentConfig.load(folder / "config.yaml")
        cfg.backbone.name = str(folder / "backbone")
        cfg.backbone.cache_dir = None
        cfg.backbone.local_files_only = True
        cfg.backbone.grad_checkpointing = False
        dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        if dtype:
            cfg.backbone.dtype = dtype
        if dev.type == "cpu" and cfg.backbone.dtype in ("bf16", "fp16"):
            cfg.backbone.dtype = "fp32"
        cfg.hierarchy.max_row_tokens = max_row_tokens  # inference: do not drop long programs
        cfg.hierarchy.max_nodes = max_nodes
        tok = load_tokenizer(cfg.backbone)
        model = HierarchicalCodeJEPA(cfg, tok)
        heads = torch.load(folder / "heads.pt", map_location="cpu", weights_only=True)
        model.heads.load_state_dict(heads["heads"])
        if model.text_heads is not None and "text_heads" in heads:
            model.text_heads.load_state_dict(heads["text_heads"])
        model.to(dev).eval()
        return cls(model, tok, cfg, dev)

    @torch.no_grad()
    def encode(self, records: list[dict], batch_size: int = 4, with_text: bool = False) -> list[dict]:
        out: list[dict] = []
        packed = []
        for rec in records:
            override = None
            if not with_text:  # descriptions are invisible to the code view; drop them to save compute
                from code_jepa.data.tree import flatten_record
                override = {n.node_id: "" for n in flatten_record(rec, step_depth=self.cfg.hierarchy.step_depth)}
            ex = pack_record(rec, self.tokenizer, self.cfg, text_override=override)
            if ex.dropped:
                out.append({"record_id": ex.record_id, "nodes": [], "dropped": ex.drop_reason})
                continue
            packed.append(ex)
            out.append(None)  # placeholder, filled below in order
        slots = [i for i, o in enumerate(out) if o is None]
        for b in range(0, len(packed), batch_size):
            chunk = packed[b : b + batch_size]
            batch = collate_packed(chunk, self.tokenizer.pad_token_id)
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            hidden = self.model._forward_hidden(
                batch["view_input_ids"], {"full_attention": batch["view_attention_mask"]}, batch["view_position_ids"]
            )
            raw = self.model._gather(hidden, batch["node_pos"])
            head = self.model._project(raw, batch["level_ids"], batch["valid_nodes"])
            if with_text:
                traw = self.model._gather(hidden, batch["pred_pos"])
                thead = self.model._project(traw, batch["level_ids"], batch["valid_nodes"], heads=self.model.text_heads)
            for j, ex in enumerate(chunk):
                nodes = []
                for k, node in enumerate(ex.nodes):
                    item = {"node_id": node.node_id, "level": node.level, "span": list(node.span),
                            "parent_id": node.parent_id,
                            "raw": raw[j, k].float().cpu(), "head": head[j, k].float().cpu()}
                    if with_text:
                        item["text_raw"] = traw[j, k].float().cpu()
                        item["text_head"] = thead[j, k].float().cpu()
                    nodes.append(item)
                out[slots[b + j]] = {"record_id": ex.record_id, "nodes": nodes, "dropped": ""}
        return out
