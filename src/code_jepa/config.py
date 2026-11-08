from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

LEVELS = ("step", "unit", "approach", "intent")

SPECIAL_NODE_TOKENS = {
    "step": "<step>",
    "unit": "<unit>",
    "approach": "<approach>",
    "intent": "<intent>",
}
SPECIAL_PRED_TOKENS = {
    "step": "<pred_step>",
    "unit": "<pred_unit>",
    "approach": "<pred_approach>",
    "intent": "<pred_intent>",
}


def _merge(dst: dict, src: dict) -> dict:
    out = dict(dst)
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _update(obj, payload: dict[str, Any] | None):
    if not payload:
        return obj
    for key, value in payload.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update(current, value)
        else:
            setattr(obj, key, value)
    return obj


@dataclass
class BackboneConfig:
    name: str = "Qwen/Qwen2.5-Coder-0.5B"
    cache_dir: str | None = "models/hf"
    local_files_only: bool = False
    lora: bool = False
    dtype: str = "bf16"
    grad_checkpointing: bool = True
    attn_implementation: str = "eager"


@dataclass
class HierarchyConfig:
    node_levels: list[str] = field(default_factory=lambda: list(LEVELS))
    node_encoding: str = "tree_mask"
    step_depth: int = 2
    max_nodes: int = 96
    max_row_tokens: int = 3072


@dataclass
class PredictorConfig:
    tokens_per_level: int = 1
    level_specific: bool = True


@dataclass
class LmObjective:
    weight: float = 1.0
    prompt: list[str] = field(default_factory=lambda: ["intent", "approach"])


@dataclass
class ViewObjective:
    weight: float = 0.5
    distance: str = "cosine"
    stop_grad_target: bool = False
    direction: str = "text_to_code"
    levels: list[str] | None = None


@dataclass
class MaskedNodesObjective:
    enabled: bool = False
    levels: list[str] = field(default_factory=lambda: ["step", "unit"])
    target: str = "no_detach"


@dataclass
class SigregConfig:
    weight: float = 0.1
    num_proj: int = 1024
    knots: int = 17
    bank_size: int = 0
    # code | text | both | target (legacy). See model.SIGREG_SIDES.
    side: str = "code"


@dataclass
class ObjectivesConfig:
    lm: LmObjective = field(default_factory=LmObjective)
    view: ViewObjective = field(default_factory=ViewObjective)
    chain: bool = False
    masked_nodes: MaskedNodesObjective = field(default_factory=MaskedNodesObjective)
    cross_solution: bool = False
    sigreg: SigregConfig = field(default_factory=SigregConfig)
    jepa_dropout: float = 0.5


@dataclass
class ProjectionConfig:
    dim: int = 256
    layers: int = 2


@dataclass
class TrainConfig:
    lr_backbone: float = 1.0e-5
    lr_new_params: float = 5.0e-5
    epochs: int = 4
    batch_size: int = 1
    grad_accum: int = 8
    max_steps: int | None = None
    seed: int = 82
    output_dir: str = "runs/code_jepa"
    log_every: int = 10
    eval_every: int = 400
    save_every: int = 500
    num_workers: int = 0
    max_grad_norm: float = 1.0


@dataclass
class DataConfig:
    jsonl: str = ""
    split: str | None = "train"
    eval_split: str | None = "val"
    max_samples: int | None = None
    eval_max_samples: int | None = None


@dataclass
class ExperimentConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    objectives: ObjectivesConfig = field(default_factory=ObjectivesConfig)
    projection_head: ProjectionConfig = field(default_factory=ProjectionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "ExperimentConfig":
        with open(path) as handle:
            raw = yaml.safe_load(handle) or {}
        if overrides:
            raw = _merge(raw, overrides)
        cfg = cls()
        _update(cfg, raw)
        return cfg

    def apply_dot_overrides(self, items: list[str]) -> "ExperimentConfig":
        for item in items:
            if "=" not in item:
                raise ValueError(f"override must be key=value, got {item!r}")
            key, text = item.split("=", 1)
            parts = key.split(".")
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            current = getattr(obj, parts[-1])
            setattr(obj, parts[-1], _coerce_override(text, current))
        return self


def _coerce_override(text: str, current):
    lowered = text.lower()
    if lowered in {"null", "none"}:
        return None
    if isinstance(current, bool):
        return lowered in {"1", "true", "yes"}
    if isinstance(current, list):
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(text))
    if isinstance(current, float):
        return float(text)
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    try:
        if "." in text or "e" in lowered:
            return float(text)
        return int(text)
    except ValueError:
        return text
