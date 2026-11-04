from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from code_jepa.config import ExperimentConfig
from code_jepa.losses import SIGReg, pairwise_distance
from code_jepa.tokenize_utils import resolve_pretrained, special_tokens

LEVEL_NAMES = ("step", "unit", "approach", "intent")


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, layers: int = 2):
        super().__init__()
        if layers <= 1:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            hidden = in_dim
            mods: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)]
            self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def build_tiny_backbone(vocab_size: int) -> AutoModelForCausalLM:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        use_cache=False,
    )
    if hasattr(config, "layer_types"):
        config.layer_types = ["full_attention"] * config.num_hidden_layers
    model = Qwen2ForCausalLM(config)
    model.config._attn_implementation = "eager"
    return model


class HierarchicalCodeJEPA(nn.Module):
    def __init__(
        self,
        cfg: ExperimentConfig,
        tokenizer: PreTrainedTokenizerBase,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        if backbone is None:
            dtype = _dtype(cfg.backbone.dtype)
            source = resolve_pretrained(cfg.backbone.name, cfg.backbone.cache_dir)
            load_kw = dict(
                cache_dir=cfg.backbone.cache_dir,
                local_files_only=cfg.backbone.local_files_only or Path(source).exists(),
                attn_implementation=cfg.backbone.attn_implementation,
                trust_remote_code=True,
            )
            try:
                backbone = AutoModelForCausalLM.from_pretrained(source, dtype=dtype, **load_kw)
            except TypeError:
                backbone = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype, **load_kw)
        self.backbone = backbone
        self.backbone.config._attn_implementation = cfg.backbone.attn_implementation
        n_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens()})
        if n_added or self.backbone.get_input_embeddings().weight.shape[0] != len(tokenizer):
            self.backbone.resize_token_embeddings(len(tokenizer))
            with torch.no_grad():
                emb = self.backbone.get_input_embeddings().weight
                mean = emb[: len(tokenizer) - len(special_tokens())].mean(dim=0)
                emb[-len(special_tokens()) :] = mean
        hidden = self.backbone.config.hidden_size
        out_dim = cfg.projection_head.dim
        self.heads = nn.ModuleDict(
            {level: ProjectionHead(hidden, out_dim, cfg.projection_head.layers) for level in LEVEL_NAMES}
        )
        self.sigreg = SIGReg(knots=cfg.objectives.sigreg.knots, num_proj=cfg.objectives.sigreg.num_proj)
        if cfg.backbone.grad_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        self.backbone.config.use_cache = False

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "output_hidden_states": True,
        }
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        outputs = self.backbone(**kwargs)
        if outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        return outputs.last_hidden_state

    def _gather(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        idx = positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        return hidden.gather(1, idx)

    def _project(self, vectors: torch.Tensor, level_ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        out = vectors.new_zeros(*vectors.shape[:2], self.cfg.projection_head.dim)
        for i, level in enumerate(LEVEL_NAMES):
            sel = valid & (level_ids == i)
            if sel.any():
                out[sel] = self.heads[level](vectors[sel].float()).to(out.dtype)
        return out

    def view_forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._forward_hidden(
            batch["view_input_ids"],
            {"full_attention": batch["view_attention_mask"]},
            batch["view_position_ids"],
        )
        c_h = self._gather(hidden, batch["node_pos"])
        p_h = self._gather(hidden, batch["pred_pos"])
        valid = batch["valid_nodes"]
        c = self._project(c_h, batch["level_ids"], valid)
        p = self._project(p_h, batch["level_ids"], valid)
        if self.cfg.objectives.view.direction == "code_to_text":
            p, c = c, p
        if self.cfg.objectives.view.stop_grad_target:
            c = c.detach()
        return p, c

    def lm_forward(self, batch: dict) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=batch["lm_input_ids"],
            attention_mask=batch["lm_attention_mask"],
            labels=batch["lm_labels"],
            use_cache=False,
        )
        return outputs.loss

    def compute_losses(self, batch: dict) -> dict[str, torch.Tensor]:
        cfg = self.cfg.objectives
        device = batch["view_input_ids"].device
        zero = torch.zeros((), device=device)
        losses = {"lm": zero, "view": zero, "sigreg": zero}
        apply_view = True
        if self.training and cfg.jepa_dropout > 0:
            apply_view = torch.rand((), device=device) >= cfg.jepa_dropout

        if apply_view and cfg.view.weight > 0:
            pred, target = self.view_forward(batch)
            valid = batch["valid_nodes"]
            level_ids = batch["level_ids"]
            view_terms = []
            for i, _level in enumerate(LEVEL_NAMES):
                sel = valid & (level_ids == i)
                if sel.any():
                    view_terms.append(pairwise_distance(pred[sel], target[sel], cfg.view.distance))
            if view_terms:
                losses["view"] = torch.stack(view_terms).mean()
            if cfg.sigreg.weight > 0:
                sig_terms = []
                for i, _level in enumerate(LEVEL_NAMES):
                    sel = valid & (level_ids == i)
                    if sel.sum() >= 2:
                        sig_terms.append(self.sigreg(target[sel].float()))
                if sig_terms:
                    losses["sigreg"] = torch.stack(sig_terms).mean()

        if cfg.lm.weight > 0:
            losses["lm"] = self.lm_forward(batch)

        losses["total"] = (
            cfg.lm.weight * losses["lm"]
            + cfg.view.weight * losses["view"]
            + cfg.sigreg.weight * losses["sigreg"]
        )
        return losses
