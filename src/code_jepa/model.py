from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from code_jepa.config import ExperimentConfig, resolve_stop_grad
from code_jepa.losses import EmbeddingBank, SIGReg, TargetBank, infonce_loss, pairwise_distance
from code_jepa.tokenize_utils import resolve_pretrained, special_tokens

LEVEL_NAMES = ("step", "unit", "approach", "intent")
# Which embeddings SIGReg regularises. "code" = node-token projections, "text" = predictor-token
# projections, both un-detached regardless of view.direction and view.stop_grad. "target" is the
# pre-2026-11-08 behaviour (whatever the view target is, detached if that side is stop-grad).
SIGREG_SIDES = {"code": ("code",), "text": ("text",), "both": ("code", "text"), "target": ("target",)}


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
        # Code side always uses self.heads; text side uses its own heads when not shared.
        self.text_heads = (
            None
            if cfg.projection_head.shared
            else nn.ModuleDict(
                {level: ProjectionHead(hidden, out_dim, cfg.projection_head.layers) for level in LEVEL_NAMES}
            )
        )
        self.stop_grad = resolve_stop_grad(cfg.objectives.view)
        self.sigreg = SIGReg(knots=cfg.objectives.sigreg.knots, num_proj=cfg.objectives.sigreg.num_proj)
        sig_on = cfg.objectives.sigreg.side
        if sig_on not in SIGREG_SIDES:
            raise ValueError(f"objectives.sigreg.side must be one of {sorted(SIGREG_SIDES)}, got {sig_on!r}")
        target_is_text = cfg.objectives.view.direction == "code_to_text"
        if sig_on == "target" and target_is_text and self.stop_grad == "text" and cfg.objectives.sigreg.weight > 0:
            warnings.warn(
                "sigreg.side=target with a detached text target: SIGReg sees a detached tensor "
                "and contributes no gradient.",
                stacklevel=2,
            )
        self.banks = {
            (side, level): EmbeddingBank(cfg.objectives.sigreg.bank_size)
            for side in SIGREG_SIDES[sig_on]
            for level in LEVEL_NAMES
        }
        if cfg.objectives.view.center not in ("none", "ema"):
            raise ValueError(f"view.center must be none | ema, got {cfg.objectives.view.center!r}")
        self.centers: dict[tuple[str, str], torch.Tensor] = {}
        self.target_banks = {level: TargetBank(cfg.objectives.view.infonce_bank) for level in LEVEL_NAMES}
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

    def _project(
        self,
        vectors: torch.Tensor,
        level_ids: torch.Tensor,
        valid: torch.Tensor,
        heads: nn.ModuleDict | None = None,
    ) -> torch.Tensor:
        heads = self.heads if heads is None else heads
        out = vectors.new_zeros(*vectors.shape[:2], self.cfg.projection_head.dim)
        for i, level in enumerate(LEVEL_NAMES):
            sel = valid & (level_ids == i)
            if sel.any():
                out[sel] = heads[level](vectors[sel].float()).to(out.dtype)
        return out

    def view_embeddings(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Projected (code, text) embeddings per node, both with gradient."""
        hidden = self._forward_hidden(
            batch["view_input_ids"],
            {"full_attention": batch["view_attention_mask"]},
            batch["view_position_ids"],
        )
        valid = batch["valid_nodes"]
        code = self._project(self._gather(hidden, batch["node_pos"]), batch["level_ids"], valid)
        text = self._project(
            self._gather(hidden, batch["pred_pos"]), batch["level_ids"], valid, heads=self.text_heads
        )
        return code, text

    def _orient(self, code: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(prediction, target) for the view loss. Only the text side is ever detached."""
        if self.stop_grad == "text":
            text = text.detach()
        if self.cfg.objectives.view.direction == "code_to_text":
            return code, text
        return text, code

    def _center(self, x: torch.Tensor, key: tuple[str, str]) -> torch.Tensor:
        """Subtract a detached running mean for (side, level); update it in training."""
        mom = self.cfg.objectives.view.center_momentum
        batch_mean = x.detach().float().mean(dim=0)
        mean = self.centers.get(key)
        if mean is None:
            mean = batch_mean
        if self.training:
            self.centers[key] = (mom * mean + (1 - mom) * batch_mean) if key in self.centers else batch_mean
        return x - mean.to(x.dtype)

    def view_forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self._orient(*self.view_embeddings(batch))

    def lm_forward(self, batch: dict) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=batch["lm_input_ids"],
            attention_mask=batch["lm_attention_mask"],
            labels=batch["lm_labels"],
            use_cache=False,
        )
        return outputs.loss

    def _level_active(self, name: str) -> bool:
        allowed = self.cfg.objectives.view.levels
        return allowed is None or name in allowed

    def compute_losses(
        self, batch: dict, with_embeddings: bool = False, grad_diag: bool = False
    ) -> dict[str, torch.Tensor]:
        """Loss terms. ``grad_diag`` adds ``losses["grad_diag"]``: mean per-node L2 norm of the
        gradient of each weighted term (view, sigreg) w.r.t. the projected code / text embeddings,
        per level (``gn_{term}_{side}_{level}``). Cheap: it only backprops through the heads' outputs."""
        cfg = self.cfg.objectives
        device = batch["view_input_ids"].device
        zero = torch.zeros((), device=device)
        losses: dict[str, torch.Tensor] = {"lm": zero, "view": zero, "sigreg": zero}
        apply_view = True
        if self.training and cfg.jepa_dropout > 0:
            apply_view = torch.rand((), device=device) >= cfg.jepa_dropout
        need_view = with_embeddings or (apply_view and (cfg.view.weight > 0 or cfg.sigreg.weight > 0))

        pred = target = None
        if need_view:
            code, text = self.view_embeddings(batch)
            pred, target = self._orient(code, text)
            valid = batch["valid_nodes"]
            level_ids = batch["level_ids"]
            view_terms = []
            for i, level in enumerate(LEVEL_NAMES):
                if not self._level_active(level):
                    continue
                sel = valid & (level_ids == i)
                if sel.any():
                    p_l, c_l = pred[sel], target[sel]
                    if cfg.view.distance == "infonce":
                        bank = self.target_banks[level]
                        term = infonce_loss(p_l, c_l, bank.get(), cfg.view.temperature)
                        if self.training:
                            bank.push(c_l)
                    else:
                        if cfg.view.center == "ema":
                            p_l = self._center(p_l, ("pred", level))
                            c_l = self._center(c_l, ("target", level))
                        term = pairwise_distance(p_l, c_l, cfg.view.distance)
                    losses[f"view_{level}"] = term
                    view_terms.append(term)
            if view_terms:
                losses["view"] = torch.stack(view_terms).mean()
            if cfg.sigreg.weight > 0:
                sides = {"code": code, "text": text, "target": target}
                sig_terms = []
                for side in SIGREG_SIDES[cfg.sigreg.side]:
                    emb = sides[side]
                    for i, level in enumerate(LEVEL_NAMES):
                        if not self._level_active(level):
                            continue
                        sel = valid & (level_ids == i)
                        current = emb[sel].float() if sel.any() else emb.new_zeros((0, emb.size(-1)))
                        if cfg.sigreg.bank_size > 0:
                            term = self.banks[(side, level)].regularize(current, self.sigreg, self.training)
                        elif sel.sum() >= 2:
                            term = self.sigreg(current)
                        else:
                            continue
                        sig_terms.append(term)
                if sig_terms:
                    losses["sigreg"] = torch.stack(sig_terms).mean()
            if grad_diag and torch.is_grad_enabled():
                diag: dict[str, float] = {}
                weighted = (("view", cfg.view.weight * losses["view"]), ("sigreg", cfg.sigreg.weight * losses["sigreg"]))
                for term_name, term in weighted:
                    grads = (None, None)
                    if term.requires_grad:
                        grads = torch.autograd.grad(term, [code, text], retain_graph=True, allow_unused=True)
                    for side, g in zip(("code", "text"), grads):
                        for i, level in enumerate(LEVEL_NAMES):
                            sel = valid & (level_ids == i)
                            if not sel.any():
                                continue
                            val = 0.0 if g is None else float(g[sel].float().norm(dim=-1).mean())
                            diag[f"gn_{term_name}_{side}_{level}"] = val
                losses["grad_diag"] = diag
            if with_embeddings:
                losses["pred"] = pred
                losses["target"] = target
                losses["code_emb"] = code
                losses["text_emb"] = text

        if cfg.lm.weight > 0:
            losses["lm"] = self.lm_forward(batch)

        losses["total"] = (
            cfg.lm.weight * losses["lm"]
            + cfg.view.weight * losses["view"]
            + cfg.sigreg.weight * losses["sigreg"]
        )
        return losses
