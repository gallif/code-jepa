from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class TeacherOutput:
    pooled: torch.Tensor  # [B, D]
    token_hidden: torch.Tensor  # [B, T, D]
    attention_mask: torch.Tensor


class FrozenTeacher:
    """Frozen code LLM; extract mid-layer hidden states."""

    def __init__(self, model_name: str, layer: int, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layer = layer
        self.device = device
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = getattr(self.model.config, "num_hidden_layers", None)

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        max_length: int = 512,
    ) -> TeacherOutput:
        tok = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = self.model(**tok, output_hidden_states=True, use_cache=False)
        # hidden_states[0] = embed; [1] = after layer 1; ...
        hs = out.hidden_states[self.layer]  # [B, T, D]
        mask = tok["attention_mask"]
        masked = hs.float() * mask.unsqueeze(-1)
        pooled = masked.sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
        return TeacherOutput(pooled=pooled, token_hidden=hs, attention_mask=mask)
