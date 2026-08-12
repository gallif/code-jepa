"""Trainable program encoder (warm-start from a code LM) for JEPA Exp 1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class EncoderOutput:
    token_hidden: torch.Tensor  # [B, T, D]
    attention_mask: torch.Tensor  # [B, T]
    pooled: torch.Tensor  # [B, D]


class ProgramEncoder(nn.Module):
    """
    HF code LM backbone used as a bidirectional-ish feature encoder.

    We take mid/last-layer hidden states as per-token program features (V-JEPA-style;
    no CLS). Causal mask of the pretrained LM remains unless the model supports
    full attention; for POC we still use teacher-forcing features on the given
    ``input_ids`` / ``position_ids``.
    """

    def __init__(
        self,
        model_name: str,
        layer: int | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.hidden_size = int(self.backbone.config.hidden_size)
        n_layers = int(getattr(self.backbone.config, "num_hidden_layers", 12))
        # None / -1 → last layer; hidden_states[0] is embeddings
        if layer is None or layer < 0:
            self.layer = n_layers
        else:
            self.layer = min(layer, n_layers)

    def tokenize(
        self,
        texts: list[str],
        max_length: int = 512,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        tok = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        if device is not None:
            tok = {k: v.to(device) for k, v in tok.items()}
        return tok

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> EncoderOutput:
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        out = self.backbone(**kwargs)
        hs = out.hidden_states[self.layer]  # [B, T, D]
        mask = attention_mask
        masked = hs.float() * mask.unsqueeze(-1)
        pooled = masked.sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
        return EncoderOutput(token_hidden=hs, attention_mask=mask, pooled=pooled)

    def encode_texts(
        self,
        texts: list[str],
        max_length: int = 512,
        device: torch.device | str | None = None,
    ) -> EncoderOutput:
        tok = self.tokenize(texts, max_length=max_length, device=device)
        return self.forward(tok["input_ids"], tok["attention_mask"])
