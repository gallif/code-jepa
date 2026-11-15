"""Dedicated code-embedding baselines (benchmark 4): whole-snippet vectors for code and queries.

  emb = load_external("unixcoder"); C = emb.code(list_of_code); Q = emb.query(list_of_text)   # L2-normalised
"""
from __future__ import annotations

import numpy as np
import torch

SPECS = {
    # name: (hf id, pooling, query prefix, code prefix, max_len, trust_remote_code)
    "unixcoder": ("microsoft/unixcoder-base", "mean", "", "", 512, False),
    "gte-modernbert": ("Alibaba-NLP/gte-modernbert-base", "cls", "", "", 1024, False),
}


class External:
    def __init__(self, name: str, device="cuda"):
        from transformers import AutoModel, AutoTokenizer

        hf, self.pool, self.qp, self.cp, self.max_len, trc = SPECS[name]
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(hf, trust_remote_code=False).to(device).eval()
        if device == "cuda":
            self.model.half()
        self.dev = device

    @torch.no_grad()
    def _embed(self, texts: list[str], bs: int = 32) -> np.ndarray:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for s in range(0, len(order), bs):
            b = order[s : s + bs]
            enc = self.tok([texts[i] for i in b], padding=True, truncation=True, max_length=self.max_len,
                           return_tensors="pt").to(self.dev)
            h = self.model(**enc).last_hidden_state.float()
            if self.pool == "cls":
                v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
            v = torch.nn.functional.normalize(v, dim=-1)
            for r, i in enumerate(b):
                out[i] = v[r].cpu().numpy()
        return np.stack(out)

    def code(self, codes: list[str]) -> np.ndarray:
        return self._embed([self.cp + c for c in codes])

    def query(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.qp + t for t in texts])


def load_external(name: str, device="cuda") -> External:
    return External(name, device)
