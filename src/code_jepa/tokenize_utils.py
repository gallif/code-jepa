from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from code_jepa.config import SPECIAL_NODE_TOKENS, SPECIAL_PRED_TOKENS, BackboneConfig


def special_tokens() -> list[str]:
    return list(SPECIAL_NODE_TOKENS.values()) + list(SPECIAL_PRED_TOKENS.values())


def resolve_pretrained(name: str, cache_dir: str | None) -> str:
    path = Path(name)
    if path.exists():
        return str(path)
    if not cache_dir:
        return name
    cache = Path(cache_dir)
    slug = "models--" + name.replace("/", "--")
    for hub in (cache / "hub" / slug, cache / slug):
        snaps = hub / "snapshots"
        if snaps.is_dir():
            candidates = sorted(snaps.iterdir())
            if candidates:
                return str(candidates[-1])
    return name


def load_tokenizer(backbone: BackboneConfig) -> PreTrainedTokenizerBase:
    source = resolve_pretrained(backbone.name, backbone.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=backbone.cache_dir,
        local_files_only=backbone.local_files_only or Path(source).exists(),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens()})
    return tokenizer
