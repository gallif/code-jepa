from pathlib import Path

import torch

from code_jepa.data.dataset import HierarchicalJsonlDataset, collate_packed
from code_jepa.model import HierarchicalCodeJEPA, build_tiny_backbone

FIXTURE = Path(__file__).resolve().parent / "fixtures/sample.jsonl"


def test_export_roundtrip_matches_model(cfg, tokenizer, tmp_path):
    import copy
    import json

    from code_jepa.encode import CodeJEPAEncoder, export_model

    c = copy.deepcopy(cfg)
    torch.manual_seed(0)
    model = HierarchicalCodeJEPA(c, tokenizer, backbone=build_tiny_backbone(len(tokenizer)))
    model.eval()
    ckpt = tmp_path / "last.pt"
    torch.save({"model": model.state_dict(), "step": 7}, ckpt)
    cfg_path = tmp_path / "cfg.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(c.to_dict()))
    # export needs the tiny backbone too: patch the loader by saving it where the config points
    tiny_dir = tmp_path / "tiny"
    model.backbone.save_pretrained(tiny_dir)
    tokenizer.save_pretrained(tiny_dir)
    c2 = copy.deepcopy(c)
    c2.backbone.name, c2.backbone.cache_dir, c2.backbone.local_files_only = str(tiny_dir), None, True
    cfg_path.write_text(yaml.safe_dump(c2.to_dict()))
    out = export_model(cfg_path, ckpt, tmp_path / "release", {"variant": "test"})
    assert json.loads((out / "metadata.json").read_text())["step"] == 7

    enc = CodeJEPAEncoder.from_pretrained(out, device="cpu", dtype="fp32")
    rec = json.loads(FIXTURE.read_text().splitlines()[0])
    feats = enc.encode([rec])[0]
    ds = HierarchicalJsonlDataset(FIXTURE, tokenizer, c, split=None)
    batch = collate_packed([ds[0]], tokenizer.pad_token_id)
    with torch.no_grad():
        code, _ = model.view_embeddings(batch)
    want = code[0, batch["valid_nodes"][0]]
    got = torch.stack([n["head"] for n in feats["nodes"]])
    assert [n["level"] for n in feats["nodes"]] == list(ds[0].node_levels)
    assert torch.allclose(got, want.float(), atol=1e-4), (got - want).abs().max()
    # descriptions do not affect code features
    feats_t = enc.encode([rec], with_text=True)[0]
    assert torch.allclose(torch.stack([n["head"] for n in feats_t["nodes"]]), got, atol=1e-4)
    assert "text_head" in feats_t["nodes"][0]
