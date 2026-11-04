import json
from pathlib import Path

from code_jepa.data.packer import pack_record, token_lines

FIXTURE = Path(__file__).resolve().parent / "fixtures/sample.jsonl"


def _record():
    return json.loads(FIXTURE.read_text().splitlines()[0])


def test_token_line_map_covers_all_lines(tokenizer):
    code = _record()["L4_code"]
    ids, line_of = token_lines(tokenizer, code)
    assert len(ids) == len(line_of)
    assert min(line_of) == 1
    assert max(line_of) == 23
    assert len(ids) > 23


def test_pack_node_and_pred_counts(tokenizer, cfg):
    packed = pack_record(_record(), tokenizer, cfg)
    assert not packed.dropped
    assert len(packed.nodes) == 14
    assert len(packed.node_seq_pos) == 14
    assert len(packed.pred_seq_pos) == 14
    assert packed.view_mask.shape[0] == len(packed.view_input_ids)
    assert (packed.view_mask.diag() == 0).all()


def test_tree_mask_step_sees_only_its_lines(tokenizer, cfg):
    record = _record()
    packed = pack_record(record, tokenizer, cfg)
    _, line_of = token_lines(tokenizer, record["L4_code"])
    n_code = packed.extra["n_code"]
    idx = next(i for i, n in enumerate(packed.nodes) if n.node_id == "u0.b4")
    row = packed.view_mask[packed.node_seq_pos[idx]]
    allowed = (row == 0).nonzero().flatten().tolist()
    code_allowed = [j for j in allowed if j < n_code]
    assert code_allowed
    assert all(line_of[j] in (8, 9) for j in code_allowed)


def test_unit_attends_to_its_step_tokens_not_other_unit(tokenizer, cfg):
    packed = pack_record(_record(), tokenizer, cfg)
    by_id = {n.node_id: i for i, n in enumerate(packed.nodes)}
    u0 = packed.node_seq_pos[by_id["u0"]]
    u1 = packed.node_seq_pos[by_id["u1"]]
    step_u0 = packed.node_seq_pos[by_id["u0.b0"]]
    step_u1 = packed.node_seq_pos[by_id["u1.b0"]]
    row = packed.view_mask[u0]
    assert row[step_u0].item() == 0
    assert row[step_u1].item() < 0
    assert packed.view_mask[packed.node_seq_pos[by_id["intent"]], u0].item() == 0
    assert packed.view_mask[packed.node_seq_pos[by_id["intent"]], u1].item() == 0
    assert packed.view_mask[packed.node_seq_pos[by_id["intent"]], step_u0].item() < 0


def test_descriptions_are_block_diagonal(tokenizer, cfg):
    packed = pack_record(_record(), tokenizer, cfg)
    n_nodes = len(packed.nodes)
    first_pred = packed.pred_seq_pos[0]
    last_pred = packed.pred_seq_pos[-1]
    assert first_pred < last_pred
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            pi, pj = packed.pred_seq_pos[i], packed.pred_seq_pos[j]
            assert packed.view_mask[pi, packed.node_seq_pos[j]].item() < 0
            assert packed.view_mask[pi, pj].item() < 0
