import json
from pathlib import Path

from code_jepa.data.tree import flatten_record, validate_tree

FIXTURE = Path(__file__).resolve().parent / "fixtures/sample.jsonl"


def test_flatten_sample_counts():
    record = json.loads(FIXTURE.read_text().splitlines()[0])
    nodes = flatten_record(record)
    counts = {level: sum(1 for n in nodes if n.level == level) for level in ("step", "unit", "approach", "intent")}
    assert counts == {"step": 10, "unit": 2, "approach": 1, "intent": 1}
    steps = [n for n in nodes if n.level == "step"]
    assert steps[0].parent_id == "u0"
    assert steps[-1].parent_id == "u1"
    warnings = validate_tree(record["L4_code"], nodes)
    assert warnings == []


def test_nested_steps():
    record = {
        "L0_intent": "task",
        "L1_approach": {"strategy": "how", "key_data_structures": [], "complexity": "O(1)"},
        "L4_code": "def f(x):\n    if x:\n        return x\n    return 0\n",
        "L2_units": [
            {
                "id": "u0",
                "span": [1, 4],
                "responsibility": "f",
                "contract": "int to int",
                "methods": [],
                "steps": [
                    {
                        "id": "u0.b0",
                        "span": [2, 3],
                        "text": "if x return x",
                        "kind": "if",
                        "children": [
                            {"id": "u0.b0.c0", "span": [3, 3], "text": "return x", "kind": "simple", "children": []}
                        ],
                    },
                    {"id": "u0.b1", "span": [4, 4], "text": "return 0", "kind": "simple", "children": []},
                ],
            }
        ],
    }
    nodes = flatten_record(record)
    steps = [n for n in nodes if n.level == "step"]
    parent = next(n for n in steps if n.node_id == "u0.b0")
    child = next(n for n in steps if n.node_id == "u0.b0.c0")
    assert child.parent_id == "u0.b0"
    assert "u0.b0.c0" in parent.children_ids
    assert not parent.is_leaf_step
    assert child.is_leaf_step
