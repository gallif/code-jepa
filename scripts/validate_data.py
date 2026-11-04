#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter

from code_jepa.config import ExperimentConfig
from code_jepa.data.packer import pack_record
from code_jepa.tokenize_utils import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate hierarchical JSONL and report token lengths")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    cfg = ExperimentConfig.load(args.config)
    tokenizer = load_tokenizer(cfg.backbone)
    n = 0
    dropped = 0
    warn = 0
    view_lens = []
    node_counts = []
    reasons = Counter()
    with open(args.data) as handle:
        for line in handle:
            if not line.strip():
                continue
            packed = pack_record(json.loads(line), tokenizer, cfg)
            n += 1
            node_counts.append(len(packed.nodes))
            if packed.dropped:
                dropped += 1
                reasons[packed.drop_reason] += 1
            else:
                view_lens.append(len(packed.view_input_ids))
            warn += len(packed.warnings)
            if args.max_samples and n >= args.max_samples:
                break
    view_lens.sort()
    node_counts.sort()

    def pct(xs, p):
        if not xs:
            return None
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    print(
        json.dumps(
            {
                "records": n,
                "dropped": dropped,
                "drop_reasons": dict(reasons),
                "warnings": warn,
                "nodes_median": pct(node_counts, 0.5),
                "nodes_p99": pct(node_counts, 0.99),
                "view_tokens_median": pct(view_lens, 0.5),
                "view_tokens_p99": pct(view_lens, 0.99),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
