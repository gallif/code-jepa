#!/usr/bin/env bash
# Score HumanEval(+) and MBPP(+) generations with EvalPlus (sanitize -> evaluate), greedy pass@1.
#   bash bench/score_evalplus.sh E0 E1 tree free
set -euo pipefail
ROOT=/mnt/storage/users/glifshitz/code-jepa
PY=$ROOT/envs/evalplus/bin/python
export XDG_CACHE_HOME=$ROOT/cache/xdg TMPDIR=$ROOT/tmp
for m in "$@"; do
  D=$ROOT/runs/bench/gen/$m
  for ds in humaneval mbpp; do
    $PY - "$D/$ds.jsonl" "$D/${ds}_samples.jsonl" <<'EOF'
import json, sys
with open(sys.argv[2], "w") as f:
    for l in open(sys.argv[1]):
        r = json.loads(l)
        f.write(json.dumps({"task_id": r["task_id"], "solution": r["prompt"] + r["completion"]}) + "\n")
EOF
    $PY -m evalplus.sanitize --samples "$D/${ds}_samples.jsonl" >/dev/null 2>&1
    rm -f "$D/${ds}_samples-sanitized_eval_results.json"
    $PY -m evalplus.evaluate --dataset $ds --samples "$D/${ds}_samples-sanitized.jsonl" --i-just-wanna-run \
        > "$D/${ds}_evalplus.log" 2>&1 || true
  done
  $PY - "$D" <<'EOF'
import json, sys, glob, re
d = sys.argv[1]; out = {}
for ds in ("humaneval", "mbpp"):
    log = open(f"{d}/{ds}_evalplus.log").read()
    base = re.search(r"\(base tests\)\s*pass@1:\s*([0-9.]+)", log)
    plus = re.search(r"\(base \+ extra tests\)\s*pass@1:\s*([0-9.]+)", log)
    out[ds] = {"pass@1": float(base.group(1)) if base else None}
    out[ds + "+"] = {"pass@1": float(plus.group(1)) if plus else None}
json.dump(out, open(f"{d}/evalplus.json", "w"), indent=1)
print(d, out)
EOF
done
