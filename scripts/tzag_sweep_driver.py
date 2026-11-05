#!/usr/bin/env python3
"""Run configs/sweep_jobs.json on every idle tzag GPU with >=40GB free."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "configs/sweep_jobs.json").read_text())
STORAGE = Path(os.environ.get("CODE_JEPA_ROOT", "/mnt/storage/users/glifshitz/code-jepa"))
RUNS = STORAGE / "runs" / "sweep"
PYTHON = Path(os.environ.get("CODE_JEPA_ENV", STORAGE / "env")) / "bin" / "python"
TRAIN = ROOT / "scripts" / "train.py"
FREE_MIB = 40000


def nvidia_free() -> list[tuple[int, int]]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        idx, free = [part.strip() for part in line.split(",")]
        rows.append((int(idx), int(free)))
    return rows


def alive(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def launch(gpu: int, job: dict) -> subprocess.Popen:
    name = job["name"]
    out_dir = RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = list(SPEC.get("common", [])) + list(job.get("overrides", []))
    overrides.append(f"train.output_dir={out_dir}")
    cmd = [
        str(PYTHON),
        str(TRAIN),
        "--config",
        str(ROOT / SPEC["base_config"]),
        "--override",
        *overrides,
    ]
    log = (RUNS / f"{name}.stdout.log").open("w")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[sweep] GPU {gpu} start {name}", flush=True)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    queue = list(SPEC["jobs"])
    slots: dict[int, subprocess.Popen | None] = {}
    print(f"[sweep] {len(queue)} jobs -> {RUNS}", flush=True)
    while queue or any(alive(p) for p in slots.values()):
        for idx, free in nvidia_free():
            if free < FREE_MIB:
                if idx in slots and not alive(slots[idx]):
                    print(f"[sweep] GPU {idx} job exited rc={slots[idx].returncode}", flush=True)
                    slots[idx] = None
                continue
            if alive(slots.get(idx)):
                continue
            if idx in slots and slots[idx] is not None and not alive(slots[idx]):
                print(f"[sweep] GPU {idx} job exited rc={slots[idx].returncode}", flush=True)
            if not queue:
                slots[idx] = None
                continue
            job = queue.pop(0)
            slots[idx] = launch(idx, job)
        time.sleep(20)
    print("[sweep] all jobs finished", flush=True)


if __name__ == "__main__":
    main()
