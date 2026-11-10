#!/usr/bin/env python3
"""Queue a jobs spec (default configs/sweep_jobs.json) on every idle tzag GPU with >=40GB free.

Spec keys: optional ``base_config``, optional ``runs_subdir`` (default "sweep"), ``common`` overrides,
and ``jobs`` with ``name``, optional per-job ``config`` (overrides base_config) and ``overrides``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser()
_ap.add_argument("--spec", default="configs/sweep_jobs.json")
_ap.add_argument("--gpus", default=None, help="comma-separated GPU indices this driver may use, e.g. 0,1")
_ap.add_argument("--min-free-mib", type=int, default=40000, help="free memory needed to start a job")
_ap.add_argument("--allow-foreign", action="store_true", help="start jobs on GPUs that other processes also use")
_ARGS = _ap.parse_args()
SPEC = json.loads((ROOT / _ARGS.spec).read_text())
ALLOWED = None if _ARGS.gpus is None else {int(g) for g in _ARGS.gpus.split(",") if g.strip()}
STORAGE = Path(os.environ.get("CODE_JEPA_ROOT", "/mnt/storage/users/glifshitz/code-jepa"))
RUNS = STORAGE / "runs" / SPEC.get("runs_subdir", "sweep")
PYTHON = Path(os.environ.get("CODE_JEPA_ENV", STORAGE / "env")) / "bin" / "python"
TRAIN = ROOT / "scripts" / "train.py"
FREE_MIB = _ARGS.min_free_mib


def nvidia_free() -> list[tuple[int, int]]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        idx, free = [part.strip() for part in line.split(",")]
        if ALLOWED is None or int(idx) in ALLOWED:
            rows.append((int(idx), int(free)))
    return rows


def foreign_busy(own_pids: set[int]) -> set[int]:
    """GPU indices running a compute process we did not launch (other users, or stray jobs)."""
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    )
    idx_of = {u.strip(): int(i) for i, u in (line.split(",") for line in uuids.strip().splitlines())}
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    busy = set()
    for line in apps.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid = [part.strip() for part in line.split(",")]
        if int(pid) not in own_pids and uuid in idx_of:
            busy.add(idx_of[uuid])
    return busy


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
        str(ROOT / job.get("config", SPEC.get("base_config"))),
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
        own = {p.pid for p in slots.values() if alive(p)}
        foreign = foreign_busy(own)
        for idx, free in nvidia_free():
            if free < FREE_MIB or (idx in foreign and not _ARGS.allow_foreign and not alive(slots.get(idx))):
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
