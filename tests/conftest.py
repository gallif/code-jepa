from pathlib import Path

import pytest

from code_jepa.config import ExperimentConfig
from code_jepa.tokenize_utils import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/sample.jsonl"
CACHE = ROOT / "models/hf"


@pytest.fixture(scope="session")
def cfg() -> ExperimentConfig:
    cfg = ExperimentConfig.load(ROOT / "configs/default.yaml")
    cfg.backbone.cache_dir = str(CACHE)
    cfg.backbone.local_files_only = True
    cfg.objectives.jepa_dropout = 0.0
    cfg.backbone.grad_checkpointing = False
    cfg.backbone.dtype = "fp32"
    return cfg


@pytest.fixture(scope="session")
def tokenizer(cfg):
    return load_tokenizer(cfg.backbone)
