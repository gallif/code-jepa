from code_jepa.config import ExperimentConfig


def test_dot_overrides():
    cfg = ExperimentConfig()
    cfg.apply_dot_overrides(
        [
            "objectives.sigreg.bank_size=256",
            "objectives.sigreg.weight=0.3",
            "objectives.view.levels=step,unit",
            "objectives.view.stop_grad=text",
            "train.max_steps=1200",
            "objectives.jepa_dropout=0.75",
        ]
    )
    assert cfg.objectives.sigreg.bank_size == 256
    assert cfg.objectives.sigreg.weight == 0.3
    assert cfg.objectives.view.levels == ["step", "unit"]
    assert cfg.objectives.view.stop_grad == "text"
    assert cfg.train.max_steps == 1200
    assert cfg.objectives.jepa_dropout == 0.75
