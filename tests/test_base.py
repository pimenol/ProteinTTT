import pytest
from pathlib import Path
from proteinttt.base import TTTConfig
import yaml

@pytest.fixture
def temp_config_yaml(tmp_path):
    config_data = {
        "lr": 0.001,
        "steps": 10,
        "msa": True,
        "score_seq_kind": "pseudo_perplexity",
        "score_seq_steps_list": [1, 5, 10]
    }
    yaml_path = tmp_path / "test_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config_data, f)
    return yaml_path

def test_tttconfig_from_yaml(temp_config_yaml):
    config = TTTConfig.from_yaml(temp_config_yaml)
    assert config.lr == 0.001
    assert config.steps == 10
    assert config.msa is True
    assert config.score_seq_kind == "pseudo_perplexity"
    assert config.score_seq_steps_list == [1, 5, 10]

def test_tttconfig_verify_valid_config():
    config = TTTConfig(
        score_seq_kind="pseudo_perplexity",
        score_seq_steps_list=[1, 2],
        msa=True,
        loss_kind="cross_entropy"
    )
    config.verify() # Should not raise any error

def test_tttconfig_verify_invalid_score_seq_steps_list():
    config = TTTConfig(score_seq_steps_list="invalid")
    with pytest.raises(ValueError, match="score_seq_steps_list must be None, an integer, or a list of integers"):
        config.verify()

    config = TTTConfig(score_seq_steps_list=[1, "invalid"])
    with pytest.raises(ValueError, match="All elements in score_seq_steps_list must be integers"):
        config.verify()

def test_tttconfig_verify_perplexity_early_stopping_without_score_seq_kind():
    config = TTTConfig(perplexity_early_stopping=0.5, score_seq_kind=None)
    with pytest.raises(ValueError, match="perplexity_early_stopping can only be used if score_seq_kind is not None"):
        config.verify()

def test_tttconfig_verify_msa_soft_labels_without_msa():
    config = TTTConfig(loss_kind="msa_soft_labels", msa=False)
    with pytest.raises(ValueError, match="msa_soft_labels loss kind can only be used if msa=True"):
        config.verify()

def test_tttconfig_verify_lora_without_installation(monkeypatch):
    monkeypatch.setattr("proteinttt.base.inject_trainable_lora", None)
    config = TTTConfig(lora_rank=8)
    with pytest.raises(ImportError, match="lora_diffusion is not installed"):
        config.verify()
