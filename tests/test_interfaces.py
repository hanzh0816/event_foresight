"""Public entry points use private dependencies and explicit configuration."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from foresight.checkpoints import load_predictor, TRAINING_FORMAT
from foresight.config import ROOT, load_config
from foresight.evaluate import read_sample_ids


def test_imports_from_unrelated_directory_without_unused_dependencies(tmp_path):
    script = '''
import importlib.abc
import sys
class BlockUnusedDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'dinov2', 'dsec_data', 'einops'}:
            raise AssertionError('forbidden runtime dependency: '+fullname)
sys.meta_path.insert(0,BlockUnusedDependencies())
import foresight.infer
assert 'lightning' not in sys.modules
import foresight.evaluate
import foresight.train
from foresight.config import load_config
assert load_config('configs/detector.yaml')['model']['encoder_checkpoint'].endswith('dinov2_vitb14_reg4_pretrain.pth')
'''
    # Bound native thread pools in this fresh process, as in the CPU model tests.
    env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_config_paths_are_project_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config("configs/detector.yaml")
    assert Path(config["model"]["encoder_checkpoint"]).is_absolute()
    assert config["data"]["image_size"] == [448, 896]
    assert config["runtime"]["encoder_chunk_size"] == 1


@pytest.mark.parametrize("key", ["detector_checkpoint", "dinof_checkpoint", "unknown_option"])
def test_config_rejects_unsupported_model_options(tmp_path, key):
    config = {"model": {"encoder_checkpoint": "unused.pth", key: None}}
    path = tmp_path / "unsupported.yaml"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=key):
        load_config(path)


def test_manifest_preserves_order_and_rejects_duplicates(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"val": {"sample_ids": ["sequence:41", "sequence:39"]}}))
    assert read_sample_ids(path, "val") == ["sequence:41", "sequence:39"]
    path.write_text(json.dumps(["a", "a"]))
    with pytest.raises(ValueError, match="unique"):
        read_sample_ids(path, "val")


def test_predictor_load_preserves_frozen_coordinate_contract(tmp_path):
    holder = nn.Module()
    holder.encoder = nn.Linear(2, 2)
    holder.architecture = {"name": "tiny-world"}
    holder.predictor = nn.Linear(1, 1)
    state = {f"forecaster.{key}": value.clone() for key, value in holder.state_dict().items()}
    state['forecaster.predictor.bias'] = state['forecaster.predictor.bias'] + 1
    saved = dict(foresight_format=TRAINING_FORMAT, stage="predictor", state_dict=state,
                 forecaster_architecture=deepcopy(holder.architecture))
    path = tmp_path / "predictor.ckpt"
    torch.save(saved, path)
    expected = state['forecaster.predictor.bias']
    load_predictor(holder.predictor, path, forecaster=holder)
    assert torch.equal(holder.predictor.bias, expected)
    before = deepcopy(holder.predictor.state_dict())
    state['forecaster.encoder.weight'] = state['forecaster.encoder.weight'] + 1
    torch.save(saved, path)
    with pytest.raises(ValueError, match="different frozen encoder"):
        load_predictor(holder.predictor, path, forecaster=holder)
    assert all(torch.equal(before[k], v) for k, v in holder.predictor.state_dict().items())
