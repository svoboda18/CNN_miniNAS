import os
import tempfile
import pytest
import yaml

from nas.io import load_config, save_results, generate_code


# ─── Helpers ────────────────────────────────────────────────────────────────

def _write_yaml(path: str, data: dict):
    with open(path, "w") as f:
        yaml.dump(data, f)


MINIMAL_CONFIG = {
    "SearchSpace": {
        "type": "LayersBased",
        "layers_types": ["Conv2d", "ReLU"],
        "layers_count": [2],
        "channels": [3],
        "kernel": [3],
        "padding": [1],
        "last_hid_mlp": [0],
        "dropout_rates": [0.1],
    },
    "SearchStrategy": {"type": "RandomSearch", "nbr_iterations": 2},
    "ValidationStrategy": {
        "type": "FullTraining",
        "optimizer": "Adam",
        "optimizer_params": {"lr": 0.01},
        "epochs": 1,
        "metrics": ["acc_val"],
        "trn_data": {"url": "data/mnist_train.csv", "batch": 32},
        "val_data": {"url": "data/mnist_test.csv", "batch": 32},
    },
}


# ─── load_config ────────────────────────────────────────────────────────────

class TestLoadConfig:

    def test_loads_valid_config(self, tmp_path):
        p = str(tmp_path / "cfg.yaml")
        _write_yaml(p, MINIMAL_CONFIG)
        cfg = load_config(p)
        assert "SearchSpace" in cfg
        assert "SearchStrategy" in cfg
        assert "ValidationStrategy" in cfg

    def test_returns_dict_with_correct_types(self, tmp_path):
        p = str(tmp_path / "cfg.yaml")
        _write_yaml(p, MINIMAL_CONFIG)
        cfg = load_config(p)
        assert isinstance(cfg["SearchSpace"]["layers_count"], list)
        assert isinstance(cfg["SearchStrategy"]["nbr_iterations"], int)

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/cfg.yaml")

    def test_raises_on_missing_top_level_key(self, tmp_path):
        bad = dict(MINIMAL_CONFIG)
        del bad["SearchStrategy"]
        p = str(tmp_path / "bad.yaml")
        _write_yaml(p, bad)
        with pytest.raises(ValueError, match="SearchStrategy"):
            load_config(p)

    def test_raises_on_missing_searchspace_field(self, tmp_path):
        import copy
        bad = copy.deepcopy(MINIMAL_CONFIG)
        del bad["SearchSpace"]["kernel"]
        p = str(tmp_path / "bad.yaml")
        _write_yaml(p, bad)
        with pytest.raises(ValueError, match="kernel"):
            load_config(p)

    def test_raises_on_unknown_layer_type(self, tmp_path):
        import copy
        bad = copy.deepcopy(MINIMAL_CONFIG)
        bad["SearchSpace"]["layers_types"] = ["Conv2d", "LSTM"]
        p = str(tmp_path / "bad.yaml")
        _write_yaml(p, bad)
        with pytest.raises(ValueError, match="LSTM"):
            load_config(p)

    def test_raises_on_invalid_nbr_iterations(self, tmp_path):
        import copy
        bad = copy.deepcopy(MINIMAL_CONFIG)
        bad["SearchStrategy"]["nbr_iterations"] = -1
        p = str(tmp_path / "bad.yaml")
        _write_yaml(p, bad)
        with pytest.raises(ValueError, match="nbr_iterations"):
            load_config(p)


# ─── save_results / generate_code ───────────────────────────────────────────

MINI_ARCH = {
    "layers": [
        {"type": "Conv2d", "channels": 3, "kernel": 3, "padding": 1,
         "padding_mode": "zeros"},
        {"type": "ReLU"},
    ],
    "last_hid_mlp": 0,
    "input_channels": 1,
    "num_classes": 10,
    "input_size": 28,
}


class TestSaveResults:

    def _make_result(self, acc: float = 0.80):
        return {
            "acc_val": acc,
            "loss_history": [1.0, 0.8, 0.6],
            "model": None,
            "architecture": MINI_ARCH,
        }

    def test_creates_accuracies_txt(self, tmp_path):
        save_results(str(tmp_path), [self._make_result()])
        assert (tmp_path / "accuracies.txt").exists()

    def test_accuracies_txt_content(self, tmp_path):
        save_results(str(tmp_path), [self._make_result(0.91), self._make_result(0.85)])
        text = (tmp_path / "accuracies.txt").read_text()
        assert "0.9100" in text
        assert "0.8500" in text

    def test_creates_loss_history_plot(self, tmp_path):
        save_results(str(tmp_path), [self._make_result()])
        assert (tmp_path / "model_0" / "loss_history.png").exists()

    def test_creates_best_model_code(self, tmp_path):
        save_results(str(tmp_path), [self._make_result(0.7), self._make_result(0.9)])
        code_file = tmp_path / "best_model_code.py"
        assert code_file.exists()
        code = code_file.read_text()
        assert "class BestModel" in code
        assert "nn.Conv2d" in code

    def test_placeholder_written_when_model_is_none(self, tmp_path):
        save_results(str(tmp_path), [self._make_result()])
        assert (tmp_path / "model_0" / "model_not_saved.txt").exists()


class TestGenerateCode:

    def test_output_has_import_and_class(self):
        code = generate_code(MINI_ARCH)
        assert "import torch" in code
        assert "class BestModel" in code

    def test_conv_layer_present(self):
        code = generate_code(MINI_ARCH)
        assert "nn.Conv2d" in code

    def test_relu_layer_present(self):
        code = generate_code(MINI_ARCH)
        assert "nn.ReLU" in code

    def test_last_hid_zero_has_single_linear(self):
        code = generate_code(MINI_ARCH)
        assert code.count("nn.Linear") == 1

    def test_last_hid_nonzero_has_two_linears(self):
        arch = dict(MINI_ARCH, last_hid_mlp=64)
        code = generate_code(arch)
        assert code.count("nn.Linear") == 2
