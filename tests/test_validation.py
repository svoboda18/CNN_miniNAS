import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, MagicMock

from nas.validation.full_training import FullTraining
from nas.model_builder import ModelBuilder


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_config(metrics=None, epochs=1):
    if metrics is None:
        metrics = ["acc_val"]
    return {
        "ValidationStrategy": {
            "type": "FullTraining",
            "optimizer": "Adam",
            "optimizer_params": {"lr": 0.01},
            "epochs": epochs,
            "metrics": metrics,
            "use_full_dataset": False,
            "trn_data": {"url": "data/mnist_train.csv", "batch": 32,
                         "max_samples": 100},
            "val_data": {"url": "data/mnist_test.csv", "batch": 32,
                         "max_samples": 50},
        }
    }


MINI_ARCH = {
    "layers": [
        {"type": "Conv2d", "channels": 3, "kernel": 3,
         "padding": 1, "padding_mode": "zeros"},
        {"type": "ReLU"},
    ],
    "last_hid_mlp": 0,
    "input_channels": 1,
    "num_classes": 10,
    "input_size": 28,
}


def _tiny_loader(n=16, num_classes=10):
    """Returns a DataLoader with n random (1,28,28) images."""
    x = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


# ─── Constructor / config parsing ───────────────────────────────────────────

class TestFullTrainingInit:

    def test_metrics_parsed(self):
        ft = FullTraining(_make_config(metrics=["acc_val", "runtime"]))
        assert "acc_val" in ft.metrics
        assert "runtime" in ft.metrics

    def test_default_metrics_is_acc_val(self):
        cfg = _make_config()
        cfg["ValidationStrategy"].pop("metrics", None)
        ft = FullTraining(cfg)
        assert ft.metrics == ["acc_val"]

    def test_raises_on_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            FullTraining(_make_config(metrics=["acc_val", "f1_score"]))

    def test_epochs_stored(self):
        ft = FullTraining(_make_config(epochs=5))
        assert ft.epochs == 5


# ─── Metrics enforcement ────────────────────────────────────────────────────

class TestMetricsEnforcement:
    """
    Validate that validate() returns exactly the requested metrics
    (plus the always-present internal keys: loss_history, model).
    """

    def _run(self, metrics):
        ft = FullTraining(_make_config(metrics=metrics))
        with patch.object(ft, "_load_data", return_value=_tiny_loader()):
            return ft.validate(MINI_ARCH)

    def test_acc_val_only_config_returns_acc_val(self):
        result = self._run(["acc_val"])
        assert "acc_val" in result

    def test_runtime_absent_when_not_requested(self):
        result = self._run(["acc_val"])
        assert "runtime" not in result

    def test_runtime_present_when_requested(self):
        result = self._run(["acc_val", "runtime"])
        assert "runtime" in result
        assert isinstance(result["runtime"], float)

    def test_loss_history_always_present(self):
        result = self._run(["acc_val"])
        assert "loss_history" in result
        assert isinstance(result["loss_history"], list)

    def test_model_always_present(self):
        result = self._run(["acc_val"])
        assert "model" in result

    def test_raises_when_acc_val_missing_from_metrics(self):
        """acc_val is required for architecture ranking."""
        with pytest.raises(ValueError, match="acc_val"):
            self._run(["runtime"])


# ─── Validate output values ──────────────────────────────────────────────────

class TestValidateOutputValues:

    def _run(self):
        ft = FullTraining(_make_config(metrics=["acc_val", "runtime"]))
        with patch.object(ft, "_load_data", return_value=_tiny_loader()):
            return ft.validate(MINI_ARCH)

    def test_acc_val_in_unit_interval(self):
        result = self._run()
        assert 0.0 <= result["acc_val"] <= 1.0

    def test_loss_history_has_correct_length(self):
        result = self._run()
        cfg = _make_config(epochs=1)
        ft = FullTraining(cfg)
        assert len(result["loss_history"]) == 1

    def test_loss_values_are_positive(self):
        result = self._run()
        for loss in result["loss_history"]:
            assert loss >= 0.0


# ─── ModelBuilder integration ────────────────────────────────────────────────

class TestModelBuilder:

    def test_build_produces_nn_module(self):
        import torch.nn as nn
        model = ModelBuilder.build_from_dict(MINI_ARCH)
        assert isinstance(model, nn.Module)

    def test_model_outputs_correct_shape(self):
        model = ModelBuilder.build_from_dict(MINI_ARCH)
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_flatten_size_positive(self):
        size = ModelBuilder.calculate_flatten_size(MINI_ARCH)
        assert size > 0
