import pytest
from nas.search_space.layers_based import LayersBased


# ─── Helpers ────────────────────────────────────────────────────────────────

BASE_CONFIG = {
    "SearchSpace": {
        "type": "LayersBased",
        "layers_types": ["Conv2d", "ReLU", "MaxPool2d", "Dropout"],
        "layers_count": [2, 3],
        "channels": [3, 5],
        "kernel": [2, 3],
        "padding": [0, 1],
        "last_hid_mlp": [0, 50],
        "dropout_rates": [0.1, 0.3],
    }
}


def make_space(cfg=None):
    return LayersBased().define_space(cfg or BASE_CONFIG)


# ─── Basic output contract ──────────────────────────────────────────────────

class TestDefineSpaceOutput:

    def test_returns_non_empty_list(self):
        archs = make_space()
        assert isinstance(archs, list)
        assert len(archs) > 0

    def test_each_arch_has_required_keys(self):
        for arch in make_space():
            assert "layers" in arch
            assert "last_hid_mlp" in arch
            assert "input_channels" in arch
            assert "input_size" in arch

    def test_layers_is_list_of_dicts(self):
        for arch in make_space():
            assert isinstance(arch["layers"], list)
            for layer in arch["layers"]:
                assert isinstance(layer, dict)
                assert "type" in layer


# ─── Layer count constraint ──────────────────────────────────────────────────

class TestLayerCount:

    def test_all_layer_counts_respected(self):
        allowed = set(BASE_CONFIG["SearchSpace"]["layers_count"])
        for arch in make_space():
            assert len(arch["layers"]) in allowed


# ─── Validity rules ──────────────────────────────────────────────────────────

class TestValidityRules:

    def test_no_consecutive_dropout(self):
        for arch in make_space():
            layers = arch["layers"]
            for i in range(1, len(layers)):
                assert not (
                    layers[i]["type"] == "Dropout"
                    and layers[i - 1]["type"] == "Dropout"
                ), f"Consecutive dropouts found in: {[l['type'] for l in layers]}"

    def test_every_arch_has_at_least_one_conv(self):
        for arch in make_space():
            types = [l["type"] for l in arch["layers"]]
            assert "Conv2d" in types, f"No Conv2d in {types}"

    def test_spatial_size_always_positive(self):
        """All returned arches should have survived the spatial validity check."""
        from nas.search_space.layers_based import LayersBased
        validator = LayersBased._is_spatially_valid
        for arch in make_space():
            assert validator(arch), (
                f"Spatially invalid arch slipped through: "
                f"{[l['type'] for l in arch['layers']]}"
            )


# ─── Layer parameter content ─────────────────────────────────────────────────

class TestLayerParameters:

    def test_conv2d_layers_have_channel_and_kernel(self):
        for arch in make_space():
            for layer in arch["layers"]:
                if layer["type"] == "Conv2d":
                    assert "channels" in layer
                    assert "kernel" in layer
                    assert "padding" in layer

    def test_conv2d_channels_within_config(self):
        allowed = set(BASE_CONFIG["SearchSpace"]["channels"])
        for arch in make_space():
            for layer in arch["layers"]:
                if layer["type"] == "Conv2d":
                    assert layer["channels"] in allowed

    def test_dropout_rate_within_config(self):
        allowed = set(BASE_CONFIG["SearchSpace"]["dropout_rates"])
        for arch in make_space():
            for layer in arch["layers"]:
                if layer["type"] == "Dropout":
                    assert layer["rate"] in allowed


# ─── last_hid_mlp values ─────────────────────────────────────────────────────

class TestLastHid:

    def test_last_hid_within_config(self):
        allowed = set(BASE_CONFIG["SearchSpace"]["last_hid_mlp"])
        for arch in make_space():
            assert arch["last_hid_mlp"] in allowed


# ─── Config validation ──────────────────────────────────────────────────────

class TestConfigValidation:

    def test_raises_on_missing_section(self):
        with pytest.raises(ValueError):
            LayersBased().define_space({})

    def test_raises_on_unknown_layer_type(self):
        import copy
        bad = copy.deepcopy(BASE_CONFIG)
        bad["SearchSpace"]["layers_types"] = ["LSTM"]
        with pytest.raises(ValueError, match="LSTM"):
            LayersBased().define_space(bad)

    def test_raises_on_negative_channels(self):
        import copy
        bad = copy.deepcopy(BASE_CONFIG)
        bad["SearchSpace"]["channels"] = [-1]
        with pytest.raises(ValueError):
            LayersBased().define_space(bad)

    def test_raises_on_invalid_dropout_rate(self):
        import copy
        bad = copy.deepcopy(BASE_CONFIG)
        bad["SearchSpace"]["dropout_rates"] = [1.5]
        with pytest.raises(ValueError):
            LayersBased().define_space(bad)

    def test_raises_on_missing_key(self):
        import copy
        bad = copy.deepcopy(BASE_CONFIG)
        del bad["SearchSpace"]["channels"]
        with pytest.raises(ValueError, match="channels"):
            LayersBased().define_space(bad)
