import itertools
from .base import SearchSpace

DEFAULT_INPUT_CHANNELS = 1
DEFAULT_NUM_CLASSES    = 10
DEFAULT_INPUT_SIZE     = 28
KNOWN_LAYER_TYPES      = {'Conv2d', 'MaxPool2d', 'ReLU', 'Dropout'}


class LayersBased(SearchSpace):

    def define_space(self, parameters: dict) -> list:
        ss = self._parse_and_validate(parameters)

        input_channels = int(ss.get('input_channels', DEFAULT_INPUT_CHANNELS))
        num_classes    = int(ss.get('num_classes',    DEFAULT_NUM_CLASSES))
        input_size     = int(ss.get('input_size',     DEFAULT_INPUT_SIZE))

        all_architectures = []

        # STAGE 1 — iterate over every allowed layer count
        for n_layers in ss['layers_count']:

            # Every possible sequence of layer types of length n_layers
            for type_sequence in itertools.product(ss['layers_types'], repeat=n_layers):

                # Skip sequences that violate structural rules
                if not self._is_valid_sequence(type_sequence):
                    continue

                # STAGE 2 — expand per-layer parameters
                for parameterised_layers in self._expand_parameters(type_sequence, ss):

                    # STAGE 3 — combine with last_hid_mlp + spatial check
                    for last_hid in ss['last_hid_mlp']:
                        arch = {
                            'layers'        : parameterised_layers,
                            'last_hid_mlp'  : last_hid,
                            'input_channels': input_channels,
                            'num_classes'   : num_classes,
                            'input_size'    : input_size,
                        }
                        if self._is_spatially_valid(arch):
                            all_architectures.append(arch)

        print(f"[LayersBased] Total valid architectures found: {len(all_architectures)}")
        return all_architectures

    # ── STAGE 1 helper ────────────────────────────────────────────────────

    @staticmethod
    def _is_valid_sequence(type_sequence: tuple) -> bool:
        """
        Rule 1: no two consecutive Dropout layers
        Rule 2: must contain at least one Conv2d
        """
        has_conv = False
        for i, lt in enumerate(type_sequence):
            if lt == 'Conv2d':
                has_conv = True
            if lt == 'Dropout' and i > 0 and type_sequence[i - 1] == 'Dropout':
                return False
        return has_conv

    # ── STAGE 2 helper ────────────────────────────────────────────────────

    @staticmethod
    def _expand_parameters(type_sequence: tuple, ss: dict):
        """
        Generator. For each position in the sequence, builds the list of
        all possible layer dicts. Then yields the Cartesian product.

        Conv2d    → all (channels × kernel × padding × padding_mode) combos
        MaxPool2d → fixed kernel=2, stride=2                (1 option)
        ReLU      → no params                               (1 option)
        Dropout   → all (rate,) combinations
        """
        per_position_choices = []

        for lt in type_sequence:
            if lt == 'Conv2d':
                choices = [
                    # Canonical Conv2d keys used throughout the project:
                    #   'channels'     – output channel count
                    #   'kernel'       – square kernel size
                    #   'padding'      – symmetric pixel count per side
                    #   'padding_mode' – fill algorithm ('zeros', 'reflect',
                    #                    'replicate', 'circular')
                    {
                        'type': 'Conv2d',
                        'channels': oc,
                        'kernel': k,
                        'padding': p,
                        'padding_mode': pm,
                    }
                    for oc, k, p, pm in itertools.product(
                        ss['channels'],
                        ss['kernel'],
                        ss['padding'],
                        ss.get('padding_mode', ['zeros']),
                    )
                ]
            elif lt == 'MaxPool2d':
                choices = [{'type': 'MaxPool2d', 'kernel': 2, 'stride': 2}]
            elif lt == 'ReLU':
                choices = [{'type': 'ReLU'}]
            elif lt == 'Dropout':
                choices = [{'type': 'Dropout', 'rate': rate} for rate in ss['dropout_rates']]
            else:
                choices = [{'type': lt}]

            per_position_choices.append(choices)

        # Cartesian product across all positions
        for combo in itertools.product(*per_position_choices):
            yield list(combo)

    # ── STAGE 3 helper ────────────────────────────────────────────────────

    @staticmethod
    def _is_spatially_valid(architecture: dict) -> bool:
        """
        Simulates spatial size layer by layer.
        Returns False if it ever drops to <= 0.

        Conv2d    → size = size + 2*padding - kernel + 1
        MaxPool2d → size = (size - kernel) // stride + 1
        ReLU/Dropout → no change
        """
        spatial = architecture['input_size']

        for layer in architecture['layers']:
            lt = layer['type']
            if lt == 'Conv2d':
                spatial = spatial + 2 * layer['padding'] - layer['kernel'] + 1
                if spatial <= 0:
                    return False
            elif lt == 'MaxPool2d':
                spatial = (spatial - layer['kernel']) // layer['stride'] + 1
                if spatial <= 0:
                    return False
        return True

    # ── Config validation ─────────────────────────────────────────────────

    def _parse_and_validate(self, parameters: dict) -> dict:
        if 'SearchSpace' not in parameters:
            raise ValueError("[LayersBased] Config missing 'SearchSpace' section.")
        ss = parameters['SearchSpace']

        self._require_keys(ss, ['layers_types','layers_count','channels',
                                 'kernel','padding','last_hid_mlp','dropout_rates'])
        self._require_non_empty(ss, ['layers_types','layers_count','channels',
                                      'kernel','padding','last_hid_mlp','dropout_rates'])

        unknown = set(ss['layers_types']) - KNOWN_LAYER_TYPES
        if unknown:
            raise ValueError(f"[LayersBased] Unknown layer types: {unknown}")

        self._validate_numeric_ranges(ss)
        return ss

    @staticmethod
    def _validate_numeric_ranges(ss):
        for n in ss['layers_count']:
            if not isinstance(n, int) or n <= 0:
                raise ValueError(f"layers_count must be positive integers. Got: {n}")
        for c in ss['channels']:
            if c <= 0:
                raise ValueError(f"channels must be > 0. Got: {c}")
        for k in ss['kernel']:
            if k < 1:
                raise ValueError(f"kernel must be >= 1. Got: {k}")
        for p in ss['padding']:
            if p < 0:
                raise ValueError(f"padding must be >= 0. Got: {p}")
        for d in ss['dropout_rates']:
            if not (0.0 < d < 1.0):
                raise ValueError(f"dropout_rates must be in (0,1). Got: {d}")

    def __repr__(self):
        return "LayersBased(SearchSpace)"