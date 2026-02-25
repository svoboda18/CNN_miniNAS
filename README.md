# CNN_Cls_miniNAS

A mini Neural Architecture Search (NAS) framework that automatically discovers
convolutional neural network (CNN) architectures for MNIST 10-class digit
classification.

---

## Objective

Automatically search a user-defined space of CNN architectures, train each
candidate from scratch on MNIST, and select the best-performing one — without
any manual architecture engineering.

---

## Dataset

| File | Role | Format |
|------|------|--------|
| `data/mnist_train.csv` | Training set (~60 000 samples) | CSV: column 0 = label, columns 1–784 = pixel values (0–255, 28×28 image flattened row-major) |
| `data/mnist_test.csv`  | Validation / test set (~10 000 samples) | Same format |

The CSV files are **not included in the repository** (excluded via `.gitignore` due to size).
Download them from Kaggle and place them in the `data/` folder:

```
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
```

---

## Project Structure

```
CNN_miniNAS/
├── nas.py                          # Entry point
├── configs/
│   └── config.yaml                  # Experiment configuration
├── data/
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── nas/
│   ├── io.py                       # Config loader + result saver
│   ├── model_builder.py            # Builds nn.Module from an arch dict
│   ├── search_space/
│   │   ├── base.py                 # Abstract SearchSpace
│   │   └── layers_based.py        # LayersBased: full combinatorial enumeration
│   ├── search_strategy/
│   │   ├── base.py                 # Abstract SearchStrategy
│   │   └── random_search.py       # RandomSearch: uniform random sampling
│   └── validation/
│       ├── base.py                 # Abstract ValidationStrategy
│       └── full_training.py       # FullTraining: train-from-scratch + eval
├── tests/
│   ├── test_io.py
│   ├── test_search_space.py
│   ├── test_search_strategy.py
│   └── test_validation.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

---

## Execution Flow

```
python nas.py configs/config.yaml [output_path]
```

### Step-by-step pipeline

```
nas.py
  │
  ├─1─ load_config(config_path)
  │       nas/io.py::load_config
  │       • Reads YAML, validates required top-level keys and field types.
  │       • Returns a plain Python dict.
  │
  ├─2─ get_search_space(config)  →  space_obj  (LayersBased instance)
  │       nas/search_space/__init__.py  (factory)
  │       nas/search_space/layers_based.py
  │       • Stage 1: iterates over every allowed layer count [5, 7].
  │       • For each count, generates every sequence of layer types
  │         (Conv2d / MaxPool2d / ReLU / Dropout) using itertools.product.
  │       • Prunes invalid sequences (no two consecutive Dropouts,
  │         at least one Conv2d).
  │       • Stage 2: expands per-layer hyper-parameters via Cartesian product:
  │           Conv2d   → channels × kernel × padding × padding_mode
  │           MaxPool2d→ fixed kernel=2, stride=2
  │           ReLU     → no params
  │           Dropout  → dropout_rates
  │       • Stage 3: pairs with last_hid_mlp values; discards any architecture
  │         where the spatial dimension goes ≤ 0 after simulating the layers.
  │       • Returns list[dict] — every valid architecture.
  │
  ├─3─ RandomSearch(architectures, config).search()  →  indices: list[int]
  │       nas/search_strategy/random_search.py
  │       • Samples nbr_iterations unique indices (without replacement
  │         while space is not exhausted) from the full architecture list.
  │
  ├─4─ get_validator(config)  →  validator  (FullTraining instance)
  │       nas/validation/__init__.py  (factory)
  │       nas/validation/full_training.py
  │
  └─5─ For each sampled index:
          arch = architectures[idx]
          │
          ├─ ModelBuilder.build_from_dict(arch)
          │     nas/model_builder.py
          │     • Constructs an nn.Module with:
          │         self.features   = nn.Sequential(Conv2d/ReLU/MaxPool2d/Dropout…)
          │         self.classifier = nn.Sequential(Flatten, Linear[, Linear])
          │     • calculate_flatten_size() simulates spatial dims to derive the
          │       correct input size for the first Linear layer.
          │
          ├─ FullTraining.validate(arch)
          │     • Loads mnist_train.csv / mnist_test.csv into DataLoaders,
          │       respecting use_full_dataset / max_samples settings.
          │     • Trains model with Adam + CrossEntropyLoss for `epochs` epochs.
          │     • Records epoch-average training loss → loss_history list.
          │     • Evaluates on test set → acc_val.
          │     • Returns {acc_val, loss_history, model, runtime}.
          │
          └─ Appends result to all_results.

  └─6─ save_results(output_path, all_results)
          nas/io.py::save_results
          • For each model_i/:
              loss_history.png   ← matplotlib plot of training loss per epoch
              model.pt           ← torch.save(model.state_dict(), …)
          • results/accuracies.txt ← one line per model: "Model i: acc_val = X"
          • results/best_model_code.py ← auto-generated Python class for the
            best architecture (ready to copy-paste or retrain).
```

### Output directory layout

```
results/
├── accuracies.txt          ← accuracy results for all evaluated architectures
├── best_model_code.py      ← auto-generated PyTorch class of top model
├── model_0/
│   ├── loss_history.png    ← loss graph image
│   └── model.pt
├── model_1/
│   ├── loss_history.png
│   └── model.pt
└── …
```

---

## Configuration (`configs/config.yaml`)

| Section | Key | Description |
|---------|-----|-------------|
| `SearchSpace` | `layers_types` | Allowed layer types in sequences |
| | `layers_count` | Layer count values to explore |
| | `channels` | Out-channel options for Conv2d |
| | `kernel` | Kernel size options |
| | `padding` | Integer pixel padding added symmetrically on each side |
| | `padding_mode` | Fill algorithm: `zeros`, `reflect`, `replicate`, `circular` |
| | `last_hid_mlp` | Hidden units of optional MLP bottleneck (0 = skip) |
| | `dropout_rates` | Dropout probability options |
| `SearchStrategy` | `nbr_iterations` | How many architectures to sample and train |
| `ValidationStrategy` | `epochs` | Training epochs per candidate |
| | `use_full_dataset` | `true` = full CSV; `false` = use `max_samples` subset |
| | `trn_data.url` | Path to training CSV |
| | `trn_data.batch` | Training batch size |
| | `trn_data.max_samples` | Max training rows when `use_full_dataset: false` |
| | `val_data.url` | Path to validation CSV |
| | `val_data.batch` | Validation batch size |
| | `val_data.max_samples` | Max validation rows when `use_full_dataset: false` |

### Reduced-dataset mode

Setting `use_full_dataset: false` enables fast experimentation by capping the
number of loaded rows per split:

```yaml
use_full_dataset: false
trn_data:
  max_samples: 5000   # ~5 000 of 60 000 training rows
val_data:
  max_samples: 1000   # ~1 000 of 10 000 test rows
```

Switch to `use_full_dataset: true` for final evaluation on the complete dataset.
The `max_samples` keys are ignored in that case.

### Padding controls

`nn.Conv2d` exposes two independent padding parameters, both configurable:

1. **`padding`** — integer specifying how many pixels to add on each side
2. **`padding_mode`** — fill algorithm: `'zeros'`, `'reflect'`, `'replicate'`, `'circular'`

The search space cross-products all listed values for both parameters.

---

## Running tests

```bash
pytest tests/ -v
```
