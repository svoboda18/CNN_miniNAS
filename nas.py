"""
nas.py — CNN_Cls_miniNAS Entry Point

Usage:
    python nas.py ./configs/config1.yaml
"""

import sys

from nas.io              import load_config, save_results
from nas.search_space    import get_search_space
from nas.search_strategy.random_search import RandomSearch
# from nas.validation    import get_validator    # uncomment when ready


def _stub_validator(architecture: dict) -> dict:
    """
    Temporary placeholder — DELETE when validation module is ready.
    Replace with:  validator = get_validator(config).validate
    """
    import random
    print("  [STUB] Returning fake metrics (validation not yet implemented).")
    return {
        "acc_val"     : round(random.uniform(0.70, 0.99), 4),
        "loss_history": [round(1.5 - i * 0.08 + random.uniform(-0.05, 0.05), 4)
                         for i in range(15)],
        "model"       : None,
        "architecture": architecture,
    }


def architecture_search(config: dict, output_path: str = "./results"):

    # STEP 1 — build the full search space (your module)
    space_obj     = get_search_space(config)
    architectures = space_obj.define_space(config)   # list[dict]
    print(f"[NAS] Search space ready — {len(architectures)} valid architectures.")

    # STEP 2 — build strategy (architectures passed into constructor)
    strategy = RandomSearch(architectures, config)

    # STEP 3 — sample: returns list of indices e.g. [42, 1337, 7, 999]
    indices = strategy.search()

    # STEP 4 — validate each sampled architecture
    validator = _stub_validator    # swap for real validator when ready

    all_results = []

    for i, idx in enumerate(indices):
        arch = architectures[idx]   # get actual dict using the index

        print(f"\n[NAS] Iteration {i+1}/{len(indices)} — index={idx} "
              f"| {len(arch['layers'])} layers "
              f"| last_hid={arch['last_hid_mlp']}")
        for layer in arch["layers"]:
            print(f"    {layer}")

        result = validator(arch)
        result["architecture"] = arch
        print(f"  acc_val = {result['acc_val']:.4f}")

        all_results.append(result)

    # STEP 5 — save everything
    save_results(output_path, all_results)

    print("\n[NAS] Done.")
    return all_results


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./configs/config1.yaml"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "./results"

    print("=" * 50)
    print("  CNN_Cls_miniNAS")
    print("=" * 50)
    print(f"  Config : {config_path}")
    print(f"  Output : {output_path}")
    print("=" * 50)

    config = load_config(config_path)
    architecture_search(config, output_path)