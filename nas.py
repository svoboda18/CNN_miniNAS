"""
nas.py — CNN_Cls_miniNAS Entry Point

Usage:
    python nas.py ./configs/config.yaml [output_path]
"""

import sys
import time

from tqdm import tqdm

from nas.io              import load_config, save_results
from nas.search_space    import get_search_space
from nas.search_strategy.random_search import RandomSearch
from nas.validation      import get_validator


def architecture_search(config: dict, output_path: str = "./results"):

    # STEP 1 — build the full search space
    print("\n[1/5] Building search space...")
    t0 = time.time()
    space_obj     = get_search_space(config)
    architectures = space_obj.define_space(config)
    print(f"      {len(architectures)} valid architectures found  ({time.time()-t0:.1f}s)")

    # STEP 2 — build strategy
    print("\n[2/5] Initialising search strategy...")
    strategy = RandomSearch(architectures, config)
    print(f"      Strategy : RandomSearch  |  iterations : {strategy.nbr_iterations}")

    # STEP 3 — sample
    print("\n[3/5] Sampling architectures...")
    indices = strategy.search()
    print(f"      Sampled indices: {indices}")

    # STEP 4 — validate
    print("\n[4/5] Validating architectures...")
    validator  = get_validator(config)
    all_results = []
    nas_bar = tqdm(indices, desc="  NAS search", unit="arch",
                   bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for i, idx in enumerate(nas_bar):
        arch = architectures[idx]
        layer_summary = ", ".join(l["type"] for l in arch["layers"])
        nas_bar.set_postfix(idx=idx, layers=len(arch["layers"]),
                            last_hid=arch["last_hid_mlp"])
        tqdm.write(f"\n  arch {i+1}/{len(indices)}  index={idx}  "
                   f"layers=[{layer_summary}]  last_hid={arch['last_hid_mlp']}")

        result = validator.validate(arch)
        result["architecture"] = arch
        tqdm.write(f"  acc_val={result['acc_val']:.4f}  "
                   f"runtime={result.get('runtime', 0):.1f}s")
        all_results.append(result)

    # STEP 5 — save
    print("\n[5/5] Saving results...")
    save_results(output_path, all_results)

    best = max(all_results, key=lambda r: r["acc_val"])
    print(f"\n{'='*50}")
    print(f"  Best acc_val : {best['acc_val']:.4f}")
    print(f"  Results saved to : {output_path}")
    print(f"{'='*50}")
    return all_results


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./configs/config.yaml"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "./results"

    w = 50
    print("=" * w)
    print(f"{'CNN_Cls_miniNAS':^{w}}")
    print("=" * w)
    print(f"  Config  : {config_path}")
    print(f"  Output  : {output_path}")
    print("=" * w)

    config = load_config(config_path)
    architecture_search(config, output_path)