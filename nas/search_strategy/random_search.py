import random
from typing import List, Dict, Any
from .base import SearchStrategy


class RandomSearch(SearchStrategy):

    def __init__(
        self,
        search_space: List[Dict[str, Any]],
        config: Dict[str, Any]
    ):
        """
        Args:
            search_space: list of ALL valid architectures
            config: SearchStrategy section from YAML
        """
        self.search_space = search_space
        # Config is the full YAML dict; strategy params live under SearchStrategy
        strat_cfg = config.get("SearchStrategy", config)
        self.nbr_iterations = strat_cfg["nbr_iterations"]

        if not self.search_space:
            raise ValueError("[RandomSearch] Search space is empty.")

    # ─────────────────────────────────────────────
    # Return N sampled architectures
    # ─────────────────────────────────────────────
    def search(self) -> List[int]:

        selected_architectures_index=[]
        visited_indices = set()

        for i in range(self.nbr_iterations):

            print(f"\n--- Sampling {i+1}/{self.nbr_iterations} ---")

            if len(visited_indices) < len(self.search_space):
                while True:
                    idx = random.randrange(len(self.search_space))
                    if idx not in visited_indices:
                        visited_indices.add(idx)
                        selected_architectures_index.append(idx)
                        break
            else:
                idx = random.randrange(len(self.search_space))
                selected_architectures_index.append(idx)
        return selected_architectures_index