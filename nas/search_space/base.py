from abc import ABC, abstractmethod

class SearchSpace(ABC):
    @abstractmethod
    def define_space(self, parameters: dict) -> list:
        pass

    @staticmethod
    def _require_keys(section, keys, context="SearchSpace"):
        missing = [k for k in keys if k not in section]
        if missing:
            raise ValueError(f"[{context}] Missing keys: {missing}")

    @staticmethod
    def _require_non_empty(section, keys, context="SearchSpace"):
        for k in keys:
            if not section.get(k):
                raise ValueError(f"[{context}] '{k}' must be a non-empty list.")