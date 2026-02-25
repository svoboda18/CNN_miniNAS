from abc import ABC, abstractmethod
from typing import List, Dict, Any

class SearchStrategy(ABC):

    @abstractmethod
    def search(self) -> List[int]:
        pass