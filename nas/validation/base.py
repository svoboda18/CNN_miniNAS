from abc import ABC, abstractmethod
from typing import Dict, Any

class ValidationStrategy(ABC):
    """
    Interface for validation strategies.
    """
    @abstractmethod
    def validate(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates an architecture based on predefined metrics.
        """
        pass