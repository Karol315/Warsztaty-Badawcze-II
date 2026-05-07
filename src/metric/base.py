from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseMetric(ABC):
    @abstractmethod
    def compute(self, pred: Any, target: Any) -> Dict[str, float]:
        """Oblicza wartości metryk i zwraca je jako słownik."""
        pass