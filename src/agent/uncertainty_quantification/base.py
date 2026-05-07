from abc import ABC, abstractmethod
from typing import Any

class BaseUncertainty(ABC):
    @abstractmethod
    def estimate(self, model: Any, memory_map: Any) -> Any:
        """Zwraca mapę niepewności o wymiarach planszy."""
        pass