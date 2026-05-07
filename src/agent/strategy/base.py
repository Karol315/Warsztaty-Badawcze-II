from abc import ABC, abstractmethod
from typing import Tuple, Any

class BaseStrategy(ABC):
    @abstractmethod
    def select_action(self, uncertainty_map: Any, reachable_mask: Any, memory_map: Any) -> Tuple[int, int]:
        """Na podstawie mapy niepewności wybiera koordynaty następnego celu."""
        pass