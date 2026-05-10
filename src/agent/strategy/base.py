from abc import ABC, abstractmethod
from typing import Tuple, Any

class BaseStrategy(ABC):
    @abstractmethod
    def select_action(self, current_pos: Tuple[int, int], uncertainty_map: Any, reachable_mask: Any, memory_map: Any) -> Tuple[int, int]:
        """Na podstawie mapy i pozycji wybiera koordynaty następnego celu."""
        pass