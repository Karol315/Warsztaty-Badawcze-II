from abc import ABC, abstractmethod
import numpy as np

class BaseMovementRule(ABC):
    @abstractmethod
    def can_traverse(self, memory_map: np.ndarray, y: int, x: int, current_dist: int) -> bool:
        """Czy agent może użyć tego pola jako korytarza, by iść dalej?"""
        pass

    @abstractmethod
    def can_finish_at(self, memory_map: np.ndarray, y: int, x: int, current_dist: int) -> bool:
        """Czy agent może zatrzymać się na tym polu jako ostatecznym celu?"""
        pass