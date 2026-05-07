import numpy as np
from .base import BaseMovementRule

class ObstacleRule(BaseMovementRule):
    """Reguła 1: Ściany. Nie można przez nie przechodzić, ani na nich stawać."""
    def can_traverse(self, memory_map: np.ndarray, y: int, x: int, current_dist: int) -> bool:
        return memory_map[y, x] != 1

    def can_finish_at(self, memory_map: np.ndarray, y: int, x: int, current_dist: int) -> bool:
        return memory_map[y, x] != 1

class FrontierRule(BaseMovementRule):
    """Reguła 2: Fronty (-1). Można do nich dojść, ale nie można iść przez nie dalej w nieznane."""
    def can_traverse(self, memory_map: np.ndarray, y: int, x: int, current_dist: int) -> bool:
        return memory_map[y, x] != -1

    def can_finish_at(self, memory_map: np.ndarray, y: int, x: int, current_dist: int) -> bool:
        return True # Na nieznanym polu można bez problemu zakończyć ruch

class BudgetRule(BaseMovementRule):
    """Reguła 3: Budżet (Radius). Limit ilości kroków R=8."""
    def __init__(self, max_distance: int):
        self.max_distance = max_distance

    def can_traverse(self, memory_map: np.ndarray, y: int, x: int, current_dist: int) -> bool:
        return current_dist < self.max_distance

    def can_finish_at(self, memory_map: np.ndarray, y: int, x: int, current_dist: int) -> bool:
        return current_dist <= self.max_distance

class NoStayRule(BaseMovementRule):
    """Reguła 4: Zakaz stania w miejscu. Agent musi wykonać fizyczny ruch."""
    def can_traverse(self, memory_map: np.ndarray, y: int, x: int, current_dist: int) -> bool:
        return True

    def can_finish_at(self, memory_map: np.ndarray, y: int, x: int, current_dist: int) -> bool:
        # Agent może zakończyć ruch tylko, jeśli przeszedł co najmniej 1 pole
        return current_dist > 0