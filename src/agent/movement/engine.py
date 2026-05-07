import numpy as np
from collections import deque
from typing import Tuple, List
from .base import BaseMovementRule


class MovementEngine:
    def __init__(self, rules: List[BaseMovementRule]):
        self.rules = rules

    def get_reachable_mask(self, memory_map: np.ndarray, start_pos: Tuple[int, int]) -> np.ndarray:
        max_y, max_x = memory_map.shape
        reachable = np.zeros((max_y, max_x), dtype=bool)

        queue = deque([(start_pos[0], start_pos[1], 0)])
        visited = set([start_pos])

        while queue:
            y, x, dist = queue.popleft()

            # 1. ZAKOŃCZENIE RUCHU (Iloczyn logiczny - wszystkie reguły muszą zwrócić True)
            if all(rule.can_finish_at(memory_map, y, x, dist) for rule in self.rules):
                reachable[y, x] = True

            # 2. PRZEJŚCIE DALEJ (Iloczyn logiczny - jeśli choć jedna zablokuje, przerywamy)
            if not all(rule.can_traverse(memory_map, y, x, dist) for rule in self.rules):
                continue

            # 3. ROZWÓJ WĘZŁA
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx

                if 0 <= ny < max_y and 0 <= nx < max_x:
                    if (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append((ny, nx, dist + 1))

        return reachable