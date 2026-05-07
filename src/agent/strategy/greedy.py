import numpy as np
from .base import BaseStrategy
from typing import Tuple, Any


class GreedyStrategy(BaseStrategy):
    def __init__(self, vision_radius: int):
        self.vision_radius = vision_radius

    def select_action(self, uncertainty_map: Any, reachable_mask: Any, memory_map: Any) -> Tuple[int, int]:
        # Pobieramy współrzędne wszystkich pól, do których agent aktualnie może dojść (np. w promieniu 8)
        valid_coords = np.argwhere(reachable_mask)

        best_score = -1.0
        best_pos = None

        # Oceniamy każde z dostępnych pól pod kątem "zysku informacyjnego" (Information Gain)
        for y, x in valid_coords:
            # Tworzymy bounding box (okienko), które symuluje pole widzenia agenta, gdyby tam stanął
            y_min = max(0, int(y - self.vision_radius))
            y_max = min(memory_map.shape[0], int(y + self.vision_radius + 1))
            x_min = max(0, int(x - self.vision_radius))
            x_max = min(memory_map.shape[1], int(x + self.vision_radius + 1))

            # Pobieramy wycinek pamięci i mapy niepewności
            window_memory = memory_map[y_min:y_max, x_min:x_max]
            window_unc = uncertainty_map[y_min:y_max, x_min:x_max]

            # KLUCZOWE: Sumujemy niepewność TYLKO na polach, których agent jeszcze NIE ZNA (-1)
            # Im więcej nieznanych i niepewnych pól w zasięgu wzroku, tym wyższy zysk (score)
            score = np.sum(window_unc[window_memory == -1])

            if score > best_score:
                best_score = score
                best_pos = (y, x)

        if best_pos is None:
            # Fallback (awaryjne wyjście), jeśli z jakiegoś powodu algorytm nic nie znajdzie
            flat_idx = np.argmax(np.where(reachable_mask, uncertainty_map, -1.0))
            best_pos = np.unravel_index(flat_idx, uncertainty_map.shape)

        return (int(best_pos[0]), int(best_pos[1]))