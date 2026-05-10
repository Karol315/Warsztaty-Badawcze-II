import numpy as np
from typing import Tuple, Any
from .base import BaseStrategy

class FrontierStrategy(BaseStrategy):
    def select_action(self, current_pos: Tuple[int, int], uncertainty_map: Any, reachable_mask: Any, memory_map: Any) -> Tuple[int, int]:
        y, x = current_pos
        
        # 1. Przestrzeń wolna (0) i nieznana (-1)
        free_space = (memory_map == 0)
        unknown = (memory_map == -1)
        
        # 2. Szybkie szukanie granic przez przesunięcia macierzy
        shifted_up = np.roll(unknown, 1, axis=0); shifted_up[0, :] = False
        shifted_down = np.roll(unknown, -1, axis=0); shifted_down[-1, :] = False
        shifted_left = np.roll(unknown, 1, axis=1); shifted_left[:, 0] = False
        shifted_right = np.roll(unknown, -1, axis=1); shifted_right[:, -1] = False
        
        has_unknown_neighbor = shifted_up | shifted_down | shifted_left | shifted_right
        
        # 3. Maska frontów: pole jest wolne, ma nieznanego sąsiada i agent może tam dojść
        frontiers = free_space & has_unknown_neighbor & reachable_mask
        valid_coords = np.argwhere(frontiers)
        
        # Awaryjnie: brak frontu = idź gdziekolwiek, gdzie można
        if len(valid_coords) == 0:
            valid_coords = np.argwhere(reachable_mask)
            if len(valid_coords) == 0:
                return current_pos # Agent całkowicie utknął
                
        # 4. Wybór najbliższego frontu (Dystans Manhattan)
        distances = np.sum(np.abs(valid_coords - np.array([y, x])), axis=1)
        best_idx = np.argmin(distances)
        best_pos = valid_coords[best_idx]
        
        return (int(best_pos[0]), int(best_pos[1]))