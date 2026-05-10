import numpy as np
from typing import Tuple

class EpisodicMemory:
    def __init__(self, map_size: int, unknown_value: int):
        self.map_size = map_size
        self.unknown_value = unknown_value
        # ZMIANA: np.float32 zamiast np.int8, żeby pamięć mogła trzymać ułamki SDF!
        self.explored_map = np.full((map_size, map_size), unknown_value, dtype=np.float32)
        self.visited_positions = []

    def update(self, current_pos: Tuple[int, int], visible_points: np.ndarray, visible_values: np.ndarray):
        self.visited_positions.append(current_pos)
        for point, value in zip(visible_points, visible_values):
            y, x = point
            self.explored_map[y, x] = value

    def get_dataset_for_model(self) -> Tuple[np.ndarray, np.ndarray]:
        known_mask = self.explored_map != self.unknown_value
        coords = np.argwhere(known_mask)
        values = self.explored_map[known_mask]
        return coords, values