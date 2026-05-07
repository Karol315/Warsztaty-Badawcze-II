import numpy as np
from typing import Tuple


class RaycastPerception:
    def __init__(self, num_rays: int, vision_radius: float):
        self.num_rays = num_rays
        self.vision_radius = vision_radius

    def observe(self, agent_pos: Tuple[int, int], ground_truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y0, x0 = agent_pos
        max_y, max_x = ground_truth.shape

        # Zbiór punktów, by uniknąć duplikatów
        visible_points = set()
        visible_points.add((y0, x0))

        # Rzucamy promienie we wszystkich kierunkach
        angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)
        for angle in angles:
            dy = np.sin(angle)
            dx = np.cos(angle)

            for r in range(1, int(self.vision_radius) + 1):
                y = int(round(y0 + r * dy))
                x = int(round(x0 + r * dx))

                # Sprawdzamy czy nie wychodzimy poza mapę
                if 0 <= y < max_y and 0 <= x < max_x:
                    visible_points.add((y, x))
                    # Jeśli trafiamy na ścianę (1), promień nie leci dalej
                    if ground_truth[y, x] == 1:
                        break
                else:
                    break

        # Formatowanie do postaci tablic (współrzędne i ich wartości z mapy)
        points_array = np.array(list(visible_points))
        values_array = ground_truth[points_array[:, 0], points_array[:, 1]]

        return points_array, values_array