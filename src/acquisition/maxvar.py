import numpy as np
from collections import deque
from acquisition.base import BaseAcquisition


class MaxVarAcquisition(BaseAcquisition):
    def __init__(self, start_pos, maze_size, vision_radius=8):
        self.pos = start_pos
        self.size = maze_size
        self.vision_radius = vision_radius
        self.explored_mask = np.zeros((maze_size, maze_size), dtype=bool)
        self.map = np.zeros((maze_size, maze_size))
        self.observed_coords = []
        self.observed_labels = []

    def normalize_coords(self, x, y):
        return (x / (self.size - 1)) * 2 - 1, (y / (self.size - 1)) * 2 - 1

    def observe(self, true_maze):
        cx, cy = self.pos
        angles = np.linspace(0, 2 * np.pi, 120)
        for angle in angles:
            for r in range(1, self.vision_radius + 1):
                nx, ny = int(round(cx + r * np.cos(angle))), int(round(cy + r * np.sin(angle)))
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    break
                if not self.explored_mask[nx, ny]:
                    self.explored_mask[nx, ny] = True
                    norm_x, norm_y = self.normalize_coords(nx, ny)
                    self.observed_coords.append([norm_x, norm_y])
                    self.observed_labels.append([true_maze[nx, ny]])
                    self.map[nx, ny] = 1 if true_maze[nx, ny] == 1 else -1
                if true_maze[nx, ny] == 1:
                    break
        if not self.explored_mask[cx, cy]:
            self.explored_mask[cx, cy] = True
            norm_x, norm_y = self.normalize_coords(cx, cy)
            self.observed_coords.append([norm_x, norm_y])
            self.observed_labels.append([true_maze[cx, cy]])
            self.map[cx, cy] = -1

    def get_next_move(self, model_uncertainty, R):
        # Tutaj model_uncertainty to nasza mapa wariancji (variance_map)
        queue = deque([self.pos])
        visited = {self.pos}
        parents = {self.pos: None}
        frontiers = []

        while queue:
            curr = queue.popleft()
            cx, cy = curr
            is_frontier = any(self.map[cx + dx, cy + dy] == 0 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)] if
                              0 <= cx + dx < self.size and 0 <= cy + dy < self.size)

            if is_frontier:
                frontiers.append((curr, model_uncertainty[cx, cy]))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in visited and self.map[nx, ny] == -1:
                    visited.add((nx, ny))
                    parents[(nx, ny)] = curr
                    queue.append((nx, ny))

        if not frontiers:
            return self.pos

        best_target = max(frontiers, key=lambda x: x[1])[0]
        path, curr = [], best_target
        while curr is not None:
            path.append(curr)
            curr = parents[curr]
        path.reverse()
        steps = min(R, len(path) - 1)
        return path[steps] if steps > 0 else self.pos