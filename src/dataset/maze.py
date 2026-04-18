import random

import numpy as np


class MazeEnvironment:
    def __init__(self, size=64):
        self.size = size
        self.maze, self.entrance, self.exits = self._generate_maze(size)

    def _generate_maze(self, size):
        maze = np.ones((size, size), dtype=np.float32)

        def carve_passages(cx, cy):
            directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 1 <= nx < size - 1 and 1 <= ny < size - 1 and maze[nx, ny] == 1:
                    maze[cx + dx // 2, cy + dy // 2] = 0
                    maze[nx, ny] = 0
                    carve_passages(nx, ny)

        start_x, start_y = (
            random.randint(1, size // 2) * 2 - 1,
            random.randint(1, size // 2) * 2 - 1,
        )
        maze[start_x, start_y] = 0
        carve_passages(start_x, start_y)

        # Dodanie wejścia
        entrance_y = random.randint(1, size // 2 - 1) * 2 - 1
        maze[0, entrance_y] = 0
        maze[1, entrance_y] = 0
        entrance = (0, entrance_y)

        exits = []
        # Uproszczone wyjście na dole, żeby agent miał cel
        exit_y = random.randint(1, size // 2 - 1) * 2 - 1
        maze[size - 1, exit_y] = 0
        maze[size - 2, exit_y] = 0
        exits.append((size - 1, exit_y))

        return maze, entrance, exits

    def normalize_coords(self, x, y):
        # Skalowanie do zakresu [-1, 1]
        return (x / (self.size - 1)) * 2 - 1, (y / (self.size - 1)) * 2 - 1
