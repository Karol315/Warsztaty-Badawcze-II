import numpy as np
from typing import Tuple, Dict
from mazelib import Maze
from mazelib.generate.Prims import Prims
from .base import BaseEnv

class DiscreteMaze(BaseEnv):
    def __init__(self, size: int, max_steps: int):
        super().__init__(size, max_steps)
        self.current_step = 0
        self.grid = None
        self.agent_pos = None

    def reset(self) -> np.ndarray:
        self.current_step = 0
        m = Maze()
        m.generator = Prims(self.size // 2, self.size // 2)
        m.generate()
        self.grid = m.grid
        self.agent_pos = (1, 1)
        return self._get_observation()

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, bool, Dict]:
        self.current_step += 1
        target_y, target_x = action
        hit_wall = False

        if self.grid[target_y, target_x] == 0:
            self.agent_pos = action
        else:
            hit_wall = True

        done = self.current_step >= self.max_steps
        info = {"agent_pos": self.agent_pos, "hit_wall": hit_wall}
        return self._get_observation(), done, info

    def _get_observation(self) -> np.ndarray:
        return self.grid

    def get_ground_truth(self) -> np.ndarray:
        return self.grid