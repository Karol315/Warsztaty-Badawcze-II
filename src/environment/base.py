from abc import ABC, abstractmethod
from typing import Any, Tuple

class BaseEnv(ABC):
    def __init__(self, size: int, max_steps: int):
        self.size = size
        self.max_steps = max_steps

    @abstractmethod
    def reset(self) -> Any:
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, bool, dict]:
        pass

    @abstractmethod
    def get_ground_truth(self) -> Any:
        pass