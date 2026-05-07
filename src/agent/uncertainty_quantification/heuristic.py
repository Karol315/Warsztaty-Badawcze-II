import numpy as np
from typing import Any
from .base import BaseUncertainty


class HeuristicUncertainty(BaseUncertainty):
    def estimate(self, model: Any, memory_map: np.ndarray) -> np.ndarray:
        # Największa niepewność (1.0) tam, gdzie mapa jest nieznana (-1)
        uncertainty = np.where(memory_map == -1, 1.0, 0.0)

        # Dodajemy mikroskopijny szum, żeby agent nie wybierał zawsze lewego górnego rogu
        uncertainty += np.random.uniform(0, 0.01, size=uncertainty.shape)
        return uncertainty