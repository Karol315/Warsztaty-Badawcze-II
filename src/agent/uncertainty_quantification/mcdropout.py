import torch
import numpy as np
from typing import Any
from .base import BaseUncertainty


class MCDropoutUncertainty(BaseUncertainty):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def estimate(self, model: Any, memory_map: np.ndarray) -> np.ndarray:
        if model is None:
            return np.zeros_like(memory_map, dtype=float)

        map_size = memory_map.shape[0]
        y, x = np.meshgrid(np.arange(map_size), np.arange(map_size), indexing='ij')
        all_coords = np.stack([y.flatten(), x.flatten()], axis=-1)
        tensor_coords = torch.tensor(all_coords, dtype=torch.float32) / (map_size - 1) * 2.0 - 1.0

        # MAGIC TRICK Pytorcha: Włączamy train(), żeby Dropout "migał",
        # ale używamy no_grad(), żeby całkowicie zamrozić wagi i nie marnować pamięci RAM.
        model.train()

        with torch.no_grad():
            samples = []
            for _ in range(self.num_samples):
                # Wynik sieci przechodzi przez Sigmoid, by otrzymać prawdopodobieństwa (0-1)
                logits = model(tensor_coords).squeeze()
                probs = torch.sigmoid(logits)
                samples.append(probs)

            samples = torch.stack(samples)  # (num_samples, liczba_pikseli)
            # Wariancja to nasza niepewność!
            variance = torch.var(samples, dim=0).numpy()

        variance_map = variance.reshape(map_size, map_size)
        return variance_map