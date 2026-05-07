import numpy as np
from typing import Dict
from .base import BaseMetric


class PSNRMetric(BaseMetric):
    def compute(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        # Środowisko binarne (0 i 1), więc max wartość to 1.0.
        mse = np.mean((pred - target) ** 2)

        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(1.0 / mse)

        return {
            "mse": mse,
            "psnr": psnr
        }