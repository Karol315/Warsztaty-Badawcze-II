import math

import torch

from .base import BaseMetric  # Dziedziczymy po klasie bazowej


class PSNRMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.total_mse = 0.0
        self.count = 0

    def update(self, preds, targets):
        # Ta funkcja będzie wywoływana w pętli uczącej dla każdego batcha
        mse = torch.nn.functional.mse_loss(preds, targets, reduction="sum")
        self.total_mse += mse.item()
        self.count += targets.numel()

    def compute_and_log(self):
        if self.count == 0:
            return 0.0
        mean_mse = self.total_mse / self.count
        if mean_mse == 0:
            return 50.0  # Idealny wynik

        # Nowy wzór: 20 * log10(MAX_I / sqrt(MSE))
        # Dla zakresu [-1, 1], MAX_I to 2.0
        psnr = 20 * math.log10(2.0 / math.sqrt(mean_mse))

        self.reset()
        return psnr
