import math
import torch
from metric.base import BaseMetric

class PSNRMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.total_mse = 0.0
        self.count = 0

    def update(self, preds, targets):
        mse = torch.nn.functional.mse_loss(preds, targets, reduction="sum")
        self.total_mse += mse.item()
        self.count += targets.numel()

    def compute_and_log(self):
        if self.count == 0:
            return 0.0
        mean_mse = self.total_mse / self.count
        if mean_mse == 0:
            return 100.0 # Idealny wynik

        # Zgodnie z dokumentacją: 10 * log10(1 / MSE) dla danych znormalizowanych
        psnr = 10 * math.log10(1.0 / mean_mse)
        self.reset()
        return psnr