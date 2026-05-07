from abc import ABC, abstractmethod
import torch.nn as nn


class BaseWorldModel(nn.Module, ABC):
    """Wspólny interfejs dla wszystkich architektur sieciowych."""

    @abstractmethod
    def forward(self, coords):
        """Każdy model musi przyjmować współrzędne i zwracać przewidywania."""
        pass