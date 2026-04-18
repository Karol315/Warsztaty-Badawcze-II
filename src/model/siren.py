import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenModel(nn.Module):
    def __init__(
        self,
        in_features=2,
        hidden_features=128,
        hidden_layers=3,
        out_features=1,
        omega_0=30,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.net = []

        # Pierwsza warstwa
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))
        self.net.append(nn.Dropout(dropout_rate))  # MCDropout!

        # Warstwy ukryte
        for _ in range(hidden_layers):
            self.net.append(
                SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0)
            )
            self.net.append(nn.Dropout(dropout_rate))  # MCDropout!

        # Warstwa wyjściowa
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / omega_0
            final_linear.weight.uniform_(-bound, bound)
            if final_linear.bias is not None:
                final_linear.bias.data.fill_(0)

        self.net.append(final_linear)
        # Używamy Sigmoid, bo chłopaki zmienili stratę na BCELoss (0 to puste, 1 to ściana)
        self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)
