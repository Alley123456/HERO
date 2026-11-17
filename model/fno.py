import itertools

import torch
from torch import nn
from torch.nn import functional as F


FNO_CONFIG = {1: (28, 28, 1), 2: (8, 20, 1), 3: (8, 4, 5)}
CONVOLUTION = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


class FourierLayer(nn.Module):
    def __init__(self, dimension, modes, width):
        super().__init__()
        self.axes = tuple(range(2, dimension + 2))
        positive = slice(0, modes)
        negative = slice(-modes, None)
        self.regions = [
            (slice(None), slice(None), *region, positive)
            for region in itertools.product((positive, negative), repeat=dimension - 1)
        ]
        shape = (len(self.regions), width, width, *((modes,) * dimension), 2)
        self.weights = nn.Parameter(torch.randn(shape) / width)
        self.pointwise = CONVOLUTION[dimension - 1](width, width, kernel_size=1)

    def forward(self, state):
        transformed = torch.fft.rfftn(state.float(), dim=self.axes)
        output = torch.zeros_like(transformed)
        for region, weight in zip(self.regions, torch.view_as_complex(self.weights)):
            output[region] = torch.einsum("bi...,io...->bo...", transformed[region], weight)
        spectral = torch.fft.irfftn(output, s=state.shape[2:], dim=self.axes)
        return F.gelu(spectral + self.pointwise(state))


class FNO(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        modes, width, depth = FNO_CONFIG[dimension]
        conv = CONVOLUTION[dimension - 1]
        self.lift = conv(1, width, kernel_size=1)
        self.layers = nn.Sequential(*(FourierLayer(dimension, modes, width) for _ in range(depth)))
        self.project = conv(width, 1, kernel_size=1)

    def forward(self, state):
        return self.project(self.layers(self.lift(state[:, None])))[:, 0]
