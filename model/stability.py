import math

import numpy as np
import torch
from torch import nn

from data.benchmarks import BENCHMARKS
from model.fno import FNO
from model.hero import EPS


CONSTRAINTS = {
    "ks": ("rms_cap", "fourier_cap"),
    "swift_hohenberg": ("swift_integrator",),
    "unbalanced_advection": ("conserve_rms", "conserve_spectrum"),
}


def training_limits(pde, train, metadata):
    if pde == "ks":
        maxima = np.abs(np.fft.rfft(train, axis=-1, norm="ortho")).max(axis=(0, 1))
        return {
            "rms_cap": 1.12 * metadata["splits"]["train"]["max_rms"],
            "mode_cap": (1.25 * maxima + 1e-6).tolist(),
        }
    return {}


class StableFNO(nn.Module):
    def __init__(self, pde, constraint, limits):
        super().__init__()
        if constraint not in CONSTRAINTS[pde]:
            raise ValueError(f"Invalid output constraint {constraint} for {pde}")
        spec = BENCHMARKS[pde]
        self.base = FNO(spec.dimension)
        self.constraint = constraint
        self.axes = tuple(range(1, spec.dimension + 1))
        if pde == "ks":
            self.rms_cap = limits["rms_cap"]
            self.register_buffer("mode_cap", torch.tensor(limits["mode_cap"], dtype=torch.float32))
        if constraint == "swift_integrator":
            spacing = spec.domain / spec.points
            full = torch.fft.fftfreq(spec.points, d=spacing, dtype=torch.float64)
            half = torch.fft.rfftfreq(spec.points, d=spacing, dtype=torch.float64)
            wave_squared = sum((2 * math.pi * axis).square()
                               for axis in torch.meshgrid(full, full, half, indexing="ij"))
            critical = spec.parameters["critical_number"]
            z = spec.dt * (spec.parameters["reactivity"] - (critical - wave_squared).square())
            phi = torch.where(z.abs() > 1e-5, torch.expm1(z) / z, 1 + z / 2 + z.square() / 6)
            self.register_buffer("linear_step", torch.exp(z).float())
            self.register_buffer("forcing_step", (spec.dt * phi).float())
            self.quadratic, self.cubic = spec.parameters["polynomial_coefficients"][2:]

    def rms(self, field):
        length = math.sqrt(math.prod(field.shape[1:]))
        norm = torch.linalg.vector_norm(field.flatten(1), dim=1) / length
        return norm.reshape(len(field), *((1,) * len(self.axes)))

    def forward(self, state):
        prediction = self.base(state)
        if self.constraint == "swift_integrator":
            start = torch.fft.rfftn(state, dim=self.axes)
            reaction = (self.quadratic + self.cubic * state) * state.square()
            force = torch.fft.rfftn(reaction + prediction, dim=self.axes)
            result = start * self.linear_step + force * self.forcing_step
            return torch.fft.irfftn(result, s=state.shape[1:], dim=self.axes)
        if self.constraint == "conserve_rms":
            mean = state.mean(self.axes, keepdim=True)
            original = state - mean
            centered = prediction - prediction.mean(self.axes, keepdim=True)
            old_rms = self.rms(original)
            rms = self.rms(centered)
            scaled = centered * (old_rms / rms.clamp_min(EPS))
            corrected = torch.where(rms > EPS, scaled, original)
            return mean + (corrected - corrected.mean(self.axes, keepdim=True))
        if self.constraint == "conserve_spectrum":
            before = torch.fft.rfftn(state, dim=self.axes, norm="ortho")
            after = torch.fft.rfftn(prediction, dim=self.axes, norm="ortho")
            magnitude = after.abs()
            projected = torch.where(
                magnitude > EPS, after * (before.abs() / magnitude.clamp_min(EPS)), before,
            )
            dc = (slice(None), *((0,) * len(self.axes)))
            projected[dc] = before[dc]
            return torch.fft.irfftn(projected, s=state.shape[1:], dim=self.axes, norm="ortho")
        if self.constraint == "fourier_cap":
            spectrum = torch.fft.rfft(prediction, dim=-1, norm="ortho")
            factor = (self.mode_cap / spectrum.abs().clamp_min(EPS)).clamp(max=1)
            prediction = torch.fft.irfft(spectrum * factor, n=state.shape[-1], dim=-1, norm="ortho")
        rms = self.rms(prediction)
        return prediction * (self.rms_cap / rms.clamp_min(EPS)).clamp(max=1)
