from copy import deepcopy

import torch
from torch.nn import functional as F


EPS = 1e-8


def relative_error(prediction, target):
    error = torch.linalg.vector_norm((prediction - target).flatten(2), dim=-1)
    scale = torch.linalg.vector_norm(target.flatten(2), dim=-1)
    return error / (scale + EPS)


def rollout(operator, initial, horizon):
    states = []
    current = initial
    for _ in range(horizon):
        current = operator(current).float()
        states.append(current)
    return torch.stack(states, dim=1)


class HERO:
    def __init__(self, operator, shape, device, start=4000, refresh=2000, ramp=2000):
        self.lagged = deepcopy(operator).eval().requires_grad_(False)
        self.start = start
        self.refresh = refresh
        self.ramp = ramp
        axes = [torch.fft.fftfreq(size, device=device) for size in shape[:-1]]
        axes.append(torch.fft.rfftfreq(shape[-1], device=device))
        radius = sum(axis.square() for axis in torch.meshgrid(*axes, indexing="ij"))
        self.mask = (radius > 0) & (radius <= 0.125**2)
        self.spatial_dims = tuple(range(1, len(shape) + 1))
        self.trajectory_dims = tuple(axis + 1 for axis in self.spatial_dims)
        self.multiplicity = torch.full((shape[-1] // 2 + 1,), 2.0, device=device)
        self.multiplicity[0] = 1.0
        if shape[-1] % 2 == 0:
            self.multiplicity[-1] = 1.0

    def begin_step(self, operator, step):
        if step >= self.start and (step - self.start) % self.refresh == 0:
            self.lagged.load_state_dict(operator.state_dict())

    def perturb(self, state):
        spectrum = torch.fft.rfftn(torch.randn_like(state), dim=self.spatial_dims)
        noise = torch.fft.irfftn(spectrum * self.mask, s=state.shape[1:], dim=self.spatial_dims)
        state_rms = state.square().mean(self.spatial_dims, keepdim=True).sqrt()
        noise_rms = noise.square().mean(self.spatial_dims, keepdim=True).sqrt()
        return state + 0.01 * state_rms * noise / (noise_rms + EPS)

    def spectrum(self, trajectory):
        return torch.fft.rfftn(trajectory, dim=self.trajectory_dims).abs()

    @torch.no_grad()
    def reference_distance(self, operator, initial, current, target):
        horizon = target.shape[1]
        candidates = (
            rollout(self.lagged, initial, horizon),
            rollout(operator, self.perturb(initial), horizon),
            current.detach(),
        )
        target_spectrum = self.spectrum(target)
        spectral_scale = (target_spectrum * self.multiplicity).flatten(2).sum(-1)
        target_energy = target.flatten(2).square().mean(-1)
        diagnostics = []
        for candidate in candidates:
            errors = relative_error(candidate, target)
            spectral_diff = (self.spectrum(candidate) - target_spectrum).abs() * self.multiplicity
            spectral_error = (spectral_diff.flatten(2).sum(-1) / (spectral_scale + EPS)).mean(1)
            energy_diff = (candidate.flatten(2).square().mean(-1) - target_energy).abs()
            energy_error = (energy_diff / (target_energy + EPS)).mean(1)
            growth = errors.diff(dim=1).clamp_min(0).mean(1)
            diagnostics.append(torch.stack((errors.mean(1), spectral_error, energy_error, growth), dim=-1))
        values = torch.stack(diagnostics, dim=1)
        minimum = values.amin(dim=1, keepdim=True)
        span = values.amax(dim=1, keepdim=True) - minimum
        selected = ((values - minimum) / (span + EPS)).mean(-1).argmax(1)
        return values[:, :, 0].gather(1, selected[:, None])[:, 0]

    def loss(self, operator, initial, target, step):
        current = rollout(operator, initial, target.shape[1])
        distance = relative_error(current, target).mean(1)
        if step < self.start:
            return distance
        failure = self.reference_distance(operator, initial, current, target)
        weight = 0.02 * min((step - self.start) / self.ramp, 1.0)
        return distance + weight * F.softplus(distance - failure + 0.02, beta=5.0)
