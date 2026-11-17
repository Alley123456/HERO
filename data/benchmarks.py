from dataclasses import dataclass, field
from math import pi


@dataclass(frozen=True)
class Benchmark:
    scenario: str
    dimension: int
    points: int
    domain: float
    dt: float
    parameters: dict = field(default_factory=dict)

    @property
    def grid(self):
        return (self.points,) * self.dimension

    def solver(self):
        from apebench.scenarios import scenario_dict

        return scenario_dict[self.scenario](
            num_spatial_dims=self.dimension,
            num_points=self.points,
            domain_extent=self.domain,
            dt=self.dt,
            **self.parameters,
        )


BENCHMARKS = {
    "dispersion": Benchmark("phy_disp", 1, 160, 1.0, 0.001, {"dipersion_coef": 0.00025}),
    "burgers": Benchmark(
        "phy_burgers_sc", 1, 160, 1.0, 0.1,
        {"diffusion_coef": 0.0003, "convection_sc_coef": -0.125},
    ),
    "kdv": Benchmark(
        "phy_kdv", 1, 160, 2 * pi, 0.01,
        {"convection_sc_coef": -2.0, "dispersion_coef": -14.0, "hyp_diffusion_coef": -9.0},
    ),
    "ks": Benchmark(
        "phy_ks", 1, 160, 60.0, 0.1,
        {"diffusion_coef": -1.0, "hyp_diffusion_coef": -1.0, "gradient_norm_coef": -1.0},
    ),
    "anisotropic_diffusion": Benchmark(
        "phy_aniso_diff", 2, 64, 1.0, 0.1,
        {"diffusion_coef_matrix": ((0.001, 0.0005), (0.0005, 0.002))},
    ),
    "kolmogorov_flow": Benchmark(
        "phy_kolm_flow", 2, 64, 2 * pi, 0.1,
        {"diffusivity": 0.01, "drag": -0.1, "injection_mode": 4, "injection_scale": 1.0},
    ),
    "navier_stokes": Benchmark("phy_decay_turb", 2, 64, 1.0, 0.1, {"diffusivity": 0.0001}),
    "swift_hohenberg": Benchmark(
        "phy_sh", 3, 32, 10 * pi, 0.1,
        {"reactivity": 0.7, "critical_number": 1.0, "polynomial_coefficients": (0.0, 0.0, 1.0, -1.0)},
    ),
    "unbalanced_advection": Benchmark(
        "phy_unbal_adv", 3, 32, 1.0, 0.1, {"advection_coef_vector": (0.01, -0.04, 0.005)},
    ),
}

STABILITY_BENCHMARKS = ("ks", "swift_hohenberg", "unbalanced_advection")

SPLITS = {
    "hero": {"train": (50, 50, 0), "val": (10, 200, 1), "test": (30, 200, 773)},
    "stability": {"train": (50, 200, 0), "val": (32, 200, 2907), "audit": (30, 200, 3119)},
}


def dataset_directory(root, preset, pde, smoke=False):
    return root / (f"{preset}_smoke" if smoke else preset) / pde
