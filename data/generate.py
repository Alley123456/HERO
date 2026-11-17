import argparse
import json
import os
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from filelock import FileLock

from data.benchmarks import BENCHMARKS, SPLITS, STABILITY_BENCHMARKS, dataset_directory


def maximum_rms(states):
    return max(float(np.sqrt(np.square(trajectory.reshape(len(trajectory), -1)).mean(1)).max())
               for trajectory in states)


def load_split(directory, metadata, split):
    specification = metadata["splits"][split]
    data = np.load(directory / f"{split}.npy", mmap_mode="r", allow_pickle=False)
    if list(data.shape) != specification["shape"] or data.dtype != np.float32:
        raise ValueError(f"Invalid {split} trajectories in {directory}")
    return data


def generate(pde, preset, output, smoke):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    spec = BENCHMARKS[pde]
    directory = dataset_directory(output, preset, pde, smoke)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "metadata.json"
    template = {
        "pde": pde,
        "preset": preset,
        "smoke": smoke,
        "benchmark": asdict(spec),
        "versions": {package: version(package) for package in ("apebench", "exponax", "jax", "numpy")},
    }
    template = json.loads(json.dumps(template))
    splits = SPLITS[preset]
    with FileLock(str(directory / "generation.lock"), timeout=0):
        if marker.exists():
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            if any(metadata.get(key) != value for key, value in template.items()):
                raise ValueError(f"Dataset configuration differs in {directory}. Choose a different output root.")
            for split in splits:
                load_split(directory, metadata, split)
            print(f"{pde}: verified {directory}", flush=True)
            return
        scenario = spec.solver()
        reference = scenario.get_ref_stepper()
        metadata = {**template, "splits": {}}
        for split, (count, horizon, seed) in splits.items():
            count = 2 if smoke else count
            values = np.asarray(
                scenario.produce_data(
                    stepper=reference,
                    num_samples=count,
                    num_warmup_steps=scenario.num_warmup_steps,
                    temporal_horizon=horizon,
                    key=jax.random.PRNGKey(seed),
                ),
                dtype=np.float32,
            )[:, :, 0]
            expected = (count, horizon + 1, *spec.grid)
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(f"{pde} {split}: expected finite trajectories of shape {expected}")
            if preset == "stability" and split == "train":
                source = dataset_directory(output, "hero", pde, smoke) / "train.npy"
                if source.exists():
                    original = np.load(source, mmap_mode="r", allow_pickle=False)
                    for first, extended in zip(original, values, strict=True):
                        np.testing.assert_allclose(extended[:len(first)], first, rtol=1e-5, atol=2e-6)
            temporary = directory / f"{split}.tmp.npy"
            np.save(temporary, values, allow_pickle=False)
            temporary.replace(directory / f"{split}.npy")
            metadata["splits"][split] = {"seed": seed, "shape": list(expected)}
            if split == "train" and preset == "stability":
                metadata["splits"][split]["max_rms"] = maximum_rms(values)
            print(f"{pde} {split}: {values.shape}", flush=True)
        pending = marker.with_suffix(".tmp")
        pending.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        pending.replace(marker)


def main():
    parser = argparse.ArgumentParser(description="Generate periodic PDE trajectories with APEBench.")
    parser.add_argument("--pde", nargs="+", choices=("all", *BENCHMARKS), default=["all"])
    parser.add_argument("--preset", choices=SPLITS, default="hero")
    parser.add_argument("--output", type=Path, default=Path("datasets"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    supported = tuple(BENCHMARKS) if args.preset == "hero" else STABILITY_BENCHMARKS
    names = supported if args.pde == ["all"] else tuple(dict.fromkeys(args.pde))
    if any(name not in supported for name in names):
        parser.error(f"Choose from {', '.join(supported)} for {args.preset}.")
    for name in names:
        generate(name, args.preset, args.output, args.smoke)


if __name__ == "__main__":
    main()
