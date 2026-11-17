import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from data.benchmarks import BENCHMARKS, STABILITY_BENCHMARKS, dataset_directory
from data.generate import load_split
from model import FNO, HERO
from model.hero import EPS, relative_error, rollout
from model.stability import CONSTRAINTS, StableFNO, training_limits


class WindowSampler:
    def __init__(self, states, horizon, batch_size, seed):
        self.states = states
        self.horizon = horizon
        self.batch_size = batch_size
        self.seed = seed
        self.windows = states.shape[1] - horizon
        self.samples = len(states) * self.windows
        self.batches = self.samples // batch_size
        self.epoch = -1

    def batch(self, step):
        epoch, offset = divmod(step, self.batches)
        if epoch != self.epoch:
            self.order = np.random.default_rng([self.seed, epoch]).permutation(self.samples)
            self.epoch = epoch
        indices = self.order[offset * self.batch_size:(offset + 1) * self.batch_size]
        trajectories, starts = np.divmod(indices, self.windows)
        frames = starts[:, None] + np.arange(self.horizon + 1)
        return self.states[trajectories[:, None], frames]


def stability_batch(states, step, seed, batch_size, horizon, lead, early):
    rng = np.random.default_rng([seed, step, 441])
    trajectories = rng.integers(0, len(states), size=batch_size)
    starts = rng.integers(0, states.shape[1] - horizon - lead, size=batch_size)
    if early:
        count = batch_size // 2
        starts[:count] = rng.integers(0, 51 - horizon - lead, size=count)
    frames = starts[:, None] + np.arange(horizon + lead + 1)
    return states[trajectories[:, None], frames]


def read_data(root, preset, pde, smoke, splits):
    directory = dataset_directory(root, preset, pde, smoke)
    marker = directory / "metadata.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Generate {preset} data for {pde} with python -m data.generate.")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if (metadata["pde"], metadata["preset"], metadata["smoke"]) != (pde, preset, smoke):
        raise ValueError(f"Unexpected dataset metadata in {directory}")
    return {name: load_split(directory, metadata, name) for name in splits}, metadata


@torch.inference_mode()
def evaluate(model, trajectories, device, batch_size, horizon):
    model.eval()
    recorded = []
    for offset in range(0, len(trajectories), batch_size):
        sample = torch.from_numpy(trajectories[offset:offset + batch_size, :horizon + 1].copy()).to(device)
        state = sample[:, 0]
        errors = []
        for step in range(1, horizon + 1):
            state = model(state).float()
            errors.append(relative_error(state[:, None], sample[:, step:step + 1])[:, 0])
        recorded.append(torch.stack(errors, dim=1).cpu().numpy())
    result = np.concatenate(recorded).astype(np.float64)
    result[~np.isfinite(result)] = np.inf
    return result


def measures(errors):
    first_100 = errors[:, :100]
    stable = (errors <= 0.1).cumprod(axis=1)
    result = {
        "gm100": np.exp(np.log(first_100 + EPS).mean(1)).mean(),
        "nrmse_at_1": first_100[:, 0].mean(),
        "nrmse_at_100": first_100[:, -1].mean(),
        "stable_step": stable[:, :100].sum(1).mean(),
    }
    if errors.shape[1] == 200:
        result.update(nrmse_at_200=errors[:, -1].mean(),
                      stable_step_200=stable.sum(1).mean(),
                      mean_nrmse_200=errors.mean())
    record = {name: float(value) if np.isfinite(value) else None for name, value in result.items()}
    if errors.shape[1] == 200:
        record["rollouts_over_10"] = int((errors > 10).any(axis=1).sum())
    return record


def validation_score(record, stage):
    if stage == "hero":
        return record["gm100"] if record["gm100"] is not None else math.inf
    mean, endpoint = record["mean_nrmse_200"], record["nrmse_at_200"]
    if mean is None or endpoint is None:
        return math.inf
    return mean + 0.25 * endpoint + 0.25 * (1 - record["stable_step_200"] / 200)


def save_json(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def save_checkpoint(path, value):
    pending = path.with_suffix(".tmp")
    torch.save(value, pending)
    pending.replace(path)


def save_errors(path, value):
    pending = path.with_suffix(".tmp.npy")
    np.save(pending, value, allow_pickle=False)
    pending.replace(path)


def weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def base_checkpoint(path, pde):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    exercise = config.get("pde", config.get("task"))
    if exercise != pde or config.get("stage", "hero") != "hero":
        raise ValueError(f"Expected a HERO checkpoint for {pde} at {path}")
    return checkpoint["model"]


def initialize_stability(pde, warm_start, limits, validation, device, eval_batch):
    origins = ("best",) if pde == "swift_hohenberg" else ("best", "last")
    candidates = []
    for origin in origins:
        source = base_checkpoint(warm_start / f"{origin}.pt", pde)
        if pde == "swift_hohenberg":
            source = dict(source)
            for name in ("project.weight", "project.bias"):
                source[name] = torch.zeros_like(source[name])
        for constraint in CONSTRAINTS[pde]:
            model = StableFNO(pde, constraint, limits).to(device)
            model.base.load_state_dict(source)
            record = measures(evaluate(model, validation, device, eval_batch, 200))
            candidates.append((validation_score(record, "stability"), origin, constraint, record, weights(model)))
    _, origin, constraint, record, initial = min(candidates, key=lambda value: value[0])
    selected = {"origin": origin, "constraint": constraint}
    print(json.dumps({"initialization": selected, "validation": record}), flush=True)
    best = None if pde == "swift_hohenberg" else {"step": 0, "validation": record, "model": initial}
    return selected, initial, best


def rate(step, steps, peak, warmup, stage):
    if stage == "stability":
        return peak * 0.5 * (1 + math.cos(math.pi * (step + 1) / steps))
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def settings(args, metadata, original_meta, device, warm_start):
    spec = BENCHMARKS[args.pde]
    stable = args.stage == "stability"
    swift = args.pde == "swift_hohenberg"
    steps = 600 if swift else 1200
    if not stable:
        steps = 10000
    batch = 10 if stable and spec.dimension == 3 else 20
    return {
        "pde": args.pde, "stage": args.stage, "seed": args.seed, "device": device.type,
        "steps": args.steps or (4 if args.smoke else steps),
        "eval_every": args.eval_every or (2 if args.smoke else 300 if stable else 500),
        "batch_size": 2 if args.smoke else batch,
        "microbatch": min(args.microbatch or batch, 2 if args.smoke else batch),
        "rollout_steps": (12 if args.pde == "ks" else 10) if stable else 5,
        "peak_lr": (0.0001 if swift else 0.0002) if stable else 0.001,
        "lr_warmup": 2 if args.smoke else 2000,
        "hero_start": 1 if args.smoke else 4000,
        "hero_refresh": 2 if args.smoke else 2000,
        "hero_ramp": 2 if args.smoke else 2000,
        "weight_decay": 0.001 if stable else 0.01,
        "data": metadata, "test_data": original_meta,
        "warm_start": str(warm_start) if stable else None,
        "torch_version": str(torch.__version__),
    }


def fit(args, out):
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select --device cpu or install a CUDA-enabled PyTorch build.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    pde, stage = args.pde, args.stage
    stable = stage == "stability"
    splits = ("train", "val", "audit") if stable else ("train", "val", "test")
    data, metadata = read_data(args.data_dir, stage, pde, args.smoke, splits)
    original_meta = metadata
    if stable:
        original, original_meta = read_data(args.data_dir, "hero", pde, args.smoke, ("test",))
        data.update(original)
    warm_start = args.warm_start or Path("runs") / ("hero_smoke" if args.smoke else "hero") / pde / f"seed_{args.seed}"
    config = settings(args, metadata, original_meta, device, warm_start)
    latest = out / "last.pt"
    resume = torch.load(latest, weights_only=True, map_location="cpu") if latest.is_file() else None
    if resume and resume["config"] != config:
        raise ValueError(f"Checkpoint configuration differs in {out}. Use a separate --out.")
    if (out / "metrics.json").is_file():
        if resume is None or resume["step"] != config["steps"]:
            raise ValueError(f"Incomplete training state beside metrics in {out}")
        print((out / "metrics.json").read_text(encoding="utf-8"), flush=True)
        return

    spec = BENCHMARKS[pde]
    eval_batch = min(config["microbatch"], 10 if stable and spec.dimension == 3 else 5)
    horizon = config["rollout_steps"]
    batch_size = config["batch_size"]
    steps = config["steps"]
    limits = training_limits(pde, data["train"], metadata) if stable else {}
    decision, best = {}, None
    start = resume["step"] if resume else 0
    if stable:
        if resume:
            decision, initial, best = resume["initialization"], resume["model"], resume["best"]
        else:
            decision, initial, best = initialize_stability(pde, warm_start, limits, data["val"], device, eval_batch)
        model = StableFNO(pde, decision["constraint"], limits).to(device)
        model.load_state_dict(initial)
        weights_by_step = torch.linspace(0.75, 1.25, horizon, device=device)
    else:
        model = FNO(spec.dimension).to(device)
        hero = HERO(model, spec.grid, device, config["hero_start"], config["hero_refresh"], config["hero_ramp"])
        sampler = WindowSampler(data["train"], horizon, batch_size, args.seed)
        if resume:
            model.load_state_dict(resume["model"])
            hero.lagged.load_state_dict(resume["lagged"])
            best = resume["best"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["peak_lr"], weight_decay=config["weight_decay"])
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        torch.set_rng_state(resume["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(resume["cuda_rng"])
        print(f"Resuming {pde} at step {start} of {steps}", flush=True)

    save_json(out / "config.json", config)
    for step in range(start, steps):
        model.train()
        optimizer.param_groups[0]["lr"] = rate(step, steps, config["peak_lr"], config["lr_warmup"], stage)
        optimizer.zero_grad(set_to_none=True)
        if stable:
            lead = 3 if step % 4 == 0 else 0
            window = stability_batch(data["train"], step, args.seed, batch_size, horizon, lead,
                                     decision["constraint"] == "swift_integrator")
        else:
            hero.begin_step(model, step)
            window = sampler.batch(step)
        total = torch.zeros((), device=device)
        for fragment in torch.from_numpy(window).to(device).split(config["microbatch"]):
            initial = fragment[:, 0]
            if stable:
                with torch.no_grad():
                    for _ in range(lead):
                        initial = model(initial)
                target = fragment[:, lead + 1:]
                error = relative_error(rollout(model, initial, horizon), target)
                loss = (error * weights_by_step).mean() * (len(fragment) / batch_size)
            else:
                loss = hero.loss(model, initial, fragment[:, 1:], step).sum() / batch_size
            loss.backward()
            total += loss.detach()
        measured = float(total)
        if not math.isfinite(measured):
            raise FloatingPointError(f"Non-finite loss at optimizer step {step + 1}")
        if stable:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        finished = step + 1
        if finished == 1 or finished % 100 == 0 or args.smoke:
            print(f"{pde} {stage} step {finished}/{steps} loss={measured:.6g}", flush=True)
        if finished % config["eval_every"] == 0 or finished == steps:
            length = 200 if stable else 100
            record = measures(evaluate(model, data["val"], device, eval_batch, length))
            score = validation_score(record, stage)
            previous = math.inf if best is None else validation_score(best["validation"], stage)
            if score < previous:
                best = {"model": weights(model), "validation": record, "step": finished}
            print(json.dumps({"step": finished, "validation": record}), flush=True)
            state = {
                "config": config, "step": finished, "model": model.state_dict(), "best": best,
                "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                "initialization": decision,
            }
            if not stable:
                state["lagged"] = hero.lagged.state_dict()
            save_checkpoint(latest, state)

    if best is None:
        raise FloatingPointError("Every validation checkpoint had non-finite error.")
    model.load_state_dict(best["model"])
    result = {
        "pde": pde, "stage": stage, "seed": args.seed, "training_steps": steps,
        "best_step": best["step"], "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "validation": best["validation"], "initialization": decision,
    }
    for split in (("test", "audit") if stable else ("test",)):
        error = evaluate(model, data[split], device, eval_batch, 200)
        save_errors(out / f"{split}_errors.npy", error)
        result[split] = measures(error)
    save_checkpoint(out / "best.pt", {
        "config": config, "model": best["model"], "step": best["step"],
        "initialization": decision, "limits": limits,
    })
    save_json(out / "metrics.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an FNO with HERO or stability tuning.")
    parser.add_argument("--pde", choices=BENCHMARKS, required=True)
    parser.add_argument("--stage", choices=("hero", "stability"), default="hero")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or any(size is not None and size < 1
                           for size in (args.threads, args.steps, args.eval_every, args.microbatch)):
        parser.error("Use a nonnegative seed and positive sizes.")
    if args.stage == "stability" and args.pde not in STABILITY_BENCHMARKS:
        parser.error(f"Stability tuning is configured for {', '.join(STABILITY_BENCHMARKS)}.")
    if args.warm_start and args.stage != "stability":
        parser.error("--warm-start is a stability-tuning option.")
    group = f"{args.stage}_smoke" if args.smoke else args.stage
    out = args.out or Path("runs") / group / args.pde / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / "training.lock"), timeout=0):
        fit(args, out)


if __name__ == "__main__":
    main()
