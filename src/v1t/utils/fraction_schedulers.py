import math
import torch

_FRACTION_SCHEDULERS = {}

def register_fraction(name):
    def add_to_dict(fn):
        _FRACTION_SCHEDULERS[name] = fn
        return fn
    return add_to_dict

@register_fraction("random")
def random_fraction(step: int, warmup_steps: int, f0: float, fmax: float) -> float:
    return float(torch.rand(1).item() * (fmax - f0) + f0)

@register_fraction("linear")
def linear_ramp_hold(step: int, warmup_steps: int, f0: float, fmax: float) -> float:
    """
    Linear ramp from f0 → fmax over `warmup_steps`, then hold fmax.
    """
    if step < warmup_steps:
        return f0 + (fmax - f0) * (step / max(1, warmup_steps))
    return fmax

@register_fraction("cosine")
def cosine_ramp_hold(step: int, warmup_steps: int, f0: float, fmax: float) -> float:
    """
    Cosine ramp from f0 → fmax over `warmup_steps`, then hold fmax.
    Smooth start/end (zero slope at boundaries).
    """
    if step < warmup_steps:
        return f0 + (fmax - f0) * (1 - math.cos(math.pi * step / max(1, warmup_steps))) / 2
    return fmax

@register_fraction("exponential")
def exponential_ramp_hold(step: int, warmup_steps: int, f0: float, fmax: float) -> float:
    """
    Exponential ramp from f0 → fmax across `warmup_steps`, then hold fmax.
    """
    if step < warmup_steps:
        alpha = (fmax / f0) ** (1 / max(1, warmup_steps))
        return min(fmax, f0 * (alpha ** step))
    return fmax

def asymptotic(step: int, warmup_steps: int, f0: float, fmax: float) -> float:
    """Asymptotically approaches fmax: f = fmax - (fmax - f0) * exp(-step / tau). Use warmup_steps as tau."""
    tau = max(1, warmup_steps)
    return fmax - (fmax - f0) * math.exp(-step / tau)

# def get_fraction_scheduler(args):
#     if not args.fraction_scheduler in _FRACTION_SCHEDULERS.keys():
#         raise NotImplementedError(f"Fraction scheduler {args.fraction_scheduler} has not been implemented.")
#     return _FRACTION_SCHEDULERS[args.fraction_scheduler]


def get_fraction_scheduler(name: str,
                           f0: float = 0.1,
                           fmax: float = 0.95,
                           warmup_steps: int = 5_000):
    """
    Returns a callable frac(step)->float using the registered schedule `name`.
    For 'asymptotic', warmup_steps is used as tau.
    """
    key = name.lower()
    if key not in _FRACTION_SCHEDULERS:
        raise ValueError(f"Unknown fraction scheduler '{name}'. "
                         f"Available: {list(_FRACTION_SCHEDULERS.keys())}")
    fn = _FRACTION_SCHEDULERS[key]
    def frac(step: int) -> float:
        return fn(step, warmup_steps, f0, fmax)
    return frac