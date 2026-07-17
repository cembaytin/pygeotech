"""Lightweight parametric / stochastic sampling helpers.

These utilities let any pyGeotech core function (or any callable) be swept
over Monte-Carlo or Latin Hypercube samples, which is the typical driver
for reliability analysis, sensitivity studies and fragility curves in
geotechnical research. They depend only on NumPy.

Examples
--------
>>> import numpy as np
>>> from pygeotech.reliability import latin_hypercube, propagate, summarize
>>> from pygeotech.foundations import ShallowFoundation
>>> def q_all(phi, cohesion):
...     return ShallowFoundation(width=2.0, depth=1.0, phi=phi,
...                              cohesion=cohesion).capacity().q_allowable_net
>>> samples = latin_hypercube(1000, {"phi": (28.0, 34.0),
...                                  "cohesion": (0.0, 10.0)}, seed=0)
>>> out = propagate(q_all, samples)
>>> stats = summarize(out)   # mean, std, percentiles
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

__all__ = ["latin_hypercube", "monte_carlo_uniform", "propagate", "summarize"]


def latin_hypercube(
    n: int,
    bounds: Dict[str, Tuple[float, float]],
    seed: int = None,
) -> Dict[str, np.ndarray]:
    """Draw a Latin Hypercube sample over uniform marginal ranges.

    Parameters
    ----------
    n
        Number of samples.
    bounds
        Mapping ``name -> (low, high)`` for each variable.
    seed
        Optional RNG seed for reproducibility.

    Returns
    -------
    dict
        Mapping ``name -> ndarray`` of length ``n``.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    rng = np.random.default_rng(seed)
    samples: Dict[str, np.ndarray] = {}
    edges = np.linspace(0.0, 1.0, n + 1)
    for name, (low, high) in bounds.items():
        if high < low:
            raise ValueError(f"bounds for {name!r} are reversed.")
        stratified = rng.uniform(edges[:-1], edges[1:])
        rng.shuffle(stratified)
        samples[name] = low + stratified * (high - low)
    return samples


def monte_carlo_uniform(
    n: int,
    bounds: Dict[str, Tuple[float, float]],
    seed: int = None,
) -> Dict[str, np.ndarray]:
    """Plain (unstratified) uniform Monte-Carlo sample."""
    if n < 1:
        raise ValueError("n must be >= 1.")
    rng = np.random.default_rng(seed)
    return {
        name: rng.uniform(low, high, size=n)
        for name, (low, high) in bounds.items()
    }


def propagate(
    func: Callable[..., float],
    samples: Dict[str, np.ndarray],
    **fixed: float,
) -> np.ndarray:
    """Evaluate ``func`` over each sample, returning an array of outputs.

    Parameters
    ----------
    func
        Callable taking the sampled names (and any ``fixed`` kwargs) as
        keyword arguments and returning a scalar.
    samples
        Mapping ``name -> ndarray`` (all equal length), e.g. from
        :func:`latin_hypercube`.
    **fixed
        Additional constant keyword arguments passed to ``func``.
    """
    names = list(samples)
    if not names:
        raise ValueError("no samples provided.")
    n = len(samples[names[0]])
    out = np.empty(n, dtype=float)
    for i in range(n):
        kwargs = {name: samples[name][i] for name in names}
        kwargs.update(fixed)
        out[i] = func(**kwargs)
    return out


def summarize(values: np.ndarray) -> Dict[str, float]:
    """Return summary statistics of a Monte-Carlo output array."""
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "cov": float(np.std(values, ddof=1) / np.mean(values))
        if values.size > 1 and np.mean(values) != 0.0 else 0.0,
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }
