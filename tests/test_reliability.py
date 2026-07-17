"""Unit tests for the reliability helpers."""

import numpy as np
import pytest

from pygeotech.reliability import (
    latin_hypercube,
    monte_carlo_uniform,
    propagate,
    summarize,
)


class TestSampling:
    def test_latin_hypercube_shape_and_bounds(self) -> None:
        s = latin_hypercube(500, {"a": (0.0, 1.0), "b": (10.0, 20.0)}, seed=0)
        assert set(s) == {"a", "b"}
        assert len(s["a"]) == 500
        assert s["a"].min() >= 0.0 and s["a"].max() <= 1.0
        assert s["b"].min() >= 10.0 and s["b"].max() <= 20.0

    def test_latin_hypercube_is_stratified(self) -> None:
        # With N samples every 1/N band should contain exactly one point.
        n = 100
        s = latin_hypercube(n, {"x": (0.0, 1.0)}, seed=1)
        counts, _ = np.histogram(s["x"], bins=n, range=(0.0, 1.0))
        assert np.all(counts == 1)

    def test_reproducible_with_seed(self) -> None:
        a = latin_hypercube(50, {"x": (0.0, 1.0)}, seed=42)
        b = latin_hypercube(50, {"x": (0.0, 1.0)}, seed=42)
        assert np.allclose(a["x"], b["x"])

    def test_monte_carlo_uniform(self) -> None:
        s = monte_carlo_uniform(1000, {"x": (-2.0, 2.0)}, seed=3)
        assert s["x"].min() >= -2.0 and s["x"].max() <= 2.0


class TestPropagate:
    def test_propagate_and_summary(self) -> None:
        samples = latin_hypercube(1000, {"a": (0.0, 2.0), "b": (0.0, 2.0)},
                                  seed=7)
        out = propagate(lambda a, b: a + b, samples)
        assert out.shape == (1000,)
        stats = summarize(out)
        assert stats["mean"] == pytest.approx(2.0, abs=0.1)
        assert stats["min"] <= stats["p50"] <= stats["max"]

    def test_propagate_with_fixed_kwargs(self) -> None:
        samples = {"x": np.array([1.0, 2.0, 3.0])}
        out = propagate(lambda x, k: x * k, samples, k=10.0)
        assert np.allclose(out, [10.0, 20.0, 30.0])
