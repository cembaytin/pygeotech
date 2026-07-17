# Contributing to pyGeotech

Thanks for your interest in improving pyGeotech! The library aims to be a
broad, trustworthy, general-purpose geotechnical toolkit, so contributions
are held to a few simple standards.

## Principles

- **Trust through validation.** Every method must be checked against a
  published benchmark or textbook example in the test suite. Cite the
  source in the docstring.
- **Layered, decoupled design.** Keep the computation core (pure
  functions / dataclasses, NumPy-vectorised, no plotting import) separate
  from the optional plotting layer. Design-code specifics belong in
  `standards`, not in the mechanics cores.
- **Code-agnostic mechanics.** Cores return characteristic (unfactored)
  quantities; apply partial / LRFD factors through `standards`.

## Style

- Python 3.9-compatible type hints (`from typing import ...`; no `X | Y`;
  no backslashes inside f-strings).
- PEP 8, enforced with `ruff`; types checked with `mypy`.
- NumPy-style docstrings carrying the governing equations.
- Figures use the shared academic style (`pygeotech.plot_style`): serif
  typography, vector-ready PDF, 600 dpi.

## Development

```bash
git clone https://github.com/cembaytin/pygeotech
cd pygeotech
pip install -e ".[dev]"
pytest            # unit tests + doctests
ruff check .
mypy pygeotech
```

## Pull requests

1. Add tests (with a validation reference) for any new method.
2. Ensure `pytest`, `ruff` and `mypy` pass.
3. Update `CHANGELOG.md` and, if you add a submodule, `ROADMAP.md`.
