# Contributing

Thanks for your interest in improving correlation-network. This guide covers
the development setup and the conventions the project follows.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and supports Python 3.10+.

```bash
git clone https://github.com/skateddu/correlation-network.git
cd correlation-network
uv sync --all-extras
```

`uv sync` installs the `dev` group (pytest, ruff) and the `typing` group (mypy
and the pandas/scipy/networkx stubs).

## Checks

CI runs the same commands; run them before opening a pull request.

```bash
# Tests (with coverage; CI requires at least 80%)
uv run pytest --cov=src/correlation_network --cov-report=term-missing

# A single file or a pattern
uv run pytest tests/unit/test_network.py
uv run pytest -k "centrality"

# Lint, format and type-check
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy
```

CI also runs the test suite against the lowest dependency versions declared in
`pyproject.toml`. To reproduce it locally, without touching your main
environment or `uv.lock`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-lowest uv sync --all-extras --no-group typing \
    --resolution lowest-direct --python 3.10
UV_PROJECT_ENVIRONMENT=.venv-lowest uv run --no-sync pytest
git checkout -- uv.lock
```

If you raise or lower a dependency's minimum, make sure that job still passes.

## Conventions

- **Branches**: `feature/`, `fix/`, `refactor/`, `chore/` or `docs/` followed by
  a short kebab-case description, branched from `main`.
- **Commits and PR titles**: [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat: ...`, `fix: ...`), imperative mood, lowercase, no trailing period.
- **Changelog**: add an entry under `[Unreleased]` in `CHANGELOG.md` in the same
  PR, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
- **Code style**: type hints on every function, NumPy-style docstrings, keyword
  arguments at call sites, comments that explain *why* rather than *what*.
- **Tests**: every bug fix comes with a regression test; new behaviour comes with
  unit tests in `tests/unit/`, mirroring the module under test.

## Adding an association measure

1. Implement a class with `compute(self, dataframe: pd.DataFrame) -> pd.DataFrame`
   returning a symmetric matrix (see `association/_base.py`).
2. Evaluate each pair on its pairwise-complete rows and raise
   `NoCompleteObservationsError` when a pair has none (`complete_pair` and
   `check_pairwise_overlap` in `_validation.py` do this).
3. Register it in `_STRATEGY_REGISTRY` (`network.py`) and in the
   `AssociationMethod` literal (`_types.py`) if it should be selectable by name.
4. Export it from `association/__init__.py` and the package `__init__.py`, and
   document it in the README's *Association Methods* section.
