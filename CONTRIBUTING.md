# Contributing to NoisiQ

Thanks for helping! NoisiQ aims to be **physically correct, well tested and easy to read**.
Please read the [maintainer guide](docs/maintainers.md) for the architecture and the design
rules (in short: never dispatch on gate names in simulators, attach noise with rules, put
math in `catalog.py`).

## Setup

```bash
git clone https://github.com/dtsaxum-oss/NoisiQ.git
cd NoisiQ
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                             # should be all green
```

## Branches

`main` is always releasable and only changes through pull requests. Work on short-lived
branches named `feature/<topic>`, `fix/<topic>` or `docs/<topic>` (lowercase, hyphenated),
branched from the latest `main`.

## Commit messages

```
<type>: <short summary in the present tense, under 72 characters>
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `chore`. Add a body after a blank
line when the *why* is not obvious.

## Code style

* Formatting and linting: `ruff format .` and `ruff check .` (configured in `pyproject.toml`; CI enforces both).
* Type hints on public functions; Google-style docstrings (`Args:`, `Returns:`, `Raises:`, `Example:`).
* Public functions explain *what* and *why*; the math for channels/metrics goes in
  `src/noisiq/catalog.py` (then run `python -m noisiq.catalog docs/math.md`).
* Prefer clear errors that tell the user what to do (e.g. which backend to use).

## Tests

Every change needs tests. Test behaviour and physics through the public API, not private
helpers. Useful patterns already in the suite:

* compare against the exact `density_matrix` backend (`tests/test_backends.py`);
* random circuits from `tests/conftest.py::random_circuit`;
* fixed seeds and tolerances sized for the number of shots.

```bash
pytest -m "not slow"     # quick loop
pytest                   # full suite, run before opening a PR
```

## Pull requests

* Keep PRs focused; describe what changed and how you verified it.
* CI (lint + tests on Python 3.10–3.13) must pass.
* Update `CHANGELOG.md` under "Unreleased" and the docs if behaviour changed.
* At least one review before merging; squash-merge into `main`.

## What not to commit

Build artifacts, notebook outputs, `examples/output/`, LaTeX auxiliary files, virtual
environments, caches, credentials. `.gitignore` covers these — check `git status` before committing.
