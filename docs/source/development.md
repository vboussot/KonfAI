# Development setup

This page covers everything needed to contribute to KonfAI or run the test
suite locally. The recommended path uses [Pixi](https://pixi.sh), which
manages Python packages, system libraries, and task runners in a single
reproducible environment.

## Prerequisites

- **Python 3.10 or later** — the minimum version declared in `pyproject.toml`
- **Pixi** — install once with:

  ```bash
  curl -fsSL https://pixi.sh/install.sh | bash
  ```

  See [pixi.sh](https://pixi.sh) for alternative installers.
- **git**

## Clone and install

```bash
git clone https://github.com/vboussot/KonfAI.git
cd KonfAI
pixi install       # resolves and installs all Pixi environments
```

`pixi install` creates isolated environments under `.pixi/` and does **not**
touch your system Python or any other virtual environment.

## Available tasks

Run tasks with `pixi run <task>`:

| Task | Command | Description |
| --- | --- | --- |
| `test` | `pytest -q tests/` | Run the full test suite |
| `test-cov` | `pytest --cov=konfai tests/` | Run tests with coverage report |
| `lint` | `ruff check konfai konfai-apps/konfai_apps` | Lint the source tree |
| `format` | `ruff format konfai konfai-apps/konfai_apps` | Auto-format source files |
| `format-check` | `ruff format --check ...` | Check formatting without modifying files |
| `typecheck` | `mypy konfai --ignore-missing-imports` | Static type checking |
| `build` | `python -m build` | Build sdist and wheel |
| `check` | lint + format-check + test | Full pre-push gate — run before finishing any change |

Always run `pixi run check` before pushing or opening a PR.

## pip fallback

If Pixi is unavailable, use an editable pip install:

```bash
pip install -e ".[dev]"
pytest -q tests/
ruff check konfai
ruff format konfai
```

## Pre-commit hooks

The repository ships a `.pre-commit-config.yaml` with both source-file checks and commit-message validation. Install
both hook types once:

```bash
# with Pixi:
pixi run pre-commit-install

# or with pip:
python -m pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

After installation, `git commit` runs file checks plus Conventional Commit and forbidden-branding validation. Run all
file checks manually with:

```bash
pre-commit run --all-files
```

## Branches, commits, and pull requests

Never commit directly to `main`. Create a focused feature branch for every change:

```bash
git switch -c fix/short-description
```

Use a Conventional Commit message such as `fix(config): improve YAML validation errors`. Commit messages must not
contain agent names, generated-by/generated-with branding, or AI co-author trailers. The `commit-msg` hooks validate
both the Conventional Commit structure and forbidden branding.

Before pushing, run `pixi run format`, `pixi run check`, and `pre-commit run --all-files`. Push the feature branch,
open a pull request, and leave it open for a maintainer to review and merge; do not merge your own PR.

## Writing and running tests

Tests live under `tests/unit/`. Follow the conventions already established
there:

- one file per module under test (e.g. `tests/unit/test_config.py`)
- use `pytest` fixtures and `monkeypatch` for environment variables
- never import `SimpleITK` or `h5py` unconditionally — guard with `pytest.importorskip`

Run a single test file:

```bash
pixi run test -- tests/unit/test_config.py -v
```

## Building the documentation

The documentation uses Sphinx with the MyST parser for Markdown files.

Build the HTML output:

```bash
pixi run -e docs build-docs
```

Or in live-reload mode during authoring:

```bash
pixi run -e docs dev-docs
```

Without Pixi:

```bash
pip install -r docs/requirements.txt
make -C docs html
```

The output lands in `docs/_build/html/`.

## AI agent rules

If you are an AI agent contributing to this repository, read `AGENTS.md` at
the repository root before making changes. It is the canonical source for branch and PR rules, Conventional Commits,
forbidden commit branding, coding norms, checks, and project-specific pitfalls.
