# Contributing

KonfAI is published as a Python package and developed directly from this
repository. This page documents the contribution workflow that is visible in the
repository itself.

## Local development setup

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/vboussot/KonfAI.git
cd KonfAI
python -m pip install -e .
```

Install test and documentation dependencies:

```bash
python -m pip install pytest pre-commit
python -m pip install -r docs/requirements.txt
```

## Running tests

The repository currently ships integration-style tests under `tests/`.

Run the full test suite with:

```bash
pytest -q
```

The GitHub Actions workflow in `.github/workflows/KonfAI_ci.yml` runs `pytest`
across Python `3.10` to `3.13` on Linux, macOS, and Windows.

## Running pre-commit

The repository includes `.pre-commit-config.yaml` and a dedicated
`.github/workflows/pre-commit.yml` workflow.

Set it up locally:

```bash
pre-commit install
pre-commit run --all-files
```

## Building the documentation

The documentation uses Sphinx from `docs/source/`.

Build it locally with:

```bash
make -C docs html
```

or:

```bash
python -m sphinx -b html docs/source docs/build/html
```

## Working on examples

Examples in `examples/` are part of the user-facing documentation of the
framework. When changing example YAML or notebooks:

- keep commands runnable from the example directory
- keep dataset group names and folder layouts explicit
- prefer adapting an existing example over inventing a new undocumented pattern

## Packaging and release notes

The repository contains a publish workflow in `.github/workflows/publish.yml`
that builds:

- `konfai`
- `impact-synth-konfai`
- `mrsegmentator-konfai`
- `totalsegmentator-konfai`

This is a useful reminder that changes to the core package may affect both the
framework and published KonfAI Apps.

## Documentation contributions

Documentation should stay aligned with the codebase, examples, and tests.

When updating the docs:

- prefer code-backed statements
- call out behavior inferred from code when needed
- avoid documenting private helpers unless they are essential extension points
- update cross-links when you rename or move pages

## See also

- {doc}`architecture`
- {doc}`reference/index`
- {doc}`usage/testing`
