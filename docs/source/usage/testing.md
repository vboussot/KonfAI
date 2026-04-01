# Testing and validation

KonfAI's repository currently exposes a lightweight automated test surface and a
set of runnable examples.

## Run the test suite

From the repository root:

```bash
pytest -q
```

At the time of writing, the repository includes an integration test for the
`konfai-apps pipeline` flow in `konfai-apps/tests/integration/test_konfai_apps.py`.

## What CI runs

The GitHub workflow `konfai_ci.yml` runs:

- editable installation of the package
- `pytest -q`

across:

- Linux
- macOS
- Windows
- Python 3.10 to 3.13

## Validate an example manually

The most practical manual validation loop is still:

1. run a shipped example
2. inspect `Checkpoints/`, `Predictions/`, `Evaluations/`, and `Statistics/`
3. confirm that the generated config copy matches the intended run

## Validate the documentation

From `docs/`:

```bash
make html
```

This uses the Sphinx configuration in `docs/source/conf.py`.

## See also

- :doc:`../contributing`
- :doc:`../examples/index`
