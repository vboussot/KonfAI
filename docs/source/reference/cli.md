# CLI reference

This page lists the main command-line entrypoints used in the repository. Use
it as the quick map of "which command should I run?".

KonfAI uses four main command-line entrypoints:

- `konfai`
- `konfai-apps`
- `konfai-apps-server`
- `konfai-cluster`

## `konfai`

Low-level workflow runner for training, prediction, and evaluation.

Use `konfai` when you are still designing a workflow directly from YAML.

### Commands

| Command | Purpose |
| --- | --- |
| `TRAIN` | Train a model from scratch. |
| `RESUME` | Resume training from a checkpoint. |
| `PREDICTION` | Run inference using one or more checkpoints. |
| `EVALUATION` | Compute metrics on saved outputs. |

### Common options

| Option | Meaning |
| --- | --- |
| `-c`, `--config` | YAML file to use. |
| `-y`, `--overwrite` | Overwrite existing outputs without prompting. |
| `--gpu` | One or more GPU ids. |
| `--cpu` | Number of CPU workers when not using GPUs. |
| `-q`, `--quiet` | Reduce console output. |
| `-tb`, `--tensorboard` | Launch TensorBoard. |

### Command-specific options

`TRAIN`

- `--checkpoints-dir`
- `--statistics-dir`

`RESUME`

- `--model`
- `--checkpoints-dir`
- `--statistics-dir`

`PREDICTION`

- `--models`
- `--predictions-dir`

`EVALUATION`

- `--evaluations-dir`

## `konfai-apps`

Higher-level packaged workflow runner.

Use `konfai-apps` when a workflow is already packaged as a KonfAI App and you
want a simpler interface than the low-level YAML CLI.

This command is provided by the standalone `konfai-apps` package.

### Commands

| Command | Purpose |
| --- | --- |
| `infer` | Run inference for an app. |
| `eval` | Run evaluation for an app. |
| `uncertainty` | Run uncertainty estimation for an app. |
| `pipeline` | Chain inference, evaluation, and optional uncertainty. |
| `fine-tune` | Fine-tune an app on a dataset. |

### Shared options

| Option | Meaning |
| --- | --- |
| `app` | App identifier or repository path. |
| `--host`, `--port`, `--token` | Switch from local app execution to remote server mode. |
| `-i`, `--inputs` | Input paths, grouped by repeated flag occurrences. |
| `-o`, `--output` | Output directory. |
| `--gpu` / `--cpu` | Device selection. |
| `-q`, `--quiet` | Reduce console output. |
| `--download` | Pre-download the full app locally. |
| `--force_update` | Force an updated app download. |

### Important command-specific options

`infer`

- `--ensemble`
- `--ensemble-models`
- `--tta`
- `--mc`
- `-uncertainty`
- `--prediction-file` (alias: `--prediction_file`)

`eval`

- `--gt`
- `--mask`
- `--evaluation-file` (alias: `--evaluation_file`)

`uncertainty`

- `--uncertainty-file` (alias: `--uncertainty_file`)

`pipeline`

- combines the options from `infer`, `eval`, and `uncertainty`

`fine-tune`

- positional `name`
- `-d`, `--dataset`
- `--epochs`
- `--it-validation`

## `konfai-apps-server`

FastAPI server exposing packaged apps remotely.

This command is the server-side counterpart of `konfai-apps --host ...`.
It is also provided by the standalone `konfai-apps` package.

Important options:

| Option | Meaning |
| --- | --- |
| `--host` | Bind address. |
| `--port` | Bind port. |
| `--auth` | `off` or `bearer`. |
| `--token-env` | Environment variable holding the token. |
| `--token` | Development-only token override. |
| `--apps` | JSON file listing the available apps. |
| `--download` | Pre-download configured apps at startup. |
| `--check` | Validate configured apps without downloading them. |

## `konfai-cluster`

Cluster-oriented wrapper around the low-level `konfai` commands.

It adds job-submission options such as:

- `--name`
- `--num-nodes`
- `--memory`
- `--time-limit`
- `--resubmit`

The cluster command depends on the optional `cluster` extra.

## See also

- {doc}`environment`
- {doc}`app-server-api`
- {doc}`../usage/apps`
