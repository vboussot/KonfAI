# Architecture and internals

This page gives a high-level map of the repository for contributors and advanced
users. It focuses on the structure that is clearly visible in the codebase.

## Package layout

| Package | Responsibility |
| --- | --- |
| `konfai.main` | CLI entrypoints for low-level workflows and cluster mode |
| `konfai.trainer` | Training workflow and training loop |
| `konfai.predictor` | Prediction workflow and export logic |
| `konfai.evaluator` | Evaluation workflow and metric export |
| `konfai.data` | Dataset discovery, transforms, augmentations, and patching |
| `konfai.network` | Model graph composition, optimizer/scheduler loaders, criterion routing |
| `konfai.metric` | Metrics, losses, and schedulers |
| `konfai.utils` | Config system, dataset helpers, distributed runtime utilities |
| `konfai_apps` | Standalone package for local/remote app execution and app server |

## Two user-facing layers

KonfAI exposes two related but distinct usage layers.

### 1. Low-level framework mode

This is the `konfai` CLI:

- `TRAIN`
- `RESUME`
- `PREDICTION`
- `EVALUATION`

It operates directly on YAML files and is the best layer for experimentation,
custom model development, and framework extension.

### 2. KonfAI Apps

This is the `konfai-apps` CLI and the remote server around it.

Apps package a stable workflow behind a simpler interface for:

- local CLI usage
- Python usage through `konfai_apps.KonfAIApp`
- remote execution through `konfai-apps-server`
- external clients such as Slicer integrations

## Configuration-driven construction

One of KonfAI's main design choices is that the runtime is built from YAML using
constructor signatures and type annotations.

The main pieces are:

- `@config(...)` to bind a class to a configuration key
- `apply_config(...)` to instantiate objects recursively
- `classpath` values to import custom implementations dynamically

The result is a constructor-driven configuration system rather than a fixed,
static schema file.

## Execution flow

At a high level, low-level workflows follow this path:

1. Parse CLI arguments in `konfai.main`
2. Set runtime environment variables and output directories
3. Instantiate the root object (`Trainer`, `Predictor`, or `Evaluator`)
4. Build datasets, transforms, models, losses, and schedulers from YAML
5. Execute the workflow
6. Write checkpoints, logs, predictions, or metrics

See also {doc}`concepts/execution-flow`.

## Model graph composition

The model system is not limited to a single monolithic network. In
`konfai.network.network`, a model can be composed from named modules with
explicit branch routing and criterion attachment.

This is what enables patterns such as:

- multi-branch architectures
- multiple output heads
- nested discriminators and generators
- patch-wise accumulation paths such as `;accu;`

## Dataset and patching layers

KonfAI has two patching levels visible in the codebase:

- dataset patching in `konfai.data.patching.DatasetPatch`
- model patching in `konfai.data.patching.ModelPatch`

The synthesis GAN example uses both levels, which is why it is a good reference
for advanced users.

## Distributed runtime

Training, prediction, and evaluation are launched through the distributed
runtime utilities in `konfai.utils.runtime`.

This is an internal detail that matters operationally:

- GPU discovery is centralized
- runtime directories are injected through environment variables
- PyTorch distributed initialization is part of the startup path

The exact control flow is documented only where it is directly visible in code;
the repository does not currently ship a separate design document for the
distributed launcher.

## KonfAI Apps server architecture

The remote app server in `konfai_apps.app_server` adds:

- upload handling
- async job management
- GPU semaphore-based scheduling
- SSE log streaming
- result packaging and download
- optional bearer-token authentication

This layer is intentionally separate from the core low-level `konfai` CLI.

## See also

- {doc}`concepts/index`
- {doc}`reference/api/index`
- {doc}`reference/app-server-api`
