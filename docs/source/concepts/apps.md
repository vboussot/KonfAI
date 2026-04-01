# KonfAI Apps

KonfAI Apps are the **deployment and reuse layer** of the framework.

Where low-level KonfAI workflows are designed directly through YAML files and
local Python modules, a KonfAI App bundles those assets into a reusable package
that can run through:

- `konfai-apps` on the command line
- Python via `konfai.app.KonfAIApp`
- a remote FastAPI server via `konfai-apps-server`
- clients such as 3D Slicer integrations

## What an app contains

A KonfAI App repository is recognized by the presence of an `app.json` file.
Typical contents are:

```text
my_app/
├── app.json
├── Prediction.yml
├── Evaluation.yml
├── Uncertainty.yml
└── checkpoint.pt
```

Depending on the app, some files are optional:

- `Prediction.yml` is the core inference entrypoint
- `Evaluation.yml` is needed for `eval`
- `Uncertainty.yml` is needed for `uncertainty`
- fine-tuning relies on training assets and checkpoint files that live next to the app

## Local and remote usage

The same app command can run:

- locally, through `KonfAIApp`
- remotely, through `KonfAIAppClient`, if `--host` is provided

In remote mode, the client:

1. uploads files to the server
2. schedules the job
3. streams logs back to the client
4. returns a zipped result bundle

## When to use apps

Use low-level YAML workflows when you are still designing or debugging a model.

Use KonfAI Apps when you want:

- a stable inference interface
- reusable packaging for a team
- distribution through Hugging Face or a private repository
- local and remote execution with the same user-facing command

## See also

- {doc}`../usage/apps`
- {doc}`../usage/remote-server`
- {doc}`../reference/app-server-api`
