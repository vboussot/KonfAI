# Execution flow

KonfAI ships three low-level workflows and one higher-level app layer.

## Low-level workflows

The `konfai` CLI dispatches to three public functions:

- `konfai.trainer.train`
- `konfai.predictor.predict`
- `konfai.evaluator.evaluate`

Each wrapper sets a small execution context through environment variables, then
instantiates the corresponding configured object:

- `Trainer`
- `Predictor`
- `Evaluator`

The key environment variables are documented in {doc}`../reference/environment`.

## What happens during training

At a high level, `TRAIN` does the following:

1. parse the YAML into a `Trainer`
2. prepare the dataset and its train/validation split
3. initialize the model graph and its losses
4. run the training loop
5. save checkpoints and TensorBoard logs
6. copy the config into the statistics directory

Outputs are written to:

- `Checkpoints/<train_name>/`
- `Statistics/<train_name>/`

## What happens during prediction

`PREDICTION`:

1. parses the YAML into a `Predictor`
2. loads one or more checkpoints
3. prepares the inference dataset
4. runs the model in prediction mode
5. writes output datasets defined in `outputs_dataset`
6. copies `Prediction.yml` into the prediction directory

Outputs are written to:

- `Predictions/<train_name>/`

## What happens during evaluation

`EVALUATION`:

1. parses the YAML into an `Evaluator`
2. loads the dataset pairs needed for metric computation
3. validates that configured output and target groups exist
4. computes per-case and aggregate metrics
5. writes JSON reports
6. copies the evaluation config into the evaluation directory

Outputs are written to:

- `Evaluations/<train_name>/Metric_TRAIN.json`
- optionally `Evaluations/<train_name>/Metric_VALIDATION.json`

## Distributed execution

KonfAI wraps these workflows with `konfai.utils.utils.run_distributed_app()`.
From the code, this wrapper is responsible for:

- setting `CUDA_VISIBLE_DEVICES`
- handling overwrite and verbosity flags
- launching TensorBoard when requested
- spawning worker processes with `torch.multiprocessing.spawn`
- initializing `torch.distributed` with a local TCP port

This means that even “local” multi-process execution uses a distributed setup.

## Apps

`konfai-apps` is the higher-level interface. It packages low-level prediction,
evaluation, uncertainty, and fine-tuning workflows into reusable app bundles.

See {doc}`apps`.

## See also

- {doc}`../usage/training`
- {doc}`../usage/prediction`
- {doc}`../usage/evaluation`
- {doc}`../usage/apps`
