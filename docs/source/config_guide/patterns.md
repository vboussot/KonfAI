# Common configuration patterns

This page gathers the conventions that are most important in day-to-day KonfAI
usage.

## `dataset_filenames`

`dataset_filenames` accepts several forms:

- `./Dataset`
- `./Dataset:mha`
- `./Dataset:a:mha`
- `./Predictions/TRAIN_01/Dataset:i:mha`

From `konfai.data.data_manager.Data.get_data()`:

- no suffix means default format `mha`
- `a` appends cases
- `i` keeps only the intersection of cases

This is especially useful during evaluation, where you often combine:

- the ground-truth dataset
- the prediction dataset

## `groups_src` / `groups_dest`

Use `groups_src` to declare what exists on disk and `groups_dest` to declare how
it should appear inside the workflow.

This allows patterns such as:

- loading `MASK` from disk but not feeding it to the model
- loading the same source group into multiple transformed destinations
- renaming groups logically inside the workflow

## `outputs_criterions`

Use `outputs_criterions` when you need:

- multi-head supervision
- multiple criteria per output
- different targets for the same output
- scheduler-weighted loss composition

The keys are model output paths. Always start from a working example and only
then introduce custom output names.

## `outputs_dataset`

Use `outputs_dataset` when you need to control:

- which prediction is exported
- how predictions are reduced across TTA or ensembles
- which final transforms run before the file is written
- which geometry should be reused for the output

## `subset` and `validation`

Use:

- `subset` to restrict the whole run to specific items
- `validation` to carve out a validation split during training or to define a
  validation report during evaluation

Supported `validation` forms are inferred from the dataset loader code:

- `0.2`
- `0:10`
- `./Validation.txt`
- `[0, 1, 2]`
- `["CASE_001", "CASE_002"]`

## Local modules next to YAML files

When you want to extend KonfAI without packaging a new Python distribution, put
Python files next to your YAML and use explicit classpaths:

```yaml
Model:
  classpath: Model:UNetpp5
```

```yaml
final_transforms:
  UnNormalize:UnNormalize: {}
```

This pattern is used heavily in `examples/Synthesis`.

## Notes on inferred behavior

The following conventions are **inferred directly from the code** rather than
being formally described elsewhere in the repository:

- `default|...` fallback values
- the precise semantics of `dataset_filenames` flags
- the way `;accu;` addresses patch-wise outputs before reassembly

These patterns are stable enough to document because they are visible in the
config loader, patching logic, and shipped examples.

## See also

- {doc}`../concepts/configuration`
- {doc}`../concepts/datasets`
- {doc}`../concepts/model-graph`
