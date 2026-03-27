# Evaluation configuration

Evaluation configuration lives under the `Evaluator` root object.

```yaml
Evaluator:
  metrics:
    SEG:
      targets_criterions:
        SEG_PRED:
          criterions_loader:
            Dice:
              labels: [1, 2, 3]
  Dataset:
    ...
  train_name: SEG_BASELINE
```

## Top-level fields

| Field | Type | Default in code | Required | Effect |
| --- | --- | --- | --- | --- |
| `metrics` | mapping | default target criterions loader | Yes in practice | Declares what metrics should be computed and between which groups. |
| `Dataset` | mapping | `DataMetric()` | Yes | Defines how targets and predictions are loaded. |
| `train_name` | string | `TRAIN_01` | Yes in practice | Names the evaluation output folder. |

## `metrics`

The evaluation structure mirrors `outputs_criterions`, but without the model.

```yaml
metrics:
  sCT:
    targets_criterions:
      CT;MASK:
        criterions_loader:
          MAE:
            reduction: mean
          PSNR:
            dynamic_range: None
```

Structure:

- output group → the predicted group to evaluate
- `targets_criterions` → one or more target groups, optionally composed with `;`
- `criterions_loader` → one or more metric implementations

Some metrics also accept attributes or write auxiliary datasets. This behavior is
implemented in `konfai.evaluator.Evaluator.update()` and `konfai.metric.measure`.

## `Evaluator.Dataset`

Evaluation datasets are instantiated through `DataMetric`.

Common fields:

| Field | Type | Effect |
| --- | --- | --- |
| `dataset_filenames` | list[str] | Pairs or merges the datasets needed for evaluation. |
| `groups_src` | mapping | Defines how the compared tensors are loaded. |
| `subset` | object | Restricts evaluated cases. |
| `validation` | string or null | Optional file listing validation cases for a separate JSON report. |

## Output files

Evaluation writes JSON files, not CSV files. The main outputs are:

- `Metric_TRAIN.json`
- optionally `Metric_VALIDATION.json`

The JSON structure contains:

- per-case values under `case`
- aggregated statistics under `aggregates`

This behavior comes from `konfai.evaluator.Statistics.write()`.

## Examples

See:

- `examples/Segmentation/Evaluation.yml`
- `examples/Synthesis/Evaluation.yml`

## See also

- {doc}`training`
- {doc}`prediction`
- {doc}`../usage/evaluation`
- {doc}`../reference/app-server-api`
