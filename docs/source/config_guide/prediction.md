# Prediction configuration

Prediction configuration lives under the `Predictor` root object.

```yaml
Predictor:
  Model:
    classpath: segmentation.UNet.UNet
    UNet:
      ...
  Dataset:
    ...
  outputs_dataset:
    ...
  train_name: SEG_BASELINE
```

## Top-level fields

| Field | Type | Default in code | Required | Effect |
| --- | --- | --- | --- | --- |
| `Model` | mapping | `ModelLoader()` | Yes | Selects the model class used for prediction. |
| `Dataset` | mapping | `DataPrediction()` | Yes | Defines inference data loading and test-time augmentation. |
| `outputs_dataset` | mapping | default output dataset | Yes in practice | Controls which outputs are written to disk and how. |
| `combine` | string | `Mean` | No | Reduces outputs across multiple checkpoints. |
| `train_name` | string | `"name"` | Yes in practice | Names the prediction run and output folder. |
| `manual_seed` | int or null | `None` | No | Optional seed. |
| `gpu_checkpoints` | list or null | `None` | No | Module placement optimization. |
| `autocast` | bool | `false` | No | Enables AMP during inference. |
| `data_log` | list or null | `None` | No | Optional TensorBoard logging. |

## `Predictor.Model`

Prediction uses the same `classpath` convention as training:

```yaml
Model:
  classpath: Model:UNetpp5
  outputs_criterions: {}
```

In most prediction configs:

- you select the architecture
- you keep only the inference-relevant parameters
- you disable or simplify training-only criteria

Checkpoint loading is controlled by the CLI argument `--models`, not by the YAML
file itself.

## `Predictor.Dataset`

Prediction datasets are instantiated through `DataPrediction`.

Key fields:

| Field | Type | Effect |
| --- | --- | --- |
| `dataset_filenames` | list[str] | Input dataset sources. |
| `groups_src` | mapping | Input groups and preprocessing transforms. |
| `augmentations` | mapping | Test-time augmentation definitions. |
| `Patch` | mapping | Sliding-window or slice-wise inference setup. |
| `subset` | object | Restricts evaluated cases. |
| `batch_size` | int | Number of patches per inference batch. |

## `outputs_dataset`

`outputs_dataset` defines how selected model outputs become files on disk.

```yaml
outputs_dataset:
  Head:Tanh:
    OutputDataset:
      name_class: OutSameAsGroupDataset
      group: sCT
      same_as_group: MR:MR
      reduction: Mean
```

Important nested fields:

| Field | Effect |
| --- | --- |
| output key | Selects the model output to export. |
| `name_class` | Selects the output dataset implementation. |
| `group` | Output group name written to disk. |
| `dataset_filename` | Destination dataset path and format. |
| `same_as_group` | Geometry reference group for exported volumes. |
| `before_reduction_transforms` | Applied before combining ensemble or TTA outputs. |
| `after_reduction_transforms` | Applied after reduction. |
| `final_transforms` | Final transforms applied before writing. |
| `reduction` | Combines multiple predictions, usually `Mean` or `Median`. |
| `patch_combine` | Optional patch reassembly strategy. |

## Examples

See:

- `examples/Segmentation/Prediction.yml`
- `examples/Synthesis/Prediction.yml`

## See also

- {doc}`training`
- {doc}`evaluation`
- {doc}`../concepts/model-graph`
- {doc}`../usage/prediction`
