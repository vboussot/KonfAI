# Training configuration

Training configuration lives under the `Trainer` root object.

```yaml
Trainer:
  Model:
    classpath: segmentation.UNet.UNet
    UNet:
      ...
  Dataset:
    ...
  train_name: SEG_BASELINE
  epochs: 100
```

## Top-level fields

| Field | Type | Default in code | Required | Effect |
| --- | --- | --- | --- | --- |
| `Model` | mapping | `ModelLoader()` | Yes | Selects and configures the model graph. |
| `Dataset` | mapping | `DataTrain()` | Yes | Defines training data loading, transforms, augmentation, and patching. |
| `train_name` | string | `TRAIN_01` | No | Names the run and its output folders. |
| `manual_seed` | int or null | `None` | No | Sets the random seed when provided. |
| `epochs` | int | `100` | No | Number of training epochs. |
| `it_validation` | int or null | `None` | No | Validation and checkpoint interval in iterations. |
| `autocast` | bool | `false` | No | Enables AMP during training. |
| `gradient_checkpoints` | list or null | `None` | No | Activates gradient checkpointing on selected modules. |
| `gpu_checkpoints` | list or null | `None` | No | Pins selected modules to dedicated GPUs. |
| `ema_decay` | float | `0` | No | Enables exponential moving average tracking when greater than zero. |
| `data_log` | list or null | `None` | No | TensorBoard logging directives for dataset groups or model outputs. |
| `EarlyStopping` | mapping or null | `None` | No | Configures early stopping. |
| `save_checkpoint_mode` | string | `BEST` | No | `BEST` keeps the best checkpoint, `ALL` keeps every saved checkpoint. |

## `Trainer.Model`

`Trainer.Model` always starts with a `classpath`, then a section named after the
selected class.

```yaml
Model:
  classpath: segmentation.UNet.UNet
  UNet:
    optimizer:
      name: AdamW
      lr: 0.001
```

Common nested fields used by built-in and local models:

| Field | Type | Required | Effect |
| --- | --- | --- | --- |
| `classpath` | string | Yes | Selects the model class to import. |
| `<SelectedClass>` | mapping | Yes | Constructor arguments for the chosen class. |
| `optimizer` | mapping | Usually | Optimizer configuration passed through `OptimizerLoader`. |
| `schedulers` | mapping | Optional | Learning-rate schedulers keyed by classpath. |
| `outputs_criterions` | mapping | Usually | Declares losses and metrics attached to specific model outputs. |
| `Patch` | mapping | Optional | Enables model-level patching via `ModelPatch`. |
| `dim` | int | Model-dependent | Declares whether the network operates in 2D or 3D. |

### `outputs_criterions`

This is the most important training structure after the dataset definition.

```yaml
outputs_criterions:
  UNetBlock_0:Head:Conv:
    targets_criterions:
      SEG:
        criterions_loader:
          torch:nn:CrossEntropyLoss:
            is_loss: true
            schedulers:
              Constant:
                nb_step: 0
                value: 1
```

Structure:

- output key → model output or module path
- `targets_criterions` → one or more target groups
- `criterions_loader` → one or more criteria for that target
- each criterion can define `is_loss`, `group`, `start`, `stop`, `accumulation`, and scheduler weights

## `Trainer.Dataset`

Training datasets are instantiated through `DataTrain`.

Common fields:

| Field | Type | Default in code | Effect |
| --- | --- | --- | --- |
| `dataset_filenames` | list[str] | `["./Dataset:mha"]` | Dataset sources and selection mode. |
| `groups_src` | mapping | required in practice | Maps on-disk groups to loaded tensors. |
| `augmentations` | mapping or null | one default augmentation list | Data augmentations sampled during training. |
| `inline_augmentations` | bool | `false` | Re-samples augmentations on each epoch when enabled. |
| `Patch` | mapping or null | `DatasetPatch()` | Dataset-level patch extraction. |
| `use_cache` | bool | `true` | Cache transformed data in memory. |
| `subset` | object | `TrainSubset()` | Restricts which cases are used. |
| `batch_size` | int | `1` | Batch size. |
| `validation` | float / string / list / null | `0.2` | Validation split or explicit validation set. |
| `shuffle` | bool | `true` through subset | Shuffles the training sampler. |

### `groups_src`

Each source group contains one or more destination groups:

```yaml
groups_src:
  CT:
    groups_dest:
      CT:
        transforms:
          Standardize:
            lazy: false
            mean: None
            std: None
            mask: None
            inverse: false
        patch_transforms: None
        is_input: true
```

Use this section to define:

- what exists on disk
- preprocessing transforms
- patch-specific transforms
- whether the tensor is a model input

## Examples

The most practical examples in the repository are:

- `examples/Segmentation/Config.yml`
- `examples/Synthesis/Config.yml`
- `examples/Synthesis/Config_GAN.yml`

## See also

- {doc}`patterns`
- {doc}`prediction`
- {doc}`../concepts/datasets`
- {doc}`../usage/training`
