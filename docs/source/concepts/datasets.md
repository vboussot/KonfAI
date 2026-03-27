# Datasets and groups

KonfAI works with **grouped datasets**. Each case lives in its own directory,
and each file in that directory belongs to a named group such as `CT`, `MR`,
`SEG`, or `MASK`.

## Expected layout

Typical layouts in the repository look like this:

```text
Dataset/
├── CASE_001/
│   ├── CT.mha
│   └── SEG.mha
└── CASE_002/
    ├── CT.mha
    └── SEG.mha
```

```text
Dataset/
├── CASE_001/
│   ├── MR.mha
│   ├── CT.mha
│   └── MASK.mha
└── CASE_002/
    ├── MR.mha
    ├── CT.mha
    └── MASK.mha
```

The concrete file extension is not restricted to `.mha`. KonfAI supports the
extensions listed in `konfai.utils.utils.SUPPORTED_EXTENSIONS`.

## `groups_src` and `groups_dest`

Each workflow describes how on-disk groups should be loaded through the
`Dataset.groups_src` mapping.

Example:

```yaml
Dataset:
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
          is_input: true
```

Conceptually:

- `groups_src` identifies what must exist on disk
- `groups_dest` identifies how the loaded tensors are exposed to the workflow
- `is_input: true` marks tensors that are fed into the model

The logic lives in `konfai.data.data_manager.GroupTransform` and the `Data*`
dataset classes.

## Dataset file selectors

The `dataset_filenames` field accepts strings in the form:

- `path`
- `path:format`
- `path:flag:format`

This behavior is implemented in `konfai.data.data_manager.Data.get_data()`.

The most important conventions are:

- `a` means “append / union”
- `i` means “intersection / keep only common cases”

Examples:

- `./Dataset:a:mha`
- `./Predictions/TRAIN_01/Dataset:i:mha`

## Training subsets and validation

KonfAI supports several ways to define subsets and validation sets.

From the dataset code, `validation` may be:

- a float such as `0.2`
- a slice string such as `0:10`
- a path to a text file listing case names
- a list of indices
- a list of case names
- `None`

The `subset` object is applied before validation splitting and can exclude or
include items. The exact logic is implemented by `TrainSubset` and
`PredictionSubset`.

## Caching, augmentation, and patching

At the dataset level, KonfAI can:

- cache transformed data in memory
- generate multiple augmentations per item
- split volumes into patches before they reach the model

This is handled by:

- `konfai.data.data_manager.DataTrain`
- `konfai.data.augmentation.DataAugmentationsList`
- `konfai.data.patching.DatasetPatch`

## When to use dataset patching

Use `Dataset.Patch` when:

- volumes are too large to process at once
- you want 2D, 2.5D, or 3D crops sampled from larger volumes
- you need sliding-window style training or inference

Dataset patching is separate from **model patching**, which applies inside the
network itself. See {doc}`model-graph`.

## See also

- {doc}`configuration`
- {doc}`model-graph`
- {doc}`../config_guide/training`
- {doc}`../config_guide/prediction`
