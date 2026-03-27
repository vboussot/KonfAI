# Segmentation example

The segmentation example is the smallest, most conservative training workflow
shipped with KonfAI. It is meant to be easy to read, easy to adapt, and easy to
use as a first project template.

## What is in the folder

```text
examples/Segmentation/
├── Config.yml
├── Prediction.yml
├── Evaluation.yml
└── Segmentation_demo.ipynb
```

- `Config.yml` defines the training workflow.
- `Prediction.yml` defines inference and export.
- `Evaluation.yml` computes Dice on the saved predictions.
- `Segmentation_demo.ipynb` bootstraps the example in a fresh environment.

## Expected dataset layout

```text
Dataset/
├── CASE_000/
│   ├── CT.mha
│   └── SEG.mha
└── ...
```

- `CT` is the input image.
- `SEG` is the segmentation label map.

The shipped demo assumes:

- a multiclass task
- label `0` for background
- labels `1..40` for foreground classes
- `41` classes total

## Default baseline

The baseline uses:

- the built-in `segmentation.UNet.UNet`
- 2D patch-based training
- `CrossEntropyLoss` during training
- Dice during evaluation

Training uses `CrossEntropyLoss` on purpose. It keeps the live training loop
simple and stable while final quality is still measured with Dice during
evaluation.

## Minimal workflow

Run all commands from `examples/Segmentation`.

Train:

```bash
konfai TRAIN -y --gpu 0 --config Config.yml
```

Predict:

```bash
konfai PREDICTION -y --gpu 0 --config Prediction.yml \
  --models Checkpoints/SEG_BASELINE/<checkpoint>.pt
```

Evaluate:

```bash
konfai EVALUATION -y --config Evaluation.yml
```

Outputs are written to:

- `Checkpoints/SEG_BASELINE/`
- `Statistics/SEG_BASELINE/`
- `Predictions/SEG_BASELINE/`
- `Evaluations/SEG_BASELINE/`

## What to adapt first

Most real projects will need changes in:

1. `dataset_filenames`
2. `train_name`
3. patch size
4. batch size
5. `nb_class`
6. preprocessing transforms
7. the list of Dice labels in `Evaluation.yml`

If your dataset is not a `0..40` label map, update both `nb_class` and the Dice
labels together.

## See also

- {doc}`../quickstart`
- {doc}`../config_guide/evaluation`
- {doc}`../usage/training`
