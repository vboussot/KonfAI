# Segmentation Example

This example provides a **simple multiclass segmentation baseline** for KonfAI.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vboussot/KonfAI/blob/main/examples/Segmentation/Segmentation_demo.ipynb)

It is intentionally conservative and is meant to be:

- easy to read
- easy to adapt
- easy to use as a first segmentation template

The current baseline uses:

- the built-in `segmentation.UNet.UNet`
- a 2D slice-wise setup
- patch-based training
- `CrossEntropyLoss` during training
- Dice evaluation after prediction
- `41` classes in total (`0` for background, `1..40` for labels)

## What you will find in this folder

```text
examples/Segmentation/
├── Config.yml
├── Prediction.yml
├── Evaluation.yml
├── README.md
└── Segmentation_demo.ipynb
```

- `Config.yml`: training workflow
- `Prediction.yml`: inference workflow
- `Evaluation.yml`: evaluation workflow
- `Segmentation_demo.ipynb`: guided onboarding notebook

The notebook is designed to work from a **fresh environment**, including **Google Colab**. Its setup cells can:

- clone the KonfAI repository if needed
- install KonfAI and its Python dependencies
- download the public segmentation demo subset automatically

## Expected dataset layout

```text
Dataset/
├── CASE_000/
│   ├── CT.mha
│   └── SEG.mha
├── CASE_001/
│   ├── CT.mha
│   └── SEG.mha
└── ...
```

- `CT`: input image
- `SEG`: segmentation label map

The default template assumes:

- a multiclass task
- label `0` as background
- labels `1..40` as foreground classes
- one input image per case
- `SEG` stored as a label map with integer values

## Demo data

The public Hugging Face demo dataset is available at:

- `https://huggingface.co/datasets/VBoussot/konfai-demo`

If you want the easiest first run, use `Segmentation_demo.ipynb`.

If you prefer to fetch the demo subset manually, use the Hugging Face CLI:

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download VBoussot/konfai-demo \
  --repo-type dataset \
  --include "Segmentation/**" \
  --local-dir Dataset
mv Dataset/Segmentation/* Dataset/
rmdir Dataset/Segmentation
rm -rf Dataset/.cache
```

After that, your local `Dataset/` folder should already match the structure expected by this example.

## Quick start

Run all commands from this directory:

```bash
cd examples/Segmentation
```

Once your `Dataset/` folder is ready:

For the smoothest first run on a fresh machine or in Colab, start with `Segmentation_demo.ipynb`.

### 1. Train

```bash
konfai TRAIN -y --gpu 0 --config Config.yml
```

### 2. Predict

```bash
konfai PREDICTION -y --gpu 0 --config Prediction.yml --models Checkpoints/SEG_BASELINE/<checkpoint>.pt
```

### 3. Evaluate

```bash
konfai EVALUATION -y --config Evaluation.yml
```

This produces:

- `Checkpoints/SEG_BASELINE/`
- `Predictions/SEG_BASELINE/`
- `Evaluations/SEG_BASELINE/`

## What to adapt first

For a real project, you will usually want to update:

1. `dataset_filenames`
2. `train_name`
3. patch size
4. batch size
5. number of classes
6. preprocessing transforms
7. Dice labels in `Evaluation.yml`
8. model channels and scheduler

For multiclass segmentation:

- update `nb_class`
- update `Dice.labels`
- review the label encoding in your dataset

## Why training uses CrossEntropyLoss here

This example uses `CrossEntropyLoss` during training and Dice during evaluation on purpose:

- training stays simple and stable
- the final segmentation quality is still measured with Dice

This makes the example easier to understand and monitor before moving to more advanced losses.

## Recommended usage

Use this example when you want to:

- bootstrap a new segmentation experiment quickly
- understand the minimal KonfAI structure for segmentation
- create your own YAML template before moving to stronger architectures or 3D workflows

If you want the easiest first run, start with:

- `Segmentation_demo.ipynb`
