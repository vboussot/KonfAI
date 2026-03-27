# Synthesis example

The synthesis example is the most complete low-level workflow in the
repository. It demonstrates how KonfAI combines YAML configuration, local
Python modules, preprocessing, prediction, and evaluation for a paired
image-to-image task.

The shipped task is **MR to CT synthesis**.

## What is in the folder

```text
examples/Synthesis/
├── Config.yml
├── Config_GAN.yml
├── Prediction.yml
├── Evaluation.yml
├── Model.py
├── UnNormalize.py
└── Synthesis_demo.ipynb
```

- `Config.yml` trains the baseline `UNetpp5` model.
- `Config_GAN.yml` trains a GAN with the same generator plus a 3D discriminator.
- `Prediction.yml` runs inference for both the baseline checkpoint and the
  generator stored inside a GAN checkpoint.
- `Evaluation.yml` evaluates predicted CT volumes against the reference CT.
- `Model.py` defines the local model classes loaded through `classpath`.
- `UnNormalize.py` contains a local transform used during prediction.

## Expected dataset layout

```text
Dataset/
├── CASE_001/
│   ├── MR.mha
│   ├── CT.mha
│   └── MASK.mha
└── ...
```

The example uses these groups:

- `MR`: model input
- `CT`: target image
- `MASK`: mask used by preprocessing and masked metrics

## Baseline workflow

Run all commands from `examples/Synthesis`.

Train:

```bash
konfai TRAIN -y --gpu 0 --config Config.yml
```

Predict:

```bash
konfai PREDICTION -y --gpu 0 --config Prediction.yml \
  --models Checkpoints/TRAIN_01/<checkpoint>.pt
```

Evaluate:

```bash
konfai EVALUATION -y --config Evaluation.yml
```

Outputs are written to:

- `Checkpoints/TRAIN_01/`
- `Statistics/TRAIN_01/`
- `Predictions/TRAIN_01/`
- `Evaluations/TRAIN_01/`

## GAN workflow

`Config_GAN.yml` shows a more advanced graph:

- a **2D / 2.5D generator**
- a **3D discriminator**
- a global dataset patch for the whole GAN
- a second internal patch for the generator

This is not an undocumented convention: it follows the model graph defined in
`examples/Synthesis/Model.py`.

### Why there are two patch levels

The GAN example uses two distinct patch scopes:

- `Trainer.Dataset.Patch`: extracts the **global 3D chunk** seen by the GAN
- `Trainer.Model.Gan.UNetpp5.Patch`: extracts the **internal 2D/2.5D slices**
  seen by the generator

This lets the generator operate slice-wise with local context while the
discriminator still judges a reconstructed 3D patch.

### What `;accu;` means in this example

The output key `Generator_A_to_B:;accu;Head:Tanh` refers to the generator output
**before patch reassembly**. It is used for the generator-side reconstruction
loss.

By contrast, `Discriminator_pB:Head:Conv` sees the reassembled 3D fake patch and
is therefore the right place for the adversarial loss.

This behavior is inferred directly from the shipped model graph and the way
KonfAI's accumulation path is used in `Config_GAN.yml`.

## Why prediction and evaluation are shared

The repository uses a single `Prediction.yml` and a single `Evaluation.yml` for
both synthesis variants. That works because:

- the baseline checkpoint stores `UNetpp5`
- the GAN checkpoint still exposes the same generator class name for inference

In practice, you only need to point `--models` at the right checkpoint and make
sure `train_name` is set consistently if you want separate output folders.

## What to adapt first

When adapting this example to your own project, change these sections first:

1. `dataset_filenames`
2. input and target groups
3. preprocessing transforms
4. patch sizes
5. losses and metrics
6. `train_name`
7. local model definitions in `Model.py` if the built-in modules are not enough

## See also

- {doc}`../concepts/model-graph`
- {doc}`../config_guide/training`
- {doc}`../usage/custom-models`
