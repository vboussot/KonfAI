# KonfAI Medical Image Synthesis Pipeline

This tutorial explains how to run a **medical image synthesis pipeline** using **KonfAI**.

The pipeline typically includes:

- **model training**
- **fine-tuning**
- **inference (prediction)**
- **evaluation**

This tutorial focuses on the **low-level KonfAI workflow**, where users directly modify the configuration files to define each step of the pipeline.  
This approach is mainly intended for **pipeline development and experimentation**, providing fine-grained control over the workflow and its parameters.

KonfAI also provides **KonfAI Apps**, a higher-level interface that packages models, workflows, and configurations into portable applications.

In contrast to the low-level workflow described in this tutorial, KonfAI Apps rely on **predefined configurations**, providing a **simpler way** to run **inference, evaluation, uncertainty estimation, or fine-tuning** through a unified interface (CLI, Python API, 3D Slicer, or remote servers). With a **single command**, you can perform these tasks on a new dataset, while the underlying pipeline design and configuration are abstracted and follow a **predefined methodology**.

For more information about KonfAI Apps, see:

https://github.com/vboussot/KonfAI/tree/main/apps

Example of a ready-to-use synthesis application:

- KonfAI App (CLI / Python / server):

https://github.com/vboussot/KonfAI/tree/main/apps/impact_synth

- 3D Slicer extension (graphical interface):

https://github.com/vboussot/SlicerImpactSynth

This application provides pretrained models for **MR → CT** and **CBCT → CT** synthesis.

These are the **best models we currently provide** for synthetic CT generation, and they generally perform **better than the models released for the SynthRAD challenge**.  
They were trained on **carefully aligned MR–CT and CBCT–CT pairs using IMPACT-Reg**, which reduces registration bias and improves anatomical consistency between modalities.

---

# Pipeline Overview

This tutorial assumes **KonfAI 1.5.4** is installed.

All commands in this tutorial should be executed from the following directory:

```
examples/Synthesis
```

This directory already contains the files required to run the pipeline:

```
examples/Synthesis/
├── Config.yml
├── Prediction.yml
├── Evaluation.yml
├── UNetpp.py
└── UnNormalize.py
```

These files define the model, preprocessing, postprocessing, and workflows used in this tutorial.

Inside this directory, you should place:

- the **Dataset/** folder
- optional **pretrained checkpoints**

You can use the **example dataset and example pretrained models** provided below, or replace them with your own data and checkpoints.

---

# Dataset Structure

The dataset must be organized **by case**.

Each case should contain:

- an **input image** (e.g. MR)
- a **target image** (e.g. CT)
- a **mask** defining the region used for training and evaluation

Example dataset:

```
Dataset/
├── 1HNA001/
│   ├── CT.mha
│   ├── MASK.mha
│   └── MR.mha
└── 1THA001/
    ├── CT.mha
    ├── MASK.mha
    └── MR.mha
```

| File | Description |
|-----|-------------|
| MR.mha | Input MR image |
| CT.mha | Ground-truth CT |
| MASK.mha | Mask used for preprocessing, training, and evaluation |

Images can be stored in **any format supported by SimpleITK** (e.g. `mha`, `nii`, `nii.gz`, etc.).  
KonfAI also supports **HDF5 datasets (`.h5`)**.

For this tutorial, a **small example dataset** is available on Hugging Face:

```
git clone --filter=blob:none --no-checkout https://huggingface.co/datasets/VBoussot/konfai-demo Dataset                                                                                       ✔  18s  .env Py  13:17:29 
cd Dataset
git sparse-checkout init --cone
git sparse-checkout set Synthesis
git checkout
mv Synthesis/* .
rm -r Synthesis
```

This will create the `Dataset/` folder with the structure shown above.

---

# Pretrained Models

Example pretrained models are available on Hugging Face:

```
git clone https://huggingface.co/VBoussot/Synthrad2025
```

These models were developed for the **SynthRAD 2025 challenge** and are provided here as an example of pretrained models that can be used with KonfAI.

The repository contains models trained using **5-fold cross-validation**, meaning that five checkpoints are available and can be used together for **ensemble prediction**.

Two anatomical configurations are provided:

- **AB-TH** — Abdomen / Thorax
- **HN** — Head & Neck

Example structure after download:

```
Task_1/
├── AB-TH/
│   ├── CV_0.pt
│   ├── CV_1.pt
│   ├── CV_2.pt
│   ├── CV_3.pt
│   ├── CV_4.pt
│   └── Prediction.yml
└── HN/
    ├── CV_0.pt
    ├── CV_1.pt
    ├── CV_2.pt
    ├── CV_3.pt
    ├── CV_4.pt
    └── Prediction.yml
```

Each folder contains:

| File | Description |
|-----|-------------|
| CV_*.pt | Trained model checkpoints |
| Prediction.yml | Example inference configuration provided with the pretrained models (used for the competition; in this tutorial we use the one in `examples/Synthesis`) |

These checkpoints can also be **replaced by other pretrained models**, such as the **IMPACT-Synth models** available on Hugging Face:

```
https://huggingface.co/VBoussot/ImpactSynth
```

---

# 1. Training

Before training, make sure the dataset is available in the `Dataset/` directory.

Train a model from scratch:

```bash
konfai TRAIN -y --gpu 0 --config Config.yml
```

Explanation:

| Argument | Description |
|--------|-------------|
| TRAIN | Runs the training workflow |
| -y | Automatically confirms prompts |
| --gpu 0 | GPU device used for training |
| --config Config.yml | Path to the training configuration file |

If multiple GPUs are available, you can specify them as a space-separated list:

```bash
konfai TRAIN -y --gpu 0 1 --config Config.yml
```

If no GPU is available, training can also run on CPU:

```bash
konfai TRAIN -y --cpu 1 --config Config.yml
```

| Argument | Description |
|--------|-------------|
| --cpu 1 | Number of CPU cores used for computation |

This command creates two main folders:

```
Checkpoints/TRAIN_01/
Statistics/TRAIN_01/
```

Example:

```
Checkpoints/TRAIN_01/
└── 2026_03_06_10_45_29.pt
```

```
Statistics/TRAIN_01/
├── tb/
│   └── events.out.tfevents...
├── Config_0_0.yml
├── log_0.txt
├── Train_0.txt
└── Validation_0.txt
```

Explanation:

| File / Folder | Description |
|---------------|-------------|
| Checkpoints/TRAIN_01 | Folder containing all checkpoints generated during training |
| Statistics/TRAIN_01/tb/ | TensorBoard logs |
| Statistics/TRAIN_01/Config_0_0.yml | Copies of the training configuration saved during training and resume |
| Statistics/TRAIN_01/log_0.txt | Training log |
| Statistics/TRAIN_01/Train_0.txt | List of training cases |
| Statistics/TRAIN_01/Validation_0.txt | List of validation cases |

Training can be monitored with TensorBoard:

```bash
tensorboard --logdir Statistics/TRAIN_01/tb
```

---

# 2. Resume / Fine-tuning

Resume training or fine-tune from an existing checkpoint:

```bash
konfai RESUME -y --gpu 0 --config Config.yml --model ./Task_1/AB-TH/CV_0.pt
```

Explanation:

| Argument | Description |
|--------|-------------|
| RESUME | Resumes training from a checkpoint |
| -y | Automatically confirms prompts |
| --gpu 0 | GPU device used for training |
| --config Config.yml | Path to the training configuration file |
| --model ./Task_1/AB-TH/CV_0.pt | Path to the checkpoint used to initialize the model |

This command is useful for:

- **continuing a previous training**
- **fine-tuning a pretrained model on a new dataset** (all model weights remain trainable).  
  For stricter fine-tuning strategies (e.g. freezing parts of the network), another command is available but is outside the scope of this tutorial.

Like training, it updates the `Checkpoints/` and `Statistics/` folders.

---

# 3. Inference (Prediction)

Generate synthetic CT images using one or several trained models.

Example using **5 checkpoints**:

```bash
konfai PREDICTION -y --gpu 0 --config Prediction.yml \
--models ./Task_1/AB-TH/CV_0.pt \
         ./Task_1/AB-TH/CV_1.pt \
         ./Task_1/AB-TH/CV_2.pt \
         ./Task_1/AB-TH/CV_3.pt \
         ./Task_1/AB-TH/CV_4.pt
```

Explanation:

| Argument | Description |
|--------|-------------|
| PREDICTION | Runs the inference workflow |
| -y | Automatically confirms prompts |
| --gpu 0 | GPU device used for inference |
| --config Prediction.yml | Path to the prediction configuration file |
| --models | List of model checkpoints used for inference |

Providing multiple checkpoints enables **model ensembling**, where predictions from several models are combined to improve robustness.

This command creates a `Predictions/` folder:

```
Predictions/TRAIN_01/
├── Dataset/
│   ├── 1HNA001/
│   │   └── sCT.mha
│   └── 1THA001/
│       └── sCT.mha
├── log_0.txt
└── Prediction.yml
```

Explanation:

| File / Folder | Description |
|---------------|-------------|
| Predictions/TRAIN_01/Dataset/ | Folder containing the generated outputs |
| sCT.mha | Predicted synthetic CT image |
| log_0.txt | Inference log |
| Prediction.yml | Copy of the prediction configuration |

---

# 4. Evaluation

Evaluate predictions against the ground-truth CT:

```bash
konfai EVALUATION -y --config Evaluation.yml
```

Explanation:

| Argument | Description |
|--------|-------------|
| EVALUATION | Runs the evaluation workflow |
| -y | Automatically confirms prompts |
| --config Evaluation.yml | Path to the evaluation configuration file |

This command compares the predicted outputs (e.g. `sCT`) with the reference images (e.g. `CT`) and computes evaluation metrics.

Example output:

```
Evaluations/TRAIN_01/
├── Evaluation.yml
├── log_0.txt
└── Metric_TRAIN.json
```

| File | Description |
|-----|-------------|
| Evaluation.yml | Copy of the evaluation configuration |
| log_0.txt | Evaluation log |
| Metric_TRAIN.json | File containing the computed metrics |

---

# Metrics

Metrics are computed **per case** and **aggregated across all cases**.

Typical metrics include:

- **MAE** — Mean Absolute Error
- **PSNR** — Peak Signal-to-Noise Ratio
- **SSIM** — Structural Similarity

Example JSON structure:

```json
{
  "case": {...},
  "aggregates": {...}
}
```

In some experiments, an additional file may appear:

```
Metric_EVALUATION.json
```

This contains metrics for **cases that were never used during training**.

---

# Configuration Files

KonfAI uses three configuration files:

- `Config.yml` → **training and fine-tuning**
- `Prediction.yml` → **inference**
- `Evaluation.yml` → **evaluation**

Only the main ideas are described here.  
For a complete explanation of all parameters, please refer to the **commented configuration files provided in this repository**.

---

## `Config.yml` — Training

This file defines the full training pipeline.

Main ideas:

- **Model**  
  The training uses the `UNetpp5` model, a **UNet++ architecture with 2.5D input**.  
  The model takes the current slice together with neighboring slices as input.

- **Losses**  
  Two losses are used during training:
  - `MAE` for pixel-wise reconstruction between prediction and CT
  - `IMPACTSynth`, a perceptual loss based on **SAM features**

  In addition, a **masked MAE metric** is computed for monitoring, but it is **not optimized**.

- **Input and target data**  
  The model uses:
  - `MR` as **input**
  - `CT` as **target**
  - `MASK` for **masked evaluation**

- **Preprocessing**  
  - `CT` is clipped to a valid Hounsfield Unit range, then normalized to `[-1, 1]`
  - `MR` is robustly clipped and standardized

- **Data augmentation**  
  Random flips are applied during training to improve robustness.

- **Patch-based training**  
  Training is performed on patches:
  - `patch_size: [1, 320, 320]`
  - `extend_slice: 4` adds **2 slices above and 2 below** for 2.5D context

- **Optimization**  
  Training uses:
  - the `AdamW` optimizer
  - a `StepLR` learning rate scheduler

- **Dataset definition**  
  The dataset section defines:
  - where the dataset is located
  - which files are used as `MR`, `CT`, and `MASK`
  - batch size
  - validation split
  - optional dataset filtering

- **Training control**  
  The configuration also defines:
  - the experiment name (`train_name`)
  - the number of epochs
  - validation frequency (`it_validation`)
  - checkpoint saving
  - TensorBoard logging
  - early stopping

- **Outputs**  
  During training, KonfAI saves:
  - model checkpoints
  - TensorBoard logs
  - training and validation case lists
  - copies of the config files

In short, `Config.yml` defines:

1. which model is trained
2. which losses are used
3. how MR, CT, and MASK are loaded and preprocessed
4. how training is performed
5. how results are logged and saved

## Parameters you may need to modify

In most cases, only a few parameters need to be adapted to your dataset and experiment.

### Training name

```yaml
train_name: TRAIN_01
```

Defines the experiment name and the output folders where results will be saved.

---

### Dataset location

Example used for training:

```yaml
dataset_filenames:
- ./Dataset/:a:mha
```

This indicates that the dataset is located in the `Dataset/` folder.

Multiple datasets can also be concatenated:

```yaml
dataset_filenames:
- ./Dataset/AB/:a:mha
- ./Dataset/TH/:a:mha
```

For evaluation (`Evaluation.yml`), the configuration usually includes both the **ground-truth dataset** and the **predicted outputs**:

```yaml
dataset_filenames:
- ./Dataset:a:mha
- ./Predictions/TRAIN_01/Dataset:i:mha
```

Here:

- `./Dataset` contains the **ground-truth images (CT)**  
- `./Predictions/TRAIN_01/Dataset` contains the **predicted images (sCT)**

In most cases, you only need to modify the **prediction path** to match the experiment you want to evaluate.


---

### Dataset modality names

The `groups_src` section maps dataset filenames to the modalities used by the pipeline.

Each entry in `groups_src` corresponds to a modality used in the workflow, for example `MR`, `CT`, or `MASK`.

Example dataset:

```text
Dataset/
└── Patient001/
    ├── MR.mha
    ├── CT.mha
    └── MASK.mha
```

Corresponding config:

```yaml
groups_src:
  MR:
  CT:
  MASK:
```

The names defined in the configuration must match the **filenames present in each case directory**.

If the filenames in your dataset are different (for example `T1.mha`, `CT_reg.mha`, or `BodyMask.mha`), the configuration must be updated accordingly so that each modality is correctly mapped.

---

### Batch size

```yaml
batch_size: 5
```

Defines the number of samples processed in each training batch.

This parameter mainly depends on **GPU memory**.

---

### Validation frequency

```yaml
it_validation: 2500
```

Defines how often validation is performed during training, in number of iterations.

---

### Validation cases

```yaml
validation: None
```

You can optionally provide a text file listing the cases used for validation.

Example:

```yaml
validation: validation_cases.txt
```

Example file:

```text
1HNA001
1THA001
1HNA005
```

Each line corresponds to the name of a case directory in the dataset.

If `validation` is set to `None`, no validation is performed.


---

## `Prediction.yml` — Inference

This file defines how predictions are generated from a trained model.

Main ideas:

- **Model**  
  Uses the same model architecture as during training (`UNetpp5`).

- **Input data**  
  The model takes **MR images** as input.  
  A **MASK** is also used to normalize the image inside the body region and mask irrelevant areas.

- **Preprocessing**  
  The MR image is standardized, then masked before being sent to the network.

- **Test-time augmentation (TTA)**  
  Two flip augmentations are applied during inference:
  - horizontal flip
  - vertical flip

  Their predictions are combined to improve robustness.

- **Patch inference**  
  Inference is done on `512 × 512` patches with **2.5D context**:
  - current slice
  - 2 slices above
  - 2 slices below

- **Output generation**  
  The predicted output is saved as **sCT**.

- **Postprocessing**  
  Before saving:
  - CT intensity values are restored (`UnNormalize`)
  - the output is masked outside the body
  - the image is converted to `int16`

- **Output geometry**  
  The predicted sCT keeps the same geometry as the MR image:
  - spacing
  - origin
  - orientation

- **Prediction output**  
  Results are saved in the folder defined by:

  ```yaml
  train_name: TRAIN_01
  ```

  which creates:

  ```text
  Predictions/TRAIN_01/
  ```

In short, `Prediction.yml` defines:

1. which model is used
2. how MR images are preprocessed
3. how inference is performed
4. how outputs are postprocessed and saved as sCT


## `Evaluation.yml` — Evaluation

This file defines how predictions are evaluated against the ground-truth images.

Main ideas:

- **Predicted modality**  
  The evaluation compares the predicted **sCT** with the reference **CT** images.

- **Metrics**  
  Several metrics are computed to measure the quality of the prediction:
  - `MAE` — Mean Absolute Error
  - `PSNR` — Peak Signal-to-Noise Ratio
  - `SSIM` — Structural Similarity Index

  Metrics are computed **inside the MASK region** to avoid evaluating background areas.

- **Input data**  
  The evaluation uses three modalities:
  - `CT` → ground-truth reference
  - `sCT` → predicted image
  - `MASK` → region where metrics are computed

- **Preprocessing**  
  Before computing the metrics, tensors are converted to the appropriate format (`float32` for images and `uint8` for masks).

- **Dataset sources**  
  The evaluation loads data from two locations:

  ```yaml
  dataset_filenames:
  - ./Dataset:a:mha
  - ./Predictions/TRAIN_01/Dataset:i:mha
  ```

  - `./Dataset` contains the **ground-truth images**
  - `./Predictions/TRAIN_01/Dataset` contains the **predicted sCT images**

  KonfAI automatically matches cases between these datasets.

- **Evaluation output**  
  The results are saved in:

  ```
  Evaluations/TRAIN_01/
  ```

  including:
  - evaluation logs
  - aggregated metrics
  - per-case metrics

In short, `Evaluation.yml` defines:

1. which predictions are evaluated
2. which ground-truth images are used
3. which metrics are computed
4. where evaluation results are saved

### Built-in and custom components

KonfAI pipelines are defined through **YAML configuration files**, while the actual operations are implemented by **Python classes**.

The main extensible components are:

- **Metrics**
- **Transforms**
- **Augmentations**
- **Networks**

Each component type has a **base class** that must be inherited when implementing custom behavior.

---

## Metrics

Metrics are used during **training and evaluation** to compute losses or evaluation scores.

Built-in metrics are implemented in:

https://github.com/vboussot/KonfAI/blob/main/konfai/metric/measure.py

All metrics must inherit from the base class:

```python
class Criterion(torch.nn.Module, ABC):
```

Minimal example with parameters (Focal Loss):

```python
class FocalLoss(Criterion):

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        error = torch.abs(output - targets[0])
        weight = error ** self.gamma
        return torch.mean(weight * error)
```

The parameter can then be defined in the YAML configuration:

```yaml
metric:FocalLoss:
  gamma: 2.0
```

In this example:

- `gamma` controls how strongly larger errors are emphasized
- the parameter is automatically passed from the YAML configuration to the metric constructor

---

## Transforms

Transforms are used for **preprocessing and postprocessing** of data.

Built-in transforms are implemented in:

https://github.com/vboussot/KonfAI/blob/main/konfai/data/transform.py

All transforms must inherit from:

```python
class TransformInverse(NeedDevice, ABC)
```

Minimal example of a transform modifying the tensor shape:

```python
class CustomTransform(TransformInverse):

    def __init__(self, pad: int = 2):
        super().__init__()
        self.pad = pad

    def transform_shape(self, group_src, name, shape, cache_attribute):
        shape[-1] += 2 * self.pad
        shape[-2] += 2 * self.pad
        return shape

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.nn.functional.pad(tensor, (self.pad, self.pad, self.pad, self.pad))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor[..., self.pad:-self.pad, self.pad:-self.pad]
```

The parameter can then be defined in the YAML configuration:

```yaml
CustomTransform:
  pad: 4
```

In this example:

- `transform_shape(...)` updates the expected tensor shape
- `__call__(...)` applies the corresponding padding to the tensor
- `inverse(...)` removes the padding and restores the original tensor shape
- the parameter `pad` controls the number of pixels added on each side

---

## Data augmentations

Augmentations are used to **randomly modify data during training or inference**.

Built-in augmentations are implemented in:

https://github.com/vboussot/KonfAI/blob/main/konfai/data/augmentation.py

All augmentations must inherit from:

```python
class DataAugmentation(NeedDevice, ABC)
```

Three internal methods define how an augmentation works:

```python
_state_init(...)
_compute(...)
_inverse(...)
```

Example structure:

```python
class CustomAugmentation(DataAugmentation):

    def _state_init(self, index, shapes, caches_attribute):
        return shapes

    def _compute(self, name, index, tensors):
        return tensors

    def _inverse(self, index, a, tensor):
        return tensor
```

---

### Example: horizontal flip augmentation

```python
class HorizontalFlip(DataAugmentation):

    def __init__(self):
        super().__init__()

    def _state_init(self, index, shapes, caches_attribute):
        return shapes

    def _compute(self, name, index, tensors):
        return [torch.flip(tensor, dims=[-1]) for tensor in tensors]

    def _inverse(self, index, a, tensor):
        return torch.flip(tensor, dims=[-1])
```

This augmentation flips the image horizontally.

The same operation is used in `_inverse(...)` because applying the flip twice restores the original image.

---

### Role of the internal methods

**`_state_init(...)`**

Initializes the **random parameters of the augmentation** (for example rotation angles, translation offsets, etc.).  
These parameters are stored so that the **same transformation is applied to all tensors of a sample** (e.g. MR, CT, MASK), ensuring that the different groups remain spatially aligned.

For example, a translation augmentation may sample random offsets and build the associated transformation matrices.

---

**`_compute(...)`**

Applies the **actual transformation** to the tensors.

For spatial augmentations (rotation, translation, etc.), this typically uses operations such as `grid_sample` to resample the image using the transformation defined in `_state_init(...)`.

---

**`_inverse(...)`**

This is mainly used for **test-time augmentation (TTA)**, where predictions are transformed back to the original orientation before being combined.

---

Augmentations are referenced in the YAML configuration inside the `augmentations` section.

---

## Networks

Networks can be defined either using **built-in models** provided by KonfAI or by implementing **custom Python classes**.

Several built-in architectures are available in the KonfAI repository:

https://github.com/vboussot/KonfAI/tree/main/konfai/models

These models can be used directly in configuration files.

---

### Custom networks

Custom networks can also be implemented as Python classes.

In this tutorial, the model architecture is implemented in:

```
UNetpp.py
```

and used in the configuration with:

```yaml
classpath: UNetpp:UNetpp5
```

This means:

- `UNetpp` → Python module (`UNetpp.py`)
- `UNetpp5` → class inside that module

KonfAI will automatically load the module and instantiate the specified class.

This mechanism allows **any custom neural network architecture** to be integrated directly into the KonfAI pipeline.

---

## Summary

KonfAI pipelines combine:

```
YAML configuration
        +
Python components
```

Where:

- **YAML** defines the workflow
- **Python classes** implement the operations

Custom components can be implemented for:

- metrics
- transforms
- augmentations
- networks

---

## Contact

If you have questions about KonfAI or this tutorial, feel free to contact me:

**boussot.v@gmail.com**

For bug reports, feature requests, or contributions, please open an **issue on GitHub** or submit a **pull request**.