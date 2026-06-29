# KonfAI — Agent Guide

This is the canonical reference for humans and AI agents working on KonfAI. Read it before making changes. For the current code-quality audit and known issues, see [`AUDIT.md`](AUDIT.md).

## 1. What KonfAI is

KonfAI is a **modular, extensible, and fully configurable deep-learning framework for medical imaging** (Boussot & Dillenseger, 2025 — arXiv:2508.09823). The central idea: a model, its data pipeline, losses/metrics, optimizers/schedulers, augmentations, post-processing, and the whole train/predict/evaluate workflow are described in **structured YAML config files, without modifying the underlying code**. YAML is mapped onto Python objects by a **reflection engine** (`konfai/utils/config.py`). You describe *what* you want and KonfAI instantiates it; the config doubles as a complete, self-contained, reproducible record of the experiment.

**Why it exists:** to separate *experimental intent* (the YAML) from *implementation details* (the code), improving reproducibility, transparency, and traceability while still allowing advanced, non-standard workflows that other frameworks (PyTorch Lightning, MONAI, nnU-Net, NiftyNet) make hard. KonfAI has produced top-ranking results in MICCAI challenges (SynthRAD 2025, TrackRAD 2025, CURVAS 2024/2025, PANTHER) across segmentation, registration, and image synthesis.

**Three design principles** (from the paper):
1. **Declarative configuration** — every component is identified by its Python class path and *all* constructor arguments live in the YAML; the pipeline is reconstructable from config alone (no hardcoded values).
2. **Modularity & controlled extensibility** — interchangeable components (models, losses, transforms, schedulers, augmentations) are instantiated at runtime through a unified registry; each inherits a shared base (`Network`, `Criterion`/loss, `Transform`, …). Extend by subclassing and referencing the new class by path in YAML — no core edits.
3. **Experiment traceability & workspace management** — each experiment lives in a dedicated workspace (config snapshot, logs, predictions, checkpoints, evaluation JSON).

**Advanced capabilities** it abstracts natively (the reason it's worth the complexity): patch-based learning in **2D / 2.5D / 3D**, **test-time augmentation (TTA)**, **model ensembling**, **deep supervision / multi-head / intermediate-feature access**, and **multi-model setups** (GANs, teacher–student/adversarial). The key enabler is that models declare *every* submodule explicitly via `add_module`, so any layer/head/skip/attention block is **referenceable by name** in YAML (e.g. for a loss, a perceptual feature, or a deep-supervision head) — standard PyTorch `forward()` hides these.

Two design pillars run through the entire codebase:

1. **Config-by-reflection.** `apply_config(path)` inspects a callable's signature at call time and recursively builds its arguments from the YAML subtree it owns (`@config("Key")`). Defaults are *materialised back* into the YAML file so a run leaves a fully-resolved config on disk.
2. **Lazy, patch-based imaging.** Volumes are never loaded whole into RAM. Data is read as overlapping patches, optionally streamed chunk-by-chunk, and predictions are reassembled with overlap blending. **Preserving this invariant is mandatory.**

A third, newer pillar is emerging:

3. **Declarative models.** Networks are routed module graphs assembled via `add_module` (with `in_branch`/`out_branch` routing metadata). Models can be written as Python classes in `konfai/models/` *or* described entirely in a `.yml` file via the **YAML model builder** (`konfai/utils/model_builder.py`). See §7.

## 2. Repository layout

| Path | Role |
|---|---|
| `konfai/` | Core package (config, data, network, workflows, utils) |
| `konfai-apps/` | **Separate** installable package `konfai_apps` (app management, HF repos, FastAPI server) — own `pyproject.toml` |
| `apps/` | Ready-to-use model app bundles (excluded from the `konfai` wheel) |
| `examples/` | Runnable `Segmentation` and `Synthesis` workflows (assume CWD = the example dir) |
| `docs/` | Sphinx documentation site |
| `tests/` | Core test suite (`tests/unit`, `tests/integration`) |
| `konfai-apps/tests/` | Apps test suite (run separately — see §8) |

### Core package map

| Module | Role |
|---|---|
| `konfai/utils/config.py` | **The reflection engine.** `Config` context manager + `@config` + `apply_config`. Read before any config change. |
| `konfai/utils/errors.py` | Typed exception hierarchy (`KonfAIError` → `ConfigError`, `TrainerError`, `DatasetManagerError`, …). Use these; do not invent new ones. |
| `konfai/utils/dataset.py` | Storage abstraction. `Dataset` + `AbstractFile` backends (`SitkFile`, `H5File`, `OmeZarrFile`, `DicomFile`) + the `Attribute` geometry sidecar. |
| `konfai/utils/dicom.py` | DICOM series reader/writer (pydicom). |
| `konfai/utils/ome_zarr.py` | OME-Zarr / OME-NGFF reader/writer (zarr; ngff-zarr is a declared but currently *unused* dep — see AUDIT.md). |
| `konfai/utils/ITK.py` | SimpleITK helpers (transform composition, resampling, mask-bbox cropping). |
| `konfai/utils/runtime.py` | `DistributedObject`, distributed setup/teardown, GPU/VRAM, TensorBoard process, logging. |
| `konfai/utils/utils.py` | `SUPPORTED_EXTENSIONS`, `get_module`, patch-slice math, `split_path_spec`. |
| `konfai/data/data_manager.py` | `DataManager`/`DatasetIter`: builds the flat (dataset, augmentation, patch) index, subset/validation selection, DDP sharding, DataLoaders. |
| `konfai/data/patching.py` | `ModelPatch`/`Accumulator`: sliding-window slicing + overlap-blended reassembly; `DatasetManager` per-case cache/stream layer. |
| `konfai/data/transform.py` | Preprocessing/postprocessing transforms (`Normalize`, `Standardize`, `Resample…`, `Mask`, `Crop`, …) + `TransformLoader`. |
| `konfai/data/augmentation.py` | Data augmentation primitives + ADA for DiffusionGAN. |
| `konfai/network/network.py` | `ModuleArgsDict` (routed graph), `Network` (training machinery), `ModelLoader`, `Measure`. The heart of the model system. |
| `konfai/network/blocks.py` | Reusable blocks (`ConvBlock`, `ResBlock`, `Attention`, `Concat`, `get_torch_module`, …). |
| `konfai/utils/model_builder.py` | The declarative YAML→`Network` builder with a safe module registry. |
| `konfai/models/` | Concrete architectures (segmentation, classification, generation, registration, representation). |
| `konfai/metric/measure.py` | `Criterion` hierarchy (losses + metrics) loaded by name from YAML. |
| `konfai/metric/schedulers.py` | Scalar weight schedulers for criteria. |
| `konfai/trainer.py` / `predictor.py` / `evaluator.py` | The `train()` / `predict()` / `evaluate()` pipelines. |
| `konfai/main.py` | CLI: `konfai` → `main()`, `konfai-cluster` → `cluster()`. Subcommands `TRAIN`, `RESUME`, `PREDICTION`, `EVALUATION`. |
| `konfai/__init__.py` | Env-backed directory/state accessors, device utilities. |

## 3. Architecture in depth

### 3.1 Config reflection (`konfai/utils/config.py`)
- `@config("Key")` tags a class/callable with the YAML branch it owns (sets `function._key`).
- `apply_config(path)(Cls)()` opens a `Config` on the resolved dot-path, then walks `inspect.signature` params: primitives are coerced, `Path` resolved, `list`/`dict` expanded, and nested `@config` objects recursively instantiated.
- **Reading config mutates it**: resolved defaults are merged back into the YAML file on `Config.__exit__`. A successful run rewrites the active config with all defaults filled in.
- Defaults use the `"default|<value>"` marker; Python `None` round-trips through YAML as the string `"None"`.

### 3.2 Data pipeline
`Dataset` (`utils/dataset.py`) maps a logical `(group, name)` onto on-disk artefacts via a backend chosen by `file_format`: `h5`→`H5File`, `omezarr`/`ome-zarr`/`zarr`→`OmeZarrFile`, `dicom`→`DicomFile`, anything else→`SitkFile` (`.mha`/`.nii`/npy/xml/vtk/transforms). Geometry travels in an `Attribute` sidecar (`Origin`, `Spacing`, `Direction` + arbitrary metadata).

`DataManager`/`DatasetIter` (`data/data_manager.py`) build a flat `(x=dataset, a=augmentation, p=patch)` index, apply subset/validation selection, shard across DDP ranks, and feed PyTorch DataLoaders. `DatasetManager`/`ModelPatch`/`Accumulator` (`data/patching.py`) do the patch slicing and overlap-blended reassembly. Transforms (`data/transform.py`) and augmentations (`data/augmentation.py`) are config-bound and run per case.

**Data convention:** arrays are always **channel-first** `[C, (Z), Y, X]`. Geometry/spacing is **`(x, y, z)`** order (SimpleITK convention).

### 3.3 Network graph (`konfai/network/network.py`)
- `ModuleArgsDict` is a `torch.nn.Module` whose ordered children each carry routing metadata. `add_module(name, module, in_branch=[...], out_branch=[...], alias=..., ...)` wires a dataflow graph (a string-keyed branch register; branch `'0'` is the implicit input/main path).
- `Network` adds optimizer/scheduler/criterion wiring, patch-based forward, gradient/cross-GPU checkpointing, alias-based state-dict load/save, and loss/metric aggregation via `Measure`.
- `ModelLoader.get_model()` resolves the configured `classpath`: a `.yml`/`.yaml` suffix → YAML builder (§7); otherwise import the class from `konfai.models` and build it via `apply_config`. Non-`Network` classes are wrapped in `MinimalModel`.

### 3.4 Workflows & runtime
`train()`/`predict()`/`evaluate()` are wrapped by `run_distributed_app` (`utils/runtime.py`), which builds the configured `DistributedObject`, sets the `KONFAI_*` env vars, forces `KONFAI_CONFIG_MODE='Done'`, and spawns one process per GPU (or submits to SLURM via `submitit`). All disk/logging side effects are gated on `global_rank == 0`.

## 4. Execution flow (end to end)

```
CLI (main.py) ─▶ train()/predict()/evaluate()  [@run_distributed_app]
   └▶ build_*()  ─▶ configure_workflow_environment()   # sets KONFAI_config_file/ROOT/STATE + dirs
                   └▶ KONFAI_CONFIG_MODE = 'Done'
                   └▶ apply_config()(Trainer/Predictor/Evaluator)()   # YAML → object graph
   └▶ execute_distributed_object()
        └▶ mp.spawn → DistributedObject.__call__(rank)
             └▶ setup_gpu() → model.init()/init_outputs_group()/_compute_channels_trace()
             └▶ DataManager → DataLoader (lazy patches)
             └▶ training loop / inference / evaluation
```

## 4b. Commands, config files & workspace

KonfAI is command-driven. Four CLI states map to three YAML config files:

| Command | Config file | Top-level key | Purpose |
|---|---|---|---|
| `TRAIN` / `RESUME` | `Config.yml` | `Trainer:` | Model + dataset + losses + augmentations + optimizer/schedulers + training params |
| `PREDICTION` | `Prediction.yml` | `Predictor:` | Load model(s), patch/TTA/ensemble inference, `outputs_dataset` post-processing |
| `EVALUATION` | `Evaluation.yml` | `Evaluator:` | Compare predictions vs ground truth, per-case + aggregate metric JSON |

Each run creates/updates a **workspace** organised by `train_name` (model name):

```
Workspace/
  Checkpoints/<name>/<timestamp>.pt   # checkpoints (BEST or ALL)
  Setups/<name>/Config_0.yml          # full resolved config snapshot
  Statistics/<name>/events.out.*      # TensorBoard logs
  Predictions/<name>/Dataset/...      # predicted volumes + Prediction.yml
  Evaluations/<name>/Metric_*.json    # per-case + aggregate metrics + Evaluation.yml
  Dataset/<case>/<group>.<ext>        # input data tree (e.g. CT.mha, MASK.mha)
```

### Config anatomy (high-value keys an agent will touch)
- **`Trainer.Model`**: `classpath` (e.g. `segmentation.UNet.UNet` or `UNet.yml`), `Optimizer`, `schedulers`, `outputs_criterions` (loss/metric attached to a **named** module output, e.g. `UNetBlock_0:Head:Softmax`), plus the model's own constructor args.
- **`Dataset`**: `groups_src` → `groups_dest` (each group has `transforms`, `is_input`, `patch_transforms`); `dataset_filenames` (e.g. `./Dataset/:a:mha` — the `:a:`/`:i:` flags mean append/intersect and the suffix is the format); `use_cache`, `batch_size`, `subset`, `validation`, `validation_augmentations` (default `true`; set `false` to validate only on base samples — train and validation datasets are then prepared separately so augmented variants never leak into validation), `shuffle`, `inline_augmentations`.
- **`Patch`**: `patch_size`, `overlap`, `extend_slice` (>0 ⇒ 2.5D, requires `patch_size[0]==1`), `pad_value`.
- **`augmentations`**: `DataAugmentation_*` → `data_augmentations` (Flip/Rotate/…); `nb` augmentations per sample.
- **Training params**: `epochs`, `it_validation`, `autocast` (AMP), `gradient_checkpoints`, `gpu_checkpoints`, `ema_decay`, `data_log`, `save_checkpoint_mode` (`BEST`/`ALL`), `EarlyStopping`, `manual_seed`. (Training also stops automatically once the schedulers decay the optimizer learning rate to `≤ 0`, via `EarlyStoppingBase.stop()`.)
- **`Predictor.outputs_dataset`**: per output module, `OutputDataset` with `after_reduction_transforms` (Argmax/…), `final_transforms` (TensorCast), `patch_combine`, `reduction` (TTA: Mean/Median/…), `inverse_transform`; top-level `combine` does model ensembling.
- **`Evaluator.metrics`**: per target group, `targets_criterions` → `criterions_loader` (e.g. `Dice`).

## 5. Environment variables

These are process-global state set by `configure_workflow_environment`; accessors in `konfai/__init__.py` read them.

| Variable | Meaning |
|---|---|
| `KONFAI_config_file` | Path to the active YAML config (note the **lowercase** `config_file` suffix — easy to mistype). Read directly from `os.environ` with no fallback. |
| `KONFAI_CONFIG_MODE` | Config behaviour: `Done` (workflows require this), `default`, `Import` (suppresses `apply_config` side effects during module import), `interactive`, `remove`. **Note:** `interactive`/`remove` are effectively dead in the current product (see AUDIT.md). |
| `KONFAI_ROOT` / `KONFAI_STATE` | Workflow root dir / current `State` (TRAIN/RESUME/PREDICTION/EVALUATION). |
| `KONFAI_CONFIG_PATH` | Hidden channel set by `apply_config`, read back by `PerceptualLoss`. |
| `CUDA_VISIBLE_DEVICES`, `KONFAI_MASTER_PORT`, `KONFAI_CLUSTER`, `KONFAI_OVERWRITE` | Distributed/cluster control. |

## 6. Imaging formats (DICOM & OME-Zarr)

Both are first-class dataset backends, dispatched by `file_format` and reachable from config (`SUPPORTED_EXTENSIONS` validates the format). Verified working end-to-end (round-trips, patch reads, geometry):

- **DICOM** (`utils/dicom.py`): groups by `SeriesInstanceUID`, sorts slices by `ImagePositionPatient` projected on the `ImageOrientationPatient` normal, derives `(origin, spacing(x,y,z), direction(9))`, applies HU rescale, supports header-only metadata + lazy per-slice patch reads, and writes uncompressed series. Geometry round-trips **byte-identically to SimpleITK's GDCM reader**. Caveats: write quantises floats to int16 and always uses the CT SOP class; left-handed directions normalise to right-handed on round-trip (standard DICOM behaviour); multi-frame/enhanced DICOM and irregular slice spacing are not handled. See AUDIT.md.
- **OME-Zarr** (`utils/ome_zarr.py`): a thin adapter over **`ngff-zarr`** (no hand-rolled NGFF parsing) — `ngff_zarr.from_ngff_zarr` for lazy chunk-wise reads, `to_ngff_zarr` for writes; output is `ngff-zarr`-interoperable (verified). **Resolution selection:** a multiscale pyramid level can be chosen per source via the dataset-spec suffix **`omezarr@<level>`** (e.g. `./Dataset:a:omezarr@2` reads pyramid level 2, coarser), parsed by `split_format_level()` and threaded to `OmeZarrFile` — independent of any transform; level 0 (full res) by default. The chosen level's `scale` becomes the KonfAI `Spacing`. Caveat: `Direction` is not representable in NGFF and round-trips only via a proprietary `konfai` group attr; KonfAI's own writer emits a single level (multi-level pyramids come from `ngff_zarr.to_multiscales`).

## 7. Declarative YAML model builder

`konfai/utils/model_builder.py` builds a routed `Network` from a `.yml` file (`name`/`parameters`/`network`/`modules`). It is **safe by construction**: node types must come from two curated registries (`_MODULE_REGISTRY` for `nn.Module` factories, `_OBJECT_REGISTRY` for config objects) — no `eval`/import injection. Features: `${param}` exact-string references (preserve Python values), `{$multiply: [...]}`, `{$object: BlockConfig, args: {...}}`, nested graphs, full `add_module` routing.

`ModelLoader` activates it when `classpath` ends in `.yml`/`.yaml` (resolved relative to `KONFAI_config_file`, or under `konfai_root()/models` for `default|…`). See `examples/Segmentation/UNet.yml`.

**Status (verified):** the example `UNet.yml` builds a model **identical** to the Python `UNet` (1,934,299 params, matching forward). **It does not yet replace `konfai/models/`:**
- The registry has only 13 module types. Replacing the existing models needs many more (`Linear`, `Add`/`Multiply`, `Attention`, `Upsample`, `View`/`Select`, `LatentDistribution`, norm/activation factories, …) and the enum config objects (`DownsampleMode`/`UpsampleMode`/`NormMode`).
- Roughly 70–80% of `models/` is pure `add_module` wiring (UNet, NestedUNet, ResNet, GAN, VAE) and is migratable once the registry grows. The rest carries genuine Python logic that the builder cannot express: custom `forward`/sampling/training loops in `ddpm`, `diffusionGan`, `cStyleGan`, `registration` (spatial-transformer grid math), `convNeXt` (LayerNorm/DropPath), `representation`.

So: **the YAML builder complements `models/` today and can replace the feed-forward subset after the registry is expanded.** See AUDIT.md for the migration plan.

**Model → YAML migration matrix** (how much of each zoo model the builder can express today):

| Model | Migratable now? | Blocker |
|---|---|---|
| UNet, NestedUNet/UNet++, ResNet | ✅ Yes | Pure `add_module` wiring; registry already covers it. |
| GAN/CycleGAN, VAE, VoxelMorph | ⚠️ Partial | Needs registry growth (`Linear`/`Add`/`LatentDistribution`/…) + channel-math/factory constructs. |
| ConvNeXt, DDPM, DiffusionGAN, cStyleGAN, Representation | ⛔ No | Custom Python `forward`/sampling/training logic the builder has no construct for. |

## 7b. How a run actually works (mental model)

Tracing one config key end-to-end makes the reflection engine concrete. Given `Config.yml`:
```yaml
Trainer:
  Model:
    classpath: segmentation.UNet.UNet
    UNet:
      dim: 2
      channels: [1, 32, 64, 128, 256]
```
1. `train()` → `build_train()` → `configure_workflow_environment()` sets `KONFAI_config_file=Config.yml`, then forces `KONFAI_CONFIG_MODE='Done'`.
2. `apply_config("Trainer")(Trainer)()` opens a `Config` on the `Trainer` subtree and walks `Trainer.__init__`'s signature. The `model: ModelLoader` param (a `@config`-tagged type) is recursively built.
3. `ModelLoader.get_model()` reads `classpath`. No `.yml` suffix → `get_module("segmentation.UNet.UNet", "konfai.models")` imports the class, then `apply_config("Trainer.Model.UNet")(UNet)()` binds `dim`, `channels`, … from the `UNet:` subtree (defaults materialised back into the file).
4. `UNet.__init__` issues `add_module(...)` calls → a routed `ModuleArgsDict` graph.
5. On `Config.__exit__`, the resolved subtree is merged back to `Config.yml` (so the file now records every default).

Swap `classpath: UNet.yml` and step 3 instead calls `build_model_from_yaml` (§7) — same `add_module` graph, no Python class.

## 7c. Extending KonfAI (custom components)

Every extension point is "subclass a base, reference it by class path in YAML" — no core edits. The base classes inherit the config + (for modules) the routing machinery.

- **Custom model:** subclass `konfai.network.network.Network`; build the graph in `__init__` via `add_module(name, module, in_branch=[...], out_branch=[...])`. Reference with `classpath: my_module.MyNet` (importable) or, for a pure feed-forward graph, write a `.yml` and register any new block via `model_builder.register_module`.
- **Custom loss/metric:** subclass the criterion base in `konfai/metric/measure.py`; `forward` returns a `Tensor` (loss) or a tuple (metric). Reference under `outputs_criterions`/`metrics` by class path, attached to a **named module output** (e.g. `UNetBlock_0:Head:Softmax`).
- **Custom transform:** subclass `konfai.data.transform.Transform`; implement `__call__(name, tensor, cache_attribute)` **and** `transform_shape()` (must predict the output spatial shape exactly — patch planning depends on it). Pair `inverse()` symmetrically if `apply_inverse`.
- **Custom augmentation:** subclass `konfai.data.augmentation.DataAugmentation`; implement `_state_init` (sample params per case index) and `_compute` (apply lazily). Return one shape per input; only `Mask`/`Permute` may change the shape.
- **New imaging format:** add a `Dataset.AbstractFile` backend in `konfai/utils/dataset.py`, dispatch it in `File.__enter__`, and register its aliases in `SUPPORTED_EXTENSIONS` (`utils/utils.py`). Keep the heavy reader lib import-guarded.

**Routing rules to respect** (`add_module`): branch `'0'` is the implicit input; an `in_branch` must be produced by an earlier module (execution = insertion order, no topo-sort); `out_branch: [-1]` marks a terminal/deep-supervision head; module names must contain no `.`; `alias` lists are positional and load-bearing for pretrained weight remapping.

## 7d. Apps & packaged models (`konfai-apps`, `apps/*`)

`konfai-apps` is a **separate package** layering remote/app/packaged-model functionality on top of the core public API (it never reaches into core internals; core never imports it). An "app" bundles `app.json` metadata + a KonfAI config (`Prediction.yml`, …) + custom `.py` + `.pt` weight checkpoints. Apps are resolved from a **Local** directory, a **HuggingFace** repo, or a **Remote** server. The `apps/*` bundles (`totalsegmentator`, `mrsegmentator`, `impact_synth`) are thin CLI wrappers that resolve an HF app and call `KonfAIApp.pipeline()`. Pretrained models are distributed as `.pt` checkpoints downloaded on demand. There is also a FastAPI server (`app_server.py`) with job lifecycle, GPU-semaphore scheduling, SSE log streaming, and TTL'd results.

> ⚠️ **Trust model (read before resolving any app).** Resolving an app **copies the app's `.py` files into the working directory and imports them**, and installs the app's `requirements.txt` via a `pip install` subprocess. A downloaded app therefore runs **arbitrary code and arbitrary dependency installs** on your machine. This is inherent to "packaged model = code + weights + config". **Only resolve apps from sources you trust** (your own repos, vetted HF orgs). Do not point the loader at untrusted HuggingFace IDs or remote servers. See AUDIT.md §4b.

## 7e. Metrics & criteria

Criteria live in `konfai/metric/measure.py` (`Criterion` hierarchy, loaded by class path from `outputs_criterions`/`metrics`, weight-scheduled via `konfai/metric/schedulers.py`). Notes for agents:

- **Optional-dependency criteria** import heavy packages lazily through `_require_optional(module, criterion=…, extra=…)`, which raises an actionable `MeasureError` (with the `pip install konfai[<extra>]` hint) at construction. `SSIM` needs `konfai[ssim]` (`scikit-image`); `FID` needs `konfai[fid]` (`scipy`+`torchvision`); `LPIPS` needs `konfai[lpips]`. Add new optional-dep criteria the same way — never a bare `import` that fails mid-run.
- **`Criterion.forward` is typed `-> Tensor` but several subclasses return a `(loss, dict)` tuple** (metrics); consumers `isinstance`-branch. Follow the existing pattern of the criterion you extend.
- **`update_scheduler`** selects the active weight scheduler for the current iteration; an empty schedule raises `ConfigError`, and iterations past the last window clamp to the last scheduler.

## 8. Running things

### Tests
The dev environment must have the imaging extras installed to exercise the real DICOM/OME-Zarr/ITK paths (the Pixi `dev` env does; a bare `pip install .[dev]` does **not** — see AUDIT.md tooling drift).

```bash
pixi run test                 # core unit + integration (tests/) — pytest -q
pixi run --environment dev python -m pytest tests/unit -q       # core unit only
pip install -e ./konfai-apps && pixi run --environment dev python -m pytest konfai-apps/tests   # apps suite
pixi run test-cov             # with coverage
```

Baseline: `tests/unit` green, `tests/integration` green, `konfai-apps/tests/unit` 25 passed.

> **Caveat:** root `pytest testpaths=['tests']` excludes `konfai-apps/tests`, so `pixi run test` does **not** run the apps suite. Run it explicitly. `konfai-apps` is an **independent package**: the core dev env no longer carries its runtime deps (fastapi/uvicorn/python-multipart) — install the package itself with `pip install -e ./konfai-apps` (which pulls them) before running its suite, exactly as its CI does.

### Lint / format / types / build / docs
```bash
pixi run --environment dev lint           # ruff check konfai konfai-apps/konfai_apps
pixi run --environment dev format-check    # ruff format --check
pixi run --environment dev typecheck       # mypy konfai
pixi run --environment dev build           # python -m build
pixi run --environment docs build-docs     # Sphinx HTML
pixi run check                             # lint + format-check + test (run before finalising)
```
(`lint`/`format`/`format-check` exist in both the `dev` and `lint` envs; pass `--environment`.)

## 9. Optional dependencies

Install via `pip install konfai[<extra>]` or Pixi (the `dev` env bundles the runtime extras; docs deps live in the `docs` env).

| Extra | Packages | Use |
|---|---|---|
| `itk` | `SimpleITK` | ITK image I/O + transforms |
| `hdf5` | `h5py` | HDF5 datasets |
| `dicom` | `pydicom` | DICOM series backend |
| `omezarr` | `zarr`, `ngff-zarr` | OME-Zarr / NGFF backend |
| `imaging` | `SimpleITK`, `h5py`, `pydicom`, `zarr`, `ngff-zarr` | All imaging backends |
| `monitoring` | `nvidia-ml-py` (imports as `pynvml`) | GPU VRAM monitoring |
| `tensorboard` | `tensorboard` | Training visualisation |
| `vtk` / `lpips` / `cluster` | `vtk` / `lpips` / `submitit` | Mesh I/O / perceptual loss / SLURM |
| `ssim` / `fid` | `scikit-image` / `scipy`+`torchvision` | SSIM criterion / FID criterion |
| `all` | every runtime extra | Full install |
| `dev` | pytest, ruff, build, Sphinx, … | Development |

> Heavy deps (`SimpleITK`, `h5py`, `pydicom`, `zarr`, `pynvml`, `tensorboard`) are **optional** and import-guarded — code must fail at point-of-use with an install hint, not at import.

## 10. Invariants — do NOT break these

- **Never load a full volume into RAM.** Use lazy/patch/streaming access (`can_stream_patch`, `read_data_slice`).
- **Channel-first arrays** `[C,(Z),Y,X]`; **spacing/geometry in `(x,y,z)`** order. `Attribute` geometry keys are `Origin`/`Spacing`/`Direction`.
- **`Attribute` stringifies every value** and reparses geometry via `np.fromstring(s[1:-1], sep=" ")` — only flat scalars/1-D arrays round-trip (nested arrays break; this is a real bug, see AUDIT.md). Read via `__getitem__`/`get_np_array`; do not pre-suffix keys with `_`.
- **`KONFAI_config_file` and `KONFAI_CONFIG_MODE` must be set** before any `Config()`; tests must `monkeypatch.setenv` both. Workflows require `KONFAI_CONFIG_MODE='Done'`.
- **Patch ordering** must match between `disassemble` (read) and `Accumulator` (write); for PREDICTION/EVALUATION, all patches of a case must stay on the same DDP rank.
- **`outputs_criterions` keys** must equal a module's dotted path (e.g. `UNetBlock_0:Head:Argmax`); the `:`/`.` separators are load-bearing.
- **`state_dict` load/save deliberately does not recurse into nested `Network`s** (each owns its optimizer/state); alias lists are positional and load-bearing for pretrained weights.
- **YAML model builder is the trusted-untrusted boundary**: only registry types may be instantiated; module names contain no `.`.
- **Format aliasing**: `ome-zarr`/`ome_zarr`/`zarr` → `omezarr`; keep `SUPPORTED_EXTENSIONS` consistent.
- **`konfai-apps` is a separate package**; `apps/` is excluded from the `konfai` wheel.

## 11. Coding conventions
- Line length 120 (Ruff). Type annotations on new public functions. Apache-2.0 SPDX header on every new source file.
- No wildcard imports; prefer `pathlib.Path`. Use existing error classes from `konfai/utils/errors.py`.
- Do not import heavy optional deps (`SimpleITK`, `h5py`, `pydicom`, `zarr`) at module top level in code paths that don't need them — guard with `try/except ImportError` + a `_require_*()` helper.

## 12. Commit conventions
- **Conventional Commits** are enforced in CI (`cz check`): `type(scope): subject` (`feat`, `fix`, `perf`, `docs`, `build`, `ci`, `refactor`, `test`, `chore`).
- A `commit-msg` hook + CI reject AI-agent branding (`maestro`, `claude`, `codex`, "generated by/with") — **avoid these words even in file names referenced in the subject**.
- Imperative present tense, subject < 72 chars. No AI co-author trailers.

## 13. Rules for AI agents
1. **Read before editing.** Open every file you change.
2. **Keep diffs small.** One logical change per PR; no unrelated reformats.
3. **Run `pixi run check`** (and the apps suite if you touched `konfai-apps`) before finalising.
4. **No new runtime dependencies** without explicit request + matching `pyproject.toml` update (declare the dep in the *same* commit as the code that uses it).
5. **Never load imaging datasets fully into RAM.**
6. **Update docs** when changing user-facing CLI/config behaviour, and update `tests/unit/test_config.py` when changing config binding.
7. **Use existing error types**; do not invent exceptions.
8. **Do not skip pre-commit hooks** with `--no-verify`.

## 14. Common pitfalls
- `Config.__init__` reads `KONFAI_config_file` straight from `os.environ`; tests must set both env vars via `monkeypatch.setenv`.
- `nvidia-ml-py` imports as `pynvml` — not the same string.
- Format readers (DICOM/OME-Zarr/ITK) live in `konfai/utils/` and are imported by `konfai/data/` — do not move them into `data/`.
- `pixi run test` does not run `konfai-apps/tests`; run them explicitly.
- The pip `[dev]` extra and the Pixi `dev` feature define *different* dev environments (see AUDIT.md) — prefer Pixi for full coverage.
- Reading a config rewrites it on disk (defaults materialised on `Config.__exit__`).
