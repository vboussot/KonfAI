# KonfAI — Code Audit

**Scope:** the whole repository (`konfai/`, `konfai-apps/`, tests, docs, packaging), with deep focus on the three areas requested: OME-Zarr support, DICOM support, and the declarative YAML model builder.

**Method.** Every subsystem was read in depth (16 parallel readers over the full tree), then the high-severity findings were **adversarially verified** — independently re-checked against the code and, where runnable, with a minimal reproduction in the Pixi `dev` environment (torch 2.12, SimpleITK 2.5.5, pydicom 3.0.2, **zarr 3.2.1**, ngff-zarr 0.37, numpy 2.5). Findings below carry a verdict (`confirmed` / `partially-confirmed` / `overstated`) and were promoted to this document only after that check. Two reader claims were **downgraded** by verification (see §8) — they are reported honestly rather than dropped.

**Test baseline (this revision):** `tests/` 167 passed, `konfai-apps/tests/unit` 25 passed; ruff lint + format clean.

---

## 1. Executive summary

KonfAI is a genuinely capable, well-architected framework: the config-by-reflection engine, the routed `ModuleArgsDict` graph, and the lazy patch/stream data layer are coherent and powerful, and the three focus areas **work end-to-end**. The main risks are not in the headline features but in (a) a handful of **confirmed correctness bugs** in augmentation/transform/VAE/CLI/trainer code, (b) **pervasive low-level smells** (mutable default args, stringly-typed `Attribute`, `os.environ` used as control-flow state, name-string coupling), (c) **dead code & duplication** (notably `ITK.py`, the dead OME-Zarr multiscale path, repeated SimpleITK transform-serialization), and (d) **tooling/test-coverage gaps** (apps tests not in the default run; imaging happy-paths not in CI; pip-`dev` vs Pixi-`dev` divergence).

| Focus area | Status | Headline |
|---|---|---|
| **OME-Zarr** | ✅ Works; ⚠️ reinvents ngff-zarr | Round-trips verified; output is ngff-zarr–interoperable; but `ngff-zarr` is a declared-yet-unused dep and ~half the module (multiscale read/`select_level`) is not wired into the pipeline. |
| **DICOM** | ✅ Works (matches SimpleITK) | Geometry round-trips **byte-identical to SimpleITK's GDCM reader**, incl. left-handed/flipped-z normalization. Caveats: lossy int16 write, CT-only SOP class, no multi-frame, single-gap z-spacing. |
| **YAML model builder** | ✅ Works; ⛔ cannot yet replace `models/` | `examples/Segmentation/UNet.yml` builds a model **identical** to the Python `UNet` (1,934,299 params). Registry (13 types) is too small and custom-`forward` models can't be expressed — see §4. |

---

## 2. OME-Zarr (`konfai/utils/ome_zarr.py`)

**Verified working:** write→read round-trips for 3-D `(C,Z,Y,X)`, 2-D `(C,Y,X)`, patch/slice reads, axis canonicalization to channel-first, and scale/translation↔Spacing/Origin conversion are all correct against **zarr 3.2.1** (a `create_array`/`create_dataset` shim handles v2/v3). Direction (not representable in NGFF) round-trips via a proprietary `konfai` attrs key. New tests added (`tests/unit/test_imaging_roundtrip.py`).

**Issues**
- ⚠️ **Reinvents ngff-zarr (the dependency you flagged).** `ngff_zarr` is imported and probed (`_NGFF_ZARR_AVAILABLE`) but **never called** anywhere; `_parse_ngff_axes/_scale/_translation/_canonical_shape/select_level` hand-parse `multiscales[0]` — exactly what ngff-zarr exposes as typed objects. **Verified:** ngff-zarr *can* read KonfAI's output (`from_ngff_zarr` returns correct dims/scale/translation/data), so the formats are compatible and a migration is low-risk. → **Recommended PR** (see §7); I added an interop regression test rather than do the rewrite in this pass.
- 🔵 **Dead code:** `read_ome_zarr_slice` (documented as the "primary entry point") and `select_level` are used only by tests, never by `dataset.py` (which uses `read_ome_zarr_data_slice`). Multiscale pyramids and the time axis are effectively unsupported despite the API surface; write only emits a single level.
- 🟡 **Standards/interop gaps:** `Direction` is invisible to any standards-compliant reader (only the proprietary key carries it); store is written as zarr-v3 storage carrying an NGFF "0.4" (a v2-era) version label — a hybrid that happens to read back.
- 🟢 **Error-type consistency (fixed this pass):** `read_ome_zarr_data_slice` raised a bare `ValueError` on slice-arity mismatch → now `DatasetManagerError` (matches AGENTS.md rule), with a test.
- 🔵 `_open_level` re-opens the group and re-reads attrs on every slice/info call — no store/group caching; costly for remote stores the docstring advertises.

## 3. DICOM (`konfai/utils/dicom.py`)

**Verified working:** series discovery by `SeriesInstanceUID`, position-based slice ordering, geometry extraction `(origin, spacing(x,y,z), direction(9))`, HU rescale, lazy per-slice patch reads, and write. **Cross-validated:** KonfAI's reader output is **byte-identical to SimpleITK's `ImageSeriesReader`** for origin/spacing/direction/data, including a **left-handed (feet-first, z-down) input**, which both normalize to the same right-handed frame with a flipped array — i.e. KonfAI is consistent with the reference DICOM reader. New tests added.

**Issues**
- 🟡 **Lossy/limited write:** floats are quantised to int16 via a derived slope/intercept; the SOP class is **always `CTImageStorage`** even when `Modality` is `OT`; `ImageType` is hardcoded `DERIVED/PRIMARY/AXIAL`.
- 🟡 **Edge cases not handled:** inter-slice spacing is taken from the **first two slices only** (irregular spacing/localizers/missing slices → wrong z-spacing, no consistency check); **multi-frame / enhanced DICOM** (one file, many frames) is treated as one slice and would mis-stack; single-slice series silently default z-spacing to 1.0 mm.
- 🟡 **Destructive write:** `write_dicom_series` deletes **all** `*.dcm` in the target directory first — undocumented; dangerous if multiple groups share a folder.
- 🟢 **Error-type consistency (fixed this pass):** `read_dicom_series_slice` raised `ValueError` on slice-arity mismatch → now `DatasetManagerError`, with a test.
- 🔵 **Performance:** a single patch read parses the whole series' headers 3–4× (`get_dicom_info` + two `_select_series_files` + sorts); `file_to_data_statistics` calls the slice reader per z-slice → O(N²) header reads for one volume's statistics. `discover_series` swallows all exceptions (incl. corrupt-but-present series → misleading "no DICOM found").
- 🔵 Dead imports: `DicomSequence`, the empty `if TYPE_CHECKING: pass`.

## 4. YAML model builder (`konfai/utils/model_builder.py`)

**Verified working:** safe-registry build, `${param}`/`$multiply`/`$object` resolution, nested routed graphs, and `add_module` routing all function; all 25 `test_model_builder.py` tests pass. **Decisive equivalence proof (new test):** building `examples/Segmentation/UNet.yml` yields a model with **the same parameter count (1,934,299) and forward output** as the Python `UNet` configured identically.

**Can it replace `konfai/models/`? Not yet — here is exactly what's missing:**
1. **Registry is too small (13 types):** `Conv/ConvTranspose/MaxPool/AvgPool/Conv1d-3d/Softmax/Identity/ArgMax/ConvBlock/ResBlock/Concat`. Shipped models also need (all exist in `blocks.py`/torch but are **not** registered): `Linear`, `Add`, `Multiply`, `Attention`, `Upsample`, `View`/`Select`/`Subset`, `LatentDistribution`, `NormalNoise`, `Const`, `Unsqueeze`/`Permute`/`ToChannels`/`ToFeatures`, plus norm/activation factories. The object registry has only `BlockConfig` — the heavily-used `DownsampleMode`/`UpsampleMode`/`NormMode` enums are not expressible.
2. **Custom-logic models can't be expressed at all.** ~70–80% of `models/` is pure `add_module` wiring (UNet, NestedUNet, ResNet, GAN, VAE — migratable once the registry grows), but the rest carries genuine Python `forward`/sampling/training logic the builder has no construct for: `ddpm` (7 custom methods — diffusion schedule/sampling), `diffusionGan`, `cStyleGan`, `registration` (spatial-transformer grid math), `convNeXt` (LayerNorm/DropPath/LayerScaler), `representation`.

**Verdict:** the builder is a sound, safe foundation that **complements** `models/` today and can **replace the feed-forward subset** after (1). A migration plan is in §7. Other notes: no looping/recursion construct, so deep nets are hand-unrolled and verbose (UNet.yml is 196 lines vs a few in Python); `register_module` mutates a **process-global** registry with no isolation/unregister (a hazard for the long-lived app server); the YAML feature has **no user-doc statement** of these limitations.

---

## 5. Confirmed bugs (adversarially verified)

Severity: 🔴 high · 🟠 medium · 🟡 low. "Status" is whether it's fixed here or recommended as a PR (kept conservative per the brief: fix only clear/useful/well-covered, no big refactor).

| # | Bug | Sev | Verdict | Status |
|---|---|---|---|---|
| 1 | **Per-epoch augmentation re-sampling is broken.** `DataAugmentation.state_init` short-circuits when an index is already in `who_index`, and `who_index` is **never cleared**, so random transform params are frozen for the object's lifetime. `docs/.../training.md:97` promises re-sampling each epoch — false. (`augmentation.py:199-202`) | 🔴 | confirmed (repro: identical params across 3 epochs) | PR |
| 2 | **`load_state_dict` warm-start checks the wrong object.** `isinstance(module, torch.nn.Linear)` should be `isinstance(child, …)` (so resized `Linear` never warm-starts), and an early `return` inside the per-child loop aborts loading the rest of the subtree. Live checkpoint-load path. (`network.py:898,913`) | 🔴 | confirmed | PR |
| 3 | **VAE latent uses uniform noise.** `LatentDistributionZ.forward` uses `torch.rand_like` (U[0,1]) where the reparameterization trick needs `torch.randn_like` (N(0,1)); the KL term assumes a unit Gaussian. One-token fix. (`blocks.py:442`) | 🔴 | partially-confirmed (claim conflated `NormalNoise`, which is fine) | PR |
| 4 | **Early-stopping/BEST score uses the EMA model when EMA is on.** `trainer._log` reuses a leaked loop variable `label` (= `_EMA` after the loop) to key `measures`, so the returned score is the EMA model's, not the base model's. (`trainer.py:469,519-520`) | 🔴 | confirmed | PR |
| 5 | **`Standardize`/explicit `mean`/`std` crash.** `torch.tensor([self.mean])` on a list makes a nested tensor; the stringly-typed `Attribute` reparse (`np.fromstring(s[1:-1])`) then fails. Only the computed-scalar path works. (`transform.py:281-288`) | 🟠 | confirmed (repro) | PR |
| 6 | **`Rotate` treats degrees as radians.** Angles sampled in `[a_min,a_max]` (default `[0,360]`) are fed straight to `cos/sin` with no `deg2rad`; a "90°" rotation is actually 90 rad. Used by `diffusionGan`. (`augmentation.py:315-331,90-135`) | 🟠 | confirmed (repro) | PR |
| 7 | **CLI `-tb/--tensorboard` is silently dropped for predict/eval.** CLI dest is `tensorboard`, but `predict()`/`evaluate()` declare the param as `tb`; `run_distributed_app` filters kwargs to the signature, so TensorBoard cannot be enabled from the CLI for PREDICTION/EVALUATION. `train()` works. (`main.py:108` vs `predictor.py:1075`/`evaluator.py:528`) | 🟠 | confirmed | PR |
| 8 | **`Unsqueeze.forward(*tensor)` errors on a tensor input** (`torch.unsqueeze` gets a tuple). Used in `resnet`/`convNeXt`. (`blocks.py:254`) | 🟠 | confirmed (repro) | PR |
| 9 | **`ResampleToShape` mutates its own config across cases.** `new_shape = self.shape` (alias) with sentinel-0 substitution writes the first case's dims into the shared instance, leaking into later cases. (`transform.py:459,466`) | 🟠 | partially-confirmed (`ResampleToResolution` is fine) | PR |
| 10 | **`Crop.transform_shape` does a full volume read** (`read_data` + percentile) at `DatasetManager.__init__`, violating the never-load-full-volume rule. Acknowledged `TODO(perf)`. (`transform.py:1043-1055`) | 🟠 | confirmed | PR |
| 11 | **MC-dropout count ignored in local inference.** `app.py` passes `mc` into `install_inference`, but `LocalAppRepository.install_inference` never uses it (no `_set_number_of_mc`). The `--mc` flag has no local effect. (`app_repository.py:553`) | 🟡 | confirmed | PR |
| 12 | **`Select.forward` squeezes by index, not size** (`enumerate(range(...))` → tests `i==1`, not `shape[i]==1`). Dead-ish but wrong. (`blocks.py:374`) | 🟡 | confirmed | PR |
| 13 | **`ITK._open_transform` double-appends displacement-field transforms** (append inside the branch + the unconditional append) → double application in `compose_transform`. Currently in dead code. (`ITK.py:83,88`) | 🟡 | confirmed (repro) | PR |
| 14 | **`Accumulator.assemble` can `UnboundLocalError`** if patch index 0 was never filled (`result` only bound inside the index-0 branch). Live path fills index 0, so latent. (`patching.py:184-203`) | 🟡 | partially-confirmed (repro) | PR |

Each row has a verified minimal fix on file; they are batched into the PR plan (§7) because most live in code with **no existing test**, so per the brief ("fix only what's clear, useful, and well covered") they should land **with** a regression test rather than be slipped in here.

## 6. Cross-cutting audit (by dimension)

**Performance**
- Confirmed perf-positive work already in tree (in-memory best-checkpoint tracking, O(1) index-cache, once-per-batch predict logging, `get_names` cache).
- DICOM statistics are O(N²) header reads (§3); OME-Zarr re-opens the group per access (§3); `Crop` reads full volumes (#10); `data_manager.__getitem__` recomputes `needs_full_load` every sample (cheap after caching but allocates).

**Elegance / duplication / unnecessary complexity**
- **Stringly-typed `Attribute`** (`dataset.py:68-95`): every value is `str()`'d and geometry reparsed via `np.fromstring(s[1:-1], sep=" ")` — lossy, locale/printoptions-sensitive, and the root cause of bug #5. `startswith`-based key counting also cross-contaminates prefix-sharing keys (e.g. `Spacing` vs a `SpacingExtra` metadata key).
- **Duplicated SimpleITK transform serialization** appears 3× (`dataset.py` H5 write / Sitk read / `read_transform`) with a latent `UnboundLocalError` on unknown transform types.
- **`os.environ` as control-flow state:** `Network.to` increments `os.environ['device']`; `get_layers` writes `KONFAI_DEBUG_LAST_LAYER`; `PerceptualLoss.forward` does `del os.environ['device']` (raises if unset). Not thread/process-safe; order-dependent.
- **Name-string coupling:** channel tracing keys `ToChannels/ToFeatures` and `ReduceLROnPlateau` by `__class__.__name__` rather than `isinstance` — renames silently break routing.
- **`named_forward` nested out-branch propagation** (`network.py:701-708`) is the highest-complexity, untested, string-surgery (`split('.')`, `;accu;`) function in the codebase.
- Dead config machinery: the `interactive` and `remove` `KONFAI_CONFIG_MODE` modes are read but never set anywhere (~⅓ of `get_value`/`_get_input*` is unreachable). `check_konfai_install`/`KonfAIPackagesError`/`_KONFAI_DEPS` (~60 lines) have no callers.
- **`ITK.py`** is ~13/15 functions unused in-repo (large untested dead surface).

**Architecture / API**
- `Criterion.forward` is typed `-> torch.Tensor` but many subclasses return tuples — the base contract is effectively a lie; consumers `isinstance`-check.
- `SitkFile.file_to_data` returns an lxml `Element` for `.xml` (violates the declared `tuple[np.ndarray, Attribute]`).
- Backend ABC is not honored uniformly (`SitkFile.get_names/get_group` raise `NotImplementedError`; `OmeZarr/Dicom.get_names` ignore the `group` arg).
- Two different `run_distributed_app` decorators (apps vs core) with different semantics are in scope together — confusing; rename one.
- `read_data` opens H5 with `read=False` (writable handle) even on read paths — concurrency/perf hazard.

**Typing**
- Pervasive `image: sitk.Image = None` / `h5_group: h5py.Group = None` non-Optional defaults (typing lies); `get_torch_module`/`get_conv` annotated `-> torch.nn.Module` but return classes/`None`; `Network.forward` annotated `-> torch.Tensor` but returns the last yielded value.

**Mutable default arguments** — pervasive across the public config API (`schedulers={...}`, `outputs_criterions={...}`, `alias=[[],[],[]]`, `patch_size=[128,128,128]`, `ModelPatch()` defaults, `data_augmentations={'default|Flip': Prob(1)}`). Mostly masked because the Config layer overrides them and they aren't mutated — but a real footgun and Ruff-B006 trap.

**Missing tests** (highest value first)
- The patch math (`Accumulator`, `PathCombine/Mean/Cosinus`, overlap blending, 2.5-D `extend_slice`) has **no direct unit tests** — the riskiest reconstruction code is only exercised indirectly.
- `named_forward` branch routing has no direct test.
- Transforms are largely untested with explicit args (`Standardize`/`Clip` percentile/lists — bug #5 slipped through), `Rotate`/affine augmentations untested (bug #6), DDP/multi-rank paths untested.
- No test that `--tensorboard` reaches predict/eval (bug #7); no RESUME test.

**Documentation**
- `docs/source/konfai.utils.rst` omits `dicom`, `ome_zarr`, `model_builder`, `runtime`, `errors`; there is **no `konfai.models.rst`** at all. `docs/.../apps.rst` autodocs `konfai_apps`, which Read the Docs doesn't install → build warnings/empty output. `examples/README.md` lists a non-existent `UNetpp.py`. `installation.md` under-describes the `imaging` extra (omits pydicom/zarr/ngff-zarr).

**Packaging / tooling** (see also the `pixi`/`ruff` discussion)
- **Two divergent dev environments:** pip `[project.optional-dependencies].dev` has Sphinx but **no ruff/mypy/build/imaging deps**; the Pixi `dev` feature has those but **no Sphinx**. `pip install -e .[dev]` cannot run ruff/mypy or imaging tests.
- **`ruff==0.15.2` pinned in 4 places** (Pixi `dev` + `lint` features, pre-commit, CI) that must stay in lockstep. Pixi-idiomatic fix: declare ruff + tasks once in the `lint` feature and compose `dev = { features = ["dev","lint"] }`; prefer conda `dependencies` over `pypi-dependencies` for conda-forge tools.
- **`pixi run test` does not run `konfai-apps/tests`** (root `testpaths=['tests']`); only the apps CI does. Add a Pixi task for the apps suite.
- CI installs `.[dev]` (no imaging extras) then runs `pytest tests`, so the imaging happy-paths run **only locally** — add an imaging-extra CI job.

## 7. Recommended PRs (prioritized)

1. **Correctness bug-fix PR (high):** bugs #1–#4 (aug re-sampling, `load_state_dict`, VAE noise, EMA early-stopping) — each with a regression test. Small diffs, high impact.
2. **Transform/augmentation correctness PR (medium):** bugs #5, #6, #9 (`Standardize` explicit stats, `Rotate` deg→rad, `ResampleToShape` aliasing) + a typed/robust `Attribute` serialization (fixes the #5 root cause) — with tests for explicit-arg transforms.
3. **CLI/apps PR (medium):** bug #7 (`tb`→`tensorboard`), bug #11 (`--mc` local), + a `--tensorboard reaches predict/eval` test and a RESUME test.
4. **OME-Zarr ↔ ngff-zarr migration (medium):** replace the hand-rolled NGFF parse/write with `ngff-zarr` (`from_ngff_zarr`/`to_ngff_zarr`), keep zarr for chunked array access; drop dead `read_ome_zarr_slice`/`select_level` or wire multiscale into the live path. Interop already verified, so risk is bounded; gate behind the existing round-trip + interop tests.
5. **Model-builder registry expansion (medium):** register the missing blocks/torch layers + enum objects (§4.1), add per-build registry isolation/unregister, document the custom-`forward` limitation. Then migrate ResNet/NestedUNet/GAN/VAE to `.yml` one at a time, each guarded by a param-count/forward-equivalence test like the one added here.
6. **Patch-math test PR (high value, no code change):** direct unit tests for `Accumulator`/overlap blending/`PathCombine`/2.5-D before any refactor; then bugs #13, #14 with coverage.
7. **Tooling PR (low):** unify the dev environments and the single-source ruff pin; add a Pixi task + CI job for `konfai-apps/tests` and an imaging-extra CI job; clean `ITK.py` dead code; fix the docs module reference.
8. **DICOM robustness PR (medium):** irregular-spacing detection, multi-frame rejection/support, configurable SOP class, non-destructive write — each with a test.

## 8. Claims downgraded by verification (reported for honesty)
- **DICOM "geometrically wrong" on flipped-z — overstated.** KonfAI is **byte-identical to SimpleITK**; left-handed → right-handed normalization is correct DICOM behavior, not a bug. (Now locked by a regression test.)
- **`Config.get_value` `next()` StopIteration — overstated.** Real in isolation but the branch is unreachable given how `key_tmp` is built; defensive-only.
- **VAE noise / `ResampleToResolution` / `Accumulator` — partially confirmed:** the defect is real but narrower than first claimed (sibling classes/paths are fine); see #3, #9, #14.

## 9. Changes applied in this pass (conservative)
- `konfai/utils/ome_zarr.py`, `konfai/utils/dicom.py`: bare `ValueError` on slice-arity mismatch → `DatasetManagerError` (AGENTS.md error-type rule), with tests.
- `tests/unit/test_imaging_roundtrip.py` (new): DICOM↔SimpleITK geometry consistency, left-handed normalization, OME-Zarr↔ngff-zarr interop, non-identity-direction backend round-trip, error-type checks.
- `tests/unit/test_yaml_model_equivalence.py` (new): `UNet.yml` builds + forwards, and matches the Python `UNet` parameter count.
- `AGENTS.md`: rewritten/expanded from the whole-codebase + paper understanding.

No larger refactor was performed: the confirmed bugs live in code without existing tests, so they are queued as test-backed PRs (§7) rather than slipped in untested.
