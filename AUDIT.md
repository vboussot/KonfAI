# KonfAI — Code Audit

**Scope.** The whole repository, with deep focus on OME-Zarr, DICOM, and the declarative YAML model builder.

**Method.** Every subsystem was read in depth; high-severity findings were verified against the code and, where runnable, reproduced in the Pixi `dev` environment. Each finding carries a verdict (`confirmed` / `partially-confirmed` / `overstated`).

**Status.** The modernization stack (PRs #6–#9) and the audit follow-up (PR #10) are merged to `main`. About half the confirmed bugs are fixed with regression tests; the rest are a prioritized backlog (below). `tests/` and `konfai-apps/tests` are green; lint/format clean.

## Focus areas (all work end-to-end)

| Area | Verdict |
|---|---|
| **OME-Zarr** (`utils/ome_zarr.py`) | ✅ Round-trips verified, `ngff-zarr`-interoperable. ⚠️ Re-implements parts of `ngff-zarr`; some multiscale read paths are dead code; `Direction` rides a proprietary attr (not NGFF-standard). |
| **DICOM** (`utils/dicom.py`) | ✅ Geometry round-trips **byte-identical to SimpleITK's GDCM reader** (incl. left-handed → right-handed normalization). Caveats: lossy int16 write, CT-only SOP class, single-gap z-spacing. |
| **YAML model builder** (`utils/model_builder.py`) | ✅ `examples/Segmentation/UNet.yml` builds a model **identical** to the Python `UNet` (1,934,299 params, matching forward). ⛔ Cannot yet replace `models/`: registry too small (13 types) and custom-`forward` models are inexpressible. |

## Fixed (merged to `main`)

**Modernization (PRs #6–#9):** OME-Zarr + DICOM dataset backends, the declarative YAML model builder, performance/elegance pass, and docs.

**Audit follow-up (PR #10):**

- **#1** per-epoch augmentation re-sampling — `DataAugmentation.reset_state()` clears per-case sampling so params are re-drawn each epoch.
- **#2** `load_state_dict` warm-start checks `isinstance(child, (Conv, Linear))`, guards on the checkpoint key, and `continue`s instead of `return`ing.
- **#10** `Crop.transform_shape` reuses a persisted box to skip the full-volume read (mitigated; a fully-lazy variant is out of scope).
- **#14** `Accumulator.assemble` seeds the output from the first present patch and raises a typed `PatchError`.
- **#15** `update_scheduler` raises `ConfigError` on an empty schedule.
- **#16** SSIM/FID/LPIPS declared as `konfai[ssim]`/`konfai[fid]`/`konfai[lpips]` extras + an actionable `MeasureError` at construction.
- **#17** `LinearVAE` rebuilt on a real `LatentDistribution` bottleneck; `VAE` documented as a deterministic autoencoder.
- **#18** `Representation.Adaptation` sets `requires_grad` in `__init__` (forward is pure again).
- **#19** debug blocks (`Print`/`Write`/`Exit`) gated + documented; `Write` takes an explicit `path`.
- **DICOM hardening:** non-destructive `write_dicom_series`; multi-frame + irregular-spacing detection.
- **`os.environ` device control-flow removed:** `Network.to` threads an explicit GPU-index counter.
- **#11** reclassified as a reserved future feature (documented).

Each fixed item landed with a regression test where the behaviour is testable (`tests/unit/test_audit_fixes.py`, `test_patching.py`, `test_named_forward.py`, `test_imaging_formats.py`, `test_early_stopping.py`).

## Open backlog — confirmed bugs, not yet fixed

Severity: 🔴 high · 🟠 medium · 🟡 low. Each is a small, well-scoped, test-backed fix.

| # | Bug | Sev | Where |
|---|---|---|---|
| 3 | VAE latent uses `torch.rand_like` (U[0,1]) where the reparameterization trick needs `randn_like` (N(0,1)) | 🔴 | `network/blocks.py` |
| 4 | Early-stopping/BEST score keys on a leaked loop variable, so it scores the **EMA** model when EMA is on, not the base model | 🔴 | `trainer.py` |
| 5 | Explicit `Standardize` `mean`/`std` crashes (`torch.tensor([list])` → nested tensor, then the stringly-typed `Attribute` reparse fails) | 🟠 | `data/transform.py` |
| 6 | `Rotate` feeds degrees straight to `cos/sin` with no `deg2rad` (a "90°" rotation is 90 rad) | 🟠 | `data/augmentation.py` |
| 7 | CLI `-tb/--tensorboard` is silently dropped for predict/eval (`tb` vs `tensorboard` param name) | 🟠 | `main.py` / `predictor.py` / `evaluator.py` |
| 8 | `Unsqueeze.forward(*tensor)` errors on a tensor input (`torch.unsqueeze` gets a tuple) | 🟠 | `network/blocks.py` |
| 9 | `ResampleToShape` mutates its own config across cases (`new_shape = self.shape` aliasing) | 🟠 | `data/transform.py` |
| 12 | `Select.forward` squeezes by index, not by size | 🟡 | `network/blocks.py` |
| 13 | `ITK._open_transform` double-appends displacement-field transforms (currently in dead code) | 🟡 | `utils/ITK.py` |

## Cross-cutting (deferred refactors, each PR-sized)

- **Typed `Attribute` sidecar.** Every value is `str()`'d and geometry reparsed via `np.fromstring(s[1:-1])` — lossy and the root cause of #5. The active bug is patched; the serialization rework is its own change.
- **Perceptual-loss duplication.** `IMPACTReg`/`IMPACTSynth`/`SAM_Perceptual` are ~70-line near-duplicates with *different* mask/normalization — needs characterization tests first.
- **Smells:** pervasive mutable default args (masked by the Config layer); `Criterion.forward` typed `-> Tensor` but several subclasses return tuples; `ITK.py` is largely dead; the `interactive`/`remove` config modes are unreachable.
- **Tests still thin** on DDP/multi-rank, RESUME, and explicit-arg transforms (where #5/#6 slipped through).

## Claims downgraded by verification

- **DICOM "geometrically wrong on flipped-z" — overstated.** KonfAI is byte-identical to SimpleITK; left-handed → right-handed normalization is correct DICOM behaviour (locked by a regression test).
- **VAE noise / `ResampleToResolution` / `Accumulator` / `Config.get_value` StopIteration — partially confirmed:** the defect is real but narrower than first claimed (sibling classes/paths are fine).
