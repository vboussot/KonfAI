# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the audit bug-fixes (see AUDIT.md §5)."""

import inspect
import os

os.environ.setdefault("KONFAI_config_file", "/tmp/konfai-none.yml")
os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")

import torch  # noqa: E402
from konfai.data.augmentation import Rotate  # noqa: E402
from konfai.data.transform import ResampleToShape, Standardize  # noqa: E402
from konfai.network.blocks import Select, Unsqueeze  # noqa: E402
from konfai.utils.dataset import Attribute  # noqa: E402


def test_vae_latent_uses_gaussian_noise():
    """#3 LatentDistributionZ must sample N(0,1), not U[0,1]."""
    from konfai.network.blocks import LatentDistribution

    layer = LatentDistribution.LatentDistributionZ()
    mu = torch.zeros(200_000)
    log_std = torch.zeros(200_000)
    z = layer(mu, log_std)  # == epsilon
    assert abs(float(z.mean())) < 0.05, "mean should be ~0"
    assert abs(float(z.std()) - 1.0) < 0.05, "std should be ~1 (Gaussian), not ~0.29 (uniform)"


def test_standardize_explicit_scalar_stats():
    """#5 Standardize with explicit scalar mean/std must not crash."""
    t = Standardize(lazy=False, mean=[10.0], std=[2.0])
    x = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    out = t("c", x.clone(), Attribute())
    assert torch.allclose(out, (x - 10.0) / 2.0)


def test_standardize_explicit_per_channel_stats():
    """#5 Per-channel mean/std broadcast over the channel axis."""
    t = Standardize(lazy=False, mean=[10.0, 20.0], std=[2.0, 4.0])
    x = torch.zeros(2, 3, 4)
    x[0] = 10.0
    x[1] = 20.0
    out = t("c", x.clone(), Attribute())
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_rotate_converts_degrees_to_radians():
    """#6 A 90-degree rotation must yield [[0,-1],[1,0]], not cos/sin of 90 radians."""
    rot = Rotate(a_min=90.0, a_max=90.0, is_quarter=False)
    rot._state_init(0, [[8, 8]], [Attribute()])
    block = rot.matrix[0][0][0, :2, :2]
    assert torch.allclose(block, torch.tensor([[0.0, -1.0], [1.0, 0.0]]), atol=1e-5)


def test_predict_evaluate_expose_tensorboard_param():
    """#7 CLI -tb/--tensorboard (dest 'tensorboard') must reach predict()/evaluate()."""
    from konfai.evaluator import evaluate
    from konfai.predictor import predict

    for fn in (predict, evaluate):
        params = inspect.signature(fn).parameters
        assert "tensorboard" in params, f"{fn.__name__} must accept 'tensorboard'"
        assert "tb" not in params, f"{fn.__name__} must not use the old 'tb' name"


def test_unsqueeze_forward_accepts_tensor():
    """#8 Unsqueeze.forward(tensor) must work on a single tensor."""
    assert Unsqueeze(dim=1)(torch.randn(3, 4)).shape == (3, 1, 4)


def test_resample_to_shape_does_not_mutate_config():
    """#9 transform_shape must not write resolved dims back into the shared instance config."""
    resampler = ResampleToShape(shape=[0, 16, 16])
    before = resampler.shape.clone()
    out = resampler.transform_shape("CT", "case", [8, 16, 16], Attribute())
    assert out[0] == 8  # sentinel 0 resolved to the input dim for this call
    assert torch.equal(resampler.shape, before), "self.shape must stay [0, 16, 16] for the next case"


def test_select_squeezes_size_one_dims_by_size():
    """#12 Select must squeeze dimensions whose size is 1, not the dim at index 1."""
    out = Select([slice(0, 1), slice(None), slice(None)])(torch.randn(1, 5, 6))
    assert out.shape == (5, 6)
    # a tensor with no size-1 dims is unchanged
    out2 = Select([slice(None), slice(None)])(torch.randn(4, 5))
    assert out2.shape == (4, 5)
