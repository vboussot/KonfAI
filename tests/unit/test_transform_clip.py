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

import pytest
import torch
from konfai.data.transform import Clip
from konfai.utils.dataset import Attribute


def test_clip_resolves_min_and_percentile_bounds() -> None:
    # ``min`` (torch scalar) and ``percentile:<p>`` (numpy scalar) bounds must be coerced to float
    # so the in-place clip assignments are valid for a torch tensor.
    tensor = torch.arange(0, 100, dtype=torch.float32)
    clip = Clip(min_value="min", max_value="percentile:90")

    out = clip("case", tensor.clone(), Attribute())

    assert out.min().item() == pytest.approx(0.0)
    assert out.max().item() == pytest.approx(89.1)
    assert out.dtype == torch.float32


def test_clip_fixed_numeric_bounds() -> None:
    tensor = torch.arange(-50, 50, dtype=torch.float32)
    clip = Clip(min_value=-10.0, max_value=10.0)

    out = clip("case", tensor.clone(), Attribute())

    assert out.min().item() == pytest.approx(-10.0)
    assert out.max().item() == pytest.approx(10.0)
