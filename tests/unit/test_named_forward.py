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

"""Unit tests for ModuleArgsDict branch routing (named_forward / forward)."""

import os

os.environ.setdefault("KONFAI_config_file", "/tmp/konfai-none.yml")
os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")

import torch  # noqa: E402

from konfai.network.blocks import Add  # noqa: E402
from konfai.network.network import ModuleArgsDict  # noqa: E402


class _MulConst(torch.nn.Module):
    """Deterministic test module: multiplies its input by a fixed constant."""

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.factor


class _TwoInputGraph(ModuleArgsDict):
    """A(in 0)→branch 0, B(in 1)→branch 1, Sum(in 0,1)→branch 2."""

    def __init__(self) -> None:
        super().__init__()
        self.add_module("A", _MulConst(3.0), in_branch=[0], out_branch=[0])
        self.add_module("B", _MulConst(10.0), in_branch=[1], out_branch=[1])
        self.add_module("Sum", Add(), in_branch=[0, 1], out_branch=[2])


class _Inner(ModuleArgsDict):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("Scale", _MulConst(2.0))


class _NestedGraph(ModuleArgsDict):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("Pre", _MulConst(5.0), in_branch=[0], out_branch=[0])
        self.add_module("Block", _Inner(), in_branch=[0], out_branch=[0])


def test_forward_routes_two_inputs_through_branches():
    graph = _TwoInputGraph()
    a = torch.ones(1, 1, 2, 2)
    b = torch.full((1, 1, 2, 2), 2.0)
    out = graph(a, b)  # 3*a + 10*b = 3 + 20 = 23
    assert torch.allclose(out, torch.full_like(out, 23.0))


def test_named_forward_exposes_every_intermediate():
    graph = _TwoInputGraph()
    a = torch.ones(1, 1, 2, 2)
    b = torch.full((1, 1, 2, 2), 2.0)
    outputs = {name: float(tensor.flatten()[0]) for name, tensor in graph.named_forward(a, b)}
    assert outputs == {"A": 3.0, "B": 20.0, "Sum": 23.0}


def test_named_forward_uses_dotted_names_for_nested_graphs():
    graph = _NestedGraph()
    x = torch.ones(1, 1, 2, 2)
    names = [name for name, _ in graph.named_forward(x)]
    assert "Pre" in names
    assert "Block.Scale" in names  # nested submodule addressable by dotted path
    out = graph(x)  # 5 then *2 = 10
    assert torch.allclose(out, torch.full_like(out, 10.0))


def test_out_branch_isolation_preserves_a_branch_for_later_use():
    """A branch written by one module must remain available to a later consumer."""

    class _SkipGraph(ModuleArgsDict):
        def __init__(self) -> None:
            super().__init__()
            # Keep the raw input on branch 1, transform branch 0, then combine.
            self.add_module("Identity", torch.nn.Identity(), in_branch=[0], out_branch=[1])
            self.add_module("Scale", _MulConst(4.0), in_branch=[0], out_branch=[0])
            self.add_module("Sum", Add(), in_branch=[0, 1], out_branch=[0])

    graph = _SkipGraph()
    x = torch.ones(1, 1, 2, 2)
    out = graph(x)  # 4*x + x = 5
    assert torch.allclose(out, torch.full_like(out, 5.0))
