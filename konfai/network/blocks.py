import ast
import importlib
from collections.abc import Callable
from enum import Enum

import SimpleITK as sitk  # noqa: N813
import torch

from konfai.network import network
from konfai.utils.config import config


class NormMode(Enum):
    NONE = (0,)
    BATCH = 1
    INSTANCE = 2
    GROUP = 3
    LAYER = 4
    SYNCBATCH = 5
    INSTANCE_AFFINE = 6


def get_norm(norm_mode: Enum, channels: int, dim: int) -> torch.nn.Module | None:
    if norm_mode == NormMode.BATCH:
        return get_torch_module("BatchNorm", dim=dim)(channels, affine=True, track_running_stats=True)
    if norm_mode == NormMode.INSTANCE:
        return get_torch_module("InstanceNorm", dim=dim)(channels, affine=False, track_running_stats=False)
    if norm_mode == NormMode.INSTANCE_AFFINE:
        return get_torch_module("InstanceNorm", dim=dim)(channels, affine=True, track_running_stats=False)
    if norm_mode == NormMode.SYNCBATCH:
        return torch.nn.SyncBatchNorm(channels, affine=True, track_running_stats=True)
    if norm_mode == NormMode.GROUP:
        return torch.nn.GroupNorm(num_groups=32, num_channels=channels)
    if norm_mode == NormMode.LAYER:
        return torch.nn.GroupNorm(num_groups=1, num_channels=channels)
    return None


class UpsampleMode(Enum):
    CONV_TRANSPOSE = (0,)
    UPSAMPLE = (1,)


class DownsampleMode(Enum):
    MAXPOOL = (0,)
    AVGPOOL = (1,)
    CONV_STRIDE = 2


def get_torch_module(name_fonction: str, dim: int | None = None) -> torch.nn.Module:
    return getattr(
        importlib.import_module("torch.nn"),
        f"{name_fonction}" + (f"{dim}d" if dim is not None else ""),
    )


@config("BlockConfig")
class BlockConfig:

    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias=True,
        activation: str | Callable[[], torch.nn.Module] | None = "ReLU",
        norm_mode: str | NormMode | Callable[[int], torch.nn.Module] = "NONE",
    ) -> None:
        self.kernel_size = kernel_size
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.norm_mode = norm_mode
        self.norm: NormMode | Callable[[int], torch.nn.Module] | None = None
        if isinstance(norm_mode, str):
            self.norm = NormMode[norm_mode]
        else:
            self.norm = norm_mode

    def get_conv(self, in_channels: int, out_channels: int, dim: int) -> torch.nn.Conv3d:
        return get_torch_module("Conv", dim=dim)(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )

    def get_norm(self, channels: int, dim: int) -> torch.nn.Module:
        if self.norm is None:
            return None
        return get_norm(self.norm, channels, dim) if isinstance(self.norm, NormMode) else self.norm(channels)

    def get_activation(self) -> torch.nn.Module:
        if self.activation is None:
            return None
        if isinstance(self.activation, str):
            return (
                get_torch_module(self.activation.split(";")[0])(
                    *[ast.literal_eval(value) for value in self.activation.split(";")[1:]]
                )
                if self.activation != "None"
                else torch.nn.Identity()
            )
        return self.activation()


class ConvBlock(network.ModuleArgsDict):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_configs: list[BlockConfig],
        dim: int,
        alias: list[list[str]] = [[], [], []],
    ) -> None:
        super().__init__()
        for i, block_config in enumerate(block_configs):
            self.add_module(
                f"Conv_{i}",
                block_config.get_conv(in_channels, out_channels, dim),
                alias=alias[0],
            )
            norm = block_config.get_norm(out_channels, dim)
            if norm is not None:
                self.add_module(f"Norm_{i}", norm, alias=alias[1])
            activation = block_config.get_activation()
            if activation is not None:
                self.add_module(f"Activation_{i}", activation, alias=alias[2])
            in_channels = out_channels


class ResBlock(network.ModuleArgsDict):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_configs: list[BlockConfig],
        dim: int,
        alias: list[list[str]] = [[], [], [], [], []],
    ) -> None:
        super().__init__()
        for i, block_config in enumerate(block_configs):
            self.add_module(
                f"Conv_{i}",
                block_config.get_conv(in_channels, out_channels, dim),
                alias=alias[0],
            )
            norm = block_config.get_norm(out_channels, dim)
            if norm is not None:
                self.add_module(f"Norm_{i}", norm, alias=alias[1])
            activation = block_config.get_activation()
            if activation is not None:
                self.add_module(f"Activation_{i}", activation, alias=alias[2])

            if in_channels != out_channels:
                self.add_module(
                    "Conv_skip",
                    get_torch_module("Conv", dim)(
                        in_channels,
                        out_channels,
                        1,
                        block_config.stride,
                        bias=block_config.bias,
                    ),
                    alias=alias[3],
                    in_branch=[1],
                    out_branch=[1],
                )
                self.add_module(
                    "Norm_skip",
                    block_config.get_norm(out_channels, dim),
                    alias=alias[4],
                    in_branch=[1],
                    out_branch=[1],
                )
            in_channels = out_channels

        self.add_module("Add", Add(), in_branch=[0, 1])
        self.add_module(f"Norm_{i + 1}", torch.nn.ReLU(inplace=True))


def downsample(in_channels: int, out_channels: int, downsample_mode: DownsampleMode, dim: int) -> torch.nn.Module:
    if downsample_mode == DownsampleMode.MAXPOOL:
        return get_torch_module("MaxPool", dim=dim)(2)
    if downsample_mode == DownsampleMode.AVGPOOL:
        return get_torch_module("AvgPool", dim=dim)(2)
    if downsample_mode == DownsampleMode.CONV_STRIDE:
        return get_torch_module("Conv", dim)(in_channels, out_channels, kernel_size=2, stride=2, padding=0)


def upsample(
    in_channels: int,
    out_channels: int,
    upsample_mode: UpsampleMode,
    dim: int,
    kernel_size: int | list[int] = 2,
    stride: int | list[int] = 2,
):
    if upsample_mode == UpsampleMode.CONV_TRANSPOSE:
        return get_torch_module("ConvTranspose", dim=dim)(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
    else:
        if dim == 3:
            upsample_method = "trilinear"
        if dim == 2:
            upsample_method = "bilinear"
        if dim == 1:
            upsample_method = "linear"
        return torch.nn.Upsample(scale_factor=2, mode=upsample_method.lower(), align_corners=False)


class Unsqueeze(torch.nn.Module):

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, *tensor: torch.Tensor) -> torch.Tensor:
        return torch.unsqueeze(tensor, self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"


class Permute(torch.nn.Module):

    def __init__(self, dims: list[int]):
        super().__init__()
        self.dims = dims

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.permute(tensor, self.dims)

    def extra_repr(self):
        return f"dims={self.dims}"


class ToChannels(Permute):

    def __init__(self, dim: int):
        super().__init__([0, dim + 1, *[i + 1 for i in range(dim)]])


class ToFeatures(Permute):

    def __init__(self, dim: int):
        super().__init__([0, *[i + 2 for i in range(dim)], 1])


class Add(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *tensor: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.stack(tensor), dim=0)


class Multiply(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *tensor: torch.Tensor) -> torch.Tensor:
        return torch.mul(*tensor)


class Concat(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(tensor, dim=1)


class Print(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        print(tensor.shape)
        return tensor


class Write(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:

        sitk.WriteImage(sitk.GetImageFromArray(tensor.clone()[0][0].cpu().numpy()), "./Data.mha")
        return tensor


class Exit(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        exit(0)


class Detach(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach()


class Negative(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return -tensor


class GetShape(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(tensor.shape)


class ArgMax(torch.nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.argmax(tensor, dim=self.dim).unsqueeze(self.dim)


class Select(torch.nn.Module):

    def __init__(self, slices: list[slice]) -> None:
        super().__init__()
        self.slices = tuple(slices)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        result = tensor[self.slices]
        for i, s in enumerate(range(len(result.shape))):
            if s == 1:
                result = result.squeeze(dim=i)
        return result


class NormalNoise(torch.nn.Module):

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.dim is not None:
            return torch.randn(self.dim).to(tensor.device)
        else:
            return torch.randn_like(tensor).to(tensor.device)


class Const(torch.nn.Module):

    def __init__(self, shape: list[int], std: float) -> None:
        super().__init__()
        self.noise = torch.nn.parameter.Parameter(torch.randn(shape) * std)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.noise.to(tensor.device)


class Subset(torch.nn.Module):
    def __init__(self, slices: list[slice]):
        super().__init__()
        self.slices = [slice(None, None), slice(None, None)] + slices

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[self.slices]


class View(torch.nn.Module):
    def __init__(self, size: list[int]):
        super().__init__()
        self.size = size

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(self.size)


class LatentDistribution(network.ModuleArgsDict):

    class LatentDistributionLinear(torch.nn.Module):

        def __init__(self, shape: list[int], latent_dim: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(torch.prod(torch.tensor(shape)), latent_dim)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return torch.unsqueeze(self.linear(tensor), 1)

    class LatentDistributionDecoder(torch.nn.Module):

        def __init__(self, shape: list[int], latent_dim: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(latent_dim, torch.prod(torch.tensor(shape)))
            self.shape = shape

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return self.linear(tensor).view(-1, *[int(i) for i in self.shape])

    class LatentDistributionZ(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()

        def forward(self, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
            return torch.exp(log_std / 2) * torch.rand_like(mu) + mu

    def __init__(self, shape: list[int], latent_dim: int) -> None:
        super().__init__()
        self.add_module("Flatten", torch.nn.Flatten(1))
        self.add_module(
            "mu",
            LatentDistribution.LatentDistributionLinear(shape, latent_dim),
            out_branch=[1],
        )
        self.add_module(
            "log_std",
            LatentDistribution.LatentDistributionLinear(shape, latent_dim),
            out_branch=[2],
        )

        self.add_module(
            "z",
            LatentDistribution.LatentDistributionZ(),
            in_branch=[1, 2],
            out_branch=[3],
        )
        self.add_module("Concat", Concat(), in_branch=[1, 2, 3])
        self.add_module(
            "DecoderInput",
            LatentDistribution.LatentDistributionDecoder(shape, latent_dim),
            in_branch=[3],
        )


class Attention(network.ModuleArgsDict):

    def __init__(self, f_g: int, f_l: int, f_int: int, dim: int):
        super().__init__()
        self.add_module(
            "W_x",
            get_torch_module("Conv", dim=dim)(in_channels=f_l, out_channels=f_int, kernel_size=1, stride=2, padding=0),
            in_branch=[0],
            out_branch=[0],
        )
        self.add_module(
            "W_g",
            get_torch_module("Conv", dim=dim)(in_channels=f_g, out_channels=f_int, kernel_size=1, stride=1, padding=0),
            in_branch=[1],
            out_branch=[1],
        )
        self.add_module("Add", Add(), in_branch=[0, 1])
        self.add_module("ReLU", torch.nn.ReLU(inplace=True))
        self.add_module(
            "Conv",
            get_torch_module("Conv", dim=dim)(in_channels=f_int, out_channels=1, kernel_size=1, stride=1, padding=0),
        )
        self.add_module("Sigmoid", torch.nn.Sigmoid())
        self.add_module("Upsample", torch.nn.Upsample(scale_factor=2))
        self.add_module("Multiply", Multiply(), in_branch=[2, 0])
