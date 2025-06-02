from enum import Enum
import importlib
from typing import Callable
import torch
from scipy.interpolate import interp1d
import numpy as np
import ast
from typing import Union

from konfai.utils.config import config
from konfai.network import network

class NormMode(Enum):
    NONE = 0,
    BATCH = 1
    INSTANCE = 2
    GROUP = 3
    LAYER = 4
    SYNCBATCH = 5
    INSTANCE_AFFINE = 6

def getNorm(normMode: Enum, channels : int, dim: int) -> torch.nn.Module:
    if normMode == NormMode.BATCH:
        return getTorchModule("BatchNorm", dim = dim)(channels, affine=True, track_running_stats=True)
    if normMode == NormMode.INSTANCE:
        return getTorchModule("InstanceNorm", dim = dim)(channels, affine=False, track_running_stats=False)
    if normMode == NormMode.INSTANCE_AFFINE:
        return getTorchModule("InstanceNorm", dim = dim)(channels, affine=True, track_running_stats=False)
    if normMode == NormMode.SYNCBATCH:
        return torch.nn.SyncBatchNorm(channels, affine=True, track_running_stats=True)
    if normMode == NormMode.GROUP:
        return torch.nn.GroupNorm(num_groups=32, num_channels=channels)
    if normMode == NormMode.LAYER:
        return torch.nn.GroupNorm(num_groups=1, num_channels=channels)
    return torch.nn.Identity()

class UpSampleMode(Enum):
    CONV_TRANSPOSE = 0,
    UPSAMPLE_NEAREST = 1,
    UPSAMPLE_LINEAR = 2,
    UPSAMPLE_BILINEAR = 3,
    UPSAMPLE_BICUBIC = 4,
    UPSAMPLE_TRILINEAR = 5

class DownSampleMode(Enum):
    MAXPOOL = 0,
    AVGPOOL = 1,
    CONV_STRIDE = 2

def getTorchModule(name_fonction : str, dim : Union[int, None] = None) -> torch.nn.Module:
    return getattr(importlib.import_module("torch.nn"), "{}".format(name_fonction) + ("{}d".format(dim) if dim is not None else ""))

class BlockConfig():

    @config("BlockConfig")
    def __init__(self, kernel_size : int = 3, stride : int = 1, padding : int = 1, bias = True, activation : Union[str, Callable[[], torch.nn.Module]] = "ReLU", normMode : Union[str, NormMode, Callable[[int], torch.nn.Module]] = "NONE") -> None:
        self.kernel_size = kernel_size
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.normMode = normMode

        if isinstance(normMode, str):
            self.norm = NormMode._member_map_[normMode]
        elif isinstance(normMode, NormMode):
            self.norm = normMode

    def getConv(self, in_channels : int, out_channels : int, dim : int) -> torch.nn.Conv3d:
        return getTorchModule("Conv", dim = dim)(in_channels = in_channels, out_channels = out_channels, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, bias=self.bias)
    
    def getNorm(self, channels : int, dim: int) -> torch.nn.Module:
        return getNorm(self.norm, channels, dim) if isinstance(self.norm, NormMode) else self.norm(channels)

    def getActivation(self) -> torch.nn.Module:
        if isinstance(self.activation, str):
            return getTorchModule(self.activation.split(";")[0])(*[ast.literal_eval(value) for value in self.activation.split(";")[1:]]) if self.activation != "None" else torch.nn.Identity()
        return self.activation()
    
class ConvBlock(network.ModuleArgsDict):
    
    def __init__(self, in_channels : int, out_channels : int, blockConfigs : list[BlockConfig], dim : int, alias : list[list[str]]=[[], [], []]) -> None:
        super().__init__()
        for i, blockConfig in enumerate(blockConfigs):
            self.add_module("Conv_{}".format(i), blockConfig.getConv(in_channels, out_channels, dim), alias=alias[0])
            self.add_module("Norm_{}".format(i), blockConfig.getNorm(out_channels, dim), alias=alias[1])
            self.add_module("Activation_{}".format(i), blockConfig.getActivation(), alias=alias[2])
            in_channels = out_channels

class ResBlock(network.ModuleArgsDict):
    
    def __init__(self, in_channels : int, out_channels : int, nb_conv: int, blockConfig : BlockConfig, dim : int, alias : list[list[str]]=[[], [], [], []]) -> None:
        super().__init__()
        for i in range(nb_conv):
            self.add_module("ConvBlock_{}".format(i), ConvBlock(in_channels, out_channels, nb_conv=1, blockConfig=blockConfig, dim=dim, alias=alias[:3]))
            if in_channels != out_channels:
                self.add_module("Conv_skip_{}".format(i), blockConfig.getConv(in_channels, out_channels, dim), alias=alias[3], in_branch=[1], out_branch=[1])    
            in_channels = out_channels
            self.add_module("Add_{}".format(i), Add(), in_branch=[0,1], out_branch=[0,1])
    
def downSample(in_channels: int, out_channels: int, downSampleMode: DownSampleMode, dim: int) -> torch.nn.Module:
    if downSampleMode == DownSampleMode.MAXPOOL:
        return getTorchModule("MaxPool", dim = dim)(2)
    if downSampleMode == DownSampleMode.AVGPOOL:
        return getTorchModule("AvgPool", dim = dim)(2)
    if downSampleMode == DownSampleMode.CONV_STRIDE:
        return getTorchModule("Conv", dim)(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

def upSample(in_channels: int, out_channels: int, upSampleMode: UpSampleMode, dim: int, kernel_size: Union[int, list[int]] = 2, stride: Union[int, list[int]] = 2):
    if upSampleMode == UpSampleMode.CONV_TRANSPOSE:
        return getTorchModule("ConvTranspose", dim = dim)(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = 0)
    else:
        return torch.nn.Upsample(scale_factor=2, mode=upSampleMode.name.replace("UPSAMPLE_", "").lower(), align_corners=False)

class Unsqueeze(torch.nn.Module):

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim
    
    def forward(self, *input : torch.Tensor) -> torch.Tensor:
        return torch.unsqueeze(input, self.dim)
    
    def extra_repr(self):
        return "dim={}".format(self.dim)

class Permute(torch.nn.Module):

    def __init__(self, dims : list[int]):
        super().__init__()
        self.dims = dims

    def forward(self, input : torch.Tensor) -> torch.Tensor:
        return torch.permute(input, self.dims)
    
    def extra_repr(self):
        return "dims={}".format(self.dims)

class ToChannels(Permute):

    def __init__(self, dim):
        super().__init__([0, dim+1, *[i+1 for i in range(dim)]])
        
class ToFeatures(Permute):

    def __init__(self, dim):
        super().__init__([0, *[i+2 for i in range(dim)], 1])        

class Add(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, *input : torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.stack(input), dim=0)
    
class Multiply(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *input : torch.Tensor) -> torch.Tensor:
        return torch.mul(*input)

class Concat(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *input : torch.Tensor) -> torch.Tensor:
        return torch.cat(input, dim=1)

class Print(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        print(input.shape)
        return input

class Write(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        import SimpleITK as sitk
        sitk.WriteImage(sitk.GetImageFromArray(input.clone()[0][0].cpu().numpy()), "./Data.mha")
        return input
    
class Exit(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        exit(0)
    
class Detach(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.detach()
    
class Negative(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return -input

class GetShape(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.tensor(input.shape)

class ArgMax(torch.nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.argmax(input, dim=self.dim).unsqueeze(self.dim)
    
class Select(torch.nn.Module):

    def __init__(self, slices: list[slice]) -> None:
        super().__init__()
        self.slices = tuple(slices)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = input[self.slices]
        for i, s in enumerate(range(len(result.shape))):
            if s == 1:
              result = result.squeeze(dim=i)  
        return result

class NormalNoise(torch.nn.Module):

    def __init__(self, dim: Union[int, None] = None) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.dim is not None:
            return torch.randn(self.dim).to(input.device)
        else:
            return torch.randn_like(input).to(input.device)
    
class Const(torch.nn.Module):

    def __init__(self, shape: list[int], std: float) -> None:
        super().__init__()
        self.noise = torch.nn.parameter.Parameter(torch.randn(shape)*std)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.noise.to(input.device)

class HistogramNoise(torch.nn.Module):

    def __init__(self, n: int, sigma: float) -> None:
        super().__init__()
        self.x = np.linspace(0, 1, num=n, endpoint=True)
        self.sigma = sigma
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.function = interp1d(self.x, self.x+np.random.normal(0, self.sigma, self.x.shape[0]), kind='cubic')
        result = torch.empty_like(input)

        for value in torch.unique(input):
            x = self.function(value.cpu())
            result[torch.where(input == value)] = torch.tensor(x, device=input.device).float()
        return result

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

    class LatentDistribution_Linear(torch.nn.Module):

        def __init__(self, shape: list[int], latentDim: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(torch.prod(torch.tensor(shape)), latentDim)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.unsqueeze(self.linear(input), 1)
        
    class LatentDistribution_Decoder(torch.nn.Module):
        
        def __init__(self, shape: list[int], latentDim: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(latentDim, torch.prod(torch.tensor(shape)))
            self.shape = shape

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.linear(input).view(-1, *[int(i) for i in self.shape])
    
    class LatentDistribution_Z(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()

        def forward(self, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
            return torch.exp(log_std/2)*torch.rand_like(mu)+mu
    
    def __init__(self, shape: list[int], latentDim: int) -> None:
        super().__init__()        
        self.add_module("Flatten", torch.nn.Flatten(1))
        self.add_module("mu", LatentDistribution.LatentDistribution_Linear(shape, latentDim), out_branch = [1])
        self.add_module("log_std", LatentDistribution.LatentDistribution_Linear(shape, latentDim), out_branch = [2])

        self.add_module("z", LatentDistribution.LatentDistribution_Z(), in_branch=[1,2], out_branch=[3])
        self.add_module("Concat", Concat(), in_branch=[1,2,3])
        self.add_module("DecoderInput", LatentDistribution.LatentDistribution_Decoder(shape, latentDim), in_branch=[3])

class Attention(network.ModuleArgsDict):

    def __init__(self, F_g : int, F_l : int, F_int : int, dim : int):
        super().__init__()
        self.add_module("W_x", getTorchModule("Conv", dim = dim)(in_channels = F_l, out_channels = F_int, kernel_size=1, stride=2, padding=0), in_branch=[0], out_branch=[0])
        self.add_module("W_g", getTorchModule("Conv", dim = dim)(in_channels = F_g, out_channels = F_int, kernel_size=1, stride=1, padding=0), in_branch=[1], out_branch=[1])
        self.add_module("Add", Add(), in_branch=[0,1])
        self.add_module("ReLU", torch.nn.ReLU(inplace=True))
        self.add_module("Conv", getTorchModule("Conv", dim = dim)(in_channels = F_int, out_channels = 1, kernel_size=1,stride=1, padding=0))
        self.add_module("Sigmoid", torch.nn.Sigmoid())
        self.add_module("Upsample", torch.nn.Upsample(scale_factor=2))
        self.add_module("Multiply", Multiply(), in_branch=[2,0])