import importlib
import torch

from konfai.network import network, blocks
from konfai.utils.config import config
from konfai.data.HDF5 import ModelPatch    

class MappingNetwork(network.ModuleArgsDict):
    def __init__(self, z_dim: int, c_dim: int, w_dim: int, num_layers: int, embed_features: int, layer_features: int):
        super().__init__()
        
        self.add_module("Concat_1", blocks.Concat(), in_branch=[0,1])
        
        features = [z_dim + embed_features if c_dim > 0 else 0] + [layer_features] * (num_layers - 1) + [w_dim]    
        if c_dim > 0:
            self.add_module("Linear", torch.nn.Linear(c_dim, embed_features), out_branch=["Embed"])

        self.add_module("Noise", blocks.NormalNoise(z_dim), in_branch=["Embed"])
        if c_dim > 0:
            self.add_module("Concat", blocks.Concat(), in_branch=[0,"Embed"])
        
        for i, (in_features, out_features) in enumerate(zip(features, features[1:])):
            self.add_module("Linear_{}".format(i), torch.nn.Linear(in_features, out_features))
    
class ModulatedConv(torch.nn.Module):

    class _ModulatedConv(torch.nn.Module):

        def __init__(self, w_dim: int, conv: torch.nn.modules.conv._ConvNd, dim: int) -> None:
            super().__init__()
            self.affine = torch.nn.Linear(w_dim, conv.in_channels)
            self.isConv = True
            self.in_channels = conv.in_channels
            self.out_channels = conv.out_channels
            self.padding = conv.padding
            self.stride = conv.stride
            if isinstance(conv, torch.nn.modules.conv._ConvTransposeNd):
                self.weight = torch.nn.parameter.Parameter(torch.randn((conv.in_channels, conv.out_channels, *conv.kernel_size)))
                self.isConv = False
            else:
                self.weight = torch.nn.parameter.Parameter(torch.randn((conv.out_channels, conv.in_channels, *conv.kernel_size)))
            conv.forward = self.forward
            self.styles = None
            self.dim = dim

        def setStyle(self, styles: torch.Tensor) -> None:
            self.styles = styles

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            b = input.shape[0]
            self.affine.to(input.device)
            styles = self.affine(self.styles)
            w1 = styles.reshape(b, -1, 1, *[1 for _ in range(self.dim)]) if not self.isConv else styles.reshape(b, 1, -1, *[1 for _ in range(self.dim)])
            w2 = self.weight.unsqueeze(0).to(input.device)
            weights = w2 * (w1 + 1)

            d = torch.rsqrt((weights ** 2).sum(dim=tuple([i+2 for i in range(len(weights.shape)-2)]), keepdim=True) + 1e-8)
            weights = weights * d
            
            input = input.reshape(1, -1, *input.shape[2:])

            _, _, *ws = weights.shape
            if not self.isConv:
                out = getattr(importlib.import_module("torch.nn.functional"), "conv_transpose{}d".format(self.dim))(input, weights.reshape(b * self.in_channels, *ws), stride=self.stride, padding=self.padding, groups=b)
            else:
                out = getattr(importlib.import_module("torch.nn.functional"), "conv{}d".format(self.dim))(input, weights.reshape(b * self.out_channels, *ws), padding=self.padding, groups=b, stride=self.stride)
            
            out = out.reshape(-1, self.out_channels, *out.shape[2:])
            return out
    
    def __init__(self, w_dim: int, module: torch.nn.Module) -> None:
        super().__init__()
        self.w_dim = w_dim
        self.module = module
        self.convs = torch.nn.ModuleList() 
        self.module.apply(self.apply)
        
    def forward(self, input: torch.Tensor, styles: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            conv.setStyle(styles.clone())
        return self.module(input)

    def apply(self, module: torch.nn.Module):
        if isinstance(module, torch.nn.modules.conv._ConvNd):
            delattr(module, "weight")
            module.bias = None
            
            str_dim = module.__class__.__name__[-2:]
            dim = 1
            if str_dim == "2d":
                dim = 2
            elif str_dim == "3d":
                dim = 3
            self.convs.append(ModulatedConv._ModulatedConv(self.w_dim, module, dim=dim))
        
class UNetBlock(network.ModuleArgsDict):

    def __init__(self, w_dim: int, channels: list[int], nb_conv_per_stage: int, blockConfig: blocks.BlockConfig, downSampleMode: blocks.DownSampleMode, upSampleMode: blocks.UpSampleMode, attention : bool, dim: int, i : int = 0) -> None:
        super().__init__()
        if i > 0:
            self.add_module(downSampleMode.name, blocks.downSample(in_channels=channels[0], out_channels=channels[1], downSampleMode=downSampleMode, dim=dim))
        self.add_module("DownConvBlock", blocks.ConvBlock(in_channels=channels[1 if downSampleMode == blocks.DownSampleMode.CONV_STRIDE and i > 0 else 0], out_channels=channels[1], nb_conv=nb_conv_per_stage, blockConfig=blockConfig, dim=dim))
        if len(channels) > 2:
            self.add_module("UNetBlock_{}".format(i+1), UNetBlock(w_dim, channels[1:], nb_conv_per_stage, blockConfig, downSampleMode, upSampleMode, attention, dim, i+1), in_branch=[0,1])
            self.add_module("UpConvBlock", ModulatedConv(w_dim, blocks.ConvBlock((channels[1]+channels[2]) if upSampleMode != blocks.UpSampleMode.CONV_TRANSPOSE else channels[1]*2, out_channels=channels[1], nb_conv=nb_conv_per_stage, blockConfig=blockConfig, dim=dim)), in_branch=[0,1])
        if i > 0:
            if attention:
                self.add_module("Attention", blocks.Attention(F_g=channels[1], F_l=channels[0], F_int=channels[0], dim=dim), in_branch=["Skip", 0], out_branch=["Skip"])
            self.add_module(upSampleMode.name, ModulatedConv(w_dim, blocks.upSample(in_channels=channels[1], out_channels=channels[0], upSampleMode=upSampleMode, dim=dim)), in_branch=[0, 1])
            self.add_module("SkipConnection", blocks.Concat(), in_branch=[0, "Skip"])

class Generator(network.Network):

    class GeneratorHead(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1))
            self.add_module("Tanh", torch.nn.Tanh())

    @config("Generator")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    patch : ModelPatch = ModelPatch(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    nb_batch_per_step: int = 64,
                    z_dim: int = 512,
                    c_dim: int = 1,
                    w_dim: int = 512,
                    dim : int = 3) -> None:
        super().__init__(optimizer=optimizer, in_channels=channels[0], schedulers=schedulers, patch=patch, outputsCriterions=outputsCriterions, dim=dim, nb_batch_per_step=nb_batch_per_step)

        self.add_module("MappingNetwork", MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_layers=8, embed_features=w_dim, layer_features=w_dim), in_branch=[1,2], out_branch=["Style"])
        nb_conv_per_stage = 2
        blockConfig = blocks.BlockConfig(kernel_size=3, stride=1, padding=1, bias=True, activation="ReLU", normMode="INSTANCE") 
        self.add_module("UNetBlock_0", UNetBlock(w_dim, channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode.MAXPOOL, upSampleMode=blocks.UpSampleMode.CONV_TRANSPOSE, attention=False, dim=dim), in_branch=[0, "Style"])
        self.add_module("Head", Generator.GeneratorHead(in_channels=channels[1], out_channels=1, dim=dim))