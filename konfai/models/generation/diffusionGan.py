from functools import partial
from typing import Union
import torch
import numpy as np

from konfai.network import network, blocks
from konfai.utils.config import config
from konfai.data.HDF5 import ModelPatch, Attribute
from konfai.data import augmentation
from konfai.models.segmentation import UNet, NestedUNet
from konfai.models.generation.ddpm import DDPM

class Discriminator(network.Network):
        
    class DiscriminatorNLayers(network.ModuleArgsDict):

        def __init__(self, channels: list[int], strides: list[int], dim: int) -> None:
            super().__init__()
            blockConfig = partial(blocks.BlockConfig, kernel_size=4, padding=1, bias=False, activation=partial(torch.nn.LeakyReLU, negative_slope = 0.2, inplace=True), normMode=blocks.NormMode.SYNCBATCH)
            for i, (in_channels, out_channels, stride) in enumerate(zip(channels, channels[1:], strides)):
                self.add_module("Layer_{}".format(i), blocks.ConvBlock(in_channels, out_channels, [blockConfig(stride=stride)], dim))
    
    class DiscriminatorHead(network.ModuleArgsDict):

        def __init__(self, channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels=channels, out_channels=1, kernel_size=4, stride=1, padding=1))
            #self.add_module("AdaptiveAvgPool", blocks.getTorchModule("AdaptiveAvgPool", dim)(tuple([1]*dim)))
            #self.add_module("Flatten", torch.nn.Flatten(1))

    class DiscriminatorBlock(network.ModuleArgsDict):

        def __init__(self,  channels: list[int] = [1, 16, 32, 64, 64],
                            strides: list[int] = [2,2,2,1],
                            dim : int = 3) -> None:
            super().__init__()
            self.add_module("Layers", Discriminator.DiscriminatorNLayers(channels, strides, dim))
            self.add_module("Head", Discriminator.DiscriminatorHead(channels[-1], dim))

    @config("Discriminator")
    def __init__(self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    channels: list[int] = [1, 16, 32, 64, 64],
                    strides: list[int] = [2,2,2,1],
                    nb_batch_per_step: int = 1,
                    dim : int = 3) -> None:
        super().__init__(in_channels = 1, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, nb_batch_per_step=nb_batch_per_step, dim=dim, init_type="kaiming")
        self.add_module("DiscriminatorModel", Discriminator.DiscriminatorBlock(channels, strides, dim))


class Discriminator_ADA(network.Network):

    class DDPM_TE(torch.nn.Module):

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.linear_0 = torch.nn.Linear(in_channels, out_channels)
            self.siLU = torch.nn.SiLU()
            self.linear_1 = torch.nn.Linear(out_channels, out_channels)
        
        def forward(self, input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return input + self.linear_1(self.siLU(self.linear_0(t))).reshape(input.shape[0], -1, *[1 for _ in range(len(input.shape)-2)]) 
        
    class DiscriminatorNLayers(network.ModuleArgsDict):

        def __init__(self, channels: list[int], strides: list[int], time_embedding_dim: int, dim: int) -> None:
            super().__init__()
            blockConfig = partial(blocks.BlockConfig, kernel_size=4, padding=1, bias=False, activation=partial(torch.nn.LeakyReLU, negative_slope = 0.2, inplace=True), normMode=blocks.NormMode.SYNCBATCH)
            for i, (in_channels, out_channels, stride) in enumerate(zip(channels, channels[1:], strides)):
                self.add_module("Te_{}".format(i), Discriminator_ADA.DDPM_TE(time_embedding_dim, in_channels), in_branch=[0, 1])
                self.add_module("Layer_{}".format(i), blocks.ConvBlock(in_channels, out_channels, [blockConfig(stride=stride)], dim))
    
    class DiscriminatorHead(network.ModuleArgsDict):

        def __init__(self, channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels=channels, out_channels=1, kernel_size=4, stride=1, padding=1))
            #self.add_module("AdaptiveAvgPool", blocks.getTorchModule("AdaptiveAvgPool", dim)(tuple([1]*dim)))
            #self.add_module("Flatten", torch.nn.Flatten(1))

    class UpdateP(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self._it = 0
            self.n = 4
            self.ada_target = 0.25
            self.ada_interval = 0.001
            self.ada_kimg = 500

            self.measure = None
            self.names = None
            self.p = 0
        
        def setMeasure(self, measure: network.Measure, names: list[str]):
            self.measure = measure
            self.names = names
    
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if self.measure is not None and self._it % self.n == 0:
                value = sum([v for k, v in self.measure.getLastValues(self.n).items() if k in self.names])
                adjust = np.sign(self.ada_target-value) * (self.ada_interval)
                self.p += adjust
                self.p = np.clip(self.p, 0, 1)
            self._it += 1
            return torch.tensor(self.p).to(input.device)
        
    class DiscriminatorAugmentation(torch.nn.Module):

        def __init__(self, dim: int):
            super().__init__()

            self.dataAugmentations : dict[augmentation.DataAugmentation, float] = {}
            pixel_blitting = {
                augmentation.Flip([1/3]*3 if dim == 3 else [1/2]*2) : 0,
                augmentation.Rotate(a_min=0, a_max=360, is_quarter = True): 0,
                augmentation.Translate([(-0.5, 0.5)]* (3 if dim == 3 else 2), is_int=True) : 0
                }
            
            self.dataAugmentations.update(pixel_blitting)
            geometric = {
                augmentation.Scale([0.2]) : 0,
                augmentation.Rotate(a_min=0, a_max=360): 0,
                augmentation.Scale([0.2]*3 if dim == 3 else [0.2]*2) : 0,
                augmentation.Rotate(a_min=0, a_max=360): 0,
                augmentation.Translate([(-0.5, 0.5)]* (3 if dim == 3 else 2)) : 0,
                augmentation.Elastix(16, 16) : 0.5
                }
            self.dataAugmentations.update(geometric)
            color = {
                augmentation.Brightness(0.2) : 0,
                augmentation.Contrast(0.5) : 0,
                augmentation.Saturation(1): 0,
                augmentation.HUE(1) : 0,
                augmentation.LumaFlip(): 0
            }
            self.dataAugmentations.update(color)
            
            corruptions =  {
                augmentation.Noise(1) : 1,
                augmentation.CutOUT(0.5, 0.5, -1) : 0.3
            }
            self.dataAugmentations.update(corruptions)
                
        def _setP(self, prob: float):
            for augmentation, p  in self.dataAugmentations.items():
                augmentation.load(prob*p)

        def forward(self, input: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
            self._setP(prob.item())
            out = input
            for augmentation in self.dataAugmentations.keys():
                augmentation.state_init(None, [input.shape[2:]]*input.shape[0], [Attribute()]*input.shape[0])
                out = augmentation(0, [data for data in out], None)
            return torch.cat([data.unsqueeze(0) for data in out], 0)


    class DiscriminatorBlock(network.ModuleArgsDict):

        def __init__(self,  channels: list[int] = [1, 16, 32, 64, 64],
                            strides: list[int] = [2,2,2,1],
                            dim : int = 3) -> None:
            super().__init__()
            self.add_module("Prob", Discriminator_ADA.UpdateP(), out_branch=["p"])
            self.add_module("Sample", Discriminator_ADA.DiscriminatorAugmentation(dim), in_branch=[0, "p"])
            self.add_module("t", DDPM.DDPM_TimeEmbedding(1000, 100), in_branch=[0, "p"], out_branch=["te"])
            self.add_module("Layers", Discriminator_ADA.DiscriminatorNLayers(channels, strides, 100, dim), in_branch=[0, "te"])
            self.add_module("Head", Discriminator_ADA.DiscriminatorHead(channels[-1], dim))

    @config("Discriminator_ADA")
    def __init__(self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    channels: list[int] = [1, 16, 32, 64, 64],
                    strides: list[int] = [2,2,2,1],
                    nb_batch_per_step: int = 1,
                    dim : int = 3) -> None:
        super().__init__(in_channels = 1, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, nb_batch_per_step=nb_batch_per_step, dim=dim, init_type="kaiming")
        self.add_module("DiscriminatorModel", Discriminator_ADA.DiscriminatorBlock(channels, strides, dim))

    def initialized(self):
        self["DiscriminatorModel"]["Prob"].setMeasure(self.measure, ["Discriminator_B.DiscriminatorModel.Head.Conv:None:PatchGanLoss"])

"""class GeneratorV1(network.Network):

    class GeneratorStem(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("ReflectionPad2d", torch.nn.ReflectionPad2d(3))
            self.add_module("ConvBlock", blocks.ConvBlock(in_channels, out_channels, blockConfigs=[blocks.BlockConfig(kernel_size=7, padding=0, bias=False, activation="ReLU", normMode="SYNCBATCH")], dim=dim))

    class GeneratorHead(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("ConvBlock", blocks.ConvBlock(in_channels, in_channels, blockConfigs=[blocks.BlockConfig(bias=False, activation="ReLU", normMode="SYNCBATCH")], dim=dim))
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels, out_channels, kernel_size=1, bias=False))
            self.add_module("Tanh", torch.nn.Tanh())

    class GeneratorDownSample(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("ConvBlock", blocks.ConvBlock(in_channels, out_channels, blockConfigs=[blocks.BlockConfig(stride=2, bias=False, activation="ReLU", normMode="SYNCBATCH")], dim=dim))
    
    class GeneratorUpSample(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("ConvBlock", blocks.ConvBlock(in_channels, out_channels, blockConfigs=[blocks.BlockConfig(bias=False, activation="ReLU", normMode="SYNCBATCH")], dim=dim))
            self.add_module("Upsample", torch.nn.Upsample(scale_factor=2, mode="bilinear" if dim < 3 else "trilinear"))
    
    class GeneratorEncoder(network.ModuleArgsDict):
        def __init__(self, channels: list[int], dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
                self.add_module("DownSample_{}".format(i), GeneratorV1.GeneratorDownSample(in_channels=in_channels, out_channels=out_channels, dim=dim))
    
    class GeneratorResnetBlock(network.ModuleArgsDict):

        def __init__(self, channels : int, dim : int):
            super().__init__()
            self.add_module("Conv_0", blocks.getTorchModule("Conv", dim)(channels, channels, kernel_size=3, padding=1, bias=False))
            self.add_module("Norm_0", torch.nn.SyncBatchNorm(channels))
            self.add_module("Activation_0", torch.nn.LeakyReLU(0.2, inplace=True))
            self.add_module("Conv_1", blocks.getTorchModule("Conv", dim)(channels, channels, kernel_size=3, padding=1, bias=False))
            self.add_module("Norm_1", torch.nn.SyncBatchNorm(channels))
            self.add_module("Residual", blocks.Add(), in_branch=[0,1])

    class GeneratorNResnetBlock(network.ModuleArgsDict):

        def __init__(self, channels: int, nb_conv: int, dim: int) -> None:
            super().__init__()
            for i in range(nb_conv):
                self.add_module("ResnetBlock_{}".format(i), GeneratorV1.GeneratorResnetBlock(channels=channels, dim=dim))

    class GeneratorDecoder(network.ModuleArgsDict):
        def __init__(self, channels: list[int], dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels) in enumerate(zip(reversed(channels), reversed(channels[:-1]))):
                self.add_module("UpSample_{}".format(i), GeneratorV1.GeneratorUpSample(in_channels=in_channels, out_channels=out_channels, dim=dim))
    
    class GeneratorAutoEncoder(network.ModuleArgsDict):

        def __init__(self, ngf: int, dim: int) -> None:
            super().__init__()
            channels = [ngf, ngf*2]
            self.add_module("Encoder", GeneratorV1.GeneratorEncoder(channels, dim))
            self.add_module("NResBlock", GeneratorV1.GeneratorNResnetBlock(channels=channels[-1], nb_conv=6, dim=dim))
            self.add_module("Decoder", GeneratorV1.GeneratorDecoder(channels, dim))

    class GeneratorBlock(network.ModuleArgsDict):

        def __init__(self, ngf: int, dim: int) -> None:
            super().__init__()
            self.add_module("Stem", GeneratorV1.GeneratorStem(3, ngf, dim))
            self.add_module("AutoEncoder", GeneratorV1.GeneratorAutoEncoder(ngf, dim))
            self.add_module("Head", GeneratorV1.GeneratorHead(in_channels=ngf, out_channels=1, dim=dim))

    @config("GeneratorV1")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    patch : ModelPatch = ModelPatch(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    dim : int = 3) -> None:
        super().__init__(optimizer=optimizer, in_channels=3, schedulers=schedulers, patch=patch, outputsCriterions=outputsCriterions, dim=dim)
        self.add_module("GeneratorModel", GeneratorV1.GeneratorBlock(32, dim))"""

class GeneratorV1(network.Network):

    class GeneratorStem(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("ConvBlock", blocks.ConvBlock(in_channels, out_channels, blockConfigs=[blocks.BlockConfig(bias=False, activation="ReLU", normMode="SYNCBATCH")], dim=dim))

    class GeneratorHead(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("ConvBlock", blocks.ConvBlock(in_channels, in_channels, blockConfigs=[blocks.BlockConfig(bias=False, activation="ReLU", normMode="SYNCBATCH")], dim=dim))
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels, out_channels, kernel_size=1, bias=False))
            self.add_module("Tanh", torch.nn.Tanh())

    class GeneratorDownSample(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("ConvBlock", blocks.ConvBlock(in_channels, out_channels, blockConfigs=[blocks.BlockConfig(stride=2, bias=False, activation="ReLU", normMode="SYNCBATCH")], dim=dim))
    
    class GeneratorUpSample(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module("ConvBlock", blocks.ConvBlock(in_channels, out_channels, blockConfigs=[blocks.BlockConfig(bias=False, activation="ReLU", normMode="SYNCBATCH")], dim=dim))
            self.add_module("Upsample", torch.nn.Upsample(scale_factor=2, mode="bilinear" if dim < 3 else "trilinear"))
    
    class GeneratorEncoder(network.ModuleArgsDict):
        def __init__(self, channels: list[int], dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
                self.add_module("DownSample_{}".format(i), GeneratorV1.GeneratorDownSample(in_channels=in_channels, out_channels=out_channels, dim=dim))
    
    class GeneratorResnetBlock(network.ModuleArgsDict):

        def __init__(self, channels : int, dim : int):
            super().__init__()
            self.add_module("Conv_0", blocks.getTorchModule("Conv", dim)(channels, channels, kernel_size=3, padding=1, bias=False))
            self.add_module("Norm_0", torch.nn.SyncBatchNorm(channels))
            self.add_module("Activation_0", torch.nn.LeakyReLU(0.2, inplace=True))
            #self.add_module("Norm", torch.nn.LeakyReLU(0.2, inplace=True))
            
            self.add_module("Conv_1", blocks.getTorchModule("Conv", dim)(channels, channels, kernel_size=3, padding=1, bias=False))
            self.add_module("Norm_1", torch.nn.SyncBatchNorm(channels))
            self.add_module("Residual", blocks.Add(), in_branch=[0,1])

    class GeneratorNResnetBlock(network.ModuleArgsDict):

        def __init__(self, channels: int, nb_conv: int, dim: int) -> None:
            super().__init__()
            for i in range(nb_conv):
                self.add_module("ResnetBlock_{}".format(i), GeneratorV1.GeneratorResnetBlock(channels=channels, dim=dim))

    class GeneratorDecoder(network.ModuleArgsDict):
        def __init__(self, channels: list[int], dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels) in enumerate(zip(reversed(channels), reversed(channels[:-1]))):
                self.add_module("UpSample_{}".format(i), GeneratorV1.GeneratorUpSample(in_channels=in_channels, out_channels=out_channels, dim=dim))
    
    class GeneratorAutoEncoder(network.ModuleArgsDict):

        def __init__(self, ngf: int, dim: int) -> None:
            super().__init__()
            channels = [ngf, ngf*2]
            self.add_module("Encoder", GeneratorV1.GeneratorEncoder(channels, dim))
            self.add_module("NResBlock", GeneratorV1.GeneratorNResnetBlock(channels=channels[-1], nb_conv=6, dim=dim))
            self.add_module("Decoder", GeneratorV1.GeneratorDecoder(channels, dim))

    class GeneratorBlock(network.ModuleArgsDict):

        def __init__(self, ngf: int, dim: int) -> None:
            super().__init__()
            self.add_module("Stem", GeneratorV1.GeneratorStem(3, ngf, dim))
            self.add_module("AutoEncoder", GeneratorV1.GeneratorAutoEncoder(ngf, dim))
            self.add_module("Head", GeneratorV1.GeneratorHead(in_channels=ngf, out_channels=1, dim=dim))

    @config("GeneratorV1")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    patch : ModelPatch = ModelPatch(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    dim : int = 3) -> None:
        super().__init__(optimizer=optimizer, in_channels=3, schedulers=schedulers, patch=patch, outputsCriterions=outputsCriterions, dim=dim)
        self.add_module("GeneratorModel", GeneratorV1.GeneratorBlock(32, dim))
        
class GeneratorV2(network.Network):

    class NestedUNetHead(network.ModuleArgsDict):

        def __init__(self, in_channels: list[int], dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = in_channels[1], out_channels = 1, kernel_size = 1, stride = 1, padding = 0))
            self.add_module("Tanh", torch.nn.Tanh())
    
    class GeneratorBlock(network.ModuleArgsDict):

        def __init__(self, 
                    channels: list[int],
                    blockConfig: blocks.BlockConfig,
                    nb_conv_per_stage: int,
                    downSampleMode: str,
                    upSampleMode: str,
                    attention : bool,
                    blockType: str,
                    dim : int,) -> None:
            super().__init__()
            self.add_module("UNetBlock_0", NestedUNet.NestedUNetBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], attention=attention, block = blocks.ConvBlock if blockType == "Conv" else blocks.ResBlock, dim=dim), out_branch=["X_0_{}".format(j+1) for j in range(len(channels)-2)])    
            self.add_module("Head", GeneratorV2.NestedUNetHead(channels[:2], dim=dim), in_branch=["X_0_{}".format(len(channels)-2)])

    @config("GeneratorV2")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    blockType: str = "Conv",
                    dim : int = 3) -> None:
        super().__init__(in_channels = channels[0], optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim = dim)
        self.add_module("GeneratorModel", GeneratorV2.GeneratorBlock(channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, blockType, dim))

class GeneratorV3(network.Network):

    class NestedUNetHead(network.ModuleArgsDict):

        def __init__(self, in_channels: list[int], dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = in_channels[1], out_channels = 1, kernel_size = 1, stride = 1, padding = 0))
            self.add_module("Tanh", torch.nn.Tanh())
    
    class GeneratorBlock(network.ModuleArgsDict):

        def __init__(self, 
                    channels: list[int],
                    blockConfig: blocks.BlockConfig,
                    nb_conv_per_stage: int,
                    downSampleMode: str,
                    upSampleMode: str,
                    attention : bool,
                    blockType: str,
                    dim : int,) -> None:
            super().__init__()
            self.add_module("UNetBlock_0", UNet.UNetBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], attention=attention, block = blocks.ConvBlock if blockType == "Conv" else blocks.ResBlock, nb_class=1, dim=dim), out_branch=["X_0_{}".format(j+1) for j in range(len(channels)-2)])    
            self.add_module("Head", GeneratorV3.NestedUNetHead(channels[:2], dim=dim), in_branch=["X_0_{}".format(len(channels)-2)])

    @config("GeneratorV3")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    blockType: str = "Conv",
                    dim : int = 3) -> None:
        super().__init__(in_channels = channels[0], optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim = dim)
        self.add_module("GeneratorModel", GeneratorV3.GeneratorBlock(channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, blockType, dim), out_branch=["pB"])

class DiffusionGan(network.Network):

    @config("DiffusionGan")
    def __init__(self, generator : GeneratorV1 = GeneratorV1(), discriminator : Discriminator_ADA = Discriminator_ADA()) -> None:
        super().__init__()
        self.add_module("Generator_A_to_B", generator, in_branch=[0], out_branch=["pB"])
        self.add_module("Discriminator_B", discriminator, in_branch=[1], out_branch=[-1], requires_grad=True)
        self.add_module("detach", blocks.Detach(), in_branch=["pB"], out_branch=["pB_detach"])
        self.add_module("Discriminator_pB_detach", discriminator, in_branch=["pB_detach"], out_branch=[-1])
        self.add_module("Discriminator_pB", discriminator, in_branch=["pB"], out_branch=[-1], requires_grad=False)

class DiffusionGanV2(network.Network):

    @config("DiffusionGan")
    def __init__(self, generator : GeneratorV2 = GeneratorV2(), discriminator : Discriminator = Discriminator()) -> None:
        super().__init__()
        self.add_module("Generator_A_to_B", generator, in_branch=[0], out_branch=["pB"])
        self.add_module("Discriminator_B", discriminator, in_branch=[1], out_branch=[-1], requires_grad=True)
        self.add_module("detach", blocks.Detach(), in_branch=["pB"], out_branch=["pB_detach"])
        self.add_module("Discriminator_pB_detach", discriminator, in_branch=["pB_detach"], out_branch=[-1])
        self.add_module("Discriminator_pB", discriminator, in_branch=["pB"], out_branch=[-1], requires_grad=False)


class CycleGanDiscriminator(network.Network):

    @config("CycleGanDiscriminator")
    def __init__(self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    channels: list[int] = [1, 16, 32, 64, 64],
                    strides: list[int] = [2,2,2,1],
                    dim : int = 3) -> None:
        super().__init__(in_channels = 1, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim=dim)
        self.add_module("Discriminator_A", Discriminator.DiscriminatorBlock(channels, strides, dim), in_branch=[0], out_branch=[0])
        self.add_module("Discriminator_B", Discriminator.DiscriminatorBlock(channels, strides, dim), in_branch=[1], out_branch=[1])
        
    def initialized(self):
        self["Discriminator_A"]["Sample"].setMeasure(self.measure, ["Discriminator.Discriminator_A.Head.Flatten:None:PatchGanLoss"])
        self["Discriminator_B"]["Sample"].setMeasure(self.measure, ["Discriminator.Discriminator_B.Head.Flatten:None:PatchGanLoss"])

class CycleGanGeneratorV1(network.Network):

    @config("CycleGanGeneratorV1")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    dim : int = 3) -> None:
        super().__init__(in_channels = 1, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions,  patch=patch, dim=dim)
        self.add_module("Generator_A_to_B", GeneratorV1.GeneratorBlock(32, dim), in_branch=[0], out_branch=["pB"])
        self.add_module("Generator_B_to_A", GeneratorV1.GeneratorBlock(32, dim), in_branch=[1], out_branch=["pA"])

class CycleGanGeneratorV2(network.Network):

    @config("CycleGanGeneratorV2")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    blockType: str = "Conv",
                    dim : int = 3) -> None:
        super().__init__(in_channels = 1, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim=dim)
        self.add_module("Generator_A_to_B", GeneratorV2.GeneratorBlock(channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, blockType, dim), in_branch=[0], out_branch=["pB"])
        self.add_module("Generator_B_to_A", GeneratorV2.GeneratorBlock(channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, blockType, dim), in_branch=[1], out_branch=["pA"])

class CycleGanGeneratorV3(network.Network):

    @config("CycleGanGeneratorV3")
    def __init__(self, 
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    blockType: str = "Conv",
                    dim : int = 3) -> None:
        super().__init__(in_channels = 1, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim=dim)
        self.add_module("Generator_A_to_B", GeneratorV3.GeneratorBlock(channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, blockType, dim), in_branch=[0], out_branch=["pB"])
        self.add_module("Generator_B_to_A", GeneratorV3.GeneratorBlock(channels, blockConfig, nb_conv_per_stage, downSampleMode, upSampleMode, attention, blockType, dim), in_branch=[1], out_branch=["pA"])

class DiffusionCycleGan(network.Network):

    @config("DiffusionCycleGan")
    def __init__(self, generators : CycleGanGeneratorV3 = CycleGanGeneratorV3(), discriminators : CycleGanDiscriminator = CycleGanDiscriminator()) -> None:
        super().__init__()
        self.add_module("Generator", generators, in_branch=[0, 1], out_branch=["pB", "pA"])
        self.add_module("Discriminator", discriminators, in_branch=[0, 1], out_branch=[-1], requires_grad=True)
        
        self.add_module("Generator_identity", generators, in_branch=[1, 0], out_branch=[-1])
        
        self.add_module("Generator_p", generators, in_branch=["pA", "pB"], out_branch=[-1])
    
        self.add_module("detach_pA", blocks.Detach(), in_branch=["pA"], out_branch=["pA_detach"])
        self.add_module("detach_pB", blocks.Detach(), in_branch=["pB"], out_branch=["pB_detach"])

        self.add_module("Discriminator_p_detach", discriminators, in_branch=["pA_detach", "pB_detach"], out_branch=[-1])
        self.add_module("Discriminator_p", discriminators, in_branch=["pA", "pB"], out_branch=[-1], requires_grad=False)
        
        