import torch
from typing import Union

from konfai.network import network, blocks
from konfai.utils.config import config
from konfai.data.HDF5 import ModelPatch

class UNetHead(network.ModuleArgsDict):

    def __init__(self, in_channels: int, nb_class: int, dim: int) -> None:
        super().__init__()
        self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = in_channels, out_channels = nb_class, kernel_size = 1, stride = 1, padding = 0))
        self.add_module("Softmax", torch.nn.Softmax(dim=1))
        self.add_module("Argmax", blocks.ArgMax(dim=1))

class UNetBlock(network.ModuleArgsDict):

    def __init__(self, channels: list[int], nb_conv_per_stage: int, blockConfig: blocks.BlockConfig, downSampleMode: blocks.DownSampleMode, upSampleMode: blocks.UpSampleMode, attention : bool, block: type, nb_class: int, dim: int, i : int = 0, mri: bool = False) -> None:
        super().__init__()
        blockConfig_stride = blockConfig
        if i > 0:
            if downSampleMode != blocks.DownSampleMode.CONV_STRIDE:
                self.add_module(downSampleMode.name, blocks.downSample(in_channels=channels[0], out_channels=channels[1], downSampleMode=downSampleMode, dim=dim))
            else:
                blockConfig_stride = blocks.BlockConfig(blockConfig.kernel_size, (1,2,2) if mri and i > 4 else 2, blockConfig.padding, blockConfig.bias, blockConfig.activation, blockConfig.normMode)
        self.add_module("DownConvBlock", block(in_channels=channels[0], out_channels=channels[1], blockConfigs=[blockConfig_stride]+[blockConfig]*(nb_conv_per_stage-1), dim=dim))
        if len(channels) > 2:
            self.add_module("UNetBlock_{}".format(i+1), UNetBlock(channels[1:], nb_conv_per_stage, blockConfig, downSampleMode, upSampleMode, attention, block, nb_class, dim, i+1, mri=mri))
            self.add_module("UpConvBlock", block(in_channels=(channels[1]+channels[2]) if upSampleMode != blocks.UpSampleMode.CONV_TRANSPOSE else channels[1]*2, out_channels=channels[1], blockConfigs=[blockConfig]*nb_conv_per_stage, dim=dim))
            if nb_class > 0:
                self.add_module("Head", UNetHead(channels[1], nb_class, dim), out_branch=[-1])
        if i > 0:
            if attention:
                self.add_module("Attention", blocks.Attention(F_g=channels[1], F_l=channels[0], F_int=channels[0], dim=dim), in_branch=[1, 0], out_branch=[1])
            self.add_module(upSampleMode.name, blocks.upSample(in_channels=channels[1], out_channels=channels[0], upSampleMode=upSampleMode, dim=dim, kernel_size=(1,2,2) if mri and i > 4 else 2, stride=(1,2,2) if mri and i > 4 else 2))
            self.add_module("SkipConnection", blocks.Concat(), in_branch=[0, 1])

class UNet(network.Network):

    @config("UNet")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    dim : int = 3,
                    channels: list[int]=[1, 64, 128, 256, 512, 1024],
                    nb_class: int = 2,
                    blockConfig: blocks.BlockConfig = blocks.BlockConfig(),
                    nb_conv_per_stage: int = 2,
                    downSampleMode: str = "MAXPOOL",
                    upSampleMode: str = "CONV_TRANSPOSE",
                    attention : bool = False,
                    blockType: str = "Conv",
                    mri: bool = False) -> None:
        super().__init__(in_channels = channels[0], optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim = dim)        
        self.add_module("UNetBlock_0", UNetBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], attention=attention, block = blocks.ConvBlock if blockType == "Conv" else blocks.ResBlock, nb_class=nb_class, dim=dim, mri = mri))    
        