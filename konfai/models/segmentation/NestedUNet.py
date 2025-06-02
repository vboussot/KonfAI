from typing import Union
import torch

from konfai.network import network, blocks
from konfai.utils.config import config
from konfai.data.HDF5 import ModelPatch

     
class NestedUNetBlock(network.ModuleArgsDict):

    def __init__(self, channels: list[int], nb_conv_per_stage: int, blockConfig: blocks.BlockConfig, downSampleMode: blocks.DownSampleMode, upSampleMode: blocks.UpSampleMode, attention : bool, block: type, dim: int, i : int = 0) -> None:
        super().__init__()
        if i > 0:
            self.add_module(downSampleMode.name, blocks.downSample(in_channels=channels[0], out_channels=channels[1], downSampleMode=downSampleMode, dim=dim))
        
        self.add_module("X_{}_{}".format(i, 0), block(in_channels=channels[1 if downSampleMode == blocks.DownSampleMode.CONV_STRIDE and i > 0 else 0], out_channels=channels[1], blockConfigs=[blockConfig]*nb_conv_per_stage, dim=dim), out_branch=["X_{}_{}".format(i, 0)])
        if len(channels) > 2:
            self.add_module("UNetBlock_{}".format(i+1), NestedUNetBlock(channels[1:], nb_conv_per_stage, blockConfig, downSampleMode, upSampleMode, attention, block, dim, i+1), in_branch=["X_{}_{}".format(i, 0)], out_branch=["X_{}_{}".format(i+1, j) for j in range(len(channels)-2)])
            for j in range(len(channels)-2):
                self.add_module("X_{}_{}_{}".format(i, j+1, upSampleMode.name), blocks.upSample(in_channels=channels[2], out_channels=channels[1], upSampleMode=upSampleMode, dim=dim), in_branch=["X_{}_{}".format(i+1, j)], out_branch=["X_{}_{}".format(i+1, j)])
                self.add_module("SkipConnection_{}_{}".format(i, j+1), blocks.Concat(), in_branch=["X_{}_{}".format(i+1, j)]+["X_{}_{}".format(i, r) for r in range(j+1)], out_branch=["X_{}_{}".format(i, j+1)])
                self.add_module("X_{}_{}".format(i, j+1), block(in_channels=(channels[1]*(j+1)+channels[2]) if upSampleMode != blocks.UpSampleMode.CONV_TRANSPOSE else channels[1]*(j+2), out_channels=channels[1], blockConfigs=[blockConfig]*nb_conv_per_stage, dim=dim), in_branch=["X_{}_{}".format(i, j+1)], out_branch=["X_{}_{}".format(i, j+1)])

class NestedUNet(network.Network):

    class NestedUNetHead(network.ModuleArgsDict):

        def __init__(self, in_channels: int, nb_class: int, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = in_channels, out_channels = nb_class, kernel_size = 1, stride = 1, padding = 0))
            self.add_module("Softmax", torch.nn.Softmax(dim=1))
            self.add_module("Argmax", blocks.ArgMax(dim=1))

    @config("NestedUNet")
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
                    blockType: str = "Conv") -> None:
        super().__init__(in_channels = channels[0], optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim = dim)
                
        self.add_module("UNetBlock_0", NestedUNetBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], attention=attention, block = blocks.ConvBlock if blockType == "Conv" else blocks.ResBlock, dim=dim), out_branch=["X_0_{}".format(j+1) for j in range(len(channels)-2)])    
        for j in range(len(channels)-2):
            self.add_module("Head_{}".format(j), NestedUNet.NestedUNetHead(in_channels=channels[1], nb_class=nb_class, dim=dim), in_branch=["X_0_{}".format(j+1)], out_branch=[-1])