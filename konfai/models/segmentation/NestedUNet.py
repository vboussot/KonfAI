from konfai.network import network, blocks
from typing import Union
import torch
from konfai.data.patching import ModelPatch

class NestedUNet(network.Network):

    class NestedUNetBlock(network.ModuleArgsDict):

        def __init__(self, channels: list[int], nb_conv_per_stage: int, blockConfig: blocks.BlockConfig, downSampleMode: blocks.DownSampleMode, upSampleMode: blocks.UpSampleMode, attention : bool, block: type, dim: int, i : int = 0) -> None:
            super().__init__()
            if i > 0:
                self.add_module(downSampleMode.name, blocks.downSample(in_channels=channels[0], out_channels=channels[1], downSampleMode=downSampleMode, dim=dim))
            
            self.add_module("X_{}_{}".format(i, 0), block(in_channels=channels[1 if downSampleMode == blocks.DownSampleMode.CONV_STRIDE and i > 0 else 0], out_channels=channels[1], blockConfigs=[blockConfig]*nb_conv_per_stage, dim=dim), out_branch=["X_{}_{}".format(i, 0)])
            if len(channels) > 2:
                self.add_module("UNetBlock_{}".format(i+1), NestedUNet.NestedUNetBlock(channels[1:], nb_conv_per_stage, blockConfig, downSampleMode, upSampleMode, attention, block, dim, i+1), in_branch=["X_{}_{}".format(i, 0)], out_branch=["X_{}_{}".format(i+1, j) for j in range(len(channels)-2)])
                for j in range(len(channels)-2):
                    self.add_module("X_{}_{}_{}".format(i, j+1, upSampleMode.name), blocks.upSample(in_channels=channels[2], out_channels=channels[1], upSampleMode=upSampleMode, dim=dim), in_branch=["X_{}_{}".format(i+1, j)], out_branch=["X_{}_{}".format(i+1, j)])
                    self.add_module("SkipConnection_{}_{}".format(i, j+1), blocks.Concat(), in_branch=["X_{}_{}".format(i+1, j)]+["X_{}_{}".format(i, r) for r in range(j+1)], out_branch=["X_{}_{}".format(i, j+1)])
                    self.add_module("X_{}_{}".format(i, j+1), block(in_channels=(channels[1]*(j+1)+channels[2]) if upSampleMode != blocks.UpSampleMode.CONV_TRANSPOSE else channels[1]*(j+2), out_channels=channels[1], blockConfigs=[blockConfig]*nb_conv_per_stage, dim=dim), in_branch=["X_{}_{}".format(i, j+1)], out_branch=["X_{}_{}".format(i, j+1)])


    class NestedUNetHead(network.ModuleArgsDict):

        def __init__(self, in_channels: int, nb_class: int, activation: str, dim: int) -> None:
            super().__init__()
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = in_channels, out_channels = nb_class, kernel_size = 1, stride = 1, padding = 0))
            if activation == "Softmax":
                self.add_module("Softmax", torch.nn.Softmax(dim=1))
                self.add_module("Argmax", blocks.ArgMax(dim=1))
            elif activation == "Tanh":
                self.add_module("Tanh", torch.nn.Tanh())
            
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
                    activation: str = "Softmax") -> None:
        super().__init__(in_channels = channels[0], optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim = dim)
                
        self.add_module("UNetBlock_0", NestedUNet.NestedUNetBlock(channels, nb_conv_per_stage, blockConfig, downSampleMode=blocks.DownSampleMode._member_map_[downSampleMode], upSampleMode=blocks.UpSampleMode._member_map_[upSampleMode], attention=attention, block = blocks.ConvBlock if blockType == "Conv" else blocks.ResBlock, dim=dim), out_branch=["X_0_{}".format(j+1) for j in range(len(channels)-2)])    
        for j in range(len(channels)-2):
            self.add_module("Head_{}".format(j), NestedUNet.NestedUNetHead(in_channels=channels[1], nb_class=nb_class, activation=activation, dim=dim), in_branch=["X_0_{}".format(j+1)], out_branch=[-1])


class UNetpp(network.Network):

    class ResNetEncoderLayer(network.ModuleArgsDict):

        def __init__(self, in_channel: int, out_channel: int, nb_block: int, dim: int, downSampleMode : blocks.DownSampleMode):
            super().__init__()
            for i in range(nb_block):
                if downSampleMode == blocks.DownSampleMode.MAXPOOL and i == 0:
                    self.add_module("DownSample", blocks.getTorchModule("MaxPool", dim)(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))
                self.add_module("ResBlock_{}".format(i), blocks.ResBlock(in_channel, out_channel, [blocks.BlockConfig(3, 2 if downSampleMode == blocks.DownSampleMode.CONV_STRIDE and i == 0 else 1, 1, False, "ReLU;True", blocks.NormMode.BATCH), blocks.BlockConfig(3, 1, 1, False, None, blocks.NormMode.BATCH)], dim=dim))
                in_channel = out_channel
            
    def resNetEncoder(channels: list[int], layers: list[int], dim: int) -> list[torch.nn.Module]:
        modules = []
        modules.append(blocks.ConvBlock(channels[0], channels[1], [blocks.BlockConfig(7, 2, 3, False, "ReLU", blocks.NormMode.BATCH)], dim=dim))
        for i, (in_channel, out_channel, layer) in enumerate(zip(channels[1:], channels[2:], layers)):
            modules.append(UNetpp.ResNetEncoderLayer(in_channel, out_channel, layer, dim, blocks.DownSampleMode.MAXPOOL if i==0 else blocks.DownSampleMode.CONV_STRIDE))
        return modules

    class UNetPPBlock(network.ModuleArgsDict):

        def __init__(self, encoder_channels: list[int], decoder_channels: list[int], encoders: list[torch.nn.Module], upSampleMode: blocks.UpSampleMode, dim: int, i : int = 0) -> None:
            super().__init__()
            self.add_module("X_{}_{}".format(i, 0), encoders[0], out_branch=["X_{}_{}".format(i, 0)])
            if len(encoder_channels) > 2:
                self.add_module("UNetBlock_{}".format(i+1), UNetpp.UNetPPBlock(encoder_channels[1:], decoder_channels[1:], encoders[1:], upSampleMode, dim, i+1), in_branch=["X_{}_{}".format(i, 0)], out_branch=["X_{}_{}".format(i+1, j) for j in range(len(encoder_channels)-2)])
                for j in range(len(encoder_channels)-2):
                    self.add_module("X_{}_{}_{}".format(i, j+1, upSampleMode.name), blocks.upSample(in_channels=encoder_channels[2], out_channels=decoder_channels[2] if j == len(encoder_channels)-3 else encoder_channels[1], upSampleMode=upSampleMode, dim=dim), in_branch=["X_{}_{}".format(i+1, j)], out_branch=["X_{}_{}".format(i+1, j)])
                    self.add_module("SkipConnection_{}_{}".format(i, j+1), blocks.Concat(), in_branch=["X_{}_{}".format(i+1, j)]+["X_{}_{}".format(i, r) for r in range(j+1)], out_branch=["X_{}_{}".format(i, j+1)])
                    self.add_module("X_{}_{}".format(i, j+1), blocks.ConvBlock(in_channels=encoder_channels[1]*j+((encoder_channels[1]+encoder_channels[2]) if upSampleMode != blocks.UpSampleMode.CONV_TRANSPOSE else (decoder_channels[1]+decoder_channels[2] if j == len(encoder_channels)-3 else encoder_channels[1]+encoder_channels[1])), out_channels=decoder_channels[2] if j == len(encoder_channels)-3 else encoder_channels[1], blockConfigs=[blocks.BlockConfig(3,1,1,False, "ReLU;True", blocks.NormMode.BATCH)]*2, dim=dim), in_branch=["X_{}_{}".format(i, j+1)], out_branch=["X_{}_{}".format(i, j+1)])

    class UNetPPHead(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, nb_class: int, dim: int) -> None:
            super().__init__()
            self.add_module("Upsample", blocks.upSample(in_channels=in_channels, out_channels=out_channels, upSampleMode=blocks.UpSampleMode.UPSAMPLE_BILINEAR, dim=dim))
            self.add_module("ConvBlock", blocks.ConvBlock(in_channels=in_channels, out_channels=out_channels, blockConfigs=[blocks.BlockConfig(3,1,1,False, "ReLU;True", blocks.NormMode.BATCH)]*2, dim=dim))
            self.add_module("Conv", blocks.getTorchModule("Conv", dim)(in_channels = out_channels, out_channels = nb_class, kernel_size = 3, stride = 1, padding = 1))
            if nb_class > 1:
                self.add_module("Softmax", torch.nn.Softmax(dim=1))
                self.add_module("Argmax", blocks.ArgMax(dim=1))
            else:
                self.add_module("Tanh", torch.nn.Tanh())
                
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : Union[ModelPatch, None] = None,
                    encoder_channels: list[int] = [1,64,64,128,256,512],
                    decoder_channels: list[int] = [256,128,64,32,16,1],
                    layers: list[int] = [3,4,6,3],
                    dim : int = 2) -> None:
        super().__init__(in_channels = encoder_channels[0], optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, patch=patch, dim = dim)
        self.add_module("Block_0", UNetpp.UNetPPBlock(encoder_channels, decoder_channels[::-1], UNetpp.resNetEncoder(encoder_channels, layers, dim), blocks.UpSampleMode.UPSAMPLE_BILINEAR, dim=dim), out_branch=["X_0_{}".format(j+1) for j in range(len(encoder_channels)-2)])    
        self.add_module("Head", UNetpp.UNetPPHead(in_channels=decoder_channels[-3], out_channels=decoder_channels[-2], nb_class=decoder_channels[-1], dim=dim), in_branch=["X_0_{}".format(len(encoder_channels)-2)], out_branch=[-1])

