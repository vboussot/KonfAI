from abc import ABC
from typing import Type
import torch
from konfai.network import network, blocks
from konfai.utils.config import config
from konfai.data.HDF5 import ModelPatch

"""
'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', dim = 2, in_channels = 3, depths=[2, 2, 2, 2], widths = [64, 64, 128, 256, 512],  num_classes=1000, useBottleneck=False

'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', dim = 2, in_channels = 3, depths=[3, 4, 6, 3], widths = [64, 64, 128, 256, 512],  num_classes=1000, useBottleneck=False

'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', dim = 2, in_channels = 3, depths=[3, 4, 6, 3], widths = [64, 256, 512, 1024, 2048],  num_classes=1000, useBottleneck=True

'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', dim = 2, in_channels = 3, depths=[3, 4, 23, 3], widths = [64, 256, 512, 1024, 2048],  num_classes=1000, useBottleneck=True

'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', dim = 2, in_channels = 3, depths=[3, 8, 36, 3], widths = [64, 256, 512, 1024, 2048],  num_classes=1000, useBottleneck=True

'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
"""
class AbstractResBlock(network.ModuleArgsDict, ABC):

    def __init__(self, in_channels: int, out_channels: int, downsample: bool, dim: int):
        super().__init__()

class ResBlock(AbstractResBlock):

    def __init__(self, in_channels: int, out_channels: int, downsample: bool, dim: int):
        super().__init__(in_channels, out_channels, downsample, dim)
        if downsample:
            self.add_module("Shortcut", blocks.ConvBlock(in_channels, out_channels, 1, blocks.BlockConfig(kernel_size=1, stride=2, padding=0, bias=False, activation="None", normMode=blocks.NormMode.BATCH.name), dim=dim, alias=[["0"], ["1"], []]), in_branch=[1], out_branch=[1], alias=["downsample"])     
                   
        self.add_module("ConvBlock_0", blocks.ConvBlock(in_channels, out_channels, 1, blocks.BlockConfig(kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False, activation="ReLU", normMode=blocks.NormMode.BATCH.name), dim=dim, alias=[["conv1"], ["bn1"], []]))
        self.add_module("ConvBlock_1", blocks.ConvBlock(out_channels, out_channels, 1, blocks.BlockConfig(kernel_size=3, stride=1, padding=1, bias=False, activation="None", normMode=blocks.NormMode.BATCH.name), dim=dim, alias=[["conv2"], ["bn2"], []]))
        self.add_module("Residual", blocks.Add(), in_branch=[0,1])
        self.add_module("ReLU", torch.nn.ReLU())

class ResBottleneckBlock(AbstractResBlock):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool, dim: int):
        super().__init__(in_channels, out_channels, downsample, dim)
        self.add_module("ConvBlock_0", blocks.ConvBlock(in_channels, out_channels//4, 1, blocks.BlockConfig(kernel_size=1, stride=1, padding=0, bias=False, activation="ReLU", normMode=blocks.NormMode.BATCH.name), dim=dim, alias=[["conv1"], ["bn1"], []]))
        self.add_module("ConvBlock_1", blocks.ConvBlock(out_channels//4, out_channels//4, 1, blocks.BlockConfig(kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False, activation="ReLU", normMode=blocks.NormMode.BATCH.name), dim=dim, alias=[["conv2"], ["bn2"], []]))
        self.add_module("ConvBlock_2", blocks.ConvBlock(out_channels//4, out_channels, 1, blocks.BlockConfig(kernel_size=1, stride=1, padding=0, bias=False, activation="ReLU", normMode=blocks.NormMode.BATCH.name), dim=dim, alias=[["conv3"], ["bn3"], []]))
        
        if downsample or in_channels != out_channels:
            self.add_module("Shortcut", blocks.ConvBlock(in_channels, out_channels, 1, blocks.BlockConfig(kernel_size=1, stride=2 if downsample else 1, padding=0, bias=False, activation="None", normMode=blocks.NormMode.BATCH.name), dim=dim, alias=[["0"], ["1"], []]), in_branch=[1], out_branch=[1], alias=["downsample"])
        
        self.add_module("Residual", blocks.Add(), in_branch=[0,1])
        self.add_module("ReLU", torch.nn.ReLU())

class ResNetStage(network.ModuleArgsDict):

    def __init__(self, 
                    in_channels: int,
                    out_channels: int,
                    depth: int,
                    block : Type[AbstractResBlock],
                    downsample : bool,
                    dim : int):
        super().__init__()
        self.add_module("BottleNeckBlock_0", block(in_channels=in_channels, out_channels=out_channels, downsample=downsample, dim=dim), alias=["0"])
        for i in range(1, depth):
            self.add_module("BottleNeckBlock_{}".format(i), block(in_channels=out_channels, out_channels=out_channels, downsample=False, dim=dim), alias=["{}".format(i)])

class ResNetStem(network.ModuleArgsDict):

    def __init__(self, in_channels: int, out_features: int, dim: int):
        super().__init__()
        self.add_module("ConvBlock", blocks.ConvBlock(in_channels, out_features, 1, blocks.BlockConfig(kernel_size=7, stride=2, padding=3, bias=False, activation="ReLU", normMode=blocks.NormMode.BATCH.name), dim=dim, alias=[["conv1"], ["bn1"], []]))
        self.add_module("MaxPool", blocks.getTorchModule("MaxPool", dim)(kernel_size=3, stride=2, padding=1)) 

class ResNetEncoder(network.ModuleArgsDict):
    
    def __init__(self,
                    in_channels: int,
                    depths: list[int],
                    widths: list[int],
                    useBottleneck : bool,
                    dim : int):
        super().__init__()
        self.add_module("ResNetStem", ResNetStem(in_channels, widths[0], dim=dim))

        for i, (in_channels, out_channels, depth) in enumerate(list(zip(widths[:], widths[1:], depths))):
            self.add_module("ResNetStage_{}".format(i), ResNetStage(in_channels=in_channels, out_channels=out_channels, depth=depth, block=ResBottleneckBlock if useBottleneck else ResBlock, downsample=i!=0, dim=dim), alias=["layer{}".format(i+1)])

class Head(network.ModuleArgsDict):
    
    def __init__(self, in_features : int, num_classes : int, dim : int):
        super().__init__()
        self.add_module("AdaptiveAvgPool", blocks.getTorchModule("AdaptiveAvgPool", dim)(1))
        self.add_module("Flatten", torch.nn.Flatten(1))
        self.add_module("Linear", torch.nn.Linear(in_features, num_classes), pretrained=False, alias=["fc"])
        self.add_module("Unsqueeze", blocks.Unsqueeze(2))

        
class ResNet(network.Network):

    @config("ResNet")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    patch : ModelPatch = ModelPatch(),
                    dim : int = 3,
                    in_channels: int = 1,
                    depths: list[int] = [2, 2, 2, 2],
                    widths: list[int] = [64, 64, 128, 256, 512],
                    num_classes: int = 10, 
                    useBottleneck=False):
        super().__init__(in_channels = in_channels, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, dim = dim, patch=patch, init_type = "trunc_normal", init_gain=0.02)
        self.add_module("ResNetEncoder", ResNetEncoder(in_channels=in_channels, depths=depths, widths=widths, useBottleneck=useBottleneck, dim=dim))        
        self.add_module("Head", Head(in_features=widths[-1], num_classes=num_classes, dim=dim))
        