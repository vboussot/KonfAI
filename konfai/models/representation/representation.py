import torch

from konfai.network import network, blocks
from konfai.utils.config import config

class ConvBlock(torch.nn.Module):
    
    def __init__(self, in_channels : int, out_channels : int) -> None:
        super().__init__()
        self.Conv_0 = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.Norm_0 = torch.nn.InstanceNorm3d(num_features=out_channels, affine=True)
        self.Activation_0 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.Conv_1 = torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.Norm_1 = torch.nn.InstanceNorm3d(num_features=out_channels, affine=True)
        self.Activation_1 = torch.nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.Conv_0(input)
        output = self.Norm_0(output)
        output = self.Activation_0(output)
        output = self.Conv_1(output)
        output = self.Norm_1(output)
        output = self.Activation_1(output)
        return output

class UnetCPP_1_Layers(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.DownConvBlock_0 = ConvBlock(in_channels=1, out_channels=32)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.DownConvBlock_0(input)

class Adaptation(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.Encoder_1 = UnetCPP_1_Layers()
        self.ToFeatures = blocks.ToFeatures(3)
        self.FCT_1 = torch.nn.Linear(32, 32, bias=True)

    def forward(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> list[torch.Tensor]:
        self.Encoder_1.requires_grad_(False)
        self.FCT_1.requires_grad_(True)
        return self.FCT_1(self.ToFeatures(self.Encoder_1(A))), self.FCT_1(self.ToFeatures(self.Encoder_1(B))), self.FCT_1(self.ToFeatures(self.Encoder_1(C)))

class Representation(network.Network):

    @config("Representation")
    def __init__(   self,
                    optimizer : network.OptimizerLoader = network.OptimizerLoader(),
                    schedulers : network.LRSchedulersLoader = network.LRSchedulersLoader(),
                    outputsCriterions: dict[str, network.TargetCriterionsLoader] = {"default" : network.TargetCriterionsLoader()},
                    dim : int = 3):
        super().__init__(in_channels = 1, optimizer = optimizer, schedulers = schedulers, outputsCriterions = outputsCriterions, dim=dim, init_type="kaiming")
        self.add_module("Model", Adaptation(), in_branch=[0,1,2])
