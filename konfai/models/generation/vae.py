import torch

from konfai.network import blocks, network


class VAE(network.Network):

    class AutoEncoderBlock(network.ModuleArgsDict):

        def __init__(
            self,
            channels: list[int],
            nb_conv_per_stage: int,
            block_config: blocks.BlockConfig,
            downsample_mode: blocks.DownsampleMode,
            upsample_mode: blocks.UpsampleMode,
            dim: int,
            block: type,
            i: int = 0,
        ) -> None:
            super().__init__()
            if i > 0:
                self.add_module(
                    downsample_mode.name,
                    blocks.downsample(
                        in_channels=channels[0],
                        out_channels=channels[1],
                        downsample_mode=downsample_mode,
                        dim=dim,
                    ),
                )
            self.add_module(
                "DownBlock",
                block(
                    in_channels=channels[(1 if downsample_mode == blocks.DownsampleMode.CONV_STRIDE and i > 0 else 0)],
                    out_channels=channels[1],
                    block_configs=[block_config] * nb_conv_per_stage,
                    dim=dim,
                ),
            )
            if len(channels) > 2:
                self.add_module(
                    f"AutoEncoder_{i + 1}",
                    VAE.AutoEncoderBlock(
                        channels[1:],
                        nb_conv_per_stage,
                        block_config,
                        downsample_mode,
                        upsample_mode,
                        dim,
                        block,
                        i + 1,
                    ),
                )
                self.add_module(
                    "UpBlock",
                    block(
                        in_channels=(
                            channels[2] if upsample_mode != blocks.UpsampleMode.CONV_TRANSPOSE else channels[1]
                        ),
                        out_channels=channels[1],
                        block_configs=[block_config] * nb_conv_per_stage,
                        dim=dim,
                    ),
                )
            if i > 0:
                self.add_module(
                    upsample_mode.name,
                    blocks.upsample(
                        in_channels=channels[1],
                        out_channels=channels[0],
                        upsample_mode=upsample_mode,
                        dim=dim,
                    ),
                )

    class VAEHead(network.ModuleArgsDict):

        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module(
                "Conv",
                blocks.get_torch_module("Conv", dim)(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
            self.add_module("Tanh", torch.nn.Tanh())

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        dim: int = 3,
        channels: list[int] = [1, 64, 128, 256, 512, 1024],
        block_config: blocks.BlockConfig = blocks.BlockConfig(),
        nb_conv_per_stage: int = 2,
        downsample_mode: str = "MAXPOOL",
        upsample_mode: str = "CONV_TRANSPOSE",
        block_type: str = "Conv",
    ) -> None:

        super().__init__(
            in_channels=channels[0],
            init_type="normal",
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=dim,
            nb_batch_per_step=1,
        )
        self.add_module(
            "AutoEncoder_0",
            VAE.AutoEncoderBlock(
                channels,
                nb_conv_per_stage,
                block_config,
                downsample_mode=blocks.DownsampleMode[downsample_mode],
                upsample_mode=blocks.UpsampleMode[upsample_mode],
                dim=dim,
                block=blocks.ConvBlock if block_type == "Conv" else blocks.ResBlock,
            ),
        )
        self.add_module("Head", VAE.VAEHead(channels[1], channels[0], dim))


class LinearVAE(network.Network):

    class LinearVAEDenseLayer(network.ModuleArgsDict):

        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.add_module("Linear", torch.nn.Linear(in_features, out_features))
            # self.add_module("Norm", torch.nn.BatchNorm1d(out_features))
            self.add_module("Activation", torch.nn.LeakyReLU())

    class LinearVAEHead(network.ModuleArgsDict):

        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.add_module("Linear", torch.nn.Linear(in_features, out_features))
            self.add_module("Tanh", torch.nn.Tanh())

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
    ) -> None:
        super().__init__(
            in_channels=1,
            init_type="normal",
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=1,
            nb_batch_per_step=1,
        )
        self.add_module("DenseLayer_0", LinearVAE.LinearVAEDenseLayer(23343, 5))
        # self.add_module("Head", LinearVAE.DenseLayer(100, 28590))
        self.add_module("Head", LinearVAE.LinearVAEHead(5, 23343))
        # self.add_module("DenseLayer_5", LinearVAE.DenseLayer(5000, 28590))
