import copy
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from konfai.data.patching import ModelPatch
from konfai.network.blocks import LatentDistribution
from konfai.network.network import ModelLoader, Network
from konfai.utils.config import apply_config
from konfai.utils.utils import get_module

models_register = {}


class Criterion(torch.nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()

    def init(self, model: torch.nn.Module, output_group: str, target_group: str) -> str:
        return output_group

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        pass


class MaskedLoss(Criterion):

    def __init__(
        self,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        mode_image_masked: bool,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.mode_image_masked = mode_image_masked

    @staticmethod
    def get_mask(targets: list[torch.Tensor]) -> torch.Tensor | None:
        result = None
        if len(targets) > 0:
            result = targets[0]
            for mask in targets[1:]:
                result = result * mask
        return result

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, float]:
        loss = torch.tensor(0, dtype=torch.float32).to(output.device)
        true_loss = 0
        true_nb = 0
        mask = MaskedLoss.get_mask(list(targets[1:]))
        if mask is not None:
            for batch in range(output.shape[0]):
                if self.mode_image_masked:
                    loss_b = self.loss(
                        output[batch, ...].float() * torch.where(mask == 1, 1, 0),
                        targets[0][batch, ...].float() * torch.where(mask == 1, 1, 0),
                    )
                else:
                    index = mask[batch, ...] == 1
                    loss_b = self.loss(
                        torch.masked_select(output[batch, ...], index).float(),
                        torch.masked_select(targets[0][batch, ...], index).float(),
                    )

                loss += loss_b
                if torch.any(mask[batch] == 1):
                    true_loss += loss_b.item()
                    true_nb += 1
        else:
            loss_tmp = self.loss(output.float(), targets[0].float())
            loss += loss_tmp
            true_loss += loss_tmp.item()
            true_nb += 1
        return loss / output.shape[0], np.nan if true_nb == 0 else true_loss / true_nb


class MSE(MaskedLoss):

    @staticmethod
    def _loss(reduction: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.MSELoss(reduction=reduction)(x, y)

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(partial(MSE._loss, reduction), False)


class MAE(MaskedLoss):

    @staticmethod
    def _loss(reduction: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.L1Loss(reduction=reduction)(x, y)

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(partial(MAE._loss, reduction), False)


class MAESaveMap(MAE):

    def __init__(self, reduction: str = "mean", dataset: str | None = None, group: str | None = None) -> None:
        super().__init__(reduction)
        self.dataset = dataset
        self.group = group

    def forward(self, output: torch.Tensor, *targets: torch.Tensor):  # type: ignore[override]
        loss, true_loss = super().forward(output, *targets)
        if len(targets) == 2:
            error_map = (
                torch.nn.L1Loss(reduction="none")(
                    output.float() * torch.where(targets[1] == 1, 1, 0),
                    targets[0].float() * torch.where(targets[1] == 1, 1, 0),
                )
                .to(output.dtype)
                .cpu()
            )
        else:
            error_map = torch.nn.L1Loss(reduction="none")(output.float(), targets[0].float()).to(output.dtype).cpu()
        return loss, true_loss, error_map

    def get_name(self) -> str:
        return "MAE"


class PSNR(MaskedLoss):

    @staticmethod
    def _loss(dynamic_range: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((x - y).pow(2))
        psnr = 10 * torch.log10(dynamic_range**2 / mse)
        return psnr

    def __init__(self, dynamic_range: float | None = None) -> None:
        dynamic_range = dynamic_range if dynamic_range else 1024 + 3071
        super().__init__(partial(PSNR._loss, dynamic_range), False)


class SSIM(MaskedLoss):

    @staticmethod
    def _loss(dynamic_range: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        from skimage.metrics import structural_similarity

        return structural_similarity(
            x[0][0].detach().cpu().numpy(),
            y[0][0].cpu().numpy(),
            data_range=dynamic_range,
            gradient=False,
            full=False,
        )

    def __init__(self, dynamic_range: float | None = None) -> None:
        dynamic_range = dynamic_range if dynamic_range else 1024 + 3000
        super().__init__(partial(SSIM._loss, dynamic_range), True)


class LPIPS(MaskedLoss):

    @staticmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor)) * 2 - 1

    @staticmethod
    def preprocessing(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.repeat((1, 3, 1, 1)).to(0)

    @staticmethod
    def _loss(loss_fn_alex, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dataset_patch = ModelPatch([1, 64, 64])
        dataset_patch.load(x.shape[2:])

        patch_iterator = dataset_patch.disassemble(LPIPS.normalize(x), LPIPS.normalize(y))
        loss = 0
        with tqdm(
            iterable=enumerate(patch_iterator),
            leave=False,
            total=dataset_patch.get_size(0),
        ) as batch_iter:
            for _, patch_input in batch_iter:
                real, fake = LPIPS.preprocessing(patch_input[0]), LPIPS.preprocessing(patch_input[1])
                loss += loss_fn_alex(real, fake).item()
        return loss / dataset_patch.get_size(0)

    def __init__(self, model: str = "alex") -> None:
        import lpips

        super().__init__(partial(LPIPS._loss, lpips.LPIPS(net=model).to(0)), True)


class TRE(Criterion):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, output: torch.Tensor, *targets: torch.Tensor):
        loss = torch.linalg.norm(output - targets[0], dim=2)
        return loss.mean(), {f"Landmarks_{i}": v.item() for i, v in enumerate(loss.mean(0))}


class Dice(Criterion):

    @staticmethod
    def flatten(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute((1, 0) + tuple(range(2, tensor.dim()))).contiguous().view(tensor.size(1), -1)

    @staticmethod
    def dice_per_channel(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tensor = Dice.flatten(tensor)
        target = Dice.flatten(target)
        return (2.0 * (tensor * target).sum() + 1e-6) / (tensor.sum() + target.sum() + 1e-6)

    @staticmethod
    def _loss(labels: list[int] | None, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = F.interpolate(targets[0], output.shape[2:], mode="nearest")
        result = {}
        loss = torch.tensor(0, dtype=torch.float32).to(output.device)
        labels = labels if labels is not None else torch.unique(target)
        for label in labels:
            tp = target == label
            if tp.any().item():
                if output.shape[1] > 1:
                    pp = output[:, label].unsqueeze(1)
                else:
                    pp = output == label
                loss_tmp = Dice.dice_per_channel(pp.float(), tp.float())
                loss += loss_tmp
                result[label] = loss_tmp.item()
            else:
                result[label] = np.nan
        return 1 - loss / len(labels), result

    def __init__(self, labels: list[int] | None = None) -> None:
        super().__init__()
        self.loss = partial(Dice._loss, labels)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, float]:
        mask = MaskedLoss.get_mask(list(targets[1:]))
        if mask is not None:
            return self.loss(
                (output * torch.where(targets[1] == 1, 1, 0)).to(torch.uint8),
                (targets[0] * torch.where(targets[1] == 1, 1, 0)).to(torch.uint8),
            )
        else:
            return self.loss(output, targets[0])


class DiceSaveMap(Dice):

    def __init__(self, labels: list[int] | None = None, dataset: str | None = None, group: str | None = None) -> None:
        super().__init__(labels)
        self.dataset = dataset
        self.group = group

    def forward(self, output: torch.Tensor, *targets: torch.Tensor):  # type: ignore[override]
        loss, true_loss = super().forward(output, *targets)
        if len(targets) == 2:
            error_map = (
                torch.nn.L1Loss(reduction="none")(
                    output * torch.where(targets[1] == 1, 1, 0), targets[0] * torch.where(targets[1] == 1, 1, 0)
                )
                .to(torch.uint8)
                .cpu()
            )
        else:
            error_map = torch.nn.L1Loss(reduction="none")(output, targets[0]).to(torch.uint8).cpu()
        return loss, true_loss, error_map

    def get_name(self) -> str:
        return "Dice"


class GradientImages(Criterion):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _image_gradient_2d(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]
        dy = image[:, :, :, 1:] - image[:, :, :, :-1]
        return dx, dy

    @staticmethod
    def _image_gradient_3d(
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dz = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]
        return dx, dy, dz

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target_0 = targets[0]
        if len(output.shape) == 5:
            dx, dy, dz = GradientImages._image_gradient_3d(output)
            if target_0 is not None:
                dx_tmp, dy_tmp, dz_tmp = GradientImages._image_gradient_3d(target_0)
                dx -= dx_tmp
                dy -= dy_tmp
                dz -= dz_tmp
            return dx.norm() + dy.norm() + dz.norm()
        else:
            dx, dy = GradientImages._image_gradient_2d(output)
            if target_0 is not None:
                dx_tmp, dy_tmp = GradientImages._image_gradient_2d(target_0)
                dx -= dx_tmp
                dy -= dy_tmp
            return dx.norm() + dy.norm()


class BCE(Criterion):

    def __init__(self, target: float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.register_buffer("target", torch.tensor(target).type(torch.float32))

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = self._buffers["target"]
        return self.loss(output, target.to(output.device).expand_as(output))


class PatchGanLoss(Criterion):

    def __init__(self, target: float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.register_buffer("target", torch.tensor(target).type(torch.float32))

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = self._buffers["target"]
        return self.loss(output, (torch.ones_like(output) * target).to(output.device))


class WGP(Criterion):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        return torch.mean((output - 1) ** 2)


class Gram(Criterion):

    @staticmethod
    def compute_gram(tensor: torch.Tensor):
        (b, ch, w) = tensor.size()
        with torch.amp.autocast("cuda", enabled=False):
            return tensor.bmm(tensor.transpose(1, 2)).div(ch * w)

    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.L1Loss(reduction="sum")

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = targets[0]
        if len(output.shape) > 3:
            output = output.view(output.shape[0], output.shape[1], int(np.prod(output.shape[2:])))
        if len(target.shape) > 3:
            target = target.view(target.shape[0], target.shape[1], int(np.prod(target.shape[2:])))
        return self.loss(Gram.compute_gram(output), Gram.compute_gram(target))


class PerceptualLoss(Criterion):

    class Module:

        def __init__(self, losses: dict[str, float] = {"Gram": 1, "torch:nn:L1Loss": 1}) -> None:
            self.losses = losses
            self.konfai_args = os.environ["KONFAI_CONFIG_PATH"] if "KONFAI_CONFIG_PATH" in os.environ else ""

        def get_loss(self) -> dict[torch.nn.Module, float]:
            result: dict[torch.nn.Module, float] = {}
            for loss, loss_value in self.losses.items():
                module, name = get_module(loss, "konfai.metric.measure")
                result[apply_config(self.konfai_args)(getattr(module, name))()] = loss_value
            return result

    def __init__(
        self,
        model_loader: ModelLoader = ModelLoader(),
        path_model: str = "name",
        modules: dict[str, Module] = {
            "UNetBlock_0.DownConvBlock.Activation_1": Module({"Gram": 1, "torch:nn:L1Loss": 1})
        },
        shape: list[int] = [128, 128, 128],
    ) -> None:
        super().__init__()
        self.path_model = path_model
        if self.path_model not in models_register:
            self.model = model_loader.get_model(
                train=False,
                konfai_args=os.environ["KONFAI_CONFIG_PATH"].split("PerceptualLoss")[0] + "PerceptualLoss.Model",
                konfai_without=[
                    "optimizer",
                    "schedulers",
                    "nb_batch_per_step",
                    "init_type",
                    "init_gain",
                    "outputs_criterions",
                    "drop_p",
                ],
            )
            if path_model.startswith("https"):
                state_dict = torch.hub.load_state_dict_from_url(path_model)
                state_dict = {"Model": {self.model.get_name(): state_dict["model"]}}
            else:
                state_dict = torch.load(path_model, weights_only=True)
            self.model.load(state_dict)
            models_register[self.path_model] = self.model
        else:
            self.model = models_register[self.path_model]

        self.shape = shape
        self.mode = "trilinear" if len(shape) == 3 else "bilinear"
        self.modules_loss: dict[str, dict[torch.nn.Module, float]] = {}
        for name, losses in modules.items():
            self.modules_loss[name.replace(":", ".")] = losses.get_loss()

        self.model.eval()
        self.model.requires_grad_(False)
        self.models: dict[int, torch.nn.Module] = {}

    def preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        # if not all([tensor.shape[-i-1] == size for i, size in enumerate(reversed(self.shape[2:]))]):
        #    tensor = F.interpolate(tensor, mode=self.mode,
        # size=tuple(self.shape), align_corners=False).type(torch.float32)
        # if tensor.shape[1] != self.model.in_channels:
        #    tensor = tensor.repeat(tuple([1,self.model.in_channels] + [1 for _ in range(len(self.shape))]))
        # tensor = (tensor - torch.min(tensor))/(torch.max(tensor)-torch.min(tensor))
        # tensor = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
        return tensor

    def _compute(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        output_preprocessing = self.preprocessing(output)
        targets_preprocessing = [self.preprocessing(target) for target in targets]
        for zipped_output in zip([output_preprocessing], *[[target] for target in targets_preprocessing]):
            output = zipped_output[0]
            targets = zipped_output[1:]

            for zipped_layers in list(
                zip(
                    self.models[output.device.index].get_layers([output], set(self.modules_loss.keys()).copy()),
                    *[
                        self.models[output.device.index].get_layers([target], set(self.modules_loss.keys()).copy())
                        for target in targets
                    ],
                )
            ):
                output_layer = zipped_layers[0][1].view(
                    zipped_layers[0][1].shape[0],
                    zipped_layers[0][1].shape[1],
                    int(np.prod(zipped_layers[0][1].shape[2:])),
                )
                for (loss_function, loss_value), target_layer in zip(
                    self.modules_loss[zipped_layers[0][0]].items(), zipped_layers[1:]
                ):
                    target_layer = target_layer[1].view(
                        target_layer[1].shape[0],
                        target_layer[1].shape[1],
                        int(np.prod(target_layer[1].shape[2:])),
                    )
                    loss = (
                        loss
                        + loss_value * loss_function(output_layer.float(), target_layer.float()) / output_layer.shape[0]
                    )
        return loss

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        if output.device.index not in self.models:
            del os.environ["device"]
            self.models[output.device.index] = Network.to(copy.deepcopy(self.model).eval(), output.device.index).eval()
        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        if len(output.shape) == 5 and len(self.shape) == 2:
            for i in range(output.shape[2]):
                loss = loss + self._compute(output[:, :, i, ...], [t[:, :, i, ...] for t in targets]) / output.shape[2]
        else:
            loss = self._compute(output, targets)
        return loss.to(output)


class KLDivergence(Criterion):

    def __init__(self, shape: list[int], dim: int = 100, mu: float = 0, std: float = 1) -> None:
        super().__init__()
        self.latent_dim = dim
        self.mu = torch.Tensor([mu])
        self.std = torch.Tensor([std])
        self.modelDim = 3
        self.shape = shape
        self.loss = torch.nn.KLDivLoss()

    def init(self, model: Network, output_group: str, target_group: str) -> str:
        super().init(model, output_group, target_group)
        model._compute_channels_trace(model, model.in_channels, None, None)

        last_module = model
        for name in output_group.split(".")[:-1]:
            last_module = last_module[name]

        modules = last_module._modules.copy()
        last_module._modules.clear()

        for name, value in modules.items():
            last_module._modules[name] = value
            if name == output_group.split(".")[-1]:
                last_module.add_module(
                    "LatentDistribution",
                    LatentDistribution(shape=self.shape, latent_dim=self.latent_dim),
                )
        return ".".join(output_group.split(".")[:-1]) + ".LatentDistribution.Concat"

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        mu = output[:, 0, :]
        log_std = output[:, 1, :]
        return torch.mean(-0.5 * torch.sum(1 + log_std - mu**2 - torch.exp(log_std), dim=1), dim=0)


class Accuracy(Criterion):

    def __init__(self) -> None:
        super().__init__()
        self.n: int = 0
        self.corrects = torch.zeros(1)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target_0 = targets[0]
        self.n += output.shape[0]
        self.corrects += (torch.argmax(torch.softmax(output, dim=1), dim=1) == target_0).sum().float().cpu()
        return self.corrects / self.n


class TripletLoss(Criterion):

    def __init__(self) -> None:
        super().__init__()
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        return self.triplet_loss(output[0], output[1], output[2])


class L1LossRepresentation(Criterion):

    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def _variance(self, features: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.clamp(1 - torch.var(features, dim=0), min=0))

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        return self.loss(output[0], output[1]) + self._variance(output[0]) + self._variance(output[1])


class FocalLoss(Criterion):

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: list[float] = [0.5, 2.0, 0.5, 0.5, 1],
        reduction: str = "mean",
    ):
        super().__init__()
        raw_alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = raw_alpha / raw_alpha.sum() * len(raw_alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        target = F.interpolate(targets[0], output.shape[2:], mode="nearest").long()

        logpt = F.log_softmax(output, dim=1)
        pt = torch.exp(logpt)

        logpt = logpt.gather(1, target)
        pt = pt.gather(1, target)

        at = self.alpha.to(target.device)[target].unsqueeze(1)
        loss = -at * ((1 - pt) ** self.gamma) * logpt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FID(Criterion):

    class InceptionV3(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()

            from torchvision.models import Inception_V3_Weights, inception_v3

            self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
            self.model.fc = torch.nn.Identity()
            self.model.eval()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

    def __init__(self) -> None:
        super().__init__()
        self.inception_model = FID.InceptionV3().cuda()

    @staticmethod
    def preprocess_images(image: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            F.resize(image, (299, 299)).repeat((1, 3, 1, 1)),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ).cuda()

    @staticmethod
    def get_features(images: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
        with torch.no_grad():
            features = model(images).cpu().numpy()
        return features

    @staticmethod
    def calculate_fid(real_features: np.ndarray, generated_features: np.ndarray) -> float:
        mu1 = np.mean(real_features, axis=0)
        sigma1 = np.cov(real_features, rowvar=False)
        mu2 = np.mean(generated_features, axis=0)
        sigma2 = np.cov(generated_features, rowvar=False)

        diff = mu1 - mu2
        from scipy import linalg

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        real_images = FID.preprocess_images(targets[0].squeeze(0).permute([1, 0, 2, 3]))
        generated_images = FID.preprocess_images(output.squeeze(0).permute([1, 0, 2, 3]))

        real_features = FID.get_features(real_images, self.inception_model)
        generated_features = FID.get_features(generated_images, self.inception_model)

        return FID.calculate_fid(real_features, generated_features)


class MutualInformationLoss(torch.nn.Module):
    def __init__(
        self,
        num_bins: int = 23,
        sigma_ratio: float = 0.5,
        smooth_nr: float = 1e-7,
        smooth_dr: float = 1e-7,
    ) -> None:
        super().__init__()
        bin_centers = torch.linspace(0.0, 1.0, num_bins)
        sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
        self.num_bins = num_bins
        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers[None, None, ...]
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def parzen_windowing(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_weight, pred_probability = self.parzen_windowing_gaussian(pred)
        target_weight, target_probability = self.parzen_windowing_gaussian(target)
        return pred_weight, pred_probability, target_weight, target_probability

    def parzen_windowing_gaussian(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.clamp(img, 0, 1)
        img = img.reshape(img.shape[0], -1, 1)  # (batch, num_sample, 1)
        weight = torch.exp(
            -self.preterm.to(img) * (img - self.bin_centers.to(img)) ** 2
        )  # (batch, num_sample, num_bin)
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # (batch, num_sample, num_bin)
        probability = torch.mean(weight, dim=-2, keepdim=True)  # (batch, 1, num_bin)
        return weight, probability

    def forward(self, pred: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        wa, pa, wb, pb = self.parzen_windowing(pred, targets[0])  # (batch, num_sample, num_bin), (batch, 1, num_bin)
        pab = torch.bmm(wa.permute(0, 2, 1), wb.to(wa)).div(wa.shape[1])  # (batch, num_bins, num_bins)
        papb = torch.bmm(pa.permute(0, 2, 1), pb.to(pa))  # (batch, num_bins, num_bins)
        mi = torch.sum(
            pab * torch.log((pab + self.smooth_nr) / (papb + self.smooth_dr) + self.smooth_dr),
            dim=(1, 2),
        )  # (batch)
        return torch.mean(mi).neg()  # average over the batch and channel ndims


class IMPACTSynth(Criterion):  # Feature-Oriented Comparison for Unpaired Synthesis

    class Weights:

        def __init__(self, weights: list[float] = [0, 1]) -> None:
            self.weights = weights

    def __init__(
        self,
        model_name: str,
        shape: list[int] = [0, 0],
        in_channels: int = 1,
        losses: dict[str, Weights] = {"torch:nn:L1Loss": Weights([0, 1]), "Gram": Weights([0, 1])},
    ) -> None:
        super().__init__()
        if model_name is None:
            return
        self.in_channels = in_channels
        self.weighted_losses = {}
        for loss, weights in losses.items():
            module, name = get_module(loss, "konfai.metric.measure")
            self.weighted_losses[
                apply_config(".".join(os.environ["KONFAI_CONFIG_PATH"].split(".")[0:-1]) + "." + loss)(
                    getattr(module, name)
                )()
            ] = weights.weights
        self.model_path = hf_hub_download(
            repo_id="VBoussot/impact-torchscript-models", filename=model_name, repo_type="model", revision=None
        )  # nosec B615

        self.model: torch.nn.Module = torch.jit.load(self.model_path, map_location=torch.device("cpu"))  # nosec B614
        self.dim = len(shape)
        self.shape = shape if all(s > 0 for s in shape) else None
        self.modules_loss: dict[str, dict[torch.nn.Module, float]] = {}

        dummy_input = torch.zeros((1, self.in_channels, *(self.shape if self.shape else [224] * self.dim))).to(0)
        try:
            out = self.model.to(0)(dummy_input)
            if not isinstance(out, (list, tuple)):
                raise TypeError(f"Expected model output to be a list or tuple, but got {type(out)}.")
            for name, weights in losses.items():
                if len(weights.weights) != len(out):
                    raise ValueError(
                        f"Loss '{name}': mismatch between the number of weights "
                        f"({len(weights.weights)}) and the number of model outputs "
                        f"({len(out)}). Each output must have a corresponding weight."
                    )
        except Exception as e:
            msg = (
                f"[Model Sanity Check Failed]\n"
                f"Input shape attempted: {dummy_input.shape}\n"
                f"Error: {type(e).__name__}: {e}"
            )
            raise RuntimeError(msg) from e
        self.model = None

    def preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.shape is not None and not all(
            tensor.shape[-i - 1] == size for i, size in enumerate(reversed(self.shape[2:]))
        ):
            tensor = torch.nn.functional.interpolate(
                tensor, mode=self.mode, size=tuple(self.shape), align_corners=False
            ).type(torch.float32)
        if tensor.shape[1] != self.in_channels:
            tensor = tensor.repeat(tuple([1, 3] + [1 for _ in range(self.dim)]))
        return tensor

    def _compute(self, output: torch.Tensor, targets: list[torch.Tensor]) -> torch.Tensor:
        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        output = self.preprocessing(output)
        targets = [self.preprocessing(target) for target in targets]
        self.model.to(output.device)
        for i, zipped_output in enumerate(zip(self.model(output), *[self.model(target) for target in targets])):
            output_feature = zipped_output[0]
            targets_features = zipped_output[1:]
            for target_feature, (sub_loss, weights) in zip(targets_features, self.weighted_losses.items()):
                loss += weights[i] * sub_loss(output_feature, target_feature)
        return loss

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            self.model = torch.jit.load(self.model_path)  # nosec B614
        loss = torch.zeros((1), requires_grad=True).to(output.device, non_blocking=False).type(torch.float32)
        if len(output.shape) == 5 and self.dim == 2:
            for i in range(output.shape[2]):
                loss += self._compute(output[:, :, i, ...], [t[:, :, i, ...] for t in targets])
            loss /= output.shape[2]
        else:
            loss = self._compute(output, list(targets))
        return loss.to(output)


class Variance(Criterion):

    def __init__(self, name: str = "Variance") -> None:
        super().__init__()
        self.name = name

    def get_name(self):
        return self.name

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        return output.float().var(1).mean(), output.float().var(1).mean().item()


class Mean(Criterion):

    def __init__(self, name: str = "Mean") -> None:
        super().__init__()
        self.name = name

    def get_name(self):
        return self.name

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        loss = output.float().mean()
        return loss, loss.item()
