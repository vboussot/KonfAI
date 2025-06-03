from abc import ABC
import importlib
import numpy as np
import torch


import torch.nn.functional as F
import os
import tqdm

from typing import Callable, Union
from functools import partial
import torch.nn.functional as F
from skimage.metrics import structural_similarity
import copy
from abc import abstractmethod

from konfai.utils.config import config
from konfai.utils.utils import _getModule
from konfai.data.HDF5 import ModelPatch
from konfai.network.blocks import LatentDistribution
from konfai.network.network import ModelLoader, Network

modelsRegister = {}

class Criterion(torch.nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()

    def init(self, model : torch.nn.Module, output_group : str, target_group : str) -> str:
        return output_group

    @abstractmethod
    def forward(self, output: torch.Tensor, *targets : list[torch.Tensor]) -> torch.Tensor:
        pass

class MaskedLoss(Criterion):

    def __init__(self, loss : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], mode_image_masked: bool) -> None:
        super().__init__()
        self.loss = loss
        self.mode_image_masked = mode_image_masked

    def getMask(self, targets: list[torch.Tensor]) -> torch.Tensor:
        result = None
        if len(targets) > 0:
            result = targets[0]
            for mask in targets[1:]:
                result = result*mask
        return result

    def forward(self, output: torch.Tensor, *targets : list[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0, dtype=torch.float32).to(output.device)
        mask = self.getMask(targets[1:])
        for batch in range(output.shape[0]):
            if mask is not None:
                if self.mode_image_masked:
                    for i in torch.unique(mask[batch]):
                        if i != 0:
                            loss += self.loss(output[batch, ...]*torch.where(mask == i, 1, 0), targets[0][batch, ...]*torch.where(mask == i, 1, 0))
                else:
                    for i in torch.unique(mask[batch]):
                        if i != 0:
                            index = mask[batch, ...] == i
                            loss += self.loss(torch.masked_select(output[batch, ...], index), torch.masked_select(targets[0][batch, ...], index))
            else:
                loss += self.loss(output[batch, ...], targets[0][batch, ...])
        return loss/output.shape[0]
    
class MSE(MaskedLoss):

    def _loss(reduction: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.MSELoss(reduction=reduction)(x, y)

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(partial(MSE._loss, reduction), False)

class MAE(MaskedLoss):

    def _loss(reduction: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.L1Loss(reduction=reduction)(x, y)
    
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(partial(MAE._loss, reduction), False)

class PSNR(MaskedLoss):

    def _loss(dynamic_range: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
        mse = torch.mean((x - y).pow(2))
        psnr = 10 * torch.log10(dynamic_range**2 / mse) 
        return psnr 

    def __init__(self, dynamic_range: Union[float, None] = None) -> None:
        dynamic_range = dynamic_range if dynamic_range else 1024+3000
        super().__init__(partial(PSNR._loss, dynamic_range), False)
    
class SSIM(MaskedLoss):
    
    def _loss(dynamic_range: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return structural_similarity(x[0][0].detach().cpu().numpy(), y[0][0].cpu().numpy(), data_range=dynamic_range, gradient=False, full=False)
    
    def __init__(self, dynamic_range: Union[float, None] = None) -> None:
        dynamic_range = dynamic_range if dynamic_range else 1024+3000
        super().__init__(partial(SSIM._loss, dynamic_range), True)


class LPIPS(MaskedLoss):

    def normalize(input: torch.Tensor) -> torch.Tensor:
        return (input-torch.min(input))/(torch.max(input)-torch.min(input))*2-1 
        
    def preprocessing(input: torch.Tensor) -> torch.Tensor:
        return input.repeat((1,3,1,1))
    
    def _loss(loss_fn_alex, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        datasetPatch = ModelPatch([1, 64, 64])
        datasetPatch.load(x.shape[2:])

        patchIterator = datasetPatch.disassemble(LPIPS.normalize(x), LPIPS.normalize(y))
        loss = 0
        with tqdm.tqdm(iterable = enumerate(patchIterator), leave=False, total=datasetPatch.getSize(0)) as batch_iter:
            for i, patch_input in batch_iter:
                real, fake = LPIPS.preprocessing(patch_input[0]), LPIPS.preprocessing(patch_input[1])
                loss += loss_fn_alex(real, fake).item()
        return loss/datasetPatch.getSize(0)

    def __init__(self, model: str = "alex") -> None:
        import lpips
        super().__init__(partial(LPIPS._loss, lpips.LPIPS(net=model)), True)

class Dice(Criterion):
    
    def __init__(self, smooth : float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth
    
    def flatten(self, tensor : torch.Tensor) -> torch.Tensor:
        return tensor.permute((1, 0) + tuple(range(2, tensor.dim()))).contiguous().view(tensor.size(1), -1)

    def dice_per_channel(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = self.flatten(input)
        target = self.flatten(target)
        return (2.*(input * target).sum() + self.smooth)/(input.sum() + target.sum() + self.smooth)

    def forward(self, output: torch.Tensor, *targets : list[torch.Tensor]) -> torch.Tensor:
        target = targets[0]
        target = F.one_hot(target.type(torch.int64), num_classes=output.shape[1]).permute(0, len(target.shape), *[i+1 for i in range(len(target.shape)-1)]).float().squeeze(2)
        return 1-torch.mean(self.dice_per_channel(output, target))

class GradientImages(Criterion):

    def __init__(self):
        super().__init__()
    
    @staticmethod
    def _image_gradient2D(image : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]
        dy = image[:, :, :, 1:] - image[:, :, :, :-1]
        return dx, dy

    @staticmethod
    def _image_gradient3D(image : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
        dy = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
        dz = image[:, :, :, :, 1:] - image[:, :, :, :, :-1]
        return dx, dy, dz
        
    def forward(self, output: torch.Tensor, *targets : list[torch.Tensor]) -> torch.Tensor:
        target_0 = targets[0]
        if len(output.shape) == 5:
            dx, dy, dz = GradientImages._image_gradient3D(output)
            if target_0 is not None:
                dx_tmp, dy_tmp, dz_tmp = GradientImages._image_gradient3D(target_0)
                dx -= dx_tmp
                dy -= dy_tmp
                dz -= dz_tmp
            return dx.norm() + dy.norm() + dz.norm()
        else:
            dx, dy = GradientImages._image_gradient2D(output)
            if target_0 is not None:
                dx_tmp, dy_tmp = GradientImages._image_gradient2D(target_0)
                dx -= dx_tmp
                dy -= dy_tmp
            return dx.norm() + dy.norm()
        
class BCE(Criterion):

    def __init__(self, target : float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.register_buffer('target', torch.tensor(target).type(torch.float32))

    def forward(self, output: torch.Tensor, *targets : list[torch.Tensor]) -> torch.Tensor:
        target = self._buffers["target"]
        return self.loss(output, target.to(output.device).expand_as(output))

class PatchGanLoss(Criterion):

    def __init__(self, target : float = 0) -> None:
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.register_buffer('target', torch.tensor(target).type(torch.float32))

    def forward(self, output: torch.Tensor, *targets : list[torch.Tensor]) -> torch.Tensor:
        target = self._buffers["target"]
        return self.loss(output, (torch.ones_like(output)*target).to(output.device))

class WGP(Criterion):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, output: torch.Tensor, *targets : list[torch.Tensor]) -> torch.Tensor:
        return torch.mean((output - 1)**2)

class Gram(Criterion):

    def computeGram(input : torch.Tensor):
        (b, ch, w) = input.size()
        with torch.amp.autocast('cuda', enabled=False):
            return input.bmm(input.transpose(1, 2)).div(ch*w)

    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.L1Loss(reduction='sum')

    def forward(self, output: torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        if len(output.shape) > 3:
            output = output.view(output.shape[0], output.shape[1], int(np.prod(output.shape[2:])))
        if len(target.shape) > 3:
            target = target.view(target.shape[0], target.shape[1], int(np.prod(target.shape[2:])))
        return self.loss(Gram.computeGram(output), Gram.computeGram(target))

class PerceptualLoss(Criterion):
    
    class Module():
        
        @config(None)
        def __init__(self, losses: dict[str, float] = {"Gram": 1, "torch_nn_L1Loss": 1}) -> None:
            self.losses = losses
            self.DL_args = os.environ['DEEP_LEARNING_API_CONFIG_PATH'] if "DEEP_LEARNING_API_CONFIG_PATH" in os.environ else ""

        def getLoss(self) -> dict[torch.nn.Module, float]:
            result: dict[torch.nn.Module, float] = {}
            for loss, l in self.losses.items():
                module, name = _getModule(loss, "metric.measure")
                result[config(self.DL_args)(getattr(importlib.import_module(module), name))(config=None)] = l   
            return result
        
    def __init__(self, modelLoader : ModelLoader = ModelLoader(), path_model : str = "name", modules : dict[str, Module] = {"UNetBlock_0.DownConvBlock.Activation_1": Module({"Gram": 1, "torch_nn_L1Loss": 1})}, shape: list[int] = [128, 128, 128]) -> None:
        super().__init__()
        self.path_model = path_model
        if self.path_model not in modelsRegister:
            self.model = modelLoader.getModel(train=False, DL_args=os.environ['DEEP_LEARNING_API_CONFIG_PATH'].split("PerceptualLoss")[0]+"PerceptualLoss.Model", DL_without=["optimizer", "schedulers", "nb_batch_per_step", "init_type", "init_gain", "outputsCriterions", "drop_p"])
            if path_model.startswith("https"):
                state_dict = torch.hub.load_state_dict_from_url(path_model)
                state_dict = {"Model": {self.model.getName() : state_dict["model"]}}
            else:
                state_dict = torch.load(path_model, weights_only=True)
            self.model.load(state_dict)
            modelsRegister[self.path_model] = self.model
        else:
            self.model = modelsRegister[self.path_model]

        self.shape = shape
        self.mode = "trilinear" if  len(shape) == 3 else "bilinear"
        self.modules_loss: dict[str, dict[torch.nn.Module, float]] = {}
        for name, losses in modules.items():
            self.modules_loss[name.replace(":", ".")] = losses.getLoss()

        self.model.eval()
        self.model.requires_grad_(False)
        self.models: dict[int, torch.nn.Module] = {}

    def preprocessing(self, input: torch.Tensor) -> torch.Tensor:
        #if not all([input.shape[-i-1] == size for i, size in enumerate(reversed(self.shape[2:]))]):
        #    input = F.interpolate(input, mode=self.mode, size=tuple(self.shape), align_corners=False).type(torch.float32)
        #if input.shape[1] != self.model.in_channels:
        #    input = input.repeat(tuple([1,self.model.in_channels] + [1 for _ in range(len(self.shape))]))   
        #input = (input - torch.min(input))/(torch.max(input)-torch.min(input))
        #input = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input)
        return input
    
    def _compute(self, output: torch.Tensor, targets: list[torch.Tensor]) -> torch.Tensor:
        loss = torch.zeros((1), requires_grad = True).to(output.device, non_blocking=False).type(torch.float32)
        output = self.preprocessing(output)
        targets = [self.preprocessing(target) for target in targets]
        for zipped_output in zip([output], *[[target] for target in targets]):
            output = zipped_output[0]
            targets = zipped_output[1:]

            for zipped_layers in list(zip(self.models[output.device.index].get_layers([output], set(self.modules_loss.keys()).copy()), *[self.models[output.device.index].get_layers([target], set(self.modules_loss.keys()).copy()) for target in targets])):
                output_layer = zipped_layers[0][1].view(zipped_layers[0][1].shape[0], zipped_layers[0][1].shape[1], int(np.prod(zipped_layers[0][1].shape[2:])))
                for (loss_function, l), target_layer in zip(self.modules_loss[zipped_layers[0][0]].items(), zipped_layers[1:]):
                    target_layer = target_layer[1].view(target_layer[1].shape[0], target_layer[1].shape[1], int(np.prod(target_layer[1].shape[2:])))
                    loss = loss+l*loss_function(output_layer.float(), target_layer.float())/output_layer.shape[0]
        return loss
    
    def forward(self, output: torch.Tensor, *targets : list[torch.Tensor]) -> torch.Tensor:
        if output.device.index not in self.models:
            del os.environ["device"]
            self.models[output.device.index] = Network.to(copy.deepcopy(self.model).eval(), output.device.index).eval()
        loss = torch.zeros((1), requires_grad = True).to(output.device, non_blocking=False).type(torch.float32)
        if len(output.shape) == 5 and len(self.shape) == 2:
            for i in range(output.shape[2]):
                loss = loss + self._compute(output[:, :, i, ...], [t[:, :, i, ...] for t in targets])/output.shape[2]
        else:
            loss = self._compute(output, targets)
        return loss.to(output)

class KLDivergence(Criterion):
    
    def __init__(self, shape: list[int], dim : int = 100, mu : float = 0, std : float = 1) -> None:
        super().__init__()
        self.latentDim = dim
        self.mu = torch.Tensor([mu])
        self.std = torch.Tensor([std])
        self.modelDim = 3
        self.shape = shape
        self.loss = torch.nn.KLDivLoss()
        
    def init(self, model : Network, output_group : str, target_group : str) -> str:
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
                last_module.add_module("LatentDistribution", LatentDistribution(shape = self.shape, latentDim=self.latentDim))
        return ".".join(output_group.split(".")[:-1])+".LatentDistribution.Concat"

    def forward(self, output: torch.Tensor, targets : list[torch.Tensor]) -> torch.Tensor:
        mu = output[:, 0, :]
        log_std = output[:, 1, :]
        return torch.mean(-0.5 * torch.sum(1 + log_std - mu**2 - torch.exp(log_std), dim = 1), dim = 0)

    """
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        mu = input[:, 0, :]
        log_std = input[:, 1, :]

        z = input[:, 2, :]

        q = torch.distributions.Normal(mu, log_std)

        target_mu = torch.ones((self.latentDim)).to(input.device)*self.mu.to(input.device)
        target_std = torch.ones((self.latentDim)).to(input.device)*self.std.to(input.device)

        p = torch.distributions.Normal(target_mu, target_std)
        
        log_pz = p.log_prob(z)
        log_qzx = q.log_prob(z)

        kl = (log_pz - log_qzx)
        kl = kl.sum(-1)
        return kl
    """
    
class Accuracy(Criterion):

    def __init__(self) -> None:
        super().__init__()
        self.n : int = 0
        self.corrects = torch.zeros((1))

    def forward(self, output: torch.Tensor, *targets : list[torch.Tensor]) -> torch.Tensor:
        target_0 = targets[0]
        self.n += output.shape[0]
        self.corrects += (torch.argmax(torch.softmax(output, dim=1), dim=1) == target_0).sum().float().cpu()
        return self.corrects/self.n

class TripletLoss(Criterion):

    def __init__(self) -> None:
        super().__init__()
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        return self.triplet_loss(output[0], output[1], output[2])

class L1LossRepresentation(Criterion):

    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def _variance(self, features: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.clamp(1 - torch.var(features, dim=0), min=0))

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        return self.loss(output[0], output[1])+ self._variance(output[0]) + self._variance(output[1])

"""import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import functional as F
from scipy import linalg
import numpy as np
    
class FID(Criterion):

    class InceptionV3(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
            self.model.fc = nn.Identity()
            self.model.eval()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)
        
    def __init__(self) -> None:
        super().__init__()
        self.inception_model = FID.InceptionV3().cuda()
        
    def preprocess_images(image: torch.Tensor) -> torch.Tensor:
        return F.normalize(F.resize(image, (299, 299)).repeat((1,3,1,1)), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).cuda()

    def get_features(images: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
        with torch.no_grad():
            features = model(images).cpu().numpy()
        return features

    def calculate_fid(real_features: np.ndarray, generated_features: np.ndarray) -> float:
        # Calculate mean and covariance statistics
        mu1 = np.mean(real_features, axis=0)
        sigma1 = np.cov(real_features, rowvar=False)
        mu2 = np.mean(generated_features, axis=0)
        sigma2 = np.cov(generated_features, rowvar=False)

        # Calculate FID score
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    def forward(self, output: torch.Tensor, *targets : list[torch.Tensor]) -> torch.Tensor:
        real_images = FID.preprocess_images(targets[0].squeeze(0).permute([1, 0, 2, 3]))
        generated_images = FID.preprocess_images(output.squeeze(0).permute([1, 0, 2, 3]))

        real_features = FID.get_features(real_images, self.inception_model)
        generated_features = FID.get_features(generated_images, self.inception_model)

        return FID.calculate_fid(real_features, generated_features)
        """

"""class MutualInformationLoss(torch.nn.Module):
    def __init__(self, num_bins: int = 23, sigma_ratio: float = 0.5, smooth_nr: float = 1e-7, smooth_dr: float = 1e-7) -> None:
        super().__init__()
        bin_centers = torch.linspace(0.0, 1.0, num_bins)
        sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
        self.num_bins = num_bins
        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers[None, None, ...]
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def parzen_windowing(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_weight, pred_probability = self.parzen_windowing_gaussian(pred)
        target_weight, target_probability = self.parzen_windowing_gaussian(target)
        return pred_weight, pred_probability, target_weight, target_probability


    def parzen_windowing_gaussian(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.clamp(img, 0, 1)
        img = img.reshape(img.shape[0], -1, 1)  # (batch, num_sample, 1)
        weight = torch.exp(-self.preterm.to(img) * (img - self.bin_centers.to(img)) ** 2)  # (batch, num_sample, num_bin)
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # (batch, num_sample, num_bin)
        probability = torch.mean(weight, dim=-2, keepdim=True)  # (batch, 1, num_bin)
        return weight, probability

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        wa, pa, wb, pb = self.parzen_windowing(pred, target)  # (batch, num_sample, num_bin), (batch, 1, num_bin)
        pab = torch.bmm(wa.permute(0, 2, 1), wb.to(wa)).div(wa.shape[1])  # (batch, num_bins, num_bins)
        papb = torch.bmm(pa.permute(0, 2, 1), pb.to(pa))  # (batch, num_bins, num_bins)
        mi = torch.sum(pab * torch.log((pab + self.smooth_nr) / (papb + self.smooth_dr) + self.smooth_dr), dim=(1, 2))  # (batch)
        return torch.mean(mi).neg()  # average over the batch and channel ndims"""