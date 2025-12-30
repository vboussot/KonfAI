import tempfile
from abc import ABC, abstractmethod
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
import torch.nn.functional as F  # noqa: N812

from konfai import cuda_visible_devices
from konfai.utils.config import apply_config
from konfai.utils.dataset import Attribute, Dataset, data_to_image, image_to_data
from konfai.utils.utils import NeedDevice, TransformError, _affine_matrix, _resample_affine, get_module


class Transform(NeedDevice, ABC):

    def __init__(self) -> None:
        NeedDevice.__init__(self)
        self.datasets: list[Dataset] = []

    def set_datasets(self, datasets: list[Dataset]):
        self.datasets = datasets

    def transform_shape(self, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return shape

    @abstractmethod
    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass


class TransformInverse(Transform, ABC):

    def __init__(self, inverse: bool) -> None:
        super().__init__()
        self.apply_inverse = inverse

    @abstractmethod
    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass


class TransformLoader:

    def __init__(self) -> None:
        pass

    def get_transform(self, classpath: str, konfai_args: str) -> Transform:
        module, name = get_module(classpath, "konfai.data.transform")
        return apply_config(f"{konfai_args}.{classpath}")(getattr(module, name))()


class Clip(Transform):

    def __init__(
        self,
        min_value: float | str = -1024,
        max_value: float | str = 1024,
        save_clip_min: bool = False,
        save_clip_max: bool = False,
        mask: str | None = None,
    ) -> None:
        super().__init__()
        if isinstance(min_value, float) and isinstance(max_value, float) and max_value <= min_value:
            raise ValueError(
                f"[Clip] Invalid clipping range: max_value ({max_value}) must be greater than min_value ({min_value})"
            )
        self.min_value = min_value
        self.max_value = max_value
        self.save_clip_min = save_clip_min
        self.save_clip_max = save_clip_max
        self.mask = mask

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        mask = None
        if self.mask is not None:
            for dataset in self.datasets:
                if dataset.is_dataset_exist(self.mask, name):
                    mask, _ = dataset.read_data(self.mask, name)
                    break
        if mask is None and self.mask is not None:
            raise ValueError(
                f"Requested mask '{self.mask}' is not present in any dataset. "
                "Check your dataset group names or configuration."
            )
        if mask is None:
            tensor_masked = tensor
        else:
            tensor_masked = tensor[mask == 1]

        if isinstance(self.min_value, str):
            if self.min_value == "min":
                min_value = torch.min(tensor_masked)
            elif self.min_value.startswith("percentile:"):
                try:
                    percentile = float(self.min_value.split(":")[1])
                    min_value = np.percentile(tensor_masked, percentile)
                except (IndexError, ValueError):
                    raise ValueError(f"Invalid format for min_value: '{self.min_value}'. Expected 'percentile:<float>'")
            else:
                raise TypeError(
                    f"Unsupported string for min_value: '{self.min_value}'."
                    "Must be a float, 'min', or 'percentile:<float>'."
                )
        else:
            min_value = self.min_value

        if isinstance(self.max_value, str):
            if self.max_value == "max":
                max_value = torch.max(tensor_masked)
            elif self.max_value.startswith("percentile:"):
                try:
                    percentile = float(self.max_value.split(":")[1])
                    max_value = np.percentile(tensor_masked, percentile)
                except (IndexError, ValueError):
                    raise ValueError(f"Invalid format for max_value: '{self.max_value}'. Expected 'percentile:<float>'")
            else:
                raise TypeError(
                    f"Unsupported string for max_value: '{self.max_value}'."
                    " Must be a float, 'max', or 'percentile:<float>'."
                )
        else:
            max_value = self.max_value

        tensor[torch.where(tensor < min_value)] = min_value
        tensor[torch.where(tensor > max_value)] = max_value
        if self.save_clip_min:
            cache_attribute["Min"] = min_value
        if self.save_clip_max:
            cache_attribute["Max"] = max_value
        return tensor


class Normalize(TransformInverse):

    def __init__(
        self,
        lazy: bool = False,
        channels: list[int] | None = None,
        min_value: float = -1,
        max_value: float = 1,
        inverse: bool = True,
    ) -> None:
        super().__init__(inverse)
        if max_value <= min_value:
            raise ValueError(
                f"[Normalize] Invalid range: max_value ({max_value}) must be greater than min_value ({min_value})"
            )
        self.lazy = lazy
        self.min_value = min_value
        self.max_value = max_value
        self.channels = channels

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Min" not in cache_attribute:
            if self.channels:
                cache_attribute["Min"] = torch.min(tensor[self.channels])
            else:
                cache_attribute["Min"] = torch.min(tensor)
        if "Max" not in cache_attribute:
            if self.channels:
                cache_attribute["Max"] = torch.max(tensor[self.channels])
            else:
                cache_attribute["Max"] = torch.max(tensor)
        if not self.lazy:
            input_min = float(cache_attribute["Min"])
            input_max = float(cache_attribute["Max"])
            norm = input_max - input_min

            if norm == 0:
                print(f"[WARNING] Norm is zero for case '{name}': input is constant with value = {self.min_value}.")
                if self.channels:
                    for channel in self.channels:
                        tensor[channel].fill_(self.min_value)
                else:
                    tensor.fill_(self.min_value)
            else:
                if self.channels:
                    for channel in self.channels:
                        tensor[channel] = (self.max_value - self.min_value) * (
                            tensor[channel] - input_min
                        ) / norm + self.min_value
                else:
                    tensor = (self.max_value - self.min_value) * (tensor - input_min) / norm + self.min_value

        return tensor

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.lazy:
            return tensor
        else:
            input_min = float(cache_attribute.pop("Min"))
            input_max = float(cache_attribute.pop("Max"))
            return (tensor - self.min_value) * (input_max - input_min) / (self.max_value - self.min_value) + input_min


class UnNormalize(Transform):

    def __init__(self, min_value: int = -1024, max_value: int = 3071) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return (tensor + 1) / 2 * (self.max_value - self.min_value) + self.min_value


class Standardize(TransformInverse):

    def __init__(
        self,
        lazy: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        mask: str | None = None,
        inverse: bool = True,
    ) -> None:
        super().__init__(inverse)
        self.lazy = lazy
        self.mean = mean
        self.std = std
        self.mask = mask

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        mask = None
        if self.mask is not None:
            for dataset in self.datasets:
                if dataset.is_dataset_exist(self.mask, name):
                    mask, _ = dataset.read_data(self.mask, name)
                    break
        if mask is None and self.mask is not None:
            raise ValueError(
                f"Requested mask '{self.mask}' is not present in any dataset."
                " Check your dataset group names or configuration."
            )
        if mask is None:
            tensor_masked = tensor
        else:
            tensor_masked = tensor[mask == 1]

        if "Mean" not in cache_attribute:
            cache_attribute["Mean"] = (
                torch.tensor([torch.mean(tensor_masked.type(torch.float32))])
                if self.mean is None
                else torch.tensor([self.mean])
            )

        if "Std" not in cache_attribute:
            cache_attribute["Std"] = (
                torch.tensor([torch.std(tensor_masked.type(torch.float32))])
                if self.std is None
                else torch.tensor([self.std])
            )
        if self.lazy:
            return tensor
        else:
            mean = cache_attribute.get_tensor("Mean")
            std = cache_attribute.get_tensor("Std")
            return (tensor - mean) / std

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.lazy:
            return tensor
        else:
            mean = cache_attribute.pop_tensor("Mean")
            std = cache_attribute.pop_tensor("Std")
            return tensor * std + mean


class TensorCast(TransformInverse):

    def __init__(self, dtype: str = "float32", inverse: bool = True) -> None:
        super().__init__(inverse)
        self.dtype: torch.dtype = getattr(torch, dtype)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        cache_attribute["dtype"] = str(tensor.dtype).replace("torch.", "")
        return tensor.type(self.dtype)

    @staticmethod
    def safe_dtype_cast(dtype_str: str) -> torch.dtype:
        try:
            return getattr(torch, dtype_str)
        except AttributeError:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.to(TensorCast.safe_dtype_cast(cache_attribute.pop("dtype")))


class Padding(TransformInverse):

    def __init__(self, padding: list[int] = [0, 0, 0, 0, 0, 0], mode: str = "constant", inverse: bool = True) -> None:
        super().__init__(inverse)
        self.padding = padding
        self.mode = mode

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute:
            origin = torch.tensor(cache_attribute.get_np_array("Origin"))
            matrix = torch.tensor(cache_attribute.get_np_array("Direction").reshape((len(origin), len(origin))))
            origin = torch.matmul(origin, matrix)
            for dim in range(len(self.padding) // 2):
                origin[-dim - 1] -= self.padding[dim * 2] * cache_attribute.get_np_array("Spacing")[-dim - 1]
            cache_attribute["Origin"] = torch.matmul(origin, torch.inverse(matrix))
        result = F.pad(
            tensor.unsqueeze(0),
            tuple(self.padding),
            self.mode.split(":")[0],
            float(self.mode.split(":")[1]) if len(self.mode.split(":")) == 2 else 0,
        ).squeeze(0)
        return result

    def transform_shape(self, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        for dim in range(len(self.padding) // 2):
            shape[-dim - 1] += sum(self.padding[dim * 2 : dim * 2 + 2])
        return shape

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: dict[str, torch.Tensor]) -> torch.Tensor:
        if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute:
            cache_attribute.pop("Origin")
        slices = [slice(0, shape) for shape in tensor.shape]
        for dim in range(len(self.padding) // 2):
            slices[-dim - 1] = slice(self.padding[dim * 2], tensor.shape[-dim - 1] - self.padding[dim * 2 + 1])
        result = tensor[tuple(slices)]
        return result


class Squeeze(TransformInverse):

    def __init__(self, dim: int, inverse: bool = True) -> None:
        super().__init__(inverse)
        self.dim = dim

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.squeeze(self.dim)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: dict[str, Any]) -> torch.Tensor:
        return tensor.unsqueeze(self.dim)


class Resample(TransformInverse, ABC):

    def __init__(self, inverse: bool) -> None:
        super().__init__(inverse)

    def _resample(self, tensor: torch.Tensor, size: list[int]) -> torch.Tensor:
        if tensor.dtype == torch.uint8:
            mode = "nearest"
        elif len(tensor.shape) < 4:
            mode = "bilinear"
        else:
            mode = "trilinear"

        return (
            F.interpolate(tensor.type(torch.float32).unsqueeze(0), size=tuple(size), mode=mode)
            .squeeze(0)
            .type(tensor.dtype)
            .cpu()
        )

    @abstractmethod
    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass

    @abstractmethod
    def transform_shape(self, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        pass

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        cache_attribute.pop_np_array("Size")
        size_1 = cache_attribute.pop_np_array("Size")
        _ = cache_attribute.pop_np_array("Spacing")
        return self._resample(tensor, [int(size) for size in size_1])


class ResampleToResolution(Resample):

    def __init__(self, spacing: list[float] = [1.0, 1.0, 1.0], inverse: bool = True) -> None:
        super().__init__(inverse)
        self.spacing = torch.tensor([0 if s < 0 else s for s in spacing])

    def transform_shape(self, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        if "Spacing" not in cache_attribute:
            TransformError(
                "Missing 'Spacing' in cache attributes, the data is likely not a valid image.",
                "Make sure your input is a image (e.g., .nii, .mha) with proper metadata.",
            )
        if len(shape) != len(self.spacing):
            TransformError("Shape and spacing dimensions do not match: shape={shape}, spacing={self.spacing}")
        image_spacing = cache_attribute.get_tensor("Spacing")
        spacing = self.spacing

        for i, s in enumerate(self.spacing):
            if s == 0:
                spacing[i] = image_spacing[i]
        resize_factor = spacing / image_spacing
        return [int(x) for x in (torch.tensor(shape) * 1 / resize_factor.flip(0))]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        image_spacing = cache_attribute.get_tensor("Spacing")
        spacing = self.spacing
        for i, s in enumerate(self.spacing):
            if s == 0:
                spacing[i] = image_spacing[i]
        resize_factor = spacing / cache_attribute.get_tensor("Spacing")
        cache_attribute["Spacing"] = spacing
        cache_attribute["Size"] = np.asarray([int(x) for x in torch.tensor(tensor.shape[1:])])
        size = [int(x) for x in (torch.tensor(tensor.shape[1:]) * 1 / resize_factor.flip(0))]
        cache_attribute["Size"] = np.asarray(size)
        return self._resample(tensor, size)


class ResampleToShape(Resample):

    def __init__(self, shape: list[float] = [100, 256, 256], inverse: bool = True) -> None:
        super().__init__(inverse)
        self.shape = torch.tensor([0 if s < 0 else s for s in shape])

    def transform_shape(self, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        if "Spacing" not in cache_attribute:
            TransformError(
                "Missing 'Spacing' in cache attributes, the data is likely not a valid image.",
                "Make sure your input is a image (e.g., .nii, .mha) with proper metadata.",
            )
        if len(shape) != len(self.shape):
            TransformError("Shape and spacing dimensions do not match: shape={shape}, spacing={self.spacing}")
        new_shape = self.shape
        for i, s in enumerate(self.shape):
            if s == 0:
                new_shape[i] = shape[i]
        return new_shape

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        shape = self.shape
        image_shape = torch.tensor([int(x) for x in torch.tensor(tensor.shape[1:])])
        for i, s in enumerate(self.shape):
            if s == 0:
                shape[i] = image_shape[i]
        if "Spacing" in cache_attribute:
            cache_attribute["Spacing"] = torch.flip(
                image_shape / shape * torch.flip(cache_attribute.get_tensor("Spacing"), dims=[0]),
                dims=[0],
            )
        cache_attribute["Size"] = image_shape
        cache_attribute["Size"] = shape
        return self._resample(tensor, shape)


class ResampleTransform(TransformInverse):

    def __init__(self, transforms: dict[str, bool], inverse: bool = True) -> None:
        super().__init__(inverse)
        self.transforms = transforms

    def transform_shape(self, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return shape

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if len(tensor.shape) != 4:
            raise NameError("Input size should be 5 dim")
        image = data_to_image(tensor, cache_attribute)

        vectors = [torch.arange(0, s) for s in tensor.shape[1:]]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)

        transforms = []
        for transform_group, invert in self.transforms.items():
            transform = None
            for dataset in self.datasets:
                if dataset.is_dataset_exist(transform_group, name):
                    transform = dataset.read_transform(transform_group, name)
                    break
            if transform is None:
                raise NameError(f"Tranform : {transform_group}/{name} not found")
            if isinstance(transform, sitk.BSplineTransform):
                if invert:
                    transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
                    transform_to_displacement_field_filter.SetReferenceImage(image)
                    displacement_field = transform_to_displacement_field_filter.Execute(transform)
                    iterative_inverse_displacement_field_image_filter = (
                        sitk.IterativeInverseDisplacementFieldImageFilter()
                    )
                    iterative_inverse_displacement_field_image_filter.SetNumberOfIterations(20)
                    inverse_displacement_field = iterative_inverse_displacement_field_image_filter.Execute(
                        displacement_field
                    )
                    transform = sitk.DisplacementFieldTransform(inverse_displacement_field)
            else:
                if invert:
                    transform = transform.GetInverse()
            transforms.append(transform)
        result_transform = sitk.CompositeTransform(transforms)

        transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
        transform_to_displacement_field_filter.SetReferenceImage(image)
        transform_to_displacement_field_filter.SetNumberOfThreads(16)
        new_locs = grid + torch.tensor(
            sitk.GetArrayFromImage(transform_to_displacement_field_filter.Execute(result_transform))
        ).unsqueeze(0).permute(0, 4, 1, 2, 3)
        shape = new_locs.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        result = (
            F.grid_sample(
                tensor.to(self.device).unsqueeze(0).float(),
                new_locs.to(self.device).float(),
                align_corners=True,
                padding_mode="border",
                mode="nearest" if tensor.dtype == torch.uint8 else "bilinear",
            )
            .squeeze(0)
            .cpu()
        )
        return result.type(torch.uint8) if tensor.dtype == torch.uint8 else result

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # TODO
        return tensor


class Mask(Transform):

    def __init__(self, path: str = "./default.mha", value_outside: int = 0) -> None:
        super().__init__()
        self.path = path
        self.value_outside = value_outside

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.path.endswith(".mha"):
            mask = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(self.path))).unsqueeze(0)
        else:
            mask = None
            for dataset in self.datasets:
                if dataset.is_dataset_exist(self.path, name):
                    mask, _ = dataset.read_data(self.path, name)
                    break
            if mask is None:
                raise NameError(f"Mask : {self.path}/{name} not found")
        return torch.where(torch.tensor(mask) > 0, tensor, self.value_outside)


class Sum(Transform):

    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "number_of_channels_per_model" in cache_attribute:
            number_of_channels = cache_attribute.pop_tensor("number_of_channels_per_model")
            result = tensor[0]
            for i, t in enumerate(tensor[1:]):
                t[t != 0] += int(number_of_channels[i]) - 1
                result += t
            return result
        else:
            return torch.sum(tensor, dim=self.dim).to(tensor.dtype)


class Gradient(Transform):

    def __init__(self, per_dim: bool = False):
        super().__init__()
        self.per_dim = per_dim

    @staticmethod
    def _image_gradient_2d(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = image[:, 1:, :] - image[:, :-1, :]
        dy = image[:, :, 1:] - image[:, :, :-1]
        return torch.nn.ConstantPad2d((0, 0, 0, 1), 0)(dx), torch.nn.ConstantPad2d((0, 1, 0, 0), 0)(dy)

    @staticmethod
    def _image_gradient_3d(
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = image[:, 1:, :, :] - image[:, :-1, :, :]
        dy = image[:, :, 1:, :] - image[:, :, :-1, :]
        dz = image[:, :, :, 1:] - image[:, :, :, :-1]
        return (
            torch.nn.ConstantPad3d((0, 0, 0, 0, 0, 1), 0)(dx),
            torch.nn.ConstantPad3d((0, 0, 0, 1, 0, 0), 0)(dy),
            torch.nn.ConstantPad3d((0, 1, 0, 0, 0, 0), 0)(dz),
        )

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        result = torch.stack(
            (Gradient._image_gradient_3d(tensor) if len(tensor.shape) == 4 else Gradient._image_gradient_2d(tensor)),
            dim=1,
        ).squeeze(0)
        if not self.per_dim:
            result = torch.sigmoid(result * 3)
            result = result.norm(dim=0)
            result = torch.unsqueeze(result, 0)

        return result


class Argmax(Transform):

    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.argmax(tensor, dim=self.dim).unsqueeze(self.dim)


class Softmax(Transform):

    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.softmax(tensor, dim=self.dim)


class FlatLabel(Transform):

    def __init__(self, labels: list[int] | None = None) -> None:
        super().__init__()
        self.labels = labels

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        data = torch.zeros_like(tensor)
        if self.labels:
            for label in self.labels:
                data[torch.where(tensor == label)] = 1
        else:
            data[torch.where(tensor > 0)] = 1
        return data


class Save(Transform):

    def __init__(self, dataset: str, group: str | None = None) -> None:
        super().__init__()
        self.dataset = dataset
        self.group = group

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor


class Flatten(Transform):

    def __init__(self) -> None:
        super().__init__()

    def transform_shape(self, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return [np.prod(np.asarray(shape))]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.flatten()


class Permute(TransformInverse):

    def __init__(self, dims: str = "1|0|2", inverse: bool = True) -> None:
        super().__init__(inverse)
        self.dims = [0] + [int(d) + 1 for d in dims.split("|")]

    def transform_shape(self, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return [shape[it - 1] for it in self.dims[1:]]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.permute(tuple(self.dims))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.permute(tuple(np.argsort(self.dims)))


class Flip(TransformInverse):

    def __init__(self, dims: str = "1|0|2", inverse: bool = True) -> None:
        super().__init__(inverse)

        self.dims = [int(d) + 1 for d in str(dims).split("|")]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.flip(tuple(self.dims))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.flip(tuple(self.dims))


class Canonical(TransformInverse):

    def __init__(self, inverse: bool = True) -> None:
        super().__init__(inverse)
        self.canonical_direction = torch.diag(torch.tensor([-1, -1, 1])).to(torch.double)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        spacing = cache_attribute.get_tensor("Spacing")
        initial_matrix = cache_attribute.get_tensor("Direction").reshape(3, 3).to(torch.double)
        initial_origin = cache_attribute.get_tensor("Origin")
        cache_attribute["Direction"] = (self.canonical_direction).flatten()
        matrix = _affine_matrix(self.canonical_direction @ initial_matrix.inverse(), torch.tensor([0, 0, 0]))
        center_voxel = torch.tensor(
            [(tensor.shape[-i - 1] - 1) * spacing[i] / 2 for i in range(3)],
            dtype=torch.double,
        )
        center_physical = initial_matrix @ center_voxel + initial_origin
        cache_attribute["Origin"] = center_physical - (self.canonical_direction @ center_voxel)
        return _resample_affine(tensor, matrix.unsqueeze(0))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        cache_attribute.pop("Direction")
        cache_attribute.pop("Origin")
        matrix = _affine_matrix(
            (
                self.canonical_direction
                @ cache_attribute.get_tensor("Direction").to(torch.double).reshape(3, 3).inverse()
            ).inverse(),
            torch.tensor([0, 0, 0]),
        )
        return _resample_affine(tensor, matrix.unsqueeze(0))


class HistogramMatching(Transform):

    def __init__(self, reference_group: str) -> None:
        super().__init__()
        self.reference_group = reference_group

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        image = data_to_image(tensor, cache_attribute)
        image_ref = None
        for dataset in self.datasets:
            if dataset.is_dataset_exist(self.reference_group, name):
                image_ref = dataset.read_image(self.reference_group, name)
        if image_ref is None:
            raise NameError(f"Image : {self.reference_group}/{name} not found")
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(256)
        matcher.SetNumberOfMatchPoints(1)
        matcher.SetThresholdAtMeanIntensity(True)
        result, _ = image_to_data(matcher.Execute(image, image_ref))
        return torch.tensor(result)


class SelectLabel(Transform):

    def __init__(self, labels: list[str]) -> None:
        super().__init__()
        self.labels = [label[1:-1].split(",") for label in labels]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        data = torch.zeros_like(tensor)
        for old_label, new_label in self.labels:
            data[tensor == int(old_label)] = int(new_label)
        return data


class OneHot(TransformInverse):

    def __init__(self, num_classes: int, inverse: bool = True) -> None:
        super().__init__(inverse)
        self.num_classes = num_classes

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        result = (
            F.one_hot(tensor.type(torch.int64), num_classes=self.num_classes)
            .permute(0, len(tensor.shape), *[i + 1 for i in range(len(tensor.shape) - 1)])
            .float()
            .squeeze(0)
        )
        return result

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.argmax(tensor, dim=1).unsqueeze(1)


class KonfAIInference(Transform):

    def __init__(
        self,
        repo_id: str = "VBoussot/MRSegmentator-KonfAI",
        model_name: str = "MRSegmentator",
        number_of_ensemble: int = 0,
        number_of_tta: int = 0,
        number_of_mc_dropout: int = 0,
        per_channel: bool = False,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.model_name = model_name
        self.number_of_ensemble = number_of_ensemble
        self.number_of_tta = number_of_tta
        self.number_of_mc_dropout = number_of_mc_dropout
        self.per_channel = per_channel

    def infer_entry(self, dataset_path: Path, output_path: Path, gpu: list[int]):
        from konfai.app import KonfAIApp

        konfai_app = KonfAIApp(f"{self.repo_id}:{self.model_name}")
        konfai_app.infer(
            [[dataset_path]],
            output_path,
            self.number_of_ensemble,
            self.number_of_tta,
            self.number_of_mc_dropout,
            gpu=gpu,
        )

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        with tempfile.TemporaryDirectory() as tmpdir:

            dataset_path = Path(tmpdir) / "Dataset"
            if self.per_channel:
                for i, channel in enumerate(tensor):
                    image = data_to_image(channel.unsqueeze(0).numpy(), cache_attribute)
                    (dataset_path / f"P{i:03d}").mkdir(parents=True, exist_ok=True)
                    sitk.WriteImage(image, str(dataset_path / f"P{i:03d}" / "Volume.mha"))
            else:
                image = data_to_image(tensor.numpy(), cache_attribute)

                (dataset_path / "P000").mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(image, str(dataset_path / "P000" / "Volume.mha"))

            ctx = get_context("spawn")

            p = ctx.Process(
                target=self.infer_entry, args=(dataset_path, Path(tmpdir) / "Output", cuda_visible_devices())
            )
            p.start()
            p.join()

            if p.exitcode != 0:
                raise RuntimeError("Inference process failed")

            result = []
            for file in (Path(tmpdir) / "Output").rglob("*.mha"):
                if file.name != "InferenceStack.mha":
                    result.append(torch.from_numpy(image_to_data(sitk.ReadImage(str(file)))[0]))
            return torch.stack(result, dim=1).squeeze(0)


class InferenceStack(Transform):

    def __init__(self, dataset: str, name: str, mode: str = "mean"):
        self.dataset = None
        if dataset:
            if len(dataset.split(":")) > 1:
                filename, file_format = dataset.split(":")
            else:
                filename = dataset
                file_format = "mha"
            self.dataset = Dataset(filename, file_format)
        self.name = name
        self.mode = mode

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if tensors.shape[0] == 1:
            return tensors.squeeze(0)
        if self.mode == "Seg":
            _tensors = torch.argmax(torch.softmax(tensors, dim=1), dim=1).to(torch.uint8)
        else:
            _tensors = tensors.squeeze(1)
        dataset = self.dataset if self.dataset else self.datasets[-1]
        dataset.write("InferenceStack", name, _tensors.numpy(), cache_attribute)
        return tensors.float().mean(0).to(tensors.dtype)


class Variance(Transform):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensors.float().var(0).unsqueeze(0) if tensors.shape[0] > 1 else torch.zeros_like(tensors[0])


class TotalSegmentator(Transform):

    def __init__(self, task: str = "total"):
        super().__init__()
        self.task = task

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        from totalsegmentator.python_api import totalsegmentator

        with tempfile.TemporaryDirectory() as tmpdir:
            image = data_to_image(tensor.numpy(), cache_attribute)
            sitk.WriteImage(image, tmpdir + "/image.nii.gz")
            seg = totalsegmentator(tmpdir + "/image.nii.gz", tmpdir, task=self.task, skip_saving=True, quiet=True)
        return (
            torch.from_numpy(np.array(np.asanyarray(seg.dataobj), copy=True).astype(np.uint8, copy=False))
            .permute(2, 1, 0)
            .unsqueeze(0)
        )
