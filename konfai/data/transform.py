import importlib
import torch
import numpy as np
import SimpleITK as sitk
from abc import ABC, abstractmethod
import torch.nn.functional as F
from typing import Any, Union

from konfai.utils.utils import _getModule, NeedDevice, _resample_affine, _affine_matrix
from konfai.utils.dataset import Dataset, Attribute, data_to_image, image_to_data
from konfai.utils.config import config

class Transform(NeedDevice, ABC):
    
    def __init__(self) -> None:
        self.datasets : list[Dataset] = []
        
    def setDatasets(self, datasets: list[Dataset]):
        self.datasets = datasets

    def transformShape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return shape

    @abstractmethod
    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass

class TransformLoader:

    @config()
    def __init__(self) -> None:
        pass
    
    def getTransform(self, classpath : str, DL_args : str) -> Transform:
        module, name = _getModule(classpath, "data.transform")
        return config("{}.{}".format(DL_args, classpath))(getattr(importlib.import_module(module), name))(config = None)

class Clip(Transform):

    def __init__(self, min_value : float = -1024, max_value : float = 1024, saveClip_min: bool = False, saveClip_max: bool = False) -> None:
        assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value
        self.saveClip_min = saveClip_min
        self.saveClip_max = saveClip_max

    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        input[torch.where(input < self.min_value)] = self.min_value
        input[torch.where(input > self.max_value)] = self.max_value
        if self.saveClip_min:
            cache_attribute["Min"] = self .min_value
        if self.saveClip_max:
            cache_attribute["Max"] = self.max_value
        return input

    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input

class Normalize(Transform):

    def __init__(self, lazy : bool = False, channels: Union[list[int], None] = None, min_value : float = -1, max_value : float = 1) -> None:
        assert max_value > min_value
        self.lazy = lazy
        self.min_value = min_value
        self.max_value = max_value
        self.channels = channels

    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Min" not in cache_attribute:
            if self.channels:
                cache_attribute["Min"] = torch.min(input[self.channels])
            else:
                cache_attribute["Min"] = torch.min(input)
        if "Max" not in cache_attribute:
            if self.channels:
                cache_attribute["Max"] = torch.max(input[self.channels])
            else:
                cache_attribute["Max"] = torch.max(input)
        if not self.lazy:
            input_min = float(cache_attribute["Min"])
            input_max = float(cache_attribute["Max"])
            norm = input_max-input_min
            assert norm != 0
            if self.channels:
                for channel in self.channels:
                    input[channel] = (self.max_value-self.min_value)*(input[channel] - input_min) / norm + self.min_value
            else:
                input = (self.max_value-self.min_value)*(input - input_min) / norm + self.min_value
        return input
    
    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.lazy:
            return input
        else:
            input_min = float(cache_attribute.pop("Min"))
            input_max = float(cache_attribute.pop("Max"))
            return (input - self.min_value)*(input_max-input_min)/(self.max_value-self.min_value)+input_min

class Standardize(Transform):

    def __init__(self, lazy : bool = False, mean: Union[list[float], None] = None, std: Union[list[float], None]= None) -> None:
        self.lazy = lazy
        self.mean = mean
        self.std = std

    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Mean" not in cache_attribute:
            cache_attribute["Mean"] = torch.mean(input.type(torch.float32), dim=[i + 1 for i in range(len(input.shape)-1)]) if self.mean is None else torch.tensor([self.mean])
        if "Std" not in cache_attribute:
            cache_attribute["Std"] = torch.std(input.type(torch.float32), dim=[i + 1 for i in range(len(input.shape)-1)]) if self.std is None else torch.tensor([self.std])

        if self.lazy:
            return input
        else:
            mean = cache_attribute.get_tensor("Mean").view(-1, *[1 for _ in range(len(input.shape)-1)])
            std = cache_attribute.get_tensor("Std").view(-1, *[1 for _ in range(len(input.shape)-1)])
            return (input - mean) / std
        
    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.lazy:
            return input
        else:
            mean = float(cache_attribute.pop("Mean"))
            std = float(cache_attribute.pop("Std"))
            return input * std + mean
        
class TensorCast(Transform):

    def __init__(self, dtype : str = "default:float32,int64,int16") -> None:
        self.dtype : torch.dtype = getattr(torch, dtype)

    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        cache_attribute["dtype"] = input.dtype
        return input.type(self.dtype)
    
    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input.to(eval(cache_attribute.pop("dtype")))

class Padding(Transform):

    def __init__(self, padding : list[int] = [0,0,0,0,0,0], mode : str = "default:constant,reflect,replicate,circular") -> None:
        self.padding = padding
        self.mode = mode

    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute:
            origin = torch.tensor(cache_attribute.get_np_array("Origin"))
            matrix = torch.tensor(cache_attribute.get_np_array("Direction").reshape((len(origin),len(origin))))
            origin = torch.matmul(origin, matrix)
            for dim in range(len(self.padding)//2):
                origin[-dim-1] -= self.padding[dim*2]* cache_attribute.get_np_array("Spacing")[-dim-1]
            cache_attribute["Origin"] = torch.matmul(origin, torch.inverse(matrix))
        result = F.pad(input.unsqueeze(0), tuple(self.padding), self.mode.split(":")[0], float(self.mode.split(":")[1]) if len(self.mode.split(":")) == 2 else 0).squeeze(0)
        return result
    
    def transformShape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        for dim in range(len(self.padding)//2):
            shape[-dim-1] += sum(self.padding[dim*2:dim*2+2])
        return shape

    def inverse(self, name: str, input : torch.Tensor, cache_attribute: dict[str, torch.Tensor]) -> torch.Tensor:
        if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute:
            cache_attribute.pop("Origin")
        slices = [slice(0, shape) for shape in input.shape]
        for dim in range(len(self.padding)//2):
            slices[-dim-1] = slice(self.padding[dim*2], input.shape[-dim-1]-self.padding[dim*2+1])
        result = input[slices]
        return result

class Squeeze(Transform):

    def __init__(self, dim: int) -> None:
        self.dim = dim
    
    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input.squeeze(self.dim)

    def inverse(self, name: str, input : torch.Tensor, cache_attribute: dict[str, Any]) -> torch.Tensor:
        return input.unsqueeze(self.dim)

class Resample(Transform, ABC):

    def __init__(self) -> None:
        pass

    def _resample(self, input: torch.Tensor, size: list[int]) -> torch.Tensor:
        args = {}
        if input.dtype == torch.uint8:
            mode = "nearest"
        elif len(input.shape) < 4:
            mode = "bilinear"
        else:
            mode = "trilinear"
        return F.interpolate(input.type(torch.float32).unsqueeze(0), size=tuple(size), mode=mode).squeeze(0).type(input.dtype).cpu()

    @abstractmethod
    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass
    
    @abstractmethod
    def transformShape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        pass
    
    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        size_0 = cache_attribute.pop_np_array("Size")
        size_1 = cache_attribute.pop_np_array("Size")
        _ = cache_attribute.pop_np_array("Spacing")
        return self._resample(input, [int(size) for size in size_1])

class ResampleIsotropic(Resample):

    def __init__(self, spacing : list[float] = [1., 1., 1.]) -> None:
        self.spacing = torch.tensor(spacing, dtype=torch.float64)
        
    def transformShape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        assert "Spacing" in cache_attribute, "Error no spacing"
        resize_factor = self.spacing/cache_attribute.get_tensor("Spacing").flip(0)
        return  [int(x) for x in (torch.tensor(shape) * 1/resize_factor)]

    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        assert "Spacing" in cache_attribute, "Error no spacing"
        resize_factor = self.spacing/cache_attribute.get_tensor("Spacing").flip(0)
        cache_attribute["Spacing"] = self.spacing.flip(0)
        cache_attribute["Size"] = np.asarray([int(x) for x in torch.tensor(input.shape[1:])])
        size = [int(x) for x in (torch.tensor(input.shape[1:]) * 1/resize_factor)]
        cache_attribute["Size"] = np.asarray(size)
        return self._resample(input, size)

class ResampleResize(Resample):

    def __init__(self, size : list[int] = [100,512,512]) -> None:
        self.size = size

    def transformShape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return self.size
    
    def __call__(self, name: str, input: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Spacing" in cache_attribute:
            cache_attribute["Spacing"] = torch.flip(torch.tensor(list(input.shape[1:]))/torch.tensor(self.size)*torch.flip(cache_attribute.get_tensor("Spacing"), dims=[0]), dims=[0])
        cache_attribute["Size"] = np.asarray([int(x) for x in torch.tensor(input.shape[1:])])
        cache_attribute["Size"] = self.size
        return self._resample(input, self.size)

class ResampleTransform(Transform):

    def __init__(self, transforms : dict[str, bool]) -> None:
        self.transforms = transforms
    
    def transformShape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return shape

    def __call__V1(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        transforms = []
        image = data_to_image(input, cache_attribute)
        for transform_group, invert in self.transforms.items():
            transform = None
            for dataset in self.datasets:
                if dataset.isDatasetExist(transform_group, name):
                    transform = dataset.readTransform(transform_group, name)
                    break
            if transform is None:
                raise NameError("Tranform : {}/{} not found".format(transform_group, name))
            if isinstance(transform, sitk.BSplineTransform):
                if invert:
                    transformToDisplacementFieldFilter = sitk.TransformToDisplacementFieldFilter()
                    transformToDisplacementFieldFilter.SetReferenceImage(image)
                    displacementField = transformToDisplacementFieldFilter.Execute(transform)
                    iterativeInverseDisplacementFieldImageFilter = sitk.IterativeInverseDisplacementFieldImageFilter()
                    iterativeInverseDisplacementFieldImageFilter.SetNumberOfIterations(20)
                    inverseDisplacementField = iterativeInverseDisplacementFieldImageFilter.Execute(displacementField)
                    transform = sitk.DisplacementFieldTransform(inverseDisplacementField)
            else:
                if invert:
                    transform = transform.GetInverse()
            transforms.append(transform)
        result_transform = sitk.CompositeTransform(transforms)
        result = torch.tensor(sitk.GetArrayFromImage(sitk.Resample(image, image, result_transform, sitk.sitkNearestNeighbor if input.dtype == torch.uint8 else sitk.sitkBSpline, 0 if input.dtype == torch.uint8 else -1024))).unsqueeze(0)
        return result.type(torch.uint8) if input.dtype == torch.uint8 else result
    
    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        assert len(input.shape) == 4 , "input size should be 5 dim"
        image = data_to_image(input, cache_attribute)
        
        vectors = [torch.arange(0, s) for s in input.shape[1:]]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        
        transforms = []
        for transform_group, invert in self.transforms.items():
            transform = None
            for dataset in self.datasets:
                if dataset.isDatasetExist(transform_group, name):
                    transform = dataset.readTransform(transform_group, name)
                    break
            if transform is None:
                raise NameError("Tranform : {}/{} not found".format(transform_group, name))
            if isinstance(transform, sitk.BSplineTransform):
                if invert:
                    transformToDisplacementFieldFilter = sitk.TransformToDisplacementFieldFilter()
                    transformToDisplacementFieldFilter.SetReferenceImage(image)
                    displacementField = transformToDisplacementFieldFilter.Execute(transform)
                    iterativeInverseDisplacementFieldImageFilter = sitk.IterativeInverseDisplacementFieldImageFilter()
                    iterativeInverseDisplacementFieldImageFilter.SetNumberOfIterations(20)
                    inverseDisplacementField = iterativeInverseDisplacementFieldImageFilter.Execute(displacementField)
                    transform = sitk.DisplacementFieldTransform(inverseDisplacementField)
            else:
                if invert:
                    transform = transform.GetInverse()
            transforms.append(transform)
        result_transform = sitk.CompositeTransform(transforms)
        
        transformToDisplacementFieldFilter = sitk.TransformToDisplacementFieldFilter()        
        transformToDisplacementFieldFilter.SetReferenceImage(image)
        transformToDisplacementFieldFilter.SetNumberOfThreads(16)
        new_locs = grid + torch.tensor(sitk.GetArrayFromImage(transformToDisplacementFieldFilter.Execute(result_transform))).unsqueeze(0).permute(0, 4, 1, 2, 3)
        shape = new_locs.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        result = F.grid_sample(input.to(self.device).unsqueeze(0).float(), new_locs.to(self.device).float(), align_corners=True, padding_mode="border", mode="nearest" if input.dtype == torch.uint8 else "bilinear").squeeze(0).cpu()
        return result.type(torch.uint8) if input.dtype == torch.uint8 else result
    
    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # TODO    
        return input

class Mask(Transform):

    def __init__(self, path : str = "default:./default.mha", value_outside: int = 0) -> None:
        self.path = path
        self.value_outside = value_outside
        
    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.path.endswith(".mha"):
            mask = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(self.path))).unsqueeze(0)
        else:
            mask = None
            for dataset in self.datasets:
                if dataset.isDatasetExist(self.path, name):
                    mask, _ = dataset.readData(self.path, name)
                    break
            if mask is None:
                raise NameError("Mask : {}/{} not found".format(self.path, name))
        return torch.where(torch.tensor(mask) > 0, input, self.value_outside)

    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input
    
class Gradient(Transform):

    def __init__(self, per_dim: bool = False):
        self.per_dim = per_dim
    
    @staticmethod
    def _image_gradient2D(image : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = image[:, 1:, :] - image[:, :-1, :]
        dy = image[:, :, 1:] - image[:, :, :-1]
        return torch.nn.ConstantPad2d((0,0,0,1), 0)(dx), torch.nn.ConstantPad2d((0,1,0,0), 0)(dy)

    @staticmethod
    def _image_gradient3D(image : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = image[:, 1:, :, :] - image[:, :-1, :, :]
        dy = image[:, :, 1:, :] - image[:, :, :-1, :]
        dz = image[:, :, :, 1:] - image[:, :, :, :-1]
        return torch.nn.ConstantPad3d((0,0,0,0,0,1), 0)(dx), torch.nn.ConstantPad3d((0,0,0,1,0,0), 0)(dy), torch.nn.ConstantPad3d((0,1,0,0,0,0), 0)(dz)
        
    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        result = torch.stack(Gradient._image_gradient3D(input) if len(input.shape) == 4 else Gradient._image_gradient2D(input), dim=1).squeeze(0)
        if not self.per_dim:
            result = torch.sigmoid(result*3)
            result = result.norm(dim=0)
            result = torch.unsqueeze(result, 0)
            
        return result

    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input

class ArgMax(Transform):

    def __init__(self, dim: int = 0) -> None:
        self.dim = dim
    
    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.argmax(input, dim=self.dim).unsqueeze(self.dim)
    
    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input
    
class FlatLabel(Transform):

    def __init__(self, labels: Union[list[int], None] = None) -> None:
        self.labels = labels

    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        data = torch.zeros_like(input)
        if self.labels:
            for label in self.labels:
                data[torch.where(input == label)] = 1
        else:
            data[torch.where(input > 0)] = 1
        return data

    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input

class Save(Transform):

    def __init__(self, save: str) -> None:
        self.save = save
    
    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input

    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input

class Flatten(Transform):

    def __init__(self) -> None:
        super().__init__()

    def transformShape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return [np.prod(np.asarray(shape))]

    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input.flatten()
    
    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input

class Permute(Transform):

    def __init__(self, dims: str = "1|0|2") -> None:
        super().__init__()
        self.dims = [0]+[int(d)+1 for d in dims.split("|")]

    def transformShape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return [shape[it-1] for it in self.dims[1:]]
    
    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input.permute(tuple(self.dims))
    
    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input.permute(tuple(np.argsort(self.dims)))
    
class Flip(Transform):

    def __init__(self, dims: str = "1|0|2") -> None:
        super().__init__()

        self.dims = [int(d)+1 for d in str(dims).split("|")]

    def __call__(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input.flip(tuple(self.dims))
    
    def inverse(self, name: str, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input.flip(tuple(self.dims))

class Canonical(Transform):

    def __init__(self) -> None:
        self.canonical_direction = torch.diag(torch.tensor([-1, -1, 1])).to(torch.double)

    def __call__(self, name: str, input: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        spacing = cache_attribute.get_tensor("Spacing")
        initial_matrix = cache_attribute.get_tensor("Direction").reshape(3,3).to(torch.double)
        initial_origin = cache_attribute.get_tensor("Origin")
        cache_attribute["Direction"] = (self.canonical_direction).flatten()
        matrix = _affine_matrix(self.canonical_direction @ initial_matrix.inverse(), torch.tensor([0, 0, 0]))
        center_voxel = torch.tensor([(input.shape[-i-1] - 1) * spacing[i] / 2 for i in range(3)], dtype=torch.double)
        center_physical = initial_matrix @ center_voxel + initial_origin
        cache_attribute["Origin"] = center_physical - (self.canonical_direction @ center_voxel)
        return _resample_affine(input, matrix.unsqueeze(0))
    
    def inverse(self, name: str, input: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        cache_attribute.pop("Direction")
        cache_attribute.pop("Origin")
        matrix = _affine_matrix((self.canonical_direction @ cache_attribute.get_tensor("Direction").to(torch.double).reshape(3,3).inverse()).inverse(), torch.tensor([0, 0, 0]))
        return _resample_affine(input, matrix.unsqueeze(0))

class HistogramMatching(Transform):

    def __init__(self, reference_group: str) -> None:
        self.reference_group = reference_group

    def __call__(self, name: str, input: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        image = data_to_image(input, cache_attribute)
        image_ref = None
        for dataset in self.datasets:
            if dataset.isDatasetExist(self.reference_group, name):
                image_ref = dataset.readImage(self.reference_group, name)
        if image_ref is None:
             raise NameError("Image : {}/{} not found".format(self.reference_group, name))
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(256)
        matcher.SetNumberOfMatchPoints(1)
        matcher.SetThresholdAtMeanIntensity(True)
        result, _ = image_to_data(matcher.Execute(image, image_ref))
        return torch.tensor(result)
    
    def inverse(self, name: str, input: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input

class SelectLabel(Transform):

    def __init__(self, labels: list[str]) -> None:
        self.labels = [l[1:-1].split(",") for l in labels]
    def __call__(self, name: str, input: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        data = torch.zeros_like(input)
        for old_label, new_label in self.labels:
            data[input == int(old_label)] = int(new_label)
        return data
    
    def inverse(self, name: str, input: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return input
    
class OneHot(Transform):
    
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def __call__(self, name: str, input: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        result =  F.one_hot(input.type(torch.int64), num_classes=self.num_classes).permute(0, len(input.shape), *[i+1 for i in range(len(input.shape)-1)]).float().squeeze(2)
        print(result.shape)
        return result
    
    def inverse(self, name: str, input: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.argmax(input, dim=1).unsqueeze(1)