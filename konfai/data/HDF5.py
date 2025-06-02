from abc import ABC, abstractmethod
import SimpleITK as sitk
import numpy as np
import torch
import os
import torch.nn.functional as F
from typing import Any, Iterator
from typing import Union
import itertools
import copy 
from functools import partial
from konfai.utils.config import config
from konfai.utils.utils import get_patch_slices_from_shape
from konfai.utils.dataset import Dataset, Attribute
from konfai.data.transform import Transform, Save
from konfai.data.augmentation import DataAugmentationsList


class PathCombine(ABC):

    def __init__(self) -> None:
        self.data: torch.Tensor = None
        self.overlap: int = None
    
    """
    A = slice(0, overlap)
    B = slice(-overlap, None)
    C = slice(overlap, -overlap)
    
    1D 
        A+B
    2D : 
        AA+AB+BA+BB
        
        AC+BC
        CA+CB
    3D : 
        AAA+AAB+ABA+ABB+BAA+BAB+BBA+BBB
        
        AAC+ABC+BAC+BBC
        ACA+ACB+BCA+BCB
        CAA+CAB+CBA+CBB

        ACC+BCC
        CAC+CBC
        CCA+CCB
    
    """
    def setPatchConfig(self, patch_size: list[int], overlap: int):
        self.data = F.pad(torch.ones([size-overlap*2 for size in patch_size]), [overlap]*2*len(patch_size), mode="constant", value=0)
        self.data = self._setFunction(self.data, overlap)
        dim = len(patch_size)

        A = slice(0, overlap)
        B = slice(-overlap, None)
        C = slice(overlap, -overlap)
        
        for i in range(dim):
            slices_badge = list(itertools.product(*[[A, B] for _ in range(dim-i)]))
            for indexs in itertools.combinations([0,1,2], i):
                result = []
                for slices in slices_badge:
                    slices = list(slices)
                    for index in indexs:
                        slices.insert(index, C)    
                    result.append(tuple(slices))
                for patch, s in zip(PathCombine._normalise([self.data[s] for s in result]), result):
                    self.data[s] = patch


    @staticmethod
    def _normalise(patchs: list[torch.Tensor]) -> list[torch.Tensor]:
        data_sum = torch.sum(torch.concat([patch.unsqueeze(0) for patch in patchs], dim=0), dim=0)
        return [d/data_sum for d in patchs]
            
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return self.data.repeat([input.shape[0]]+[1]*(len(input.shape)-1)).to(input.device)*input
    
    @abstractmethod
    def _setFunction(self, data: torch.Tensor, overlap: int) -> torch.Tensor:
        pass

class Mean(PathCombine):

    @config("Mean")
    def __init__(self) -> None:
        super().__init__()

    def _setFunction(self, data: torch.Tensor, overlap: int) -> torch.Tensor:
        return torch.ones_like(self.data)
    
class Cosinus(PathCombine):

    @config("Cosinus")
    def __init__(self) -> None:
        super().__init__()

    def _function_sides(self, overlap: int, x: float):
        return np.clip(np.cos(np.pi/(2*(overlap+1))*x), 0, 1)

    def _setFunction(self, data: torch.Tensor, overlap: int) -> torch.Tensor:
        image = sitk.GetImageFromArray(np.asarray(data, dtype=np.uint8))
        danielssonDistanceMapImageFilter = sitk.DanielssonDistanceMapImageFilter()
        distance = torch.tensor(sitk.GetArrayFromImage(danielssonDistanceMapImageFilter.Execute(image)))
        return distance.apply_(partial(self._function_sides, overlap))
        
class Accumulator():

    def __init__(self, patch_slices: list[tuple[slice]], patch_size: list[int], patchCombine: Union[PathCombine, None] = None, batch: bool = True) -> None:
        self._layer_accumulator: list[Union[torch.Tensor, None]] = [None for i in range(len(patch_slices))]
        self.patch_slices = []
        for patch in patch_slices:
            slices = []
            for s, shape in zip(patch, patch_size):
                slices.append(slice(s.start, s.start+shape))
            self.patch_slices.append(tuple(slices))
        self.shape = max([[v.stop for v in patch] for patch in patch_slices])
        self.patch_size = patch_size
        self.patchCombine = patchCombine
        self.batch = batch

    def addLayer(self, index: int, layer: torch.Tensor) -> None:
        self._layer_accumulator[index] = layer
    
    def isFull(self) -> bool:
        return len(self.patch_slices) == len([v for v in self._layer_accumulator if v is not None])

    def assemble(self) -> torch.Tensor:
        if all([self._layer_accumulator[0].shape[i-len(self.patch_size)] == size for i, size in enumerate(self.patch_size)]): 
            N = 2 if self.batch else 1
            
            result = torch.zeros((list(self._layer_accumulator[0].shape[:N])+list(max([[v.stop for v in patch] for patch in self.patch_slices]))), dtype=self._layer_accumulator[0].dtype).to(self._layer_accumulator[0].device)
            for patch_slice, data in zip(self.patch_slices, self._layer_accumulator):
                slices_dest = tuple([slice(result.shape[i]) for i in range(N)] + list(patch_slice))

                for dim, s in enumerate(patch_slice):
                    if s.stop-s.start == 1:
                        data = data.unsqueeze(dim=dim+N)
                if self.patchCombine is not None:
                    result[slices_dest] += self.patchCombine(data)
                else:
                    result[slices_dest] = data
            result = result[tuple([slice(None, None)]+[slice(0, s) for s in self.shape])]
        else:
            result = torch.cat(tuple(self._layer_accumulator), dim=0)
        self._layer_accumulator.clear()
        return result

class Patch(ABC):

    def __init__(self, patch_size: list[int], overlap: Union[int, None], path_mask: Union[str, None] = None, padValue: float = 0, extend_slice: int = 0) -> None:
        self.patch_size = patch_size
        self.overlap = overlap
        self._patch_slices : dict[int, list[tuple[slice]]] = {}
        self._nb_patch_per_dim: dict[int, list[tuple[int, bool]]] = {}
        self.path_mask = path_mask
        self.mask = None
        self.padValue = padValue
        self.extend_slice = extend_slice
        if self.path_mask is not None:
            if os.path.exists(self.path_mask):
                self.mask = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(self.path_mask)))
            else:
                raise NameError('Mask file not found')
            
    def load(self, shape : dict[int, list[int]], a: int = 0) -> None:
        self._patch_slices[a], self._nb_patch_per_dim[a] = get_patch_slices_from_shape(self.patch_size, shape, self.overlap)

    def getPatch_slices(self, a: int = 0):
        return self._patch_slices[a]
    
    @abstractmethod
    def getData(self, data : torch.Tensor, index : int, a: int, isInput: bool) -> torch.Tensor:
        pass
    
    def getData(self, data : torch.Tensor, index : int, a: int, isInput: bool) -> list[torch.Tensor]:
        slices_pre = []
        for max in data.shape[:-len(self.patch_size)]:
            slices_pre.append(slice(max))
        extend_slice = self.extend_slice if isInput else 0

        bottom = extend_slice//2
        top = int(np.ceil(extend_slice/2))
        s = slice(self._patch_slices[a][index][0].start-bottom if self._patch_slices[a][index][0].start-bottom >= 0 else 0, self._patch_slices[a][index][0].stop+top if self._patch_slices[a][index][0].stop+top <= data.shape[len(slices_pre)] else data.shape[len(slices_pre)])
        slices =  [s] + list(self._patch_slices[a][index][1:])
        data_sliced = data[slices_pre+slices]
        if data_sliced.shape[len(slices_pre)] < bottom+top+1:
            pad_bottom = 0
            pad_top = 0
            if self._patch_slices[a][index][0].start-bottom < 0:
                pad_bottom = bottom-s.start
            if self._patch_slices[a][index][0].stop+top > data.shape[len(slices_pre)]:
                pad_top = self._patch_slices[a][index][0].stop+top-data.shape[len(slices_pre)]
            data_sliced = F.pad(data_sliced, [0 for _ in range((len(slices)-1)*2)]+[pad_bottom, pad_top], 'reflect')

        padding = []
        for dim_it, _slice in enumerate(reversed(slices)):
            p = 0 if _slice.start+self.patch_size[-dim_it-1] <= data.shape[-dim_it-1] else self.patch_size[-dim_it-1]-(data.shape[-dim_it-1]-_slice.start)
            padding.append(0)
            padding.append(p)

        data_sliced = F.pad(data_sliced, tuple(padding), "constant", 0 if data_sliced.dtype == torch.uint8 and self.padValue < 0 else self.padValue)
        if self.mask is not None:
            outside = torch.ones_like(data_sliced)*(0 if data_sliced.dtype == torch.uint8 and self.padValue < 0 else self.padValue)
            data_sliced = torch.where(self.mask == 0, outside, data_sliced)
        
        for d in [i for i, v in enumerate(reversed(self.patch_size)) if v == 1]:
            data_sliced = torch.squeeze(data_sliced, dim = len(data_sliced.shape)-d-1)
        return torch.cat([data_sliced[:, i, ...] for i in range(data_sliced.shape[1])], dim=0) if extend_slice > 0 else data_sliced

    def getSize(self, a: int = 0) -> int:
        return len(self._patch_slices[a])

class DatasetPatch(Patch):

    @config("Patch")
    def __init__(self, patch_size : list[int] = [128, 256, 256], overlap : Union[int, None] = None, mask: Union[str, None] = None, padValue: float = 0, extend_slice: int = 0) -> None:
        super().__init__(patch_size, overlap, mask, padValue, extend_slice)

class ModelPatch(Patch):

    @config("Patch")
    def __init__(self, patch_size : list[int] = [128, 256, 256], overlap : Union[int, None] = None, patchCombine: Union[str, None] = None, mask: Union[str, None] = None, padValue: float = 0, extend_slice: int = 0) -> None:
        super().__init__(patch_size, overlap, mask, padValue, extend_slice)
        self.patchCombine = patchCombine

    def disassemble(self, *dataList: torch.Tensor) -> Iterator[list[torch.Tensor]]:
        for i in range(self.getSize()):
            yield [self.getData(data, i, 0, True) for data in dataList]

class DatasetManager():

    def __init__(self, index: int, group_src: str, group_dest : str, name: str, dataset : Dataset, patch : Union[DatasetPatch, None], pre_transforms : list[Transform], dataAugmentationsList : list[DataAugmentationsList]) -> None:
        self.group_src = group_src
        self.group_dest = group_dest
        self.name = name
        self.index = index
        self.dataset = dataset
        self.loaded = False
        self.cache_attributes: list[Attribute] = []
        _shape, cache_attribute =  self.dataset.getInfos(self.group_src, name)
        self.cache_attributes.append(cache_attribute)
        _shape = list(_shape[1:])
        
        self.data : list[torch.Tensor] = list()

        for transformFunction in pre_transforms:
            _shape = transformFunction.transformShape(_shape, cache_attribute)
        
        self.patch = DatasetPatch(patch_size=patch.patch_size, overlap=patch.overlap, mask=patch.path_mask, padValue=patch.padValue, extend_slice=patch.extend_slice) if patch else DatasetPatch(_shape)
        self.patch.load(_shape, 0)
        self.shape = _shape
        self.dataAugmentationsList = dataAugmentationsList
        self.resetAugmentation()
        self.cache_attributes_bak = copy.deepcopy(self.cache_attributes)
    
    def resetAugmentation(self):
        i = 1
        for dataAugmentations in self.dataAugmentationsList:
            shape = []
            caches_attribute = []
            for _ in range(dataAugmentations.nb):
                shape.append(self.shape)
                caches_attribute.append(copy.deepcopy(self.cache_attributes[0]))

            for dataAugmentation in dataAugmentations.dataAugmentations:
                shape = dataAugmentation.state_init(self.index, shape, caches_attribute)
            for it, s in enumerate(shape):
                self.cache_attributes.append(caches_attribute[it])
                self.patch.load(s, i)
                i+=1

    def load(self, pre_transform : list[Transform], dataAugmentationsList : list[DataAugmentationsList], device: torch.device) -> None:
        if self.loaded:
            return
        i = len(pre_transform)
        data = None
        for transformFunction in reversed(pre_transform):
            if isinstance(transformFunction, Save):
                filename, format = transformFunction.save.split(":")
                dataset = Dataset(filename, format)
                if dataset.isDatasetExist(self.group_dest, self.name):
                    data, attrib = dataset.readData(self.group_dest, self.name)
                    self.cache_attributes[0].update(attrib)
                    break
            i-=1
        
        if i==0:
            data, _ = self.dataset.readData(self.group_src, self.name)


        data = torch.from_numpy(data)

        if len(pre_transform):
            for transformFunction in pre_transform[i:]:
                data = transformFunction(self.name, data, self.cache_attributes[0])
                if isinstance(transformFunction, Save):
                    filename, format = transformFunction.save.split(":")
                    dataset = Dataset(filename, format)
                    dataset.write(self.group_dest, self.name, data.numpy(), self.cache_attributes[0])
        self.data : list[torch.Tensor] = list()
        self.data.append(data)

        for dataAugmentations in dataAugmentationsList:
            a_data = [data.clone() for _ in range(dataAugmentations.nb)]
            for dataAugmentation in dataAugmentations.dataAugmentations:
                a_data = dataAugmentation(self.index, a_data, device)
            
            for d in a_data:
                self.data.append(d)
        self.loaded = True

    def unload(self) -> None:
        if hasattr(self, "data"):
            del self.data
        self.cache_attributes = copy.deepcopy(self.cache_attributes_bak)
        self.loaded = False
    
    def getData(self, index : int, a : int, post_transforms : list[Transform], isInput: bool) -> torch.Tensor:
        data = self.patch.getData(self.data[a], index, a, isInput)
        for transformFunction in post_transforms:
            data = transformFunction(self.name, data, self.cache_attributes[a])
        return data

    def getSize(self, a: int) -> int:
        return self.patch.getSize(a)
