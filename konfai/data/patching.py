import copy
import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import partial

import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
import torch.nn.functional as F  # noqa: N812

from konfai.data.augmentation import DataAugmentationsList
from konfai.data.transform import Save, Transform
from konfai.utils.config import apply_config, config
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.utils import get_module, get_patch_slices_from_shape


class PathCombine(ABC):

    def __init__(self) -> None:
        self.data: torch.Tensor
        self.overlap: int

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

        CAA+CAB+CBA+CBB
        ACA+ACB+BCA+BCB
        AAC+ABC+BAC+BBC

        CCA+CCB
        CAC+CBC
        ACC+BCC

    """

    def set_patch_config(self, patch_size: list[int], overlap: int):
        self.data = F.pad(
            torch.ones([size - overlap * 2 for size in patch_size]),
            [overlap] * 2 * len(patch_size),
            mode="constant",
            value=0,
        )
        self.data = self._set_function(self.data, overlap)
        dim = len(patch_size)

        a = slice(0, overlap)
        b = slice(-overlap, None)
        c = slice(overlap, -overlap)

        for i in range(dim):
            slices_badge = list(itertools.product(*[[a, b] for _ in range(dim - i)]))
            for indexs in itertools.combinations([0, 1, 2], i):
                result = []
                for slices_tuple in slices_badge:
                    slices_list = list(slices_tuple)
                    for index in indexs:
                        slices_list.insert(index, c)
                    result.append(tuple(slices_list))
                for patch, s in zip(PathCombine._normalise([self.data[s] for s in result]), result):
                    self.data[s] = patch

    @staticmethod
    def _normalise(patchs: list[torch.Tensor]) -> list[torch.Tensor]:
        data_sum = torch.sum(torch.concat([patch.unsqueeze(0) for patch in patchs], dim=0), dim=0)
        return [d / data_sum for d in patchs]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.data.repeat([tensor.shape[0]] + [1] * (len(tensor.shape) - 1)).to(tensor.device) * tensor

    @abstractmethod
    def _set_function(self, data: torch.Tensor, overlap: int) -> torch.Tensor:
        pass


class Mean(PathCombine):

    def __init__(self) -> None:
        super().__init__()

    def _set_function(self, data: torch.Tensor, overlap: int) -> torch.Tensor:
        return torch.ones_like(self.data)


class Cosinus(PathCombine):

    def __init__(self) -> None:
        super().__init__()

    def _function_sides(self, overlap: int, x: float):
        return np.clip(np.cos(np.pi / (2 * (overlap + 1)) * x), 0, 1)

    def _set_function(self, data: torch.Tensor, overlap: int) -> torch.Tensor:
        image = sitk.GetImageFromArray(np.asarray(data, dtype=np.uint8))
        danielsson_distance_map_image_filter = sitk.DanielssonDistanceMapImageFilter()
        distance = torch.tensor(sitk.GetArrayFromImage(danielsson_distance_map_image_filter.Execute(image)))
        return distance.apply_(partial(self._function_sides, overlap))


class Accumulator:

    def __init__(
        self,
        patch_slices: list[tuple[slice, ...]],
        patch_size: list[int],
        patch_combine: PathCombine | None = None,
        batch: bool = True,
    ) -> None:
        self._layer_accumulator: list[torch.Tensor | None] = [None] * len(patch_slices)
        self.patch_slices: list[tuple[slice, ...]] = []

        if patch_size is not None and not all(p == 0 for p in patch_size):
            for patch in patch_slices:
                slices: list[slice] = []
                for s, shape in zip(patch, patch_size):
                    slices.append(slice(s.start, s.start + shape))
                self.patch_slices.append(tuple(slices))
        else:
            self.patch_slices = patch_slices

        self.shape = max([[v.stop for v in patch] for patch in patch_slices])
        self.patch_size = patch_size
        self.patch_combine = patch_combine
        self.batch = batch

    def add_layer(self, index: int, layer: torch.Tensor) -> None:
        self._layer_accumulator[index] = layer

    def is_full(self) -> bool:
        return len(self.patch_slices) == len([v for v in self._layer_accumulator if v is not None])

    def assemble(self) -> torch.Tensor:
        n = 2 if self.batch else 1
        if self._layer_accumulator[0] is not None:
            result = torch.zeros(
                (
                    list(self._layer_accumulator[0].shape[:n])
                    + list(max([[v.stop for v in patch] for patch in self.patch_slices]))
                ),
                dtype=self._layer_accumulator[0].dtype,
            ).to(self._layer_accumulator[0].device)
        for patch_slice, data in zip(self.patch_slices, self._layer_accumulator):
            if data is not None:
                slices_dest = tuple([slice(result.shape[i]) for i in range(n)] + list(patch_slice))

                for dim, s in enumerate(patch_slice):
                    if s.stop - s.start == 1:
                        data = data.unsqueeze(dim=dim + n)
                if self.patch_combine is not None:
                    result[slices_dest] += self.patch_combine(data)
                else:
                    result[slices_dest] = data
        result = result[tuple([slice(None, None)] + [slice(0, s) for s in self.shape])]

        self._layer_accumulator.clear()
        return result


class Patch(ABC):

    @abstractmethod
    def __init__(
        self,
        patch_size: list[int],
        overlap: int | None,
        pad_value: float = 0,
        extend_slice: int = 0,
    ) -> None:
        self.patch_size = patch_size
        self.overlap = overlap
        if isinstance(self.overlap, int):
            if self.overlap < 0:
                self.overlap = None
        self._patch_slices: dict[int, list[tuple[slice, ...]]] = {}
        self._nb_patch_per_dim: dict[int, list[tuple[int, bool]]] = {}
        self.pad_value = pad_value
        self.extend_slice = extend_slice

    def load(self, shape: list[int], a: int = 0) -> None:
        self._patch_slices[a], self._nb_patch_per_dim[a] = get_patch_slices_from_shape(
            self.patch_size, shape, self.overlap
        )

    @abstractmethod
    def init(self, key: str):
        pass

    def get_patch_slices(self, a: int = 0):
        return self._patch_slices[a]

    def get_data(self, data: torch.Tensor, index: int, a: int, is_input: bool) -> list[torch.Tensor]:
        slices_pre = []
        for max_value in data.shape[: -len(self._patch_slices[a][0])]:
            slices_pre.append(slice(max_value))
        extend_slice = self.extend_slice if is_input else 0

        bottom = extend_slice // 2
        top = int(np.ceil(extend_slice / 2))
        s = slice(
            (
                self._patch_slices[a][index][0].start - bottom
                if self._patch_slices[a][index][0].start - bottom >= 0
                else 0
            ),
            (
                self._patch_slices[a][index][0].stop + top
                if self._patch_slices[a][index][0].stop + top <= data.shape[len(slices_pre)]
                else data.shape[len(slices_pre)]
            ),
        )
        slices = [s] + list(self._patch_slices[a][index][1:])
        data_sliced = data[tuple(slices_pre + slices)]
        if extend_slice > 0 and data_sliced.shape[len(slices_pre)] < bottom + top + 1:
            pad_bottom = 0
            pad_top = 0
            if self._patch_slices[a][index][0].start - bottom < 0:
                pad_bottom = bottom - self._patch_slices[a][index][0].start
            if self._patch_slices[a][index][0].stop + top > data.shape[len(slices_pre)]:
                pad_top = self._patch_slices[a][index][0].stop + top - data.shape[len(slices_pre)]
            data_sliced = F.pad(
                data_sliced,
                [0 for _ in range((len(slices) - 1) * 2)] + [pad_bottom, pad_top],
                "reflect",
            )

        padding = []
        if self.patch_size is not None and not all(p == 0 for p in self.patch_size):
            for dim_it, _slice in enumerate(reversed(slices)):
                p = (
                    0
                    if _slice.start + self.patch_size[-dim_it - 1] <= data.shape[-dim_it - 1]
                    else self.patch_size[-dim_it - 1] - (data.shape[-dim_it - 1] - _slice.start)
                )
                padding.append(0)
                padding.append(p)

        data_sliced = F.pad(
            data_sliced,
            tuple(padding),
            "constant",
            (0 if data_sliced.dtype == torch.uint8 and self.pad_value < 0 else self.pad_value),
        )

        if self.patch_size is not None and not all(p == 0 for p in self.patch_size):
            for d in [i for i, v in enumerate(reversed(self.patch_size)) if v == 1]:
                data_sliced = torch.squeeze(data_sliced, dim=len(data_sliced.shape) - d - 1)
        return (
            torch.cat([data_sliced[:, i, ...] for i in range(data_sliced.shape[1])], dim=0)
            if extend_slice > 0
            else data_sliced
        )

    def get_size(self, a: int = 0) -> int:
        return len(self._patch_slices[a])


@config("Patch")
class DatasetPatch(Patch):

    def __init__(
        self,
        patch_size: list[int] = [128, 128, 128],
        overlap: int | None = None,
        pad_value: float = 0,
        extend_slice: int = 0,
    ) -> None:
        super().__init__(patch_size, overlap, pad_value, extend_slice)

    def init(self, key: str = ""):
        pass


@config("Patch")
class ModelPatch(Patch):

    def __init__(
        self,
        patch_size: list[int] = [128, 128, 128],
        overlap: int | None = None,
        patch_combine: str | None = None,
        pad_value: float = 0,
        extend_slice: int = 0,
    ) -> None:
        super().__init__(patch_size, overlap, pad_value, extend_slice)
        self._patch_combine = patch_combine
        self.patch_combine: PathCombine | None = None

    def init(self, key: str):
        if self._patch_combine is not None:
            module, name = get_module(self._patch_combine, "konfai.data.patching")
            self.patch_combine = apply_config(key)(getattr(module, name))()
        if self.patch_size is not None and self.overlap is not None:
            if self.patch_combine is not None:
                self.patch_combine.set_patch_config([i for i in self.patch_size if i > 1], self.overlap)
        else:
            self.patch_combine = None

    def disassemble(self, *data_list: torch.Tensor) -> Iterator[list[torch.Tensor]]:
        for i in range(self.get_size()):
            yield [self.get_data(data, i, 0, True) for data in data_list]


class DatasetManager:

    def __init__(
        self,
        index: int,
        group_src: str,
        group_dest: str,
        name: str,
        dataset: Dataset,
        patch: DatasetPatch | None,
        transforms: list[Transform],
        data_augmentations_list: list[DataAugmentationsList],
    ) -> None:
        self.group_src = group_src
        self.group_dest = group_dest
        self.name = name
        self.index = index
        self.dataset = dataset
        self.loaded = False
        self.augmentationLoaded = False
        self.cache_attributes: list[Attribute] = []
        _shape, cache_attribute = self.dataset.get_infos(self.group_src, name)
        self.cache_attributes.append(cache_attribute)
        _shape = list(_shape[1:])

        self.data: list[torch.Tensor] = []

        for transform_function in transforms:
            _shape = transform_function.transform_shape(self.name, _shape, cache_attribute)

        self.patch = (
            DatasetPatch(
                patch_size=patch.patch_size,
                overlap=patch.overlap,
                pad_value=patch.pad_value,
                extend_slice=patch.extend_slice,
            )
            if patch
            else DatasetPatch(_shape)
        )
        self.patch.load(_shape, 0)
        self.shape = _shape
        self.data_augmentations_list = data_augmentations_list
        self.reset_augmentation()
        self.cache_attributes_bak = copy.deepcopy(self.cache_attributes)

    def reset_augmentation(self):
        self.cache_attributes[:] = self.cache_attributes[:1]
        i = 1
        for data_augmentations in self.data_augmentations_list:
            shape = []
            caches_attribute = []
            for _ in range(data_augmentations.nb):
                shape.append(self.shape)
                caches_attribute.append(copy.deepcopy(self.cache_attributes[0]))

            for data_augmentation in data_augmentations.data_augmentations:
                shape = data_augmentation.state_init(self.index, shape, caches_attribute)
            for it, s in enumerate(shape):
                self.cache_attributes.append(caches_attribute[it])
                self.patch.load(s, i)
                i += 1

    def load(
        self,
        pre_transform: list[Transform],
        data_augmentations_list: list[DataAugmentationsList],
    ) -> None:
        if not self.loaded:
            self._load(pre_transform)
        if not self.augmentationLoaded:
            self._load_augmentation(data_augmentations_list)

    def _load(self, pre_transform: list[Transform]):
        self.cache_attributes = copy.deepcopy(self.cache_attributes_bak)
        i = len(pre_transform)
        data = None
        for transform_function in reversed(pre_transform):
            if isinstance(transform_function, Save):
                if transform_function.dataset:
                    if len(transform_function.dataset.split(":")) > 1:
                        filename, file_format = transform_function.dataset.split(":")
                    else:
                        filename = transform_function.dataset
                        file_format = "mha"
                    dataset = Dataset(filename, file_format)
                else:
                    dataset = self.dataset
                group_dest = transform_function.group if transform_function.group else self.group_dest
                if dataset.is_dataset_exist(group_dest, self.name):
                    data, attrib = dataset.read_data(group_dest, self.name)
                    self.cache_attributes[0].update(attrib)
                    break
            i -= 1

        if i == 0:
            data, _ = self.dataset.read_data(self.group_src, self.name)

        data = torch.from_numpy(data)

        if len(pre_transform):
            for transform_function in pre_transform[i:]:
                data = transform_function(self.name, data, self.cache_attributes[0])
                if isinstance(transform_function, Save):
                    if transform_function.dataset:
                        if len(transform_function.dataset.split(":")) > 1:
                            filename, file_format = transform_function.dataset.split(":")
                        else:
                            filename = transform_function.dataset
                            file_format = "mha"
                        dataset = Dataset(filename, file_format)
                    else:
                        dataset = self.dataset
                    group_dest = transform_function.group if transform_function.group else self.group_dest
                    dataset.write(
                        group_dest,
                        self.name,
                        data.numpy(),
                        self.cache_attributes[0],
                    )
        self.data.append(data)

        for i in range(len(self.cache_attributes) - 1):
            self.cache_attributes[i + 1].update(self.cache_attributes[0])
        self.loaded = True

    def _load_augmentation(self, data_augmentations_list: list[DataAugmentationsList]) -> None:
        for data_augmentations in data_augmentations_list:
            a_data = [self.data[0].clone() for _ in range(data_augmentations.nb)]
            for data_augmentation in data_augmentations.data_augmentations:
                if data_augmentation.groups is None or self.group_dest in data_augmentation.groups:
                    a_data = data_augmentation(self.name, self.index, a_data)

            for d in a_data:
                self.data.append(d)
        self.augmentationLoaded = True

    def unload(self) -> None:
        self.data.clear()
        self.loaded = False
        self.augmentationLoaded = False

    def unload_augmentation(self) -> None:
        self.data[:] = self.data[:1]
        self.augmentationLoaded = False

    def get_data(self, index: int, a: int, patch_transforms: list[Transform], is_input: bool) -> torch.Tensor:
        data = self.patch.get_data(self.data[a], index, a, is_input)
        for transform_function in patch_transforms:
            data = transform_function(self.name, data, self.cache_attributes[a])
        return data

    def get_size(self, a: int) -> int:
        return self.patch.get_size(a)
