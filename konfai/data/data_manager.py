import math
import os
import random
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import cast

import numpy as np
import torch
import tqdm
from torch.cuda import device_count
from torch.utils import data
from torch.utils.data import DataLoader, Sampler

from konfai import konfai_root, konfai_state
from konfai.data.augmentation import DataAugmentationsList
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import Transform, TransformLoader
from konfai.utils.config import config
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.utils import (
    SUPPORTED_EXTENSIONS,
    DatasetManagerError,
    State,
    get_cpu_info,
    get_memory,
    get_memory_info,
    memory_forecast,
)


class GroupTransform:

    def __init__(
        self,
        transforms: dict[str, TransformLoader] | None = {
            "default|Normalize|Standardize|Unsqueeze|TensorCast|ResampleIsotropic|ResampleResize": TransformLoader()
        },
        patch_transforms: dict[str, TransformLoader] | None = {
            "default|Normalize|Standardize|Unsqueeze|TensorCast|ResampleIsotropic|ResampleResize": TransformLoader()
        },
        is_input: bool = True,
    ) -> None:
        self._transforms = transforms
        self._patch_transforms = patch_transforms
        self.transforms: list[Transform] = []
        self.patch_transforms: list[Transform] = []
        self.is_input = is_input

    def load(self, group_src: str, group_dest: str, datasets: list[Dataset]):
        if self._transforms is not None:
            for classpath, transform_loader in self._transforms.items():
                transform = transform_loader.get_transform(
                    classpath,
                    konfai_args=f"{konfai_root()}.Dataset.groups_src.{group_src}.groups_dest.{group_dest}.transforms",
                )
                transform.set_datasets(datasets)
                self.transforms.append(transform)
        if self._patch_transforms is not None:
            for classpath, transform_loader in self._patch_transforms.items():
                transform = transform_loader.get_transform(
                    classpath,
                    konfai_args=f"{konfai_root()}.Dataset.groups_src.{group_src}"
                    f".groups_dest.{group_dest}.patch_transforms",
                )
                transform.set_datasets(datasets)
                self.patch_transforms.append(transform)

    def to(self, device: int):
        for transform in self.transforms:
            transform.to(device)
        for transform in self.patch_transforms:
            transform.to(device)

    def __str__(self) -> str:
        params = {"transforms": self.transforms, "patch_transforms": self.patch_transforms}
        return str(params)

    def __repr__(self) -> str:
        return str(self)


class GroupTransformMetric(GroupTransform):

    def __init__(
        self,
        transforms: dict[str, TransformLoader] = {
            "default|Normalize|Standardize|Unsqueeze|TensorCast|ResampleIsotropic|ResampleResize": TransformLoader()
        },
    ):
        super().__init__(transforms, {})


class Group(dict[str, GroupTransform]):

    def __init__(
        self,
        groups_dest: dict[str, GroupTransform] = {"default|Labels": GroupTransform()},
    ):
        super().__init__(groups_dest)


class GroupMetric(dict[str, GroupTransformMetric]):

    def __init__(
        self,
        groups_dest: dict[str, GroupTransformMetric] = {"default|group_dest": GroupTransformMetric()},
    ):
        super().__init__(groups_dest)


class CustomSampler(Sampler[int]):

    def __init__(self, size: int, shuffle: bool = False) -> None:
        self.size = size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        return iter(torch.randperm(len(self)).tolist() if self.shuffle else list(range(len(self))))

    def __len__(self) -> int:
        return self.size


class DatasetIter(data.Dataset):

    def __init__(
        self,
        rank: int,
        data: dict[str, list[DatasetManager]],
        mapping: list[tuple[int, int, int]],
        groups_src: Mapping[str, Group | GroupMetric],
        inline_augmentations: bool,
        data_augmentations_list: list[DataAugmentationsList],
        patch_size: list[int] | None,
        overlap: int | None,
        buffer_size: int,
        use_cache=True,
    ) -> None:
        self.rank = rank
        self.data = data
        self.mapping = mapping
        self.patch_size = patch_size
        self.overlap = overlap
        self.groups_src = groups_src
        self.data_augmentations_list = data_augmentations_list
        self.use_cache = use_cache
        self.nb_dataset = len(data[list(data.keys())[0]])
        self.buffer_size = buffer_size
        self._index_cache: list[int] = []
        self.inline_augmentations = inline_augmentations

    def get_patch_config(self) -> tuple[list[int] | None, int | None]:
        return self.patch_size, self.overlap

    def to(self, device: int):
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.groups_src[group_src][group_dest].to(device)
        for data_augmentations in self.data_augmentations_list:
            for data_augmentation in data_augmentations.data_augmentations:
                data_augmentation.to(device)

    def get_dataset_from_index(self, group_dest: str, index: int) -> DatasetManager:
        return self.data[group_dest][index]

    def reset_augmentation(self, label):
        if self.inline_augmentations and len(self.data_augmentations_list) > 0:
            for index in range(self.nb_dataset):
                for group_src in self.groups_src:
                    for group_dest in self.groups_src[group_src]:
                        self.data[group_dest][index].unload_augmentation()
                        self.data[group_dest][index].reset_augmentation()
            self.load(label + " Augmentation")

    def load(self, label: str):
        if self.use_cache:
            memory_init = get_memory()

            indexs = list(range(self.nb_dataset))
            if len(indexs) > 0:
                memory_lock = threading.Lock()

                def desc(i: int = 0):
                    return (
                        f"Caching {label}: "
                        f"{get_memory_info()} | "
                        f"{memory_forecast(memory_init, i, self.nb_dataset)} | "
                        f"{get_cpu_info()}"
                    )

                pbar = tqdm.tqdm(total=len(indexs), desc=desc(), leave=False)

                def process(index):
                    self._load_data(index)
                    with memory_lock:
                        pbar.set_description(desc(pbar.n + 1))
                        pbar.update(1)

                cpu_count = os.cpu_count() or 1
                with ThreadPoolExecutor(
                    max_workers=cpu_count // (device_count() if device_count() > 0 else 1)
                ) as executor:
                    futures = [executor.submit(process, index) for index in indexs]
                    for _ in as_completed(futures):
                        pass

                pbar.close()

    def _load_data(self, index):
        if index not in self._index_cache:
            self._index_cache.append(index)
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.load_data(group_src, group_dest, index)

    def load_data(self, group_src: str, group_dest: str, index: int) -> None:
        self.data[group_dest][index].load(
            self.groups_src[group_src][group_dest].transforms,
            self.data_augmentations_list,
        )

    def _unload_data(self, index: int) -> None:
        if index in self._index_cache:
            self._index_cache.remove(index)
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.unload_data(group_dest, index)

    def unload_data(self, group_dest: str, index: int) -> None:
        return self.data[group_dest][index].unload()

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, index: int) -> dict[str, tuple[torch.Tensor, int, int, int, str, bool]]:
        data = {}
        x, a, p = self.mapping[index]
        if x not in self._index_cache:
            if len(self._index_cache) >= self.buffer_size and not self.use_cache:
                self._unload_data(self._index_cache[0])
            self._load_data(x)

        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                dataset = self.data[group_dest][x]
                data[f"{group_dest}"] = (
                    dataset.get_data(
                        p,
                        a,
                        self.groups_src[group_src][group_dest].patch_transforms,
                        self.groups_src[group_src][group_dest].is_input,
                    ),
                    x,
                    a,
                    p,
                    dataset.name,
                    self.groups_src[group_src][group_dest].is_input,
                )
        return data


class Subset:

    def __init__(
        self,
        subset: str | list[int] | list[str] | None = None,
        shuffle: bool = True,
    ) -> None:
        self.subset = subset
        self.shuffle = shuffle

    def _get_index(self, subset: str | int, names: list[str]) -> list[int]:
        size = len(names)
        index = []
        if isinstance(subset, int):
            index.append(subset)
        elif ":" in subset:
            r = np.clip(
                np.asarray([int(subset.split(":")[0]), int(subset.split(":")[1])]),
                0,
                size,
            )
            index = list(range(r[0], r[1]))
        elif os.path.exists(subset):
            train_names = []
            with open(subset) as f:
                for name in f:
                    train_names.append(name.strip())
            index = []
            for i, name in enumerate(names):
                if name in train_names:
                    index.append(i)
        elif subset.startswith("~") and os.path.exists(subset[1:]):
            exclude_names = []
            with open(subset[1:]) as f:
                for name in f:
                    exclude_names.append(name.strip())
            index = []
            for i, name in enumerate(names):
                if name not in exclude_names:
                    index.append(i)
        return index

    def __call__(self, names: list[str], infos: dict[str, tuple[list[int], Attribute]]) -> set[str]:
        names = sorted(names)
        size = len(names)

        if self.subset is None:
            index = list(range(0, size))
        elif isinstance(self.subset, list):
            index_set: set[int] = set()
            for s in self.subset:
                if len(index_set) == 0:
                    index_set.update(set(self._get_index(s, names)))
                else:
                    index_set = index_set.intersection(set(self._get_index(s, names)))
                index = list(index_set)
        else:
            index = self._get_index(self.subset, names)
        if self.shuffle:
            index = random.sample(index, len(index))  # nosec B311
        return {names[i] for i in index}

    def __str__(self):
        return "Subset : " + str(self.subset) + " shuffle : " + str(self.shuffle)


class TrainSubset(Subset):

    def __init__(
        self,
        subset: str | list[int] | list[str] | None = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__(subset, shuffle)


class PredictionSubset(Subset):

    def __init__(self, subset: str | list[int] | list[str] | None = None) -> None:
        super().__init__(subset, False)


class Data(ABC):

    @abstractmethod
    def __init__(
        self,
        dataset_filenames: list[str],
        groups_src: Mapping[str, Group | GroupMetric],
        patch: DatasetPatch | None,
        use_cache: bool,
        subset: Subset,
        batch_size: int,
        validation: float | str | list[int] | list[str] | None,
        inline_augmentations: bool,
        data_augmentations_list: dict[str, DataAugmentationsList],
    ) -> None:
        self.dataset_filenames = dataset_filenames
        self.subset = subset
        self.groups_src = groups_src
        self.patch = patch
        self.validation = validation
        self.data_augmentations_list = data_augmentations_list
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.inline_augmentations = inline_augmentations

        self.datasetIter = partial(
            DatasetIter,
            groups_src=self.groups_src,
            inline_augmentations=inline_augmentations,
            data_augmentations_list=list(self.data_augmentations_list.values()),
            patch_size=self.patch.patch_size if self.patch is not None else None,
            overlap=self.patch.overlap if self.patch is not None else None,
            buffer_size=batch_size + 1,
            use_cache=use_cache,
        )
        self.dataLoader_args = {
            "num_workers": 0,
            "pin_memory": False,
        }
        self.data: list[list[dict[str, list[DatasetManager]]]] = []
        self.mapping: list[list[list[tuple[int, int, int]]]] = []
        self.datasets: dict[str, Dataset] = {}

    def _get_datasets(
        self, names: list[str], dataset_name: dict[str, dict[str, list[str]]]
    ) -> tuple[dict[str, list[DatasetManager]], list[tuple[int, int, int]]]:
        nb_dataset = len(names)
        nb_patch: list[list[int]]
        data = {}
        mapping = []
        nb_augmentation = np.max(
            [
                int(np.sum([data_augmentation.nb for data_augmentation in self.data_augmentations_list.values()]) + 1),
                1,
            ]
        )

        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                data[group_dest] = [
                    DatasetManager(
                        i,
                        group_src,
                        group_dest,
                        name,
                        self.datasets[
                            [filename for filename, names in dataset_name[group_src].items() if name in names][0]
                        ],
                        patch=self.patch,
                        transforms=self.groups_src[group_src][group_dest].transforms,
                        data_augmentations_list=list(self.data_augmentations_list.values()),
                    )
                    for i, name in enumerate(names)
                ]
                nb_patch = [[dataset.get_size(a) for a in range(nb_augmentation)] for dataset in data[group_dest]]

        for x in range(nb_dataset):
            for y in range(nb_augmentation):
                for z in range(nb_patch[x][y]):
                    mapping.append((x, y, z))
        return data, mapping

    def get_groups_dest(self):
        groups_dest = []
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                groups_dest.append(group_dest)
        return groups_dest

    @staticmethod
    def _split(mapping: list[tuple[int, int, int]], world_size: int) -> list[list[tuple[int, int, int]]]:
        if len(mapping) == 0:
            return [[] for _ in range(world_size)]

        mappings = []
        if konfai_state() == str(State.PREDICTION) or konfai_state() == str(State.EVALUATION):
            np_mapping = np.asarray(mapping)
            unique_index = np.unique(np_mapping[:, 0])
            offset = int(np.ceil(len(unique_index) / world_size))
            if offset == 0:
                offset = 1
            for itr in range(0, len(unique_index), offset):
                mappings.append(
                    [
                        tuple(v)
                        for v in np_mapping[
                            np.where(np.isin(np_mapping[:, 0], unique_index[itr : itr + offset]))[0],
                            :,
                        ]
                    ]
                )
        else:
            offset = int(np.ceil(len(mapping) / world_size))
            if offset == 0:
                offset = 1
            for itr in range(0, len(mapping), offset):
                mappings.append(list(mapping[-offset:]) if itr + offset > len(mapping) else mapping[itr : itr + offset])
        return mappings

    def get_data(self, world_size: int) -> tuple[list[list[DataLoader]], list[str], list[str]]:
        datasets: dict[str, list[tuple[str, bool]]] = {}
        if self.dataset_filenames is None or len(self.dataset_filenames) == 0:
            raise DatasetManagerError("No dataset filenames were provided")
        for dataset_filename in self.dataset_filenames:
            if dataset_filename is None:
                raise DatasetManagerError(
                    "Invalid dataset entry: 'None' received.",
                    "Each dataset must be a valid path string (e.g., './Dataset/', './Dataset/:mha, "
                    "'./Dataset/:a:mha', './Dataset/:i:mha').",
                    "Please check your 'dataset_filenames' list for missing or null entries.",
                )
            if len(dataset_filename.split(":")) == 1:
                filename = dataset_filename
                file_format = "mha"
                append = True
            elif len(dataset_filename.split(":")) == 2:
                filename, file_format = dataset_filename.split(":")
                append = True
            else:
                filename, flag, file_format = dataset_filename.split(":")
                append = flag == "a"

            if file_format not in SUPPORTED_EXTENSIONS:
                raise DatasetManagerError(
                    f"Unsupported file format '{file_format}'.",
                    f"Supported extensions are: {', '.join(SUPPORTED_EXTENSIONS)}",
                )

            dataset = Dataset(filename, file_format)

            self.datasets[filename] = dataset
            for group in self.groups_src:
                if dataset.is_group_exist(group):
                    if group in datasets:
                        datasets[group].append((filename, append))
                    else:
                        datasets[group] = [(filename, append)]
        model_have_input = False
        for group_src in self.groups_src:
            if group_src not in datasets:

                raise DatasetManagerError(
                    f"Group source '{group_src}' not found in any dataset.",
                    f"Dataset filenames provided: {self.dataset_filenames}",
                    f"Available groups across all datasets: "
                    f"{[f'{f} {d.get_group()}' for f, d in self.datasets.items()]}\n"
                    f"Please check that an entry in the dataset with the name '{group_src}' exists.",
                )

            for group_dest in self.groups_src[group_src]:
                self.groups_src[group_src][group_dest].load(
                    group_src,
                    group_dest,
                    list(self.datasets.values()),
                )
                model_have_input |= self.groups_src[group_src][group_dest].is_input
        if self.patch is not None:
            self.patch.init()

        if not model_have_input:
            raise DatasetManagerError(
                "At least one group must be defined with 'is_input: true' to provide input to the network."
            )

        for key, data_augmentations in self.data_augmentations_list.items():
            data_augmentations.load(key, [self.datasets[filename] for filename, _ in datasets[group_src]])

        names: set[str] = set()
        dataset_name: dict[str, dict[str, list[str]]] = {}
        dataset_info: dict[str, dict[str, dict[str, tuple[list[int], Attribute]]]] = {}
        for group in self.groups_src:
            names_by_group = set()
            if group not in dataset_name:
                dataset_name[group] = {}
                dataset_info[group] = {}
            for filename, _ in datasets[group]:
                names_by_group.update(self.datasets[filename].get_names(group))
                dataset_name[group][filename] = self.datasets[filename].get_names(group)
                dataset_info[group][filename] = {
                    name: self.datasets[filename].get_infos(group, name) for name in dataset_name[group][filename]
                }
            if len(names) == 0:
                names.update(names_by_group)
            else:
                names = names.intersection(names_by_group)
        if len(names) == 0:
            raise DatasetManagerError(
                f"No data was found for groups {list(self.groups_src.keys())}: although each group contains data "
                "from a dataset, there are no common dataset names shared across all groups, the intersection is empty."
            )

        subset_names: set[str] = set()
        for group in dataset_name:
            subset_names_bygroup: set[str] = set()
            for filename, append in datasets[group]:
                if append:
                    subset_names_bygroup.update(
                        self.subset(
                            dataset_name[group][filename],
                            dataset_info[group][filename],
                        )
                    )
                else:
                    if len(subset_names_bygroup) == 0:
                        subset_names_bygroup.update(
                            self.subset(
                                dataset_name[group][filename],
                                dataset_info[group][filename],
                            )
                        )
                    else:
                        subset_names_bygroup = subset_names_bygroup.intersection(
                            self.subset(
                                dataset_name[group][filename],
                                dataset_info[group][filename],
                            )
                        )
            if len(subset_names) == 0:
                subset_names.update(subset_names_bygroup)
            else:
                subset_names = subset_names.intersection(subset_names_bygroup)

        if len(subset_names) == 0:
            raise DatasetManagerError(
                "All data entries were excluded by the subset filter.",
                f"Dataset entries found: {', '.join(names)}",
                f"Subset object applied: {self.subset}",
                f"Subset requested : {', '.join(subset_names)}",
                "None of the dataset entries matched the given subset.",
                "Please check your 'subset' configuration — it may be too restrictive or incorrectly formatted.",
                "Examples of valid subset formats:",
                "\tsubset: [0, 1]            # explicit indices",
                "\tsubset: 0:10              # slice notation",
                "\tsubset: ./Validation.txt  # external file",
                "\tsubset: None              # to disable filtering",
            )

        data, mapping = self._get_datasets(list(subset_names), dataset_name)

        index = []
        if isinstance(self.validation, float):
            if self.validation <= 0 or self.validation >= 1:
                raise DatasetManagerError(
                    "Validation must be a float between 0 and 1.",
                    f"Received: {self.validation}",
                    "Example: validation = 0.2  # for a 20% validation split",
                )
            index = [m[0] for m in mapping[int(math.floor(len(mapping) * (1 - self.validation))) :]]
        elif isinstance(self.validation, str):
            if ":" in self.validation:
                index = list(range(int(self.validation.split(":")[0]), int(self.validation.split(":")[1])))
            elif os.path.exists(self.validation):
                validation_names = []
                with open(self.validation) as f:
                    for name in f:
                        validation_names.append(name.strip())
                index = [i for i, n in enumerate(subset_names) if n in validation_names]
            else:
                raise DatasetManagerError(
                    f"Invalid string value for 'validation': '{self.validation}'",
                    "Expected one of the following formats:",
                    "\t• A slice string like '0:10'",
                    "\t• A path to a text file listing validation sample names (e.g., './val.txt')",
                    "\t• A float between 0 and 1 (e.g., 0.2)",
                    "\t• A list of sample names or indices",
                    "The provided value is neither a valid slice nor a readable file.",
                    "Please fix your 'validation' setting in the configuration.",
                )
        elif isinstance(self.validation, list):
            if isinstance(self.validation[0], int):
                index = cast(list[int], self.validation)
            elif isinstance(self.validation[0], str):
                index = [i for i, n in enumerate(subset_names) if n in self.validation]
            else:
                raise DatasetManagerError(
                    "Invalid list type for 'validation': elements of type "
                    f"'{type(self.validation[0]).__name__}' are not supported.",
                    "Supported list element types are:",
                    "\t• int  → list of indices (e.g., [0, 1, 2])",
                    "\t• str  → list of sample names (e.g., ['patient01', 'patient02'])",
                    f"Received list: {self.validation}",
                )
        train_mapping = [m for m in mapping if m[0] not in index]
        validate_mapping = [m for m in mapping if m[0] in index]

        if len(train_mapping) == 0:
            raise DatasetManagerError(
                "No data left for training after applying the validation split.",
                f"Dataset size: {len(mapping)}",
                f"Validation setting: {self.validation}",
                "Please reduce the validation size, increase the dataset, or disable validation.",
            )

        if self.validation is not None and len(validate_mapping) == 0:
            raise DatasetManagerError(
                "No data left for validation after applying the validation split.",
                f"Dataset size: {len(mapping)}",
                f"Validation setting: {self.validation}",
                "Please increase the validation size, increase the dataset, or disable validation.",
            )

        validation_names = [name for i, name in enumerate(subset_names) if i in index]
        train_names = [name for name in subset_names if name not in validation_names]
        train_mappings = Data._split(train_mapping, world_size)
        validate_mappings = Data._split(validate_mapping, world_size)

        for i, (train_mapping, validate_mapping) in enumerate(zip(train_mappings, validate_mappings)):
            mappings = [train_mapping]
            if len(validate_mapping):
                mappings += [validate_mapping]
            self.data.append([])
            self.mapping.append([])
            for mapping_tmp in mappings:
                indexs = np.unique(np.asarray(mapping_tmp)[:, 0])
                self.data[i].append({k: [v[it] for it in indexs] for k, v in data.items()})
                mapping_tmp_array = np.asarray(mapping_tmp)
                for a, b in enumerate(indexs):
                    mapping_tmp_array[np.where(np.asarray(mapping_tmp_array)[:, 0] == b), 0] = a
                self.mapping[i].append([(a, b, c) for a, b, c in mapping_tmp_array])

        data_loaders: list[list[DataLoader]] = []
        for i, (datas, mappings) in enumerate(zip(self.data, self.mapping)):
            data_loaders.append([])
            for data, mapping in zip(datas, mappings):
                data_loaders[i].append(
                    DataLoader(
                        dataset=self.datasetIter(
                            rank=i,
                            data=data,
                            mapping=mapping,
                        ),
                        sampler=CustomSampler(len(mapping), self.subset.shuffle),
                        batch_size=self.batch_size,
                        **self.dataLoader_args,
                    )
                )
        return data_loaders, train_names, validation_names

    def __str__(self) -> str:
        params = {
            "dataset_filenames": self.dataset_filenames,
            "groups_src": self.groups_src,
            "patch": self.patch,
            "use_cache": self.use_cache,
            "subset": self.subset,
            "batch_size": self.batch_size,
            "validation": self.validation,
            "inline_augmentations": self.inline_augmentations,
            "data_augmentations_list": self.data_augmentations_list,
        }
        return str(params)

    def __repr__(self) -> str:
        return str(self)


@config("Dataset")
class DataTrain(Data):

    def __init__(
        self,
        dataset_filenames: list[str] = ["default|./Dataset:mha"],
        groups_src: dict[str, Group] = {"default|Labels": Group()},
        augmentations: dict[str, DataAugmentationsList] | None = {"DataAugmentation_0": DataAugmentationsList()},
        inline_augmentations: bool = False,
        patch: DatasetPatch | None = DatasetPatch(),
        use_cache: bool = True,
        subset: TrainSubset = TrainSubset(),
        batch_size: int = 1,
        validation: float | str | list[int] | list[str] = 0.2,
    ) -> None:
        super().__init__(
            dataset_filenames,
            groups_src,
            patch,
            use_cache,
            subset,
            batch_size,
            validation,
            inline_augmentations,
            augmentations if augmentations else {},
        )


@config("Dataset")
class DataPrediction(Data):

    def __init__(
        self,
        dataset_filenames: list[str] = ["default|./Dataset"],
        groups_src: dict[str, Group] = {"default": Group()},
        augmentations: dict[str, DataAugmentationsList] | None = {"DataAugmentation_0": DataAugmentationsList()},
        patch: DatasetPatch | None = DatasetPatch(),
        subset: PredictionSubset = PredictionSubset(),
        batch_size: int = 1,
    ) -> None:

        super().__init__(
            dataset_filenames=dataset_filenames,
            groups_src=groups_src,
            patch=patch,
            use_cache=False,
            subset=subset,
            batch_size=batch_size,
            validation=None,
            inline_augmentations=False,
            data_augmentations_list=augmentations if augmentations else {},
        )


@config("Dataset")
class DataMetric(Data):

    def __init__(
        self,
        dataset_filenames: list[str] = ["default|./Dataset:mha"],
        groups_src: dict[str, GroupMetric] = {"default": GroupMetric()},
        subset: PredictionSubset = PredictionSubset(),
        validation: str | None = None,
    ) -> None:

        super().__init__(
            dataset_filenames=dataset_filenames,
            groups_src=groups_src,
            patch=None,
            use_cache=True,
            subset=subset,
            batch_size=1,
            validation=validation,
            data_augmentations_list={},
            inline_augmentations=False,
        )
