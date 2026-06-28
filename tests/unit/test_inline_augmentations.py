from typing import cast
from pathlib import Path

import numpy as np
import pytest
import torch

import konfai.data.augmentation as augmentation_module
from konfai.data.augmentation import DataAugmentation, DataAugmentationsList, Elastix, Mask
from konfai.data.data_manager import DatasetIter, Group, GroupTransform
from konfai.data.patching import DatasetManager
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import AugmentationError


class DummyDataset:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def get_infos(self, group_src: str, name: str) -> tuple[list[int], Attribute]:
        return list(self.array.shape), Attribute({"name": name, "group": group_src})

    def read_data(self, group_src: str, name: str) -> tuple[np.ndarray, Attribute]:
        return self.array.copy(), Attribute({"name": name, "group": group_src})


class CountingOffsetAugmentation(DataAugmentation):
    def __init__(self) -> None:
        super().__init__()
        self.compute_calls = 0

    def _state_init(
        self,
        index: int,
        shapes: list[list[int]],
        caches_attribute: list[Attribute],
    ) -> list[list[int]]:
        return shapes

    def _compute(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        self.compute_calls += 1
        return [tensor + (offset + 1) for offset, tensor in enumerate(tensors)]

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def test_inline_augmentations_are_loaded_on_demand() -> None:
    base = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    dataset = cast(Dataset, DummyDataset(base))
    augmentation = CountingOffsetAugmentation()
    augmentation.load(1.0)

    augmentations = DataAugmentationsList(nb=2, data_augmentations={})
    augmentations.data_augmentations = [augmentation]

    manager = DatasetManager(
        index=0,
        group_src="src",
        group_dest="dest",
        name="case_000",
        dataset=dataset,
        patch=None,
        transforms=[],
        data_augmentations_list=[augmentations],
    )
    dataset_iter = DatasetIter(
        rank=0,
        data={"dest": [manager]},
        mapping=[(0, 0, 0), (0, 1, 0), (0, 2, 0)],
        groups_src={"src": Group(groups_dest={"dest": GroupTransform(transforms=None, patch_transforms=None)})},
        inline_augmentations=True,
        data_augmentations_list=[augmentations],
        patch_size=None,
        overlap=None,
        buffer_size=1,
        use_cache=True,
    )

    base_sample = dataset_iter[0]["dest"].tensor
    assert augmentation.compute_calls == 0
    assert manager.loaded is True
    assert manager.augmentationLoaded is False
    assert torch.equal(base_sample, torch.from_numpy(base))

    first_augmented_sample = dataset_iter[1]["dest"].tensor
    assert augmentation.compute_calls == 1
    assert manager.augmentationLoaded is True
    assert torch.equal(first_augmented_sample, torch.from_numpy(base) + 1)

    second_augmented_sample = dataset_iter[2]["dest"].tensor
    assert augmentation.compute_calls == 1
    assert torch.equal(second_augmented_sample, torch.from_numpy(base) + 2)


def test_simpleitk_augmentations_fail_clearly_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(augmentation_module, "sitk", None)

    with pytest.raises(AugmentationError, match="SimpleITK"):
        Elastix()
    with pytest.raises(AugmentationError, match="SimpleITK"):
        Mask("mask.mha", 0)


def test_mask_reads_pixels_only_on_first_compute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sitk = pytest.importorskip("SimpleITK")
    mask_path = tmp_path / "mask.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.ones((2, 2), dtype=np.uint8)), str(mask_path))

    read_count = 0
    original_read_image = sitk.ReadImage

    def counting_read_image(path: str):
        nonlocal read_count
        read_count += 1
        return original_read_image(path)

    monkeypatch.setattr(augmentation_module.sitk, "ReadImage", counting_read_image)
    augmentation = Mask(str(mask_path), 0)
    augmentation._state_init(0, [[2, 2]], [Attribute()])

    assert read_count == 0
    augmentation._compute("case", 0, [torch.ones((1, 2, 2))])
    augmentation._compute("case", 0, [torch.ones((1, 2, 2))])
    assert read_count == 1
