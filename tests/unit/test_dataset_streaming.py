from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from konfai.data.data_manager import DataPrediction, DatasetIter, DataTrain, Group, GroupTransform
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import KonfAIInference, Normalize, Standardize, TransformLoader
from konfai.utils.dataset import Attribute, Dataset

SimpleITK = pytest.importorskip("SimpleITK")


def _image_attributes(origin: list[float], spacing: list[float]) -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray(origin, dtype=np.float64)
    attributes["Spacing"] = np.asarray(spacing, dtype=np.float64)
    attributes["Direction"] = np.eye(len(origin), dtype=np.float64).reshape(-1)
    return attributes


def test_dataset_read_data_slice_h5_reads_only_requested_region(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Volumes", "h5")
    volume = np.arange(1 * 4 * 5, dtype=np.float32).reshape(1, 4, 5)
    dataset.write("CT", "CASE_000", volume, _image_attributes([1.0, 2.0], [0.5, 1.5]))

    patch, _ = dataset.read_data_slice("CT", "CASE_000", (slice(None), slice(1, 3), slice(2, 5)))

    np.testing.assert_array_equal(patch, volume[:, 1:3, 2:5])


def test_dataset_read_data_statistics_h5_returns_global_stats_without_loading_full_array(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Volumes", "h5")
    volume = np.arange(1 * 4 * 5, dtype=np.float32).reshape(1, 4, 5)
    dataset.write("CT", "CASE_000", volume, _image_attributes([1.0, 2.0], [0.5, 1.5]))

    stats = dataset.read_data_statistics("CT", "CASE_000")

    assert stats["min"] == pytest.approx(float(volume.min()))
    assert stats["max"] == pytest.approx(float(volume.max()))
    assert stats["mean"] == pytest.approx(float(volume.mean()))
    assert stats["std"] == pytest.approx(float(volume.std(ddof=1)))


def test_dataset_read_data_slice_sitk_reads_requested_patch_and_updates_origin(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    volume = np.arange(1 * 4 * 5 * 6, dtype=np.float32).reshape(1, 4, 5, 6)
    origin = [10.0, 20.0, 30.0]
    spacing = [0.5, 1.5, 2.0]
    dataset.write("CT", "CASE_000", volume, _image_attributes(origin, spacing))

    patch, attributes = dataset.read_data_slice(
        "CT",
        "CASE_000",
        (slice(None), slice(1, 3), slice(2, 5), slice(3, 6)),
    )

    np.testing.assert_array_equal(patch, volume[:, 1:3, 2:5, 3:6])
    np.testing.assert_allclose(
        attributes.get_np_array("Origin"),
        np.asarray([origin[0] + 3 * spacing[0], origin[1] + 2 * spacing[1], origin[2] + 1 * spacing[2]]),
    )


class StreamingDatasetStub:

    def __init__(self, volume: np.ndarray) -> None:
        self.volume = volume
        self.full_reads = 0
        self.patch_reads = 0
        self.stats_reads = 0

    def get_infos(self, group_src: str, name: str) -> tuple[list[int], Attribute]:
        return list(self.volume.shape), _image_attributes([0.0, 0.0], [1.0, 1.0])

    def read_data(self, group_src: str, name: str) -> tuple[np.ndarray, Attribute]:
        self.full_reads += 1
        return self.volume.copy(), _image_attributes([0.0, 0.0], [1.0, 1.0])

    def read_data_slice(self, group_src: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        self.patch_reads += 1
        return self.volume[slices].copy(), _image_attributes([0.0, 0.0], [1.0, 1.0])

    def read_data_statistics(
        self,
        group_src: str,
        name: str,
        channels: list[int] | None = None,
    ) -> dict[str, float]:
        self.stats_reads += 1
        data = self.volume if channels is None else self.volume[channels]
        return {
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "std": float(data.std(ddof=1)),
        }


def test_dataset_iter_streams_patch_reads_when_cache_disabled() -> None:
    volume = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    dataset_stub = StreamingDatasetStub(volume)
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, dataset_stub),
        patch=DatasetPatch([2, 2]),
        transforms=[],
        data_augmentations_list=[],
    )
    dataset_iter = DatasetIter(
        rank=0,
        data={"CT": [manager]},
        mapping=[(0, 0, 1)],
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
        inline_augmentations=False,
        data_augmentations_list=[],
        patch_size=[2, 2],
        overlap=None,
        buffer_size=1,
        use_cache=False,
    )

    sample = dataset_iter[0]["CT"].tensor

    assert dataset_stub.full_reads == 0
    assert dataset_stub.patch_reads == 1
    assert manager.loaded is False
    np.testing.assert_array_equal(sample.numpy(), volume[:, 0:2, 2:4])


def test_data_train_enables_worker_prefetch_when_cache_is_disabled() -> None:
    dataset = DataTrain(use_cache=False, augmentations=None)

    assert cast(int, dataset.dataLoader_args["num_workers"]) >= 1
    assert dataset.dataLoader_args["prefetch_factor"] == 2
    assert dataset.dataLoader_args["persistent_workers"] is True


def test_data_prediction_disables_workers_for_konfai_inference_transforms() -> None:
    dataset = DataPrediction(
        augmentations=None,
        groups_src={
            "Volume_0": Group(
                groups_dest={
                    "MASK": GroupTransform(
                        transforms={"KonfAIInference": TransformLoader()},
                        patch_transforms=None,
                    )
                }
            )
        },
    )

    assert dataset.requires_single_process_loading is True
    assert dataset.dataLoader_args["num_workers"] == 0
    assert "prefetch_factor" not in dataset.dataLoader_args
    assert "persistent_workers" not in dataset.dataLoader_args


def test_data_prediction_disables_persistent_workers_by_default() -> None:
    dataset = DataPrediction(augmentations=None)

    assert cast(int, dataset.dataLoader_args["num_workers"]) >= 1
    assert dataset.dataLoader_args["prefetch_factor"] == 2
    assert dataset.dataLoader_args["persistent_workers"] is False


def test_konfai_inference_raises_clear_error_inside_daemon_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    transform = KonfAIInference()

    class DaemonProcess:
        daemon = True

    monkeypatch.setattr("konfai.data.transform.current_process", lambda: DaemonProcess())

    with pytest.raises(RuntimeError, match="Dataset.num_workers: 0"):
        transform("CASE_000", torch.zeros(1, 4, 4), Attribute())


def test_dataset_iter_streams_patch_reads_with_global_normalize_stats() -> None:
    volume = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    dataset_stub = StreamingDatasetStub(volume)
    normalize = Normalize()
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, dataset_stub),
        patch=DatasetPatch([2, 2]),
        transforms=[normalize],
        data_augmentations_list=[],
    )
    dataset_iter = DatasetIter(
        rank=0,
        data={"CT": [manager]},
        mapping=[(0, 0, 1)],
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
        inline_augmentations=False,
        data_augmentations_list=[],
        patch_size=[2, 2],
        overlap=None,
        buffer_size=1,
        use_cache=False,
    )

    sample = dataset_iter[0]["CT"].tensor
    expected = (2 * volume[:, 0:2, 2:4] / (volume.max() - volume.min())) - 1

    assert dataset_stub.full_reads == 0
    assert dataset_stub.patch_reads == 1
    assert dataset_stub.stats_reads == 1
    np.testing.assert_allclose(sample.numpy(), expected)


def test_dataset_iter_streams_patch_reads_with_computed_standardize_stats() -> None:
    volume = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    dataset_stub = StreamingDatasetStub(volume)
    standardize = Standardize()
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, dataset_stub),
        patch=DatasetPatch([2, 2]),
        transforms=[standardize],
        data_augmentations_list=[],
    )
    dataset_iter = DatasetIter(
        rank=0,
        data={"CT": [manager]},
        mapping=[(0, 0, 3)],
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
        inline_augmentations=False,
        data_augmentations_list=[],
        patch_size=[2, 2],
        overlap=None,
        buffer_size=1,
        use_cache=False,
    )

    sample = dataset_iter[0]["CT"].tensor
    expected = (volume[:, 2:4, 2:4] - volume.mean()) / volume.std(ddof=1)

    assert dataset_stub.full_reads == 0
    assert dataset_stub.patch_reads == 1
    assert dataset_stub.stats_reads == 1
    np.testing.assert_allclose(sample.numpy(), expected)
