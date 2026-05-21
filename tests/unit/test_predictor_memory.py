from typing import cast

import numpy as np
import torch

from konfai.data.data_manager import BatchDataItem, DatasetIter
from konfai.network.network import Network
from konfai.predictor import Mean, ModelComposite, OutSameAsGroupDataset
from konfai.utils.dataset import Attribute


class DummyPredictNetwork(Network):

    def __init__(self) -> None:
        super().__init__(in_channels=1)
        self.scale = 1.0
        self.load_history: list[float] = []
        self.load_name_history: list[str] = []

    def load(self, state_dict, init: bool = True, ema: bool = False):  # type: ignore[override]
        self.scale = float(state_dict["scale"])
        self.load_history.append(self.scale)
        self.load_name_history.append(self.get_name())

    def forward(self, batch_sample, output_layers=[]):  # type: ignore[override]
        tensor = next(iter(batch_sample.values())).tensor * self.scale
        return [("out", tensor)]


def test_model_composite_streams_ensemble_through_a_single_loaded_model() -> None:
    model = DummyPredictNetwork()
    composite = ModelComposite(model, Mean())
    composite.load([{"scale": 1.0}, {"scale": 3.0}])

    batch_sample = {
        "input": BatchDataItem(
            name=["CASE_000"],
            tensor=torch.ones(1, 1, 2, 2),
            attribute=[Attribute()],
            x=[0],
            a=[0],
            p=[0],
            is_input=True,
        )
    }

    outputs = composite(batch_sample, ["out"])
    streamed_model = composite["Model_0"]

    assert len(list(composite.keys())) == 1
    assert len(outputs) == 1
    assert outputs[0][0] == "out"
    assert outputs[0][1] == [1, 1]
    assert torch.allclose(outputs[0][2], torch.full((1, 1, 2, 2), 2.0, dtype=outputs[0][2].dtype))
    assert isinstance(streamed_model, DummyPredictNetwork)
    assert streamed_model.load_history == [1.0, 3.0]
    assert streamed_model.load_name_history == ["DummyPredictNetwork", "DummyPredictNetwork"]


def test_output_dataset_uses_batch_attributes_when_manager_cache_is_cold() -> None:
    class DummyPatch:
        patch_size = [2, 2]

        @staticmethod
        def get_patch_slices(index_augmentation: int):
            del index_augmentation
            return [(slice(0, 2), slice(0, 2))]

    class DummyManager:
        name = "CASE_000"
        patch = DummyPatch()
        cache_attributes = [Attribute({"Origin": [0.0, 0.0]})]

    class DummyGroupTransform:
        patch_transforms: list[object] = []

    class DummyDatasetIter:
        groups_src = {"src": {"dest": DummyGroupTransform()}}

        @staticmethod
        def get_dataset_from_index(group_dest: str, index: int):
            assert group_dest == "dest"
            assert index == 0
            return DummyManager()

    output_dataset = OutSameAsGroupDataset(
        same_as_group="src:dest",
        dataset_filename="./Output:mha",
        group="out",
        patch_combine=None,
        reduction="Mean",
    )

    streamed_attribute = Attribute()
    streamed_attribute["Spacing"] = np.asarray([1.0, 1.0])
    streamed_attribute["Size"] = np.asarray([4, 4])
    streamed_attribute["Size"] = np.asarray([2, 2])

    output_dataset.add_layer(
        index_dataset=0,
        index_augmentation=0,
        index_patch=0,
        layer=torch.zeros(1, 2, 2),
        dataset=cast(DatasetIter, DummyDatasetIter()),
        attribute=streamed_attribute,
    )

    assert "Size" in output_dataset.attributes[0][0][0]
    assert output_dataset.attributes[0][0][0].get_np_array("Size").tolist() == [2.0, 2.0]
