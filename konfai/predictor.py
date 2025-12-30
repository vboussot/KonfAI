import builtins
import copy
import importlib
import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from konfai import config_file, cuda_visible_devices, konfai_root, predictions_directory
from konfai.data.data_manager import DataPrediction, DatasetIter
from konfai.data.patching import Accumulator, PathCombine
from konfai.data.transform import Transform, TransformInverse, TransformLoader
from konfai.network.network import Model, ModelLoader, NetState, Network
from konfai.utils.config import apply_config, config
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.utils import (
    DataLog,
    DistributedObject,
    NeedDevice,
    PredictorError,
    State,
    description,
    get_module,
    run_distributed_app,
)


class Reduction(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, tensor: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()


class Mean(Reduction):

    def __init__(self):
        pass

    def __call__(self, tensor: list[torch.Tensor]) -> torch.Tensor:
        return torch.mean(torch.cat(tensor, dim=0).float(), dim=0).to(tensor[0].dtype).unsqueeze(0)


class Median(Reduction):

    def __init__(self):
        pass

    def __call__(self, tensor: list[torch.Tensor]) -> torch.Tensor:
        return torch.median(torch.cat(tensor, dim=0).float(), dim=0).values.to(tensor[0].dtype).unsqueeze(0)


class Concat(Reduction):

    def __init__(self):
        pass

    def __call__(self, tensor: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensor, dim=1)


class OutputDataset(Dataset, NeedDevice, ABC):

    def __init__(
        self,
        filename: str,
        group: str,
        before_reduction_transforms: dict[str, TransformLoader],
        after_reduction_transforms: dict[str, TransformLoader],
        final_transforms: dict[str, TransformLoader],
        patch_combine: str | None,
        reduction: str,
    ) -> None:
        filename, file_format = filename.split(":")
        super().__init__(filename, file_format)
        self.group = group
        self._before_reduction_transforms = before_reduction_transforms
        self._after_reduction_transforms = after_reduction_transforms
        self._final_transforms = final_transforms
        self._patch_combine = patch_combine
        self.reduction_classpath = reduction
        self.reduction: Reduction

        self.before_reduction_transforms: list[Transform] = []
        self.after_reduction_transforms: list[Transform] = []
        self.final_transforms: list[Transform] = []
        self.patch_combine: PathCombine | None = None

        self.output_layer_accumulator: dict[int, dict[int, Accumulator]] = {}
        self.attributes: dict[int, dict[int, dict[int, Attribute]]] = {}
        self.names: dict[int, str] = {}
        self.nb_data_augmentation = 0

    @abstractmethod
    def load(self, name_layer: str, datasets: list[Dataset], groups: dict[str, str]):
        transforms_type = [
            "before_reduction_transforms",
            "after_reduction_transforms",
            "final_transforms",
        ]
        for name, _transform_type, transform_type in [
            (k, getattr(self, f"_{k}"), getattr(self, k)) for k in transforms_type
        ]:

            if _transform_type is not None:
                for classpath, transform in _transform_type.items():
                    transform = transform.get_transform(
                        classpath,
                        konfai_args=f"{konfai_root()}.outputs_dataset.{name_layer}.OutputDataset.{name}",
                    )
                    transform.set_datasets(datasets + [self])
                    transform_type.append(transform)

        if self._patch_combine is not None:
            module, name = get_module(self._patch_combine, "konfai.data.patching")
            self.patch_combine = apply_config(f"{konfai_root()}.outputs_dataset.{name_layer}.OutputDataset")(
                getattr(module, name)
            )()

        module, name = get_module(self.reduction_classpath, "konfai.predictor")
        if module == "konfai.predictor":
            self.reduction = getattr(module, name)()
        else:
            self.reduction = apply_config(
                f"{konfai_root()}.outputs_dataset.{name_layer}.OutputDataset.{self.reduction_classpath}"
            )(getattr(module, name))()

    def set_patch_config(
        self,
        patch_size: list[int] | None,
        overlap: int | None,
        nb_data_augmentation: int,
    ) -> None:
        if patch_size is not None and overlap is not None:
            if self.patch_combine is not None:
                self.patch_combine.set_patch_config(patch_size, overlap)
        else:
            self.patch_combine = None
        self.nb_data_augmentation = nb_data_augmentation

    def to(self, device: torch.device):
        super().to(device)
        transforms_type = [
            "before_reduction_transforms",
            "after_reduction_transforms",
            "final_transforms",
        ]
        for transform_type in [(getattr(self, k)) for k in transforms_type]:
            if transform_type is not None:
                for transform in transform_type:
                    transform.to(device)

    @abstractmethod
    def add_layer(
        self,
        index_dataset: int,
        index_augmentation: int,
        index_patch: int,
        layer: torch.Tensor,
        dataset: DatasetIter,
    ):
        raise NotImplementedError()

    def is_done(self, index: int) -> bool:
        return len(self.output_layer_accumulator[index]) == self.nb_data_augmentation and all(
            acc.is_full() for acc in self.output_layer_accumulator[index].values()
        )

    @abstractmethod
    def get_output(self, index: int, number_of_channels_per_model: list[int], dataset: DatasetIter) -> torch.Tensor:
        raise NotImplementedError()

    def write_prediction(self, index: int, name: str, layer: torch.Tensor) -> None:
        super().write(self.group, name, layer.numpy(), self.attributes[index][0][0])
        self.attributes.pop(index)

    def __str__(self) -> str:
        params = {
            "filename": self.filename,
            "group": self.group,
            "before_reduction_transforms": self.before_reduction_transforms,
            "after_reduction_transforms": self.after_reduction_transforms,
            "final_transforms": self.final_transforms,
            "patch_combine": self.patch_combine,
            "reduction": self.patch_combine,
        }
        return str(params)

    def __repr__(self) -> builtins.str:
        return str(self)


@config("OutputDataset")
class OutSameAsGroupDataset(OutputDataset):

    def __init__(
        self,
        same_as_group: str = "default",
        dataset_filename: str = "default|./Dataset:mha",
        group: str = "default",
        before_reduction_transforms: dict[str, TransformLoader] = {"default|Normalize": TransformLoader()},
        after_reduction_transforms: dict[str, TransformLoader] = {"default|Normalize": TransformLoader()},
        final_transforms: dict[str, TransformLoader] = {"default|Normalize": TransformLoader()},
        patch_combine: str | None = None,
        reduction: str = "Mean",
    ) -> None:
        super().__init__(
            dataset_filename,
            group,
            before_reduction_transforms,
            after_reduction_transforms,
            final_transforms,
            patch_combine,
            reduction,
        )
        self.group_src, self.group_dest = same_as_group.split(":")

    def add_layer(
        self,
        index_dataset: int,
        index_augmentation: int,
        index_patch: int,
        layer: torch.Tensor,
        dataset: DatasetIter,
    ):
        if (
            index_dataset not in self.output_layer_accumulator
            or index_augmentation not in self.output_layer_accumulator[index_dataset]
        ):
            input_dataset = dataset.get_dataset_from_index(self.group_dest, index_dataset)
            if index_dataset not in self.output_layer_accumulator:
                self.output_layer_accumulator[index_dataset] = {}
                self.attributes[index_dataset] = {}
                self.names[index_dataset] = input_dataset.name
            self.attributes[index_dataset][index_augmentation] = {}

            self.output_layer_accumulator[index_dataset][index_augmentation] = Accumulator(
                input_dataset.patch.get_patch_slices(index_augmentation),
                input_dataset.patch.patch_size,
                self.patch_combine,
                batch=False,
            )

            for i in range(len(input_dataset.patch.get_patch_slices(index_augmentation))):
                self.attributes[index_dataset][index_augmentation][i] = Attribute(input_dataset.cache_attributes[0])

        for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].patch_transforms):
            if isinstance(transform, TransformInverse) and transform.apply_inverse:
                layer = transform.inverse(
                    self.names[index_dataset],
                    layer,
                    self.attributes[index_dataset][index_augmentation][index_patch],
                )
        self.output_layer_accumulator[index_dataset][index_augmentation].add_layer(index_patch, layer)

    def load(self, name_layer: str, datasets: list[Dataset], groups: dict[str, str]):
        super().load(name_layer, datasets, groups)

        if self.group_src not in groups.keys():
            raise PredictorError(f"Source group '{self.group_src}' not found. Available groups: {list(groups.keys())}.")

        if self.group_dest not in groups[self.group_src]:
            raise PredictorError(
                f"Destination group '{self.group_dest}' not found. Available groups: {groups[self.group_src]}."
            )

    def _get_output(
        self, index: int, index_augmentation: int, number_of_channels_per_model: list[int], dataset: DatasetIter
    ) -> torch.Tensor:
        layer = self.output_layer_accumulator[index][index_augmentation].assemble()
        if index_augmentation > 0:

            i = 0
            index_augmentation_tmp = index_augmentation - 1
            for data_augmentations in dataset.data_augmentations_list:
                if index_augmentation_tmp >= i and index_augmentation_tmp < i + data_augmentations.nb:
                    for data_augmentation in reversed(data_augmentations.data_augmentations):
                        layer = data_augmentation.inverse(index, index_augmentation_tmp - i, layer)
                    break
                i += data_augmentations.nb
        chunks = list(torch.split(layer, number_of_channels_per_model, dim=0))
        base_attr = self.attributes[index][index_augmentation][0]
        base_attr["number_of_channels_per_model_0"] = torch.tensor(number_of_channels_per_model)
        results = []
        for i, layer in enumerate(chunks):
            attr = base_attr if (i == len(chunks) - 1) else Attribute(base_attr)
            for transform in self.before_reduction_transforms:
                layer = transform(self.names[index], layer, Attribute(attr))
            results.append(layer)

        return torch.stack(results, dim=0)

    def get_output(self, index: int, number_of_channels_per_model: list[int], dataset: DatasetIter) -> torch.Tensor:
        results = [
            self._get_output(index, index_augmentation, number_of_channels_per_model, dataset).unsqueeze(0)
            for index_augmentation in self.output_layer_accumulator[index].keys()
        ]

        self.output_layer_accumulator.pop(index)
        result = self.reduction(results).squeeze(0)
        for transform in self.after_reduction_transforms:
            result = transform(self.names[index], result, self.attributes[index][0][0])

        for transform in reversed(dataset.groups_src[self.group_src][self.group_dest].transforms):
            if isinstance(transform, TransformInverse) and transform.apply_inverse:
                result = transform.inverse(self.names[index], result, self.attributes[index][0][0])

        for transform in self.final_transforms:
            result = transform(self.names[index], result, self.attributes[index][0][0])

        return result


@config("OutputDataset")
class OutputDatasetLoader:

    def __init__(self, name_class: str = "OutSameAsGroupDataset") -> None:
        self.name_class = name_class

    def get_output_dataset(self, layer_name: str) -> OutputDataset:
        return apply_config(f"Predictor.outputs_dataset.{layer_name}")(
            getattr(importlib.import_module("konfai.predictor"), self.name_class)
        )()


class _Predictor:
    """
    Internal class that runs distributed inference over a dataset using a composite model.

    This class handles patch-wise prediction, output accumulation, logging to TensorBoard, and
    writing final predictions to disk. It is designed to be used as a context manager and
    supports model ensembles via `ModelComposite`.

    Args:
        world_size (int): Total number of processes or GPUs used.
        global_rank (int): Rank of the current process across all nodes.
        local_rank (int): Local GPU index within a single node.
        autocast (bool): Whether to use automatic mixed precision (AMP).
        predict_path (str): Output directory path where predictions and metrics are saved.
        data_log (list[str] | None): List of logging targets in the format 'group/DataLogType/N'.
        outputs_dataset (dict[str, OutputDataset]): Dictionary of output datasets to store predictions.
        model_composite (DDP): Distributed model container that wraps the prediction model(s).
        dataloader_prediction (DataLoader): DataLoader that provides prediction batches.
    """

    def __init__(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        autocast: bool,
        predict_path: Path,
        data_log: list[str] | None,
        outputs_dataset: dict[str, OutputDataset],
        model_composite: DDP,
        dataloader_prediction: DataLoader,
    ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.local_rank = local_rank

        self.model_composite = model_composite

        self.dataloader_prediction = dataloader_prediction
        self.outputs_dataset = outputs_dataset
        self.autocast = autocast

        self.it = 0

        self.dataset: DatasetIter = self.dataloader_prediction.dataset
        patch_size, overlap = self.dataset.get_patch_config()
        for output_dataset in self.outputs_dataset.values():
            output_dataset.set_patch_config(
                [size for size in patch_size if size > 1] if patch_size else None,
                overlap,
                np.max(
                    [
                        int(
                            np.sum([data_augmentation.nb for data_augmentation in self.dataset.data_augmentations_list])
                            + 1
                        ),
                        1,
                    ]
                ),
            )
        self.data_log: dict[str, tuple[DataLog, int]] = {}
        if data_log is not None:
            for data in data_log:
                self.data_log[data.split("/")[0].replace(":", ".")] = (
                    DataLog[data.split("/")[1]],
                    int(data.split("/")[2]),
                )
        self.tb = (
            SummaryWriter(log_dir=predict_path / "Metric")
            if len(
                [
                    network
                    for network in self.model_composite.module.get_networks().values()
                    if network.measure is not None
                ]
            )
            or len(self.data_log)
            else None
        )

    def __enter__(self):
        """
        Enters the prediction context and returns the predictor instance.
        """
        return self

    def __exit__(self, exc_type, value, traceback):
        """
        Closes the TensorBoard writer upon exit.
        """
        if self.tb:
            self.tb.close()

    def get_input(
        self,
        data_dict: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor]],
    ) -> dict[tuple[str, bool], torch.Tensor]:
        return {(k, v[5][0]): v[0] for k, v in data_dict.items()}

    @torch.no_grad()
    def run(self):
        """
        Run the full prediction loop.

        Iterates over the prediction DataLoader, performs inference using the composite model,
        applies reduction (e.g., mean), and writes the final results using each `OutputDataset`.

        Also logs intermediate data and metrics to TensorBoard if enabled.
        """

        self.model_composite.eval()
        self.model_composite.module.set_state(NetState.PREDICTION)
        self.dataloader_prediction.dataset.load("Prediction")
        with tqdm.tqdm(
            iterable=enumerate(self.dataloader_prediction),
            leave=True,
            desc=f"Prediction : {description(self.model_composite)}",
            total=len(self.dataloader_prediction),
            ncols=0,
        ) as batch_iter:
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=self.autocast):
                    for _, data_dict in batch_iter:
                        input_tensor = self.get_input(data_dict)
                        for name, number_of_channels_per_model, output in self.model_composite(
                            input_tensor, list(self.outputs_dataset.keys())
                        ):
                            self._predict_log(data_dict)
                            output_dataset = self.outputs_dataset[name]
                            for i, (index, patch_augmentation, patch_index) in enumerate(
                                [
                                    (int(index), int(patch_augmentation), int(patch_index))
                                    for index, patch_augmentation, patch_index in zip(
                                        list(data_dict.values())[0][1],
                                        list(data_dict.values())[0][2],
                                        list(data_dict.values())[0][3],
                                    )
                                ]
                            ):
                                output_dataset.add_layer(
                                    index,
                                    patch_augmentation,
                                    patch_index,
                                    output[i].cpu(),
                                    self.dataset,
                                )
                                if output_dataset.is_done(index):
                                    output_dataset.write_prediction(
                                        index,
                                        self.dataset.get_dataset_from_index(
                                            list(data_dict.keys())[0], index
                                        ).name.split("/")[-1],
                                        output_dataset.get_output(index, number_of_channels_per_model, self.dataset),
                                    )

                        batch_iter.set_description(f"Prediction : {description(self.model_composite)}")
                        self.it += 1

    def _predict_log(
        self,
        data_dict: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor]],
    ):
        """
        Log prediction results to TensorBoard, including images and metrics.

        This method handles:
        - Logging image-like data (e.g., inputs, outputs, masks) using `DataLog` instances,
        based on the `data_log` configuration.
        - Logging scalar loss and metric values (if present in the network) under the `Prediction/` namespace.
        - Dynamically retrieving additional feature maps or intermediate layers if requested via `data_log`.

        Logging is performed only on the global rank 0 process and only if `TensorBoard` is active.

        Args:
            data_dict (dict): Dictionary mapping group names to 6-tuples containing:
                - input tensor,
                - index,
                - patch_augmentation,
                - patch_index,
                - metadata (list of strings),
                - `requires_grad` flag (as a tensor).
        """
        measures = DistributedObject.get_measure(
            self.world_size,
            self.global_rank,
            self.local_rank,
            {"": self.model_composite.module},
            1,
        )

        if self.global_rank == 0 and self.tb is not None:
            data_log = []
            if len(self.data_log):
                for name, data_type in self.data_log.items():
                    if name in data_dict:
                        data_type[0](
                            self.tb,
                            f"Prediction/{name}",
                            data_dict[name][0][: self.data_log[name][1]].detach().cpu().numpy(),
                            self.it,
                        )
                    else:
                        data_log.append(name.replace(":", "."))

            for name, network in self.model_composite.module.get_networks().items():
                if network.measure is not None:
                    self.tb.add_scalars(
                        f"Prediction/{name}/Loss",
                        {k: v[1] for k, v in measures[name][0].items()},
                        self.it,
                    )
                    self.tb.add_scalars(
                        f"Prediction/{name}/Metric",
                        {k: v[1] for k, v in measures[name][1].items()},
                        self.it,
                    )
                if len(data_log):
                    for name, layer, _ in self.model_composite.module.get_layers(
                        [v.to(0) for k, v in self.get_input(data_dict).items() if k[1]], data_log
                    ):
                        self.data_log[name][0](
                            self.tb,
                            f"Prediction/{name}",
                            layer[: self.data_log[name][1]].detach().cpu().numpy(),
                            self.it,
                        )


class ModelComposite(Network):
    """
    A composite model that replicates a given base network multiple times and combines their outputs.

    This class is designed to handle model ensembles or repeated predictions from the same architecture.
    It creates `nb_models` deep copies of the input `model`, each with its own name and output branch,
    and aggregates their outputs using a provided `Reduction` strategy (e.g., mean, median).

    Args:
        model (Network): The base network to replicate.
        nb_models (int): Number of copies of the model to create.
        combine (Reduction): The reduction method used to combine outputs from all model replicas.

    Attributes:
        combine (Reduction): The reduction method used during forward inference.
    """

    def __init__(self, model: Network, nb_models: int, combine: Reduction):
        super().__init__(
            model.in_channels,
            model.optimizer,
            model.lr_schedulers_loader,
            model.outputs_criterions_loader,
            model.patch,
            model.nb_batch_per_step,
            model.init_type,
            model.init_gain,
            model.dim,
        )
        self.combine = combine
        for i in range(nb_models):
            self.add_module(
                f"Model_{i}",
                copy.deepcopy(model),
                in_branch=[0],
                out_branch=[f"output_{i}"],
            )

    def load(self, state_dicts: list[dict[str, dict[str, torch.Tensor]]]):
        """
        Load weights for each sub-model in the composite from the corresponding state dictionaries.

        Args:
            state_dicts (list): A list of state dictionaries, one for each model replica.
        """
        for i, state_dict in enumerate(state_dicts):
            self[f"Model_{i}"].load(state_dict, init=False)
            self[f"Model_{i}"].set_name(f"{self[f'Model_{i}'].get_name()}_{i}")

    def forward(  # type: ignore[override]
        self,
        data_dict: dict[tuple[str, bool], torch.Tensor],
        output_layers: list[str] = [],
    ) -> list[tuple[str, list[int], torch.Tensor]]:
        """
        Perform a forward pass on all model replicas and aggregate their outputs.

        Args:
            data_dict (dict): A dictionary mapping (group_name, requires_grad) to input tensors.
            output_layers (list): List of output layer names to extract from each sub-model.

        Returns:
            list[tuple[str, torch.Tensor]]: Aggregated output for each layer, after applying the reduction.
        """
        result = {}
        for name, module in self.items():
            result[name] = module(data_dict, output_layers)

        aggregated = defaultdict(list)
        for module_outputs in result.values():
            for key, tensor in module_outputs:
                if tensor.dtype == torch.float32:
                    tensor = tensor.to(torch.float16)
                aggregated[key].append(tensor)

        final_outputs = []
        for key, tensors in aggregated.items():
            final_outputs.append((key, [t.shape[1] for t in tensors], self.combine(tensors)))

        return final_outputs


@config("Predictor")
class Predictor(DistributedObject):
    """
    KonfAI's main prediction controller.

    This class orchestrates the prediction phase by:
    - Loading model weights from checkpoint(s) or URL(s)
    - Preparing datasets and output configurations
    - Managing distributed inference with optional multi-GPU support
    - Applying transformations and saving predictions
    - Optionally logging results to TensorBoard

    Attributes:
        model (Network): The neural network model to use for prediction.
        dataset (DataPrediction): Dataset manager for prediction data.
        combine_classpath (str): Path to the reduction strategy (e.g., "Mean").
        autocast (bool): Whether to enable AMP inference.
        outputs_dataset (dict[str, OutputDataset]): Mapping from layer names to output writers.
        data_log (list[str] | None): List of tensors to log during inference.
    """

    def __init__(
        self,
        model: ModelLoader = ModelLoader(),
        dataset: DataPrediction = DataPrediction(),
        combine: str = "Mean",
        train_name: str = "name",
        manual_seed: int | None = None,
        gpu_checkpoints: list[str] | None = None,
        autocast: bool = False,
        outputs_dataset: dict[str, OutputDatasetLoader] | None = {"default|Default": OutputDatasetLoader()},
        data_log: list[str] | None = None,
    ) -> None:
        if os.environ["KONFAI_CONFIG_MODE"] != "Done":
            exit(0)
        super().__init__(train_name)
        self.manual_seed = manual_seed
        self.dataset = dataset
        module, name = get_module(combine, "konfai.predictor")
        if module.__name__ == "konfai.predictor":
            self.combine = getattr(module, name)()
        else:
            self.combine = apply_config(f"{konfai_root()}.{combine}")(getattr(module, name))()

        self.autocast = autocast
        self.model = model.get_model(train=False)
        self.it = 0
        self.outputs_dataset_loader = outputs_dataset if outputs_dataset else {}
        self.outputs_dataset = {
            name.replace(":", "."): value.get_output_dataset(name)
            for name, value in self.outputs_dataset_loader.items()
        }

        self.datasets_filename = []
        self.predict_path = predictions_directory() / self.name
        for output_dataset in self.outputs_dataset.values():
            self.datasets_filename.append(output_dataset.filename)
            output_dataset.filename = str(self.predict_path / output_dataset.filename) + "/"
        self.data_log = data_log
        modules = []
        for i, _ in self.model.named_modules():
            modules.append(i)
        if self.data_log is not None:
            for k in self.data_log:
                tmp = k.split("/")[0].replace(":", ".")
                if tmp not in self.dataset.get_groups_dest() and tmp not in modules:
                    raise PredictorError(
                        f"Invalid key '{tmp}' in `data_log`.",
                        f"This key is neither a destination group from the dataset ({self.dataset.get_groups_dest()})",
                        f"nor a valid module name in the model ({modules}).",
                        "Please check your `data_log` configuration,"
                        " it should reference either a model output or a dataset group.",
                    )

        self.gpu_checkpoints = gpu_checkpoints

    def set_models(self, path_to_models: list[Path | str]) -> None:
        self.path_to_models = path_to_models

    def _load(self) -> list[dict[str, dict[str, torch.Tensor]]]:
        """
        Load pretrained model weights from configured paths or URLs.

        This method handles both remote and local model sources:
        - If the model path is a URL (starting with "https://"), it uses `torch.hub.load_state_dict_from_url`
        to download and load the state dict.
        - If the model path is local:
            - It either loads the explicit file or resolves the latest model file in a default directory
            based on the prediction name.
        - All loaded state dicts are returned as a list of nested dictionaries mapping module names
        to parameter tensors.

        Returns:
            list[dict[str, dict[str, torch.Tensor]]]: A list of state dictionaries, one per model.

        Raises:
            Exception: If a model path does not exist or cannot be loaded.
        """
        state_dicts = []
        for path_to_model in self.path_to_models:
            if isinstance(path_to_model, str) and path_to_model.startswith("https://"):
                try:
                    state_dicts.append(
                        torch.hub.load_state_dict_from_url(url=path_to_model, map_location="cpu", check_hash=True)
                    )
                except Exception:
                    raise Exception(f"Model : {path_to_model} does not exist !")
            elif Path(path_to_model).exists():
                state_dicts.append(
                    torch.load(str(path_to_model), map_location=torch.device("cpu"), weights_only=False)  # nosec B614
                )  # nosec B614
            else:
                raise ValueError(f"Invalid model path entry: {path_to_model}")
        return state_dicts

    def setup(self, world_size: int):
        """
        Set up the predictor for inference.

        This method performs all necessary initialization steps before running predictions:
        - Ensures output directories exist, and optionally prompts the user before overwriting existing predictions.
        - Copies the current configuration file (Prediction.yml) into the output directory for reproducibility.
        - Initializes the model in prediction mode, including output configuration and channel tracing.
        - Validates that the configured output groups match existing modules in the model architecture.
        - Dynamically loads pretrained weights from local files or remote URLs.
        - Wraps the base model into a `ModelComposite` to support ensemble inference.
        - Initializes the prediction dataloader, with proper distribution across available GPUs.
        - Loads and prepares each configured `OutputDataset` object for storing predictions.

        Args:
            world_size (int): Total number of processes or GPUs used for distributed prediction.

        Raises:
            PredictorError: If an output group does not match any module in the model.
            Exception: If a specified model file or URL is invalid or inaccessible.
        """
        for dataset_filename in self.datasets_filename:
            path = self.predict_path / dataset_filename
            if os.path.exists(path) and len(list(Path(path).rglob("*.yml"))):
                if os.environ["KONFAI_OVERWRITE"] != "True":
                    accept = builtins.input(
                        f"The prediction {path} already exists ! Do you want to overwrite it (yes,no) : "
                    )
                    if accept != "yes":
                        exit(0)

            if not os.path.exists(path):
                os.makedirs(path)

        shutil.copyfile(config_file(), self.predict_path / "Prediction.yml")

        self.model.init(self.autocast, State.PREDICTION, self.dataset.get_groups_dest())
        self.model.init_outputs_group()
        self.model._compute_channels_trace(self.model, self.model.in_channels, None, self.gpu_checkpoints)

        modules = []
        for i, _, _ in self.model.named_module_args_dict():
            modules.append(i)
        for output_group in self.outputs_dataset.keys():
            if output_group.replace(";accu;", "") not in modules:
                raise PredictorError(
                    f"The output group '{output_group}' defined in 'outputs_criterions' "
                    "does not correspond to any module in the model.",
                    f"Available modules: {modules}",
                    "Please check that the name matches exactly a submodule or" "output of your model architecture.",
                )

        self.model_composite = ModelComposite(self.model, len(self.path_to_models), self.combine)
        self.model_composite.load(self._load())

        if (
            len(list(self.outputs_dataset.keys())) == 0
            and len(
                [network for network in self.model_composite.get_networks().values() if network.measure is not None]
            )
            == 0
        ):
            exit(0)

        self.size = len(self.gpu_checkpoints) + 1 if self.gpu_checkpoints else 1

        self.dataloader, _, _ = self.dataset.get_data(world_size // self.size)
        for name, output_dataset in self.outputs_dataset.items():
            output_dataset.load(
                name.replace(".", ":"),
                list(self.dataset.datasets.values()),
                {src: dest for src, inner in self.dataset.groups_src.items() for dest in inner},
            )

    def run_process(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        dataloaders: list[DataLoader],
    ):
        """
        Launch prediction on the given process rank.

        Args:
            world_size (int): Total number of processes.
            global_rank (int): Rank of the current process.
            local_rank (int): Local device rank.
            dataloaders (list[DataLoader]): List of data loaders for prediction.
        """

        model_composite = (
            Network.to(self.model_composite, local_rank * self.size)
            if len(cuda_visible_devices())
            else self.model_composite
        )
        has_trainable_params = any(p.requires_grad for p in model_composite.parameters())
        model_composite = (
            DDP(model_composite, static_graph=True)
            if dist.is_initialized() and has_trainable_params
            else Model(model_composite)
        )
        with _Predictor(
            world_size,
            global_rank,
            local_rank,
            self.autocast,
            self.predict_path,
            self.data_log,
            self.outputs_dataset,
            model_composite,
            *dataloaders,
        ) as p:
            p.run()

    def __str__(self) -> str:
        params = {
            "model": self.model,
            "dataset": self.dataset,
            "combine": self.combine,
            "train_name": self.name,
            "manual_seed": self.manual_seed,
            "gpu_checkpoints": self.gpu_checkpoints,
            "autocast": self.autocast,
            "outputs_dataset": self.outputs_dataset,
            "data_log": self.data_log,
        }
        return str(params)

    def __repr__(self) -> str:
        return str(self)


@run_distributed_app
def predict(
    models: list[Path],
    overwrite: bool = False,
    gpu: list[int] | None = cuda_visible_devices(),
    cpu: int = 1,
    quiet: bool = False,
    tb: bool = False,
    prediction_file: Path | str = Path("./Prediction.yml").resolve(),
    predictions_dir: Path | str = Path("./Predictions").resolve(),
) -> DistributedObject:
    os.environ["KONFAI_config_file"] = str(Path(prediction_file).resolve())
    os.environ["KONFAI_ROOT"] = "Predictor"
    os.environ["KONFAI_STATE"] = str(State.PREDICTION)
    os.environ["KONFAI_PREDICTIONS_DIRECTORY"] = str(Path(predictions_dir).resolve())
    predictor = apply_config()(Predictor)()
    predictor.set_models(models)
    return predictor
