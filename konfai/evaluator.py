import builtins
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from konfai import config_file, cuda_visible_devices, evaluations_directory, konfai_root
from konfai.data.data_manager import DataMetric
from konfai.utils.config import apply_config, config
from konfai.utils.dataset import Dataset
from konfai.utils.utils import (
    DistributedObject,
    EvaluatorError,
    State,
    get_module,
    run_distributed_app,
    synchronize_data,
)


class CriterionsAttr:
    """
    Container for additional metadata or configuration attributes related to a loss criterion.

    This class is currently empty but acts as a placeholder for future extension.
    It is passed along with each loss function to allow parameterization or inspection of its behavior.

    Use cases may include:
    - Weighting of individual loss terms
    - Conditional activation
    - Logging preferences
    """

    def __init__(self) -> None:
        pass


class CriterionsLoader:
    """
    Loader for multiple criterion modules to be applied between a model output and one or more targets.

    Each loss module (e.g., Dice, CrossEntropy, NCC) is dynamically loaded using its fully-qualified
    classpath and is associated with a `CriterionsAttr` configuration object.

    Args:
        criterions_loader (dict): A mapping from module classpaths (as strings) to `CriterionsAttr` instances.
                                  The module path is parsed and instantiated via `get_module`.

    """

    def __init__(
        self,
        criterions_loader: dict[str, CriterionsAttr] = {"default|torch:nn:CrossEntropyLoss|Dice|NCC": CriterionsAttr()},
    ) -> None:
        self.criterions_loader = criterions_loader

    def get_criterions(self, output_group: str, target_group: str) -> dict[torch.nn.Module, CriterionsAttr]:
        criterions = {}
        for module_classpath, criterions_attr in self.criterions_loader.items():
            module, name = get_module(module_classpath, "konfai.metric.measure")
            criterions[
                apply_config(
                    f"{konfai_root()}.metrics.{output_group}.targets_criterions.{target_group}"
                    f".criterions_loader.{module_classpath}"
                )(getattr(module, name))()
            ] = criterions_attr
        return criterions


class TargetCriterionsLoader:
    """
    Loader class for handling multiple target groups with associated criterion configurations.

    This class allows defining a set of criterion loaders (e.g., Dice, BCE, MSE) for each
    target group to be used during evaluation or training. Each target group corresponds
    to one or more loss functions, all linked to a specific model output.

    Args:
        targets_criterions (dict[str, CriterionsLoader]): Dictionary mapping each target group name
            to a `CriterionsLoader` instance that defines its associated loss functions.
    """

    def __init__(
        self,
        targets_criterions: dict[str, CriterionsLoader] = {"default": CriterionsLoader()},
    ) -> None:
        self.targets_criterions = targets_criterions

    def get_targets_criterions(self, output_group: str) -> dict[str, dict[torch.nn.Module, CriterionsAttr]]:
        """
        Retrieve the criterion modules and their attributes for a specific output group.

        This function prepares the loss functions to be applied for a given model output,
        grouped by their target group.

        Args:
            output_group (str): Name of the model output group (e.g., "output_segmentation").

        Returns:
            dict[str, dict[nn.Module, CriterionsAttr]]: A nested dictionary where the first key is the
            target group name, and the value is a dictionary mapping each loss module to its attributes.
        """
        targets_criterions = {}
        for target_group, criterions_loader in self.targets_criterions.items():
            targets_criterions[target_group] = criterions_loader.get_criterions(output_group, target_group)
        return targets_criterions


class Statistics:
    """
    Utility class to accumulate, structure, and write evaluation metric results.

    This class is used to:
    - Collect metrics for each dataset sample.
    - Compute aggregate statistics (mean, std, percentiles, etc.).
    - Export all results in a structured JSON format, including both per-case and aggregate values.

    Args:
        filename (str): Path to the output JSON file that will store the final results.
    """

    def __init__(self, filename: Path) -> None:
        self.measures: dict[str, dict[str, float]] = {}
        self.filename = filename

    def add(self, values: dict[str, float], name_dataset: str) -> None:
        """
        Add a set of metric values for a given dataset case.

        Args:
            values (dict): Dictionary of metric names and their values.
            name_dataset (str): Identifier (e.g., case name) for the sample.
        """
        for name, value in values.items():
            if name_dataset not in self.measures:
                self.measures[name_dataset] = {}
            self.measures[name_dataset][name] = value

    @staticmethod
    def get_statistic(values: list[float]) -> dict[str, float]:
        """
        Compute statistical aggregates for a list of metric values.

        Args:
            values (list of float): Values to summarize.

        Returns:
            dict[str, float]: A dictionary containing:
                - max, min, std
                - 25th, 50th, and 75th percentiles
                - mean and count
        """
        return {
            "max": float(np.nanmax(values)) if np.any(~np.isnan(values)) else np.nan,
            "min": float(np.nanmin(values)) if np.any(~np.isnan(values)) else np.nan,
            "std": float(np.nanstd(values)) if np.any(~np.isnan(values)) else np.nan,
            "25pc": float(np.nanpercentile(values, 25)) if np.any(~np.isnan(values)) else np.nan,
            "50pc": float(np.nanpercentile(values, 50)) if np.any(~np.isnan(values)) else np.nan,
            "75pc": float(np.nanpercentile(values, 75)) if np.any(~np.isnan(values)) else np.nan,
            "mean": float(np.nanmean(values)) if np.any(~np.isnan(values)) else np.nan,
            "count": float(np.count_nonzero(~np.isnan(values))) if np.any(~np.isnan(values)) else np.nan,
        }

    def write(self, outputs: list[dict[str, dict[str, Any]]]) -> None:
        """
        Write the collected and aggregated statistics to the configured output file.

        The output JSON structure contains:
        - `case`: All individual metrics per sample.
        - `aggregates`: Global statistics computed over all cases.

        Args:
            outputs (list): List of metric dictionaries to merge and serialize.
        """
        measures = {}
        for output in outputs:
            measures.update(output)
        result: dict[str, dict[str, dict[str, Any]]] = {}
        result["case"] = {}
        for name, v in measures.items():
            for metric_name, value in v.items():
                if metric_name not in result["case"]:
                    result["case"][metric_name] = {}
                result["case"][metric_name][name] = value

        result["aggregates"] = {}
        tmp: dict[str, list[float]] = {}
        for _, v in measures.items():
            for metric_name, _ in v.items():
                if metric_name not in tmp:
                    tmp[metric_name] = []
                tmp[metric_name].append(v[metric_name])
        for metric_name, values in tmp.items():
            result["aggregates"][metric_name] = Statistics.get_statistic(values)

        with open(self.filename, "w") as f:
            f.write(json.dumps(result, indent=4))

    def read(self):
        with open(self.filename) as f:
            json_data = json.load(f)

        result = {}

        aggregates = json_data.get("aggregates", {})

        for key, stats in aggregates.items():
            mean_value = stats.get("mean", None)
            if mean_value is None:
                continue

            # Nettoyage du nom
            parts = key.split(":")
            if parts[-2] == "Dice":
                continue
            else:
                metric_name = parts[-1]

            result[metric_name] = mean_value

        return result


@config("Evaluator")
class Evaluator(DistributedObject):
    """
    Distributed evaluation engine for computing metrics on model predictions.

    This class handles the evaluation of predicted outputs using predefined metric loaders.
    It supports multi-output and multi-target configurations, computes aggregated statistics
    across training and validation datasets, and synchronizes results across processes.

    Evaluation results are stored in JSON format and optionally displayed during iteration.

    Args:
        train_name (str): Unique name of the evaluation run, used for logging and output folders.
        metrics (dict[str, TargetCriterionsLoader]): Dictionary mapping output groups to loaders of target metrics.
        dataset (DataMetric): Dataset provider configured for evaluation mode.

    Attributes:
        statistics_train (Statistics): Object used to store training evaluation metrics.
        statistics_validation (Statistics): Object used to store validation evaluation metrics.
        dataloader (list[DataLoader]): DataLoaders for training and validation sets.
        metric_path (str): Path to the evaluation output directory.
        metrics (dict): Instantiated metrics organized by output and target groups.
    """

    def __init__(
        self,
        train_name: str = "default|TRAIN_01",
        metrics: dict[str, TargetCriterionsLoader] = {"default": TargetCriterionsLoader()},
        dataset: DataMetric = DataMetric(),
    ) -> None:
        if os.environ["KONFAI_CONFIG_MODE"] != "Done":
            exit(0)
        super().__init__(train_name)
        self.metric_path = evaluations_directory() / self.name
        self.metricsLoader = metrics if metrics else {}
        self.dataset = dataset
        self.metrics = {k: v.get_targets_criterions(k) for k, v in self.metricsLoader.items()}
        self.statistics_train = Statistics(self.metric_path / "Metric_TRAIN.json")
        self.statistics_validation = Statistics(self.metric_path / "Metric_VALIDATION.json")

    def update(self, data_dict: dict[str, tuple[torch.Tensor, str]], statistics: Statistics) -> dict[str, float]:
        """
        Compute metrics for a batch and update running statistics.

        Args:
            data_dict (dict): Dictionary where keys are output/target group names and values are
                            tuples of (tensor, sample name).
            statistics (Statistics): The statistics object to update (train or validation).

        Returns:
            dict[str, float]: Dictionary of computed metric values with keys in the format
                            'output_group:target_group:MetricName'.
        """
        result = {}
        for output_group in self.metrics:
            for target_group in self.metrics[output_group]:
                targets = [
                    (data_dict[group][0].to(0) if torch.cuda.is_available() else data_dict[group][0])
                    for group in target_group.split(";")
                    if group in data_dict
                ]
                name = data_dict[output_group][1][0]
                for metric in self.metrics[output_group][target_group]:
                    loss = metric(
                        (data_dict[output_group][0].to(0) if torch.cuda.is_available() else data_dict[output_group][0]),
                        *targets,
                    )
                    if isinstance(loss, tuple):
                        true_loss = loss[1]
                        if len(loss) == 3:
                            if metric.dataset:
                                if len(metric.dataset.split(":")) > 1:
                                    filename, file_format = metric.dataset.split(":")
                                else:
                                    filename = metric.dataset
                                    file_format = "mha"
                                map_dataset = Dataset(filename, file_format)
                                group = metric.group if metric.group else output_group
                                for dataset in self.dataset.datasets.values():
                                    for g in dataset.get_group():
                                        if dataset.is_dataset_exist(g, name):
                                            _, cache_attribute = dataset.get_infos(g, name)
                                            map_dataset.write(
                                                group,
                                                name,
                                                loss[2].squeeze(0).numpy(),
                                                cache_attribute,
                                            )
                    else:
                        true_loss = loss.item()

                    if isinstance(true_loss, dict):
                        loss = 0
                        c = 0
                        for k, v in true_loss.items():
                            result[f"{output_group}:{target_group}:{metric.get_name()}:{k}"] = v
                            if not np.isnan(v):
                                loss += v
                                c += 1
                        result[f"{output_group}:{target_group}:{metric.get_name()}"] = loss / c
                    else:
                        result[f"{output_group}:{target_group}:{metric.get_name()}"] = true_loss
        if len(self.metrics) > 0:
            statistics.add(result, name)
        return result

    def setup(self, world_size: int):
        """
        Prepare the evaluator for distributed metric computation.

        This method performs the following steps:
        - Checks whether previous evaluation results exist and optionally overwrites them.
        - Creates the output directory and copies the current configuration file for reproducibility.
        - Loads the evaluation dataset according to the world size.
        - Validates that all specified output and target groups used in metric definitions
        are present in the dataset group configuration.

        Args:
            world_size (int): Number of processes in the distributed evaluation setup.

        Raises:
            EvaluatorError: If any metric output or target group is missing in the dataset's group mapping.
        """
        if self.metric_path.exists() and len(list(self.metric_path.rglob("*.yml"))):
            if os.environ["KONFAI_OVERWRITE"] != "True":
                accept = builtins.input(
                    f"The metric {self.name} already exists ! Do you want to overwrite it (yes,no) : "
                )
                if accept != "yes":
                    exit(0)

                shutil.rmtree(self.metric_path)

        if not self.metric_path.exists():
            os.makedirs(self.metric_path)
        shutil.copyfile(
            config_file(),
            self.metric_path / config_file().name,
        )

        self.dataloader, _, _ = self.dataset.get_data(world_size)

        groups_dest = [group for groups in self.dataset.groups_src.values() for group in groups]
        missing_outputs = set(self.metrics.keys()) - set(groups_dest)
        if missing_outputs:
            raise EvaluatorError(
                f"The following metric output groups are missing from 'groups_dest': {sorted(missing_outputs)}. ",
                f"Available groups: {sorted(groups_dest)}",
            )

        target_groups = []
        for i in {target for targets in self.metrics.values() for target in targets}:
            for u in i.split(";"):
                target_groups.append(u)
        missing_targets = set(target_groups) - (set(groups_dest + ["None"]))
        if missing_targets:
            raise EvaluatorError(
                f"The following metric target groups are missing from 'groups_dest': {sorted(missing_targets)}. ",
                f"Available groups: {sorted(groups_dest)}",
            )

    def run_process(self, world_size: int, global_rank: int, gpu: int, dataloaders: list[DataLoader]):
        """
        Execute the distributed evaluation loop over the training and validation datasets.

        This method iterates through the provided DataLoaders (train and optionally validation),
        updates the metric statistics using the configured `metrics` dictionary, and synchronizes
        the results across all processes. On the global rank 0, the metrics are saved as JSON files.

        Metrics are displayed in real-time using `tqdm` progress bars, showing a summary of the
        current batch's computed values.

        Args:
            world_size (int): Total number of distributed processes.
            global_rank (int): Global rank of the current process (used for writing results).
            gpu (int): Local GPU ID used for synchronization.
            dataloaders (list[DataLoader]): A list containing one or two DataLoaders:
                - `dataloaders[0]` is used for training evaluation.
                - `dataloaders[1]` (optional) is used for validation evaluation.

        Notes:
            - Only the main process (`global_rank == 0`) writes final results to disk.
        """

        def description(measure):
            return (
                f"Metric TRAIN : {' | '.join(f'{k}: {v:.4f}' for k, v in measure.items())}"
                if measure is not None
                else "Metric TRAIN : "
            )

        with tqdm.tqdm(
            iterable=enumerate(dataloaders[0]),
            leave=True,
            desc=description(None),
            total=len(dataloaders[0]),
            ncols=0,
        ) as batch_iter:
            for _, data_dict in batch_iter:
                batch_iter.set_description(
                    description(
                        self.update(
                            {k: (v[0], v[4]) for k, v in data_dict.items()},
                            self.statistics_train,
                        )
                    )
                )
        outputs = synchronize_data(world_size, gpu, self.statistics_train.measures)
        if global_rank == 0:
            self.statistics_train.write(outputs)
        if len(dataloaders) == 2:

            def description(measure):
                return (
                    f"Metric VALIDATION : {' | '.join(f'{k}: {v:.2f}' for k, v in measure.items())}"
                    if measure is not None
                    else "Metric VALIDATION : "
                )

            with tqdm.tqdm(
                iterable=enumerate(dataloaders[1]),
                leave=True,
                desc=description(None),
                total=len(dataloaders[1]),
                ncols=0,
            ) as batch_iter:
                for _, data_dict in batch_iter:
                    batch_iter.set_description(
                        description(
                            self.update(
                                {k: (v[0], v[4]) for k, v in data_dict.items()},
                                self.statistics_validation,
                            )
                        )
                    )
            outputs = synchronize_data(world_size, gpu, self.statistics_validation.measures)
            if global_rank == 0:
                self.statistics_validation.write(outputs)


@run_distributed_app
def evaluate(
    overwrite: bool = False,
    gpu: list[int] | None = cuda_visible_devices(),
    cpu: int = 1,
    quiet: bool = False,
    tb: bool = False,
    evaluations_file: Path | str = Path("./Evaluation.yml").resolve(),
    evaluations_dir: Path | str = Path("./Evaluations").resolve(),
) -> DistributedObject:
    os.environ["KONFAI_config_file"] = str(Path(evaluations_file).resolve())
    os.environ["KONFAI_ROOT"] = "Evaluator"
    os.environ["KONFAI_STATE"] = str(State.EVALUATION)
    os.environ["KONFAI_EVALUATIONS_DIRECTORY"] = str(Path(evaluations_dir).resolve())
    return apply_config()(Evaluator)()
