import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from konfai import (
    checkpoints_directory,
    config_file,
    cuda_visible_devices,
    current_date,
    konfai_state,
    statistics_directory,
)
from konfai.data.data_manager import DataTrain
from konfai.network.network import Model, ModelLoader, NetState, Network
from konfai.utils.config import apply_config, config
from konfai.utils.utils import DataLog, DistributedObject, State, TrainerError, description, run_distributed_app


class EarlyStoppingBase:

    def __init__(self):
        pass

    def is_stopped(self) -> bool:

        return False

    def get_score(self, values: dict[str, float]):
        return sum(list(values.values()))

    def __call__(self, current_score: float) -> bool:
        return False


@config("EarlyStopping")
class EarlyStopping(EarlyStoppingBase):
    """
    Implements early stopping logic with configurable patience and monitored metrics.

    Attributes:
        monitor (list[str]): Metrics to monitor.
        patience (int): Number of checks with no improvement before stopping.
        min_delta (float): Minimum change to qualify as improvement.
        mode (str): "min" or "max" depending on optimization direction.
    """

    def __init__(
        self,
        monitor: list[str] | None = None,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        super().__init__()
        self.monitor = [] if monitor is None else monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def is_stopped(self) -> bool:
        return self.early_stop

    def get_score(self, values: dict[str, float]):
        if len(self.monitor) == 0:
            return super().get_score(values)
        for v in self.monitor:
            if v not in values.keys():
                raise TrainerError(
                    "Metric '{}' specified in EarlyStopping.monitor not found in logged values. ",
                    f"Available keys: {v}. Please check your configuration.",
                )
        return sum([i for v, i in values.items() if v in self.monitor])

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "min":
            improvement = self.best_score - current_score
        elif self.mode == "max":
            improvement = current_score - self.best_score
        else:
            raise TrainerError("Mode must be 'min' or 'max'.")

        if improvement > self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


class _Trainer:
    """
    Internal class for managing the training loop in a distributed or standalone setting.

    Handles:
    - Epoch iteration with training and optional validation
    - Mixed precision support (autocast)
    - Exponential Moving Average (EMA) model tracking
    - Early stopping
    - Logging to TensorBoard
    - Model checkpoint saving and selection (ALL or BEST)

    This class is intended to be used via a context manager
    (`with _Trainer(...) as trainer:`)  inside the public `Trainer` class.
    """

    def __init__(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        size: int,
        train_name: str,
        early_stopping: EarlyStopping | None,
        data_log: list[str] | None,
        save_checkpoint_mode: str,
        epochs: int,
        epoch: int,
        autocast: bool,
        it_validation: int | None,
        it: int,
        model: Model,
        model_ema: AveragedModel,
        dataloader_training: DataLoader,
        dataloader_validation: DataLoader | None = None,
    ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.size = size
        self.save_checkpoint_mode = save_checkpoint_mode
        self.train_name = train_name
        self.epochs = epochs
        self.epoch = epoch
        self.model = model
        self.dataloader_training = dataloader_training
        self.dataloader_validation = dataloader_validation
        self.autocast = autocast
        self.model_ema = model_ema
        self.early_stopping = EarlyStoppingBase() if early_stopping is None else early_stopping

        self.it_validation = len(dataloader_training) if it_validation is None else it_validation
        self.it = it
        self.tb = SummaryWriter(log_dir=statistics_directory() / self.train_name / "tb")
        self.data_log: dict[str, tuple[DataLog, int]] = {}
        if data_log is not None:
            for data in data_log:
                self.data_log[data.split("/")[0].replace(":", ".")] = (
                    DataLog[data.split("/")[1]],
                    int(data.split("/")[2]),
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        """Closes the SummaryWriter if used."""
        if self.tb is not None:
            self.tb.close()
        self.checkpoint_save(None)

    def run(self) -> None:
        """
        Launches the training loop, performing one epoch at a time.
        Triggers early stopping and resets data augmentations between epochs.
        """
        self.dataloader_training.dataset.load("Train")
        if self.dataloader_validation is not None:
            self.dataloader_validation.dataset.load("Validation")
            if State[konfai_state()] != State.TRAIN:
                self._validate()

        with tqdm.tqdm(
            iterable=range(self.epoch, self.epochs),
            leave=False,
            total=self.epochs,
            initial=self.epoch,
            desc="Progress",
        ) as epoch_tqdm:
            for self.epoch in epoch_tqdm:
                self.train()
                if self.early_stopping.is_stopped():
                    break
                self.dataloader_training.dataset.reset_augmentation("Train")

    def get_input(
        self,
        data_dict: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor]],
    ) -> dict[tuple[str, bool], torch.Tensor]:
        """
        Extracts input tensors from the data dict for model input.

        Args:
            data_dict (dict): Dictionary with data items structured as tuples.

        Returns:
            dict: Mapping from (group, bool_flag) to input tensors.
        """
        return {(k, v[5][0].item()): v[0] for k, v in data_dict.items()}

    def train(self) -> None:
        """
        Performs a full training epoch with support for:
        - mixed precision
        - DDP / CPU training
        - EMA updates
        - loss logging and checkpoint saving
        - validation at configurable iteration interval
        """
        self.model.train()
        self.model.module.set_state(NetState.TRAIN)
        if self.model_ema is not None:
            self.model_ema.eval()
            self.model_ema.module.set_state(NetState.TRAIN)

        with tqdm.tqdm(
            iterable=enumerate(self.dataloader_training),
            desc=f"Training : {description(self.model, self.model_ema)}",
            total=len(self.dataloader_training),
            leave=False,
            ncols=0,
        ) as batch_iter:
            for _, data_dict in batch_iter:
                with torch.amp.autocast("cuda", enabled=self.autocast):
                    input_data_dict = self.get_input(data_dict)
                    self.model(input_data_dict)
                    self.model.module.backward(self.model.module)
                    if self.model_ema is not None:
                        self.model_ema.update_parameters(self.model)
                        self.model_ema.module(input_data_dict)
                    self.it += 1
                    if (self.it) % self.it_validation == 0:
                        loss = self._train_log(data_dict)

                        if self.dataloader_validation is not None:
                            loss = self._validate()
                        self.model.module.update_lr()
                        score = self.early_stopping.get_score(loss)
                        self.checkpoint_save(score)
                        if self.early_stopping(score):
                            break

                batch_iter.set_description(f"Training : {description(self.model, self.model_ema)}")

    @torch.no_grad()
    def _validate(self) -> float:
        """
        Executes the validation phase, evaluates loss and metrics.
        Updates model states and resets augmentation for validation set.

        Returns:
            float: Validation loss.
        """
        if self.dataloader_validation is None:
            return 0
        self.model.eval()
        self.model.module.set_state(NetState.PREDICTION)
        if self.model_ema is not None:
            self.model_ema.module.set_state(NetState.PREDICTION)

        data_dict = None
        with tqdm.tqdm(
            iterable=enumerate(self.dataloader_validation),
            desc=f"Validation : {description(self.model, self.model_ema)}",
            total=len(self.dataloader_validation),
            leave=False,
            ncols=0,
        ) as batch_iter:
            for _, data_dict in batch_iter:
                input_data_dict = self.get_input(data_dict)
                self.model(input_data_dict)
                if self.model_ema is not None:
                    self.model_ema.module(input_data_dict)

                batch_iter.set_description(f"Validation : {description(self.model, self.model_ema)}")
        self.dataloader_validation.dataset.reset_augmentation("Validation")
        dist.barrier()
        self.model.train()
        self.model.module.set_state(NetState.TRAIN)
        if self.model_ema is not None:
            self.model_ema.module.set_state(NetState.TRAIN)
        return self._validation_log(data_dict)

    def checkpoint_save(self, loss: float | None) -> None:
        """
        Saves model and optimizer states. Keeps either all checkpoints or only the best one.

        Args:
            loss (float): Current loss used for best checkpoint selection.
        """
        if self.global_rank != 0:
            return

        path = checkpoints_directory() / self.train_name
        path.mkdir(parents=True, exist_ok=True)

        name = current_date() + ".pt"
        save_path = path / name

        save_dict = {
            "epoch": self.epoch,
            "it": self.it,
            "loss": loss if loss else 0,
            "Model": self.model.module.state_dict(),
        }

        if self.model_ema is not None:
            save_dict["Model_EMA"] = self.model_ema.module.state_dict()

        save_dict.update(
            {
                f"{name}_optimizer_state_dict": network.optimizer.state_dict()
                for name, network in self.model.module.get_networks().items()
                if network.optimizer is not None
            }
        )
        save_dict.update(
            {
                f"{name}_it": network._it
                for name, network in self.model.module.get_networks().items()
                if network.optimizer is not None
            }
        )
        save_dict.update(
            {
                f"{name}_nb_lr_update": network._nb_lr_update
                for name, network in self.model.module.get_networks().items()
                if network.optimizer is not None
            }
        )

        torch.save(save_dict, save_path)

        if self.save_checkpoint_mode == "BEST" and loss is not None:
            all_checkpoints = sorted(path.glob("*.pt"))
            best_ckpt = None
            best_loss = float("inf")

            for f in all_checkpoints:
                d = torch.load(f, map_location=torch.device("cpu"), weights_only=False)  # nosec B614
                if d.get("loss", float("inf")) < best_loss:
                    best_loss = d["loss"]
                    best_ckpt = f

            for f in all_checkpoints:
                if f != best_ckpt and f != save_path:
                    f.unlink()

    @torch.no_grad()
    def _log(
        self,
        type_log: str,
        data_dict: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor]],
    ) -> dict[str, float] | None:
        """
        Logs losses, metrics and optionally images to TensorBoard.

        Args:
            type_log (str): "Training" or "Validation".
            data_dict (dict): Dictionary of data from current batch.

        Returns:
            dict[str, float] | None: Dictionary of aggregated losses and metrics if rank == 0.
        """
        models: dict[str, Network] = {"": self.model.module}
        if self.model_ema is not None:
            models["_EMA"] = self.model_ema.module

        measures = DistributedObject.get_measure(
            self.world_size,
            self.global_rank,
            self.local_rank * self.size + self.size - 1,
            models,
            (
                self.it_validation
                if type_log == "Training" or self.dataloader_validation is None
                else len(self.dataloader_validation)
            ),
        )

        if self.global_rank == 0:
            images_log = []
            if len(self.data_log):
                for name, data_type in self.data_log.items():
                    if name in data_dict:
                        data_type[0](
                            self.tb,
                            f"{type_log}/{name}",
                            data_dict[name][0][: self.data_log[name][1]].detach().cpu().numpy(),
                            self.it,
                        )
                    else:
                        images_log.append(name.replace(":", "."))

            for label, model in models.items():
                for name, network in model.get_networks().items():
                    if network.measure is not None:
                        self.tb.add_scalars(
                            f"{type_log}/{name}/Loss/{label}",
                            {k: v[1] for k, v in measures[f"{name}{label}"][0].items()},
                            self.it,
                        )
                        self.tb.add_scalars(
                            f"{type_log}/{name}/Loss_weight/{label}",
                            {k: v[0] for k, v in measures[f"{name}{label}"][0].items()},
                            self.it,
                        )

                        self.tb.add_scalars(
                            f"{type_log}/{name}/Metric/{label}",
                            {k: v[1] for k, v in measures[f"{name}{label}"][1].items()},
                            self.it,
                        )
                        self.tb.add_scalars(
                            f"{type_log}/{name}/Metric_weight/{label}",
                            {k: v[0] for k, v in measures[f"{name}{label}"][1].items()},
                            self.it,
                        )

                    if len(images_log):
                        for name, layer, _ in model.get_layers(
                            [v.to(0) for k, v in self.get_input(data_dict).items() if k[1]],
                            images_log,
                        ):
                            self.data_log[name][0](
                                self.tb,
                                f"{type_log}/{name}{label}",
                                layer[: self.data_log[name][1]].detach().cpu().numpy(),
                                self.it,
                            )

            if type_log == "Training":
                for name, network in self.model.module.get_networks().items():
                    if network.optimizer is not None:
                        self.tb.add_scalar(
                            f"{type_log}/{name}/Learning Rate",
                            network.optimizer.param_groups[0]["lr"],
                            self.it,
                        )

        if self.global_rank == 0:
            loss = {}
            for name, network in self.model.module.get_networks().items():
                if network.measure is not None:
                    loss.update({k: v[1] for k, v in measures[f"{name}{label}"][0].items()})
                    loss.update({k: v[1] for k, v in measures[f"{name}{label}"][1].items()})
            return loss
        return None

    @torch.no_grad()
    def _train_log(self, data_dict: dict[str, tuple[torch.Tensor, int, int, int]]) -> dict[str, float]:
        """Wrapper for _log during training."""
        return self._log("Training", data_dict)

    @torch.no_grad()
    def _validation_log(self, data_dict: dict[str, tuple[torch.Tensor, int, int, int]]) -> dict[str, float]:
        """Wrapper for _log during validation."""
        return self._log("Validation", data_dict)


@config("Trainer")
class Trainer(DistributedObject):
    """
    Public API for training a model using the KonfAI framework.
    Wraps setup, checkpointing, resuming, logging, and launching distributed _Trainer.

    Main responsibilities:
    - Initialization from config (via @config)
    - Model and EMA setup
    - Checkpoint loading and saving
    - Distributed setup and launch

    Args:
        model (ModelLoader): Loader for model architecture.
        dataset (DataTrain): Training/validation dataset.
        train_name (str): Training session name.
        manual_seed (int | None): Random seed.
        epochs (int): Number of epochs to run.
        it_validation (int | None): Validation interval.
        autocast (bool): Enable AMP training.
        gradient_checkpoints (list[str] | None): Modules to use gradient checkpointing on.
        gpu_checkpoints (list[str] | None): Modules to pin on specific GPUs.
        ema_decay (float): EMA decay factor.
        data_log (list[str] | None): Logging instructions.
        early_stopping (EarlyStopping | None): Optional early stopping config.
        save_checkpoint_mode (str): Either "BEST" or "ALL".
    """

    def __init__(
        self,
        model: ModelLoader = ModelLoader(),
        dataset: DataTrain = DataTrain(),
        train_name: str = "default|TRAIN_01",
        manual_seed: int | None = None,
        epochs: int = 100,
        it_validation: int | None = None,
        autocast: bool = False,
        gradient_checkpoints: list[str] | None = None,
        gpu_checkpoints: list[str] | None = None,
        ema_decay: float = 0,
        data_log: list[str] | None = None,
        early_stopping: EarlyStopping | None = None,
        save_checkpoint_mode: str = "BEST",
    ) -> None:
        if os.environ["KONFAI_CONFIG_MODE"] != "Done":
            exit(0)
        super().__init__(train_name)
        self.manual_seed = manual_seed
        self.dataset = dataset
        self.autocast = autocast
        self.epochs = epochs
        self.epoch = 0
        self.early_stopping = early_stopping
        self.it = 0
        self.it_validation = it_validation
        self.model = model.get_model(train=True)
        self.ema_decay = ema_decay
        self.model_ema: torch.optim.swa_utils.AveragedModel | None = None
        self.data_log = data_log

        modules = []
        for i, _ in self.model.named_modules():
            modules.append(i)
        if self.data_log is not None:
            for k in self.data_log:
                tmp = k.split("/")[0].replace(":", ".")
                if tmp not in self.dataset.get_groups_dest() and tmp not in modules:
                    raise TrainerError(
                        f"Invalid key '{tmp}' in `data_log`.",
                        f"This key is neither a destination group from the dataset ({self.dataset.get_groups_dest()})",
                        f"nor a valid module name in the model ({modules}).",
                        "Please check your `data_log` configuration,"
                        " it should reference either a model output or a dataset group.",
                    )

        self.gradient_checkpoints = gradient_checkpoints
        self.gpu_checkpoints = gpu_checkpoints
        self.save_checkpoint_mode = save_checkpoint_mode
        self.config_namefile_src = config_file().name.replace(".yml", "")
        self.config_namefile = (
            statistics_directory() / self.name / f"{self.config_namefile_src.split("/")[-1]}_{self.it}.yml"
        )
        self.size = len(self.gpu_checkpoints) + 1 if self.gpu_checkpoints else 1

    def set_model(self, path_to_model: Path) -> None:
        self.path_to_model = str(path_to_model)

    def __exit__(self, exc_type, value, traceback):
        """Exit training context and trigger save of model/checkpoints."""
        super().__exit__(exc_type, value, traceback)
        self._save()

    def _load(self) -> dict[str, dict[str, torch.Tensor]]:
        """
        Loads a previously saved checkpoint from local disk or URL.

        Returns:
            dict: State dictionary loaded from checkpoint.
        """
        if self.path_to_model.startswith("https://"):
            try:
                state_dict = {
                    self.path_to_model.split(":")[1]: torch.hub.load_state_dict_from_url(
                        url=self.path_to_model.split(":")[0], map_location="cpu", check_hash=True
                    )
                }
            except Exception:
                raise Exception(f"Model : {self.path_to_model} does not exist !")
        elif Path(self.path_to_model).exists():
            state_dict = torch.load(
                str(self.path_to_model), map_location=torch.device("cpu"), weights_only=False
            )  # nosec B614
        else:
            raise ValueError(f"Invalid model path entry: {self.path_to_model}")

        if "epoch" in state_dict:
            self.epoch = state_dict["epoch"]
        if "it" in state_dict:
            self.it = state_dict["it"]
        return state_dict

    def _save(self) -> None:
        if self.config_namefile.exists():
            os.rename(
                self.config_namefile,
                self.config_namefile.parent / f"{self.config_namefile.name.replace(".yml", "")}_{self.it}.yml",
            )

    def _avg_fn(self, averaged_model_parameter: float, model_parameter, num_averaged):
        """
        EMA update rule used by AveragedModel.

        Returns:
            torch.Tensor: Blended parameter using decay factor.
        """
        return (1 - self.ema_decay) * averaged_model_parameter + self.ema_decay * model_parameter

    def setup(self, world_size: int):
        """
        Initializes the training environment:
        - Clears previous outputs (unless resuming)
        - Initializes model and EMA
        - Loads checkpoint (if resuming)
        - Prepares dataloaders

        Args:
            world_size (int): Total number of distributed processes.
        """
        state = State[konfai_state()]
        if state != State.RESUME and (checkpoints_directory() / self.name).exists():
            if os.environ["KONFAI_OVERWRITE"] != "True":
                accept = input(f"The model {self.name} already exists ! Do you want to overwrite it (yes,no) : ")
                if accept != "yes":
                    exit(0)
            for directory_path in [
                statistics_directory(),
                checkpoints_directory(),
            ]:
                if (directory_path / self.name).exists():
                    (directory_path / self.name).unlink()

        state_dict = {}
        if state != State.TRAIN:
            state_dict = self._load()

        self.model.init(self.autocast, state, self.dataset.get_groups_dest())
        self.model.init_outputs_group()
        self.model._compute_channels_trace(
            self.model,
            self.model.in_channels,
            self.gradient_checkpoints,
            self.gpu_checkpoints,
        )
        self.model.load(state_dict, init=True, ema=False)
        if self.ema_decay > 0:
            self.model_ema = AveragedModel(self.model, avg_fn=self._avg_fn)
            if state_dict is not None:
                self.model_ema.module.load(state_dict, init=False, ema=True)

        (statistics_directory() / self.name).mkdir(exist_ok=True)
        shutil.copyfile(self.config_namefile_src + ".yml", self.config_namefile)

        self.dataloader, train_names, validation_names = self.dataset.get_data(world_size // self.size)
        with open(statistics_directory() / self.name / f"Train_{self.it}.txt", "w") as f:
            for name in train_names:
                f.write(name + "\n")
        with open(statistics_directory() / self.name / f"Validation_{self.it}.txt", "w") as f:
            for name in validation_names:
                f.write(name + "\n")

    def run_process(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        dataloaders: list[DataLoader],
    ):
        """
        Launches the actual training process via internal `_Trainer` class.
        Wraps model with DDP or CPU fallback, attaches EMA, and starts training.

        Args:
            world_size (int): Total number of distributed processes.
            global_rank (int): Global rank of the current process.
            local_rank (int): Local rank within the node.
            dataloaders (list[DataLoader]): Training and validation dataloaders.
        """
        model = Network.to(self.model, local_rank * self.size) if len(cuda_visible_devices()) else self.model
        model = DDP(model, static_graph=True) if dist.is_initialized() else Model(model)

        if self.model_ema is not None:
            self.model_ema.module = Network.to(self.model_ema.module, local_rank)
        with _Trainer(
            world_size,
            global_rank,
            local_rank,
            self.size,
            self.name,
            self.early_stopping,
            self.data_log,
            self.save_checkpoint_mode,
            self.epochs,
            self.epoch,
            self.autocast,
            self.it_validation,
            self.it,
            model,
            self.model_ema,
            *dataloaders,
        ) as t:
            t.run()


@run_distributed_app
def train(
    command: State = State.TRAIN,
    overwrite: bool = False,
    model: Path | str | None = None,
    gpu: list[int] | None = cuda_visible_devices(),
    cpu: int | None = None,
    quiet: bool = False,
    tensorboard: bool = False,
    config: Path | str = Path("./Config.yml"),
    checkpoints_dir: Path | str = Path("./Checkpoints/"),
    statistics_dir: Path | str = Path("./Statistics/"),
) -> DistributedObject:
    os.environ["KONFAI_config_file"] = str(Path(config).resolve())
    os.environ["KONFAI_ROOT"] = "Trainer"
    os.environ["KONFAI_STATE"] = str(command)
    os.environ["KONFAI_CHECKPOINTS_DIRECTORY"] = str(Path(checkpoints_dir).resolve())
    os.environ["KONFAI_STATISTICS_DIRECTORY"] = str(Path(statistics_dir).resolve())
    trainer = apply_config()(Trainer)()
    if model is not None:
        trainer.set_model(Path(model))
    return trainer


if __name__ == "__main__":
    train(State.TRAIN, False, None)
