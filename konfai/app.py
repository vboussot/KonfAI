import inspect
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk  # noqa: N813

from konfai import cuda_visible_devices
from konfai.utils.dataset import Dataset
from konfai.utils.utils import SUPPORTED_EXTENSIONS, ModelDirectory, ModelHF, ModelLoad, State


def run_distributed_app(
    func: Callable[..., None],
) -> Callable[..., None]:

    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        params = sig.parameters
        kwargs_fun = {k: v for k, v in kwargs.items() if k in params}

        bound = sig.bind_partial(*args, **kwargs_fun)
        bound.apply_defaults()

        tmp_dir = bound.arguments.get("tmp_dir", Path(tempfile.mkdtemp(prefix="konfai_app_"))).resolve()

        user_dir = os.getcwd()
        try:
            os.makedirs(tmp_dir, exist_ok=True)
            os.chdir(str(tmp_dir))
            sys.path.insert(0, os.getcwd())
            func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\n[KonfAI-Apps] Manual interruption (Ctrl+C)")
            exit(0)
        finally:
            if Path(os.getcwd()).resolve() != Path(user_dir).resolve():
                tmp_dir = Path(os.getcwd()).resolve()
                if str(tmp_dir) in sys.path:
                    sys.path.remove(str(tmp_dir))
                os.chdir(user_dir)
                if tmp_dir.parent == Path(tempfile.gettempdir()):
                    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    return wrapper


class KonfAIApp:

    def __init__(self, app: str) -> None:
        self.model: ModelLoad
        if len(app.split(":")) == 2:
            self.model = ModelHF(app.split(":")[0], app.split(":")[1])
        else:
            self.model = ModelDirectory(Path(app).resolve().parent, Path(app).name)
        self._user_dir = Path(os.getcwd())

    @staticmethod
    def _match_supported(file: Path) -> bool:
        lower = file.name.lower()
        return any(lower.endswith("." + ext) for ext in SUPPORTED_EXTENSIONS)

    @staticmethod
    def _list_supported_files(paths: list[Path]) -> list[Path]:
        files = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Path does not exist: '{path}'")

            if path.is_file():
                if KonfAIApp._match_supported(path):
                    files.append(path)
                else:
                    raise FileNotFoundError(f"No supported file found: '{path.name}' is not a supported format.")
            else:

                for f in path.rglob("*"):
                    if f.is_file() and KonfAIApp._match_supported(f):
                        files.append(f)
                if not files:
                    raise FileNotFoundError(f"No supported files found in directory: '{path}'.")
        return files

    @staticmethod
    def symlink(src: Path, dst: Path) -> None:
        if dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            os.symlink(src, dst, target_is_directory=src.is_dir())
        except OSError:
            # fallback Windows / FS without symlink
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    def _write_inputs_to_dataset(self, inputs: list[list[Path]]) -> None:
        dataset_path = Path("./Dataset/")
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        for i, input_path in enumerate(inputs):
            for idx, file in enumerate(KonfAIApp._list_supported_files(input_path)):
                KonfAIApp.symlink(file, dataset_path / f"P{idx:03d}" / f"Volume_{i}{file.suffix}")

    def _write_inference_stack_to_dataset(self, inputs: list[list[Path]]) -> None:
        dataset_path = Path("./Dataset/")
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        for i, input_path in enumerate(inputs):
            for idx, file in enumerate(KonfAIApp._list_supported_files(input_path)):
                reader = sitk.ImageFileReader()
                reader.SetFileName(file)
                reader.ReadImageInformation()
                n_channels = reader.GetNumberOfComponents()
                if n_channels > 1:
                    KonfAIApp.symlink(file, dataset_path / f"P{idx:03d}" / f"Volume_{i}{file.suffix}")
                else:
                    raise FileNotFoundError(
                        "Invalid input volume for inference: a multi-channel volume stack is required, "
                        "but a single-channel volume was provided."
                    )

    def _write_gt_to_dataset(self, gt: list[list[Path]]) -> None:
        for i, gt_path in enumerate(gt):
            for idx, file in enumerate(KonfAIApp._list_supported_files(gt_path)):
                KonfAIApp.symlink(file, Path(f"./Dataset/P{idx:03d}/Reference_{i}{file.suffix}"))

    def _write_mask_or_default(self, mask: list[list[Path]] | None) -> None:
        if mask is None:
            dataset = Dataset("Dataset", "mha")
            names = dataset.get_names("Volume_0")
            for name in names:
                data, attr = dataset.read_data("Volume_0", name)
                dataset.write("Mask_0", name, np.ones_like(data), attr)
        else:
            for i, mask_path in enumerate(mask):
                for idx, file in enumerate(KonfAIApp._list_supported_files(mask_path)):
                    KonfAIApp.symlink(file, Path(f"./Dataset/P{idx:03d}/Mask_{i}{file.suffix}"))

    @run_distributed_app
    def infer(
        self,
        inputs: list[list[Path]],
        output: Path = Path("./Output/").resolve(),
        ensemble: int = 0,
        tta: int = 0,
        mc_dropout: int = 0,
        prediction_file: str = "Prediction.yml",
        gpu: list[int] = cuda_visible_devices(),
        cpu: int = 1,
        quiet: bool = False,
        tmp_dir: Path = Path(tempfile.mkdtemp(prefix="konfai_app_")),
    ) -> None:
        self._write_inputs_to_dataset(inputs)
        models_path = self.model.install_inference(tta, ensemble, mc_dropout, prediction_file)
        from konfai.predictor import predict

        predict(models_path, True, gpu, cpu, quiet, False, Path(prediction_file).resolve())
        if Path("./Predictions").absolute().exists():
            shutil.copytree(Path("./Predictions").absolute(), output, dirs_exist_ok=True)

    @run_distributed_app
    def evaluate(
        self,
        inputs: list[list[Path]],
        gt: list[list[Path]],
        output: Path = Path("./Output/"),
        mask: list[list[Path]] | None = None,
        evaluation_file: str = "Evaluation.yml",
        gpu: list[int] = cuda_visible_devices(),
        cpu: int = 0,
        quiet: bool = False,
        tmp_dir: Path = Path(tempfile.mkdtemp(prefix="konfai_app_")),
    ) -> None:
        self._write_inputs_to_dataset(inputs)
        self._write_gt_to_dataset(gt)
        self._write_mask_or_default(mask)
        self.model.install_evaluation(evaluation_file)
        from konfai.evaluator import evaluate

        evaluate(True, gpu, cpu, quiet, False, Path(evaluation_file).resolve())
        if Path("./Evaluations").exists():
            shutil.copytree("./Evaluations", output, dirs_exist_ok=True)

    @run_distributed_app
    def uncertainty(
        self,
        inputs: list[list[Path]],
        output: Path = Path("./Output/"),
        uncertainty_file: str = "Uncertainty.yml",
        gpu: list[int] = cuda_visible_devices(),
        cpu: int = 0,
        quiet: bool = False,
        tmp_dir: Path = Path(tempfile.mkdtemp(prefix="konfai_app_")),
    ) -> None:
        self._write_inference_stack_to_dataset(inputs)
        self.model.install_uncertainty(uncertainty_file)

        from konfai.evaluator import evaluate

        evaluate(True, gpu, cpu, quiet, False, Path(uncertainty_file).resolve(), Path("./Uncertainties/"))
        if Path("./Uncertainties").exists():
            shutil.copytree("./Uncertainties", output, dirs_exist_ok=True)

    def pipeline(
        self,
        inputs: list[list[Path]],
        gt: list[list[Path]],
        output: Path = Path("./Output/"),
        ensemble: int = 0,
        tta: int = 0,
        mc_dropout: int = 0,
        prediction_file: str = "Prediction.yml",
        mask: list[list[Path]] | None = None,
        evaluation_file: str = "Evaluation.yml",
        uncertainty_file: str = "Uncertainty.yml",
        gpu: list[int] = cuda_visible_devices(),
        cpu: int = 0,
        quiet: bool = False,
        tmp_dir: Path = Path(tempfile.mkdtemp(prefix="konfai_app_")),
    ) -> None:
        self.infer(inputs, output / "Predictions", ensemble, tta, mc_dropout, prediction_file, gpu, cpu, quiet, tmp_dir)
        outputs = []
        inference_stacks = []
        for f in (output / "Predictions").rglob("*"):
            if f.is_file() and KonfAIApp._match_supported(f):
                if f.name == "InferenceStack.mha":
                    inference_stacks.append(f)
                else:
                    outputs.append(f)
        self.evaluate([outputs], gt, output / "Evaluations", mask, evaluation_file, gpu, cpu, quiet, tmp_dir)
        self.uncertainty([inference_stacks], output / "Uncertainties", uncertainty_file, gpu, cpu, quiet, tmp_dir)

    @run_distributed_app
    def fine_tune(
        self,
        dataset: Path,
        name: str = "Finetune",
        output: Path = Path("./Output/"),
        epochs: int = 10,
        it_validation: int = 1000,
        gpu: list[int] = cuda_visible_devices(),
        cpu: int = 0,
        quiet: bool = False,
        config_file: str = "Config.yml",
        tmp_dir: Path = Path(tempfile.mkdtemp(prefix="konfai_app_")),
    ) -> None:
        models_path = self.model.install_fine_tune(config_file, tmp_dir, name, epochs, it_validation)
        KonfAIApp.symlink(dataset, Path("./Dataset").absolute())
        from konfai.trainer import train

        train(State.RESUME, True, models_path[0], gpu, cpu, quiet, False, config_file)
