# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Repository and metadata adapters for KonfAI Apps."""

from __future__ import annotations

import importlib.metadata
import json
import re
import shutil
import subprocess  # nosec B404
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import requests
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.hf_api import RepoFolder
from packaging.requirements import Requirement
from ruamel.yaml import YAML

from konfai import RemoteServer
from konfai.utils.errors import AppMetadataError, AppRepositoryError, ConfigError


def get_available_apps_on_remote_server(remote_server: RemoteServer) -> list[str]:
    """Return the list of app identifiers exposed by a remote KonfAI app server."""
    r = requests.get(
        f"{remote_server.get_url()}/repo_apps_list",
        headers=remote_server.get_headers(),
        timeout=remote_server.timeout,
    )
    r.raise_for_status()

    data = r.json()
    apps = data.get("apps")

    if not isinstance(apps, list):
        raise ValueError("Invalid response from remote server: expected 'apps' list.")

    return [str(a) for a in apps]


def get_available_apps_on_hf_repo(repo_id: str, force_update: bool) -> list[str]:
    """List app folders available inside a Hugging Face repository."""
    api = HfApi()
    app_names: list[str] = []

    if force_update:
        try:
            tree = api.list_repo_tree(repo_id=repo_id)
            for entry in tree:
                app_name = Path(entry.path).name
                if isinstance(entry, RepoFolder) and is_app_repo(
                    LocalAppRepositoryFromHF.get_filenames(repo_id, app_name, True)
                ):
                    app_names.append(app_name)
            return app_names
        except Exception as exc:
            raise AppRepositoryError(
                f"Failed to inspect Hugging Face repository '{repo_id}'. "
                "Unable to list its tree and detect valid application folders. "
                "Please check that the repository exists, that you have access to it, "
                "that your authentication is valid, and that your internet connection is working.\n"
                f"Original error: {exc}"
            )

    try:
        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_files_only=True,
            revision="main",
        )  # nosec B615
        root = Path(snapshot_dir)
        for path in root.iterdir():
            if path.is_dir():
                app_name = path.name
                if is_app_repo(LocalAppRepositoryFromHF.get_filenames(repo_id, app_name, False)):
                    app_names.append(app_name)
        return app_names
    except Exception:
        return get_available_apps_on_hf_repo(repo_id, True)


def is_app_repo(filenames: list[str]) -> bool:
    """Return whether the given repository file list looks like a KonfAI app."""
    return any(filename.endswith("app.json") for filename in filenames)


class VolumeType(Enum):
    SEGMENTATION = "SEGMENTATION"
    VOLUME = "VOLUME"
    FIDUCIALS = "FIDUCIALS"
    TRANSFORM = "TRANSFORM"


@dataclass
class VRAMPlanEntry:
    patch_size: list[int]
    batch_size: int


@dataclass
class TerminologyEntry:
    name: str
    color: str


@dataclass
class DataEntry:
    display_name: str
    volume_type: VolumeType
    required: bool


@dataclass(frozen=True, slots=True)
class EvaluationKey:
    display_name: str
    evaluation_file: str


class AppRepositoryInfo(ABC):
    """Common interface implemented by local, HF, and remote app repositories."""

    def __init__(
        self,
        app_name: str,
        display_name: str,
        description: str,
        short_description: str,
        checkpoints_name: list[str],
        checkpoints_name_available: list[str],
        maximum_tta: int,
        mc_dropout: int,
        inputs: dict[str, DataEntry],
        outputs: dict[str, DataEntry],
        inputs_evaluations: dict[EvaluationKey, dict[str, DataEntry]],
        terminology: dict[int, TerminologyEntry] | None = None,
        vram_plan: dict[int, VRAMPlanEntry] | None = None,
    ) -> None:
        super().__init__()
        self._app_name = app_name
        self._display_name = display_name
        self._description = description
        self._short_description = short_description
        self._checkpoints_name = checkpoints_name
        self._checkpoints_name_available = checkpoints_name_available
        self._maximum_tta = maximum_tta
        self._mc_dropout = mc_dropout
        self._inputs = inputs
        self._outputs = outputs
        self._inputs_evaluations = inputs_evaluations
        self._terminology = terminology
        self._vram_plan = vram_plan

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  app_name={self._app_name!r},\n"
            f"  display_name={self._display_name!r},\n"
            f"  description={self._description!r},\n"
            f"  short_description={self._short_description!r}\n"
            f"  checkpoints_name={self._checkpoints_name!r},\n"
            f"  maximum_tta={self._maximum_tta!r},\n"
            f"  mc_dropout={self._mc_dropout!r},\n"
            f"  inputs={self._inputs!r},\n"
            f"  outputs={self._outputs!r},\n"
            f"  inputs_evaluations={self._inputs_evaluations!r},\n"
            f"  terminology={self._terminology!r},\n"
            f"  vram_plan={self._vram_plan!r}\n"
            f")"
        )

    def get_display_name(self) -> str:
        return self._display_name

    def get_description(self) -> str:
        return self._description

    def get_short_description(self) -> str:
        return self._short_description

    def get_checkpoints_name(self) -> list[str]:
        return self._checkpoints_name

    def get_checkpoints_name_available(self) -> list[str]:
        return self._checkpoints_name_available

    def get_maximum_tta(self) -> int:
        return self._maximum_tta

    def get_mc_dropout(self) -> int:
        return self._mc_dropout

    def get_inputs(self) -> dict[str, DataEntry]:
        return self._inputs

    def get_outputs(self) -> dict[str, DataEntry]:
        return self._outputs

    def get_evaluations_inputs(self) -> dict[EvaluationKey, dict[str, DataEntry]]:
        return self._inputs_evaluations

    def get_terminology(self) -> dict[int, TerminologyEntry] | None:
        return self._terminology

    @abstractmethod
    def has_capabilities(self) -> tuple[bool, bool, bool]:
        raise NotImplementedError()

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def download_config_file(self) -> list[Path]:
        raise NotImplementedError()


class LocalAppRepository(AppRepositoryInfo):
    """Base implementation shared by local-directory and Hugging Face apps."""

    def __init__(self, app_name: str) -> None:
        self._app_name = app_name
        filenames = self._get_filenames()
        if not is_app_repo(filenames):
            raise AppRepositoryError("Missing 'app.json' in apps.")

        required_keys = ["description", "short_description", "tta", "mc_dropout", "display_name"]
        for filename in filenames:
            if not filename.endswith(".pt"):
                self._download(filename)
        metadata_file_path = self._download("app.json")
        with open(metadata_file_path, encoding="utf-8") as file:
            app_repository_metadata = json.load(file)

        missing = [key for key in required_keys if key not in app_repository_metadata]
        if missing:
            raise AppMetadataError(f"Missing keys in app.json: {', '.join(missing)}")

        inputs: dict[str, DataEntry] = {}
        if "inputs" in app_repository_metadata:
            inputs = {
                key: DataEntry(
                    display_name=value["display_name"],
                    volume_type=VolumeType(value["volume_type"]),
                    required=bool(value["required"]),
                )
                for key, value in app_repository_metadata["inputs"].items()
            }

        outputs: dict[str, DataEntry] = {}
        if "outputs" in app_repository_metadata:
            outputs = {
                key: DataEntry(
                    display_name=value["display_name"],
                    volume_type=VolumeType(value["volume_type"]),
                    required=bool(value["required"]),
                )
                for key, value in app_repository_metadata["outputs"].items()
            }

        inputs_evaluations: dict[EvaluationKey, dict[str, DataEntry]] = {}
        if "inputs_evaluations" in app_repository_metadata:
            for display_name, by_file in app_repository_metadata["inputs_evaluations"].items():
                for evaluation_file, entries in by_file.items():
                    eval_key = EvaluationKey(display_name=display_name, evaluation_file=evaluation_file)
                    inputs_evaluations[eval_key] = {
                        key: DataEntry(
                            display_name=value["display_name"],
                            volume_type=VolumeType(value["volume_type"]),
                            required=bool(value["required"]),
                        )
                        for key, value in entries.items()
                    }

        try:
            maximum_tta = int(app_repository_metadata["tta"])
        except Exception:
            raise AppMetadataError("The field 'tta' must be an integer.")

        try:
            mc_dropout = int(app_repository_metadata["mc_dropout"])
        except Exception:
            raise AppMetadataError("The field 'mc_dropout' must be an integer.")

        terminology: dict[int, TerminologyEntry] | None = None
        if "terminology" in app_repository_metadata:
            terminology = {
                int(key): TerminologyEntry(name=value["name"], color=value["color"])
                for key, value in app_repository_metadata["terminology"].items()
            }

        vram_plan: dict[int, VRAMPlanEntry] | None = None
        if "vram_plan" in app_repository_metadata:
            vram_plan = {
                int(key): VRAMPlanEntry(
                    patch_size=list(map(int, value["patch_size"])),
                    batch_size=int(value["batch_size"]),
                )
                for key, value in app_repository_metadata["vram_plan"].items()
            }

        checkpoints_name: list[str] = app_repository_metadata.get("models", [])
        checkpoints_name_available: list[str] = [
            checkpoint_name for checkpoint_name in checkpoints_name if checkpoint_name in filenames
        ]

        super().__init__(
            app_name=app_name,
            display_name=str(app_repository_metadata["display_name"]),
            description=str(app_repository_metadata["description"]),
            short_description=str(app_repository_metadata["short_description"]),
            checkpoints_name=checkpoints_name,
            checkpoints_name_available=checkpoints_name_available,
            maximum_tta=maximum_tta,
            mc_dropout=mc_dropout,
            inputs=inputs,
            outputs=outputs,
            inputs_evaluations=inputs_evaluations,
            terminology=terminology,
            vram_plan=vram_plan,
        )

    def _set_number_of_augmentation(self, inference_file_path: str, new_value: int) -> None:
        new_value = int(np.clip(new_value, 0, self._maximum_tta))
        yaml = YAML()
        with open(inference_file_path) as file:
            data = yaml.load(file)

        if new_value > 0:
            tmp = data["Predictor"]["Dataset"]["augmentations"]
            if "DataAugmentation_0" in tmp:
                tmp["DataAugmentation_0"]["nb"] = new_value
        else:
            data["Predictor"]["Dataset"]["augmentations"] = {}

        with open(inference_file_path, "w") as file:
            yaml.dump(data, file)

    def _disable_uncertainty(self, inference_file_path: str) -> None:
        yaml = YAML()
        with open(inference_file_path) as file:
            data = yaml.load(file)

        predictor = data["Predictor"]
        outputs = predictor["outputs_dataset"]

        has_inference_stack = False
        for value in outputs.values():
            after = value["OutputDataset"]["after_reduction_transforms"]
            if "InferenceStack" in after:
                has_inference_stack = True
                break
        if not has_inference_stack:
            return

        predictor["combine"] = "Mean"
        for value in outputs.values():
            value["OutputDataset"]["reduction"] = "Mean"
            if "InferenceStack" in value["OutputDataset"]["after_reduction_transforms"]:
                del value["OutputDataset"]["after_reduction_transforms"]["InferenceStack"]

        with open(inference_file_path, "w") as file:
            yaml.dump(data, file)

    def _set_patch_size_and_batch_size(
        self,
        inference_file_path: str,
        patch_size: list[int],
        batch_size: int,
    ) -> None:
        yaml = YAML()
        with open(inference_file_path) as file:
            data = yaml.load(file)

        tmp = data["Predictor"]["Dataset"]
        tmp["Patch"]["patch_size"] = patch_size
        tmp["batch_size"] = batch_size

        with open(inference_file_path, "w") as file:
            yaml.dump(data, file)

    @abstractmethod
    def _get_filenames(self) -> list[str]:
        raise NotImplementedError()

    @abstractmethod
    def _download(self, filename: str) -> Path:
        raise NotImplementedError()

    def has_capabilities(self) -> tuple[bool, bool, bool]:
        filenames = self._get_filenames()
        inference_support = len(self.get_inputs()) > 0
        evaluation_support = len(self.get_evaluations_inputs()) > 0
        uncertainty_support = any(filename == "Uncertainty.yml" for filename in filenames)
        return inference_support, evaluation_support, uncertainty_support

    def download_config_file(self) -> list[Path]:
        filenames = self._get_filenames()
        files_path: list[Path] = []
        for filename in filenames:
            if not filename.endswith(".pt"):
                files_path.append(self._download(filename))
        return files_path

    def download_inference(
        self, number_of_model: int, name_of_models: list[str], prediction_file: str
    ) -> tuple[list[Path], Path, list[Path]]:
        filenames = self._get_filenames()
        models_path: list[Path] = []
        codes_path: list[Path] = []

        inference_file_path = self._download(prediction_file)
        available_models = [name for name in filenames if name.endswith(".pt")]
        if len(name_of_models):
            for name in name_of_models:
                models_path.append(self._download(name if name.endswith(".pt") else name + ".pt"))
        else:
            if len(available_models) < number_of_model and isinstance(self, LocalAppRepositoryFromHF):
                filenames = LocalAppRepositoryFromHF.get_filenames(self._repo_id, self._app_name, True)
            available_models = [name for name in filenames if name.endswith(".pt")]
            if len(available_models) < number_of_model:
                raise AppRepositoryError(
                    f"Expected {number_of_model} model files (.pt), but found "
                    f"{len(available_models)} in the repository."
                )
            for name in available_models[:number_of_model]:
                models_path.append(self._download(name))

        for filename in filenames:
            if filename.endswith(".py"):
                codes_path.append(self._download(filename))

        if "requirements.txt" in filenames:
            with open(self._download("requirements.txt"), encoding="utf-8") as file:
                required_lines = [line.strip() for line in file if line.strip() and not line.startswith("#")]
                installed = {
                    dist.metadata["Name"].lower(): dist.version
                    for dist in importlib.metadata.distributions()
                    if dist.metadata.get("Name")
                }
                missing_or_outdated = []
                for line in required_lines:
                    req = Requirement(line)
                    name = req.name.lower()
                    installed_version_str = installed.get(name)
                    if installed_version_str is None:
                        missing_or_outdated.append(line)
                        continue
                    if req.specifier and not req.specifier.contains(installed_version_str, prereleases=True):
                        missing_or_outdated.append(line)

                if missing_or_outdated:
                    try:
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", *missing_or_outdated]
                        )  # nosec B603
                    except subprocess.CalledProcessError as exc:
                        raise AppRepositoryError(f"Failed to install packages: {exc}") from exc

        return models_path, inference_file_path, codes_path

    def download_app(self) -> list[Path]:
        filenames = self._get_filenames()
        files_path: list[Path] = []
        for filename in filenames:
            files_path.append(self._download(filename))
            print(f"[KonfAI-Apps] {filename} is ready.")
        return files_path

    def download_evaluation(self, evaluation_file: str) -> tuple[Path, list[Path]]:
        filenames = self._get_filenames()
        codes_path: list[Path] = []
        evaluation_file_path = self._download(evaluation_file)
        for filename in filenames:
            if filename.endswith(".py"):
                codes_path.append(self._download(filename))
        return evaluation_file_path, codes_path

    def download_uncertainty(self, uncertainty_file: str) -> tuple[Path, list[Path]]:
        filenames = self._get_filenames()
        codes_path: list[Path] = []
        uncertainty_file_path = self._download(uncertainty_file)
        for filename in filenames:
            if filename.endswith(".py"):
                codes_path.append(self._download(filename))
        return uncertainty_file_path, codes_path

    def install_inference(
        self,
        number_of_augmentation: int,
        number_of_model: int,
        name_of_models: list[str],
        number_of_mc_dropout: int,
        uncertainty: bool,
        prediction_file: str,
        available_vram: float | None,
    ) -> list[Path]:
        if len(name_of_models) == 0 and number_of_model == 0:
            number_of_model = len(self._checkpoints_name)

        models_path, inference_file_path, codes_path = self.download_inference(
            number_of_model, name_of_models, prediction_file
        )
        shutil.copy2(inference_file_path, prediction_file)
        self._set_number_of_augmentation(prediction_file, number_of_augmentation)
        if not uncertainty:
            self._disable_uncertainty(prediction_file)
        if self._vram_plan is not None and available_vram is not None:
            thresholds = sorted(self._vram_plan.keys())
            selected_t = thresholds[0]
            for threshold in thresholds:
                if threshold <= available_vram:
                    selected_t = threshold
                else:
                    break
            vram_plan = self._vram_plan[selected_t]
            self._set_patch_size_and_batch_size(prediction_file, vram_plan.patch_size, vram_plan.batch_size)
        for code_path in codes_path:
            if code_path.suffix == ".py":
                shutil.copy2(code_path, code_path.name)

        return models_path

    def install_evaluation(self, evaluation_file: str) -> None:
        evaluation_file_path, codes_path = self.download_evaluation(evaluation_file)
        shutil.copy2(evaluation_file_path, evaluation_file)
        for code_path in codes_path:
            if code_path.suffix == ".py":
                shutil.copy2(code_path, code_path.name)

    def install_uncertainty(self, uncertainty_file: str) -> None:
        uncertainty_file_path, codes_path = self.download_uncertainty(uncertainty_file)
        shutil.copy2(uncertainty_file_path, uncertainty_file)
        for code_path in codes_path:
            if code_path.suffix == ".py":
                shutil.copy2(code_path, code_path.name)

    def install_fine_tune(
        self,
        config_file: str,
        path: Path,
        display_name: str,
        epochs: int,
        it_validation: int | None,
    ) -> list[Path]:
        src_paths = self.download_app()
        models_path = []

        overwrite_all = None

        def ask_overwrite_cli(dest_path: Path) -> bool:
            nonlocal overwrite_all
            if overwrite_all is not None:
                return overwrite_all

            while True:
                print(f"KonfAI-Apps] File already exists: {dest_path}")
                choice = input("Overwrite? [y]es / [n]o / [a]ll / [s]kip_all: ").strip().lower()
                if choice == "y":
                    return True
                if choice == "n":
                    return False
                if choice == "a":
                    overwrite_all = True
                    return True
                if choice == "s":
                    overwrite_all = False
                    return False
                print("[KonfAI-Apps] Invalid input. Please choose y / n / a / s.")

        for src in src_paths:
            if src.is_dir():
                for item in src.rglob("*"):
                    rel = item.relative_to(src)
                    dest = path / rel
                    if item.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                        continue

                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists() and not ask_overwrite_cli(dest):
                        continue
                    shutil.copy2(item, dest)
                    if str(item).endswith(".pt"):
                        models_path.append(item)

            elif src.is_file() or src.is_symlink():
                dest = path / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if str(src).endswith(".pt"):
                    models_path.append(src)
                if dest.exists() and not ask_overwrite_cli(dest):
                    continue
                shutil.copy2(src, dest)

        metadata_file = path / "app.json"
        config_file_path = path / config_file
        if not metadata_file.exists():
            raise ConfigError(
                f"Metadata file not found: '{metadata_file}'.",
                "Ensure the metadata file exists and the provided path is correct.",
            )

        with open(metadata_file, encoding="utf-8") as file:
            app_repository_metadata = json.load(file)

        app_repository_metadata["display_name"] = display_name

        with open(metadata_file, "w", encoding="utf-8") as file:
            json.dump(app_repository_metadata, file, indent=2, ensure_ascii=False)

        if not Path(config_file_path).exists():
            raise ConfigError(
                f"Configuration file not found: '{config_file_path}'.",
                "Ensure the configuration file exists and the provided path is correct.",
            )

        yaml = YAML()
        with open(config_file_path) as file:
            data = yaml.load(file)
            data["Trainer"]["epochs"] = epochs
            data["Trainer"]["it_validation"] = it_validation

        with open(config_file_path, "w") as file:
            yaml.dump(data, file)

        return models_path


class LocalAppRepositoryFromDirectory(LocalAppRepository):
    """KonfAI app repository loaded from a local folder."""

    def __init__(self, app_directory: Path, app_name: str):
        self._app_directory = app_directory
        super().__init__(app_name)

    @staticmethod
    def get_filenames(app_directory: Path, app_name: str) -> list[str]:
        return [filename.name for filename in (app_directory / app_name).glob("*")]

    def _get_filenames(self) -> list[str]:
        return LocalAppRepositoryFromDirectory.get_filenames(self._app_directory, self._app_name)

    def _download(self, filename: str) -> Path:
        return self._app_directory / self._app_name / filename

    def get_name(self) -> str:
        return str(self._app_directory / self._app_name)


class LocalAppRepositoryFromHF(LocalAppRepository):
    """KonfAI app repository backed by a Hugging Face model repository."""

    def __init__(self, repo_id: str, app_name: str, force_update: bool):
        self._repo_id = repo_id
        self._force_update = force_update
        super().__init__(app_name)

    @staticmethod
    def get_filenames(repo_id: str, app_name: str, force_update: bool) -> list[str]:
        if force_update:
            try:
                api = HfApi()
                tree = api.list_repo_tree(repo_id=repo_id, path_in_repo=app_name)
                return sorted([Path(filename.path).name for filename in tree])
            except Exception as exc:
                raise AppRepositoryError(
                    f"Failed to list contents of '{app_name}' in Hugging Face repository '{repo_id}'. "
                    "This prevents verifying whether it is a valid application folder. "
                    "Please check that the repository exists, that the path is correct, "
                    "and that you have sufficient access rights.\n"
                    f"Original error: {exc}"
                )
        try:
            snapshot_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_files_only=True,
                revision="main",
            )  # nosec B615
            folder = Path(snapshot_dir) / app_name
            return sorted([path.name for path in folder.iterdir() if path.is_file()])
        except Exception:
            return LocalAppRepositoryFromHF.get_filenames(repo_id, app_name, True)

    def _get_filenames(self) -> list[str]:
        return LocalAppRepositoryFromHF.get_filenames(self._repo_id, self._app_name, self._force_update)

    @staticmethod
    def download(repo_id: str, filename: str, force_update: bool) -> Path:
        if force_update:
            try:
                from huggingface_hub.constants import HF_HUB_CACHE
                from huggingface_hub.file_download import repo_folder_name

                lock_path = Path(HF_HUB_CACHE) / ".locks" / repo_folder_name(repo_id=repo_id, repo_type="model")
                if lock_path.is_dir():
                    shutil.rmtree(lock_path)

                return Path(
                    hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model", revision=None)  # nosec B615
                )
            except Exception as exc:
                raise AppRepositoryError(
                    f"Failed to download '{filename}' from '{repo_id}'. "
                    "Check your internet connection or repository access."
                ) from exc
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="model",
                    revision=None,
                    local_files_only=True,
                )  # nosec B615
            )
        except Exception:
            return LocalAppRepositoryFromHF.download(repo_id, filename, True)

    def _download(self, filename: str) -> Path:
        if not filename.startswith(self._app_name):
            filename = self._app_name + "/" + filename
        return LocalAppRepositoryFromHF.download(self._repo_id, filename, self._force_update)

    def get_name(self) -> str:
        return f"{self._repo_id}:{self._app_name}"


class AppRepositoryInfoFromRemoteServer(AppRepositoryInfo):
    """Read-only repository adapter backed by a remote KonfAI app server."""

    def __init__(self, remote_server: RemoteServer, app_name: str) -> None:
        self._remote_server = remote_server
        url = f"{remote_server.get_url()}/repo_apps/{app_name}"
        response = requests.get(url, headers=remote_server.get_headers(), timeout=remote_server.timeout)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        if not data.get("available", False):
            raise AppRepositoryError(f"App '{app_name}' is not available on remote server.")
        self._has_capabilities = data["has_capabilities"]

        inputs = {
            key: DataEntry(
                display_name=str(value["display_name"]),
                volume_type=VolumeType(value["volume_type"]),
                required=bool(value["required"]),
            )
            for key, value in data["inputs"].items()
        }
        outputs = {
            key: DataEntry(
                display_name=str(value["display_name"]),
                volume_type=VolumeType(value["volume_type"]),
                required=bool(value["required"]),
            )
            for key, value in data["outputs"].items()
        }

        inputs_evaluations: dict[EvaluationKey, dict[str, DataEntry]] = {}
        for display_name, by_file in data["inputs_evaluations"].items():
            for evaluation_file, entries in by_file.items():
                eval_key = EvaluationKey(display_name=str(display_name), evaluation_file=str(evaluation_file))
                inputs_evaluations[eval_key] = {
                    key: DataEntry(
                        display_name=str(value["display_name"]),
                        volume_type=VolumeType(value["volume_type"]),
                        required=bool(value["required"]),
                    )
                    for key, value in entries.items()
                }

        terminology: dict[int, TerminologyEntry] | None = None
        if "terminology" in data:
            terminology = {
                int(key): TerminologyEntry(name=str(value["name"]), color=str(value["color"]))
                for key, value in data["terminology"].items()
            }

        super().__init__(
            app_name=data["app"],
            display_name=str(data["display_name"]),
            description=str(data["description"]),
            short_description=str(data["short_description"]),
            checkpoints_name=list(data["checkpoints_name"]),
            checkpoints_name_available=list(data["checkpoints_name_available"]),
            maximum_tta=int(data["maximum_tta"]),
            mc_dropout=int(data["mc_dropout"]),
            inputs=inputs,
            outputs=outputs,
            inputs_evaluations=inputs_evaluations,
            terminology=terminology,
        )

    def has_capabilities(self) -> tuple[bool, bool, bool]:
        return self._has_capabilities

    def get_name(self) -> str:
        return self._app_name

    def download_config_file(self) -> list[Path]:
        import tempfile
        import zipfile

        def safe_name(value: str) -> str:
            return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)

        url = f"{self._remote_server.get_url()}/repo_apps_config/{self._app_name}"
        response = requests.get(url, headers=self._remote_server.get_headers(), timeout=self._remote_server.timeout)
        response.raise_for_status()

        base_tmp = Path(tempfile.gettempdir())
        folder_name = safe_name(
            f"konfai_remote_app_{self._remote_server.host}_{self._remote_server.port}_{self._app_name}"
        )
        app_tmp = base_tmp / folder_name
        if app_tmp.exists():
            shutil.rmtree(app_tmp, ignore_errors=True)
        app_tmp.mkdir(parents=True, exist_ok=True)

        zip_filename = safe_name(f"{self._app_name}_configs.zip")
        zip_path = app_tmp / zip_filename
        with open(zip_path, "wb") as file:
            file.write(response.content)

        extract_dir = app_tmp / "configs"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)

        files = [path for path in extract_dir.iterdir() if path.is_file()]
        if not files:
            raise AppRepositoryError(f"No configuration files received for remote app '{self._app_name}'.")

        return files


def get_app_repository_info(app_id: str, force_update: bool) -> AppRepositoryInfo:
    """
    Resolve an app repository identifier into a concrete repository adapter.

    Supported formats:
    - ``repo_id:app_name`` -> Hugging Face repository
    - ``/path/to/app_repository`` -> local folder
    - ``host:port:app_name`` -> remote KonfAI app server
    - ``host:port:app_name|token`` -> remote KonfAI app server with bearer token
    """
    if app_id.count(":") >= 2:
        host, port_str, name_and_token = app_id.split(":", 2)
        name_and_token_split = name_and_token.split("|")
        name = name_and_token
        token = None
        if len(name_and_token_split) == 2:
            name, token = name_and_token_split
        if port_str.isdigit():
            remote = RemoteServer(host, int(port_str), token)
            return AppRepositoryInfoFromRemoteServer(remote, name)

    if app_id.count(":") == 1:
        repo_id, name = app_id.split(":", 1)
        return LocalAppRepositoryFromHF(repo_id, name, force_update)

    path = Path(app_id)
    if path.exists():
        return LocalAppRepositoryFromDirectory(path.parent, path.name)

    raise AppRepositoryError(
        "Invalid app_id format. Expected one of:\n"
        "  - repo_id:app_name\n"
        "  - /path/to/app_repository\n"
        "  - host:port:app_name\n"
        "  - host:port:app_name|token\n"
        f"Got: {app_id!r}"
    )
