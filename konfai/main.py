#!/usr/bin/env python3
#
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

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from konfai import RemoteServer, cuda_visible_devices
from konfai.utils.utils import State

sys.path.insert(0, os.getcwd())


def add_common_konfai_apps(parser: argparse.ArgumentParser, with_uncertainty: bool = True) -> dict[str, Any]:
    """
    Add shared CLI arguments for KonfAI "apps-style" commands and parse them.

    This helper is used for commands that operate on medical volumes/datasets and
    share the same input/output/device semantics. It adds:
    - inputs (required, multi-group via `action="append"`)
    - optional ground-truth and mask (same grouping behavior)
    - output directory
    - optional `-uncertainty` flag (when enabled by `with_uncertainty`)
    - mutually exclusive device selection:
        * `--gpu <ids...>` (default: [])
        * `--cpu <n>`     (default: None, must be > 0)
    - quiet flag

    Grouping semantics
    ------------------
    `action="append"` + `nargs="+"` means the user can provide multiple groups.
    Each occurrence of `--inputs ...` creates one group. Example:

        --inputs volA.mha volB.mha --inputs volC.mha

    yields:
        inputs = [[volA, volB], [volC]]

    Post-processing
    ---------------
    - If `--cpu` is provided, GPU usage is disabled (`gpu=[]`).
    - If `with_uncertainty` is False, `uncertainty` is forced to False.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to which arguments are added.
    with_uncertainty : bool
        Whether to expose the `-uncertainty` flag and return an `uncertainty`
        key in the parsed kwargs.

    Returns
    -------
    dict[str, Any]
        Parsed CLI arguments as a kwargs dictionary.
    """
    parser.add_argument(
        "-i",
        "--inputs",
        type=lambda x: Path(x).resolve(),
        nargs="+",
        action="append",
        required=True,
        help="Input path(s): provide one or multiple volume files, or a dataset directory.",
    )

    parser.add_argument(
        "--gt",
        type=lambda x: Path(x).resolve(),
        nargs="+",
        action="append",
        help="Ground-truth path(s): provide one or multiple data files, or a dataset directory.",
    )

    parser.add_argument(
        "--mask",
        type=lambda x: Path(x).resolve(),
        nargs="+",
        action="append",
        help="Optional evaluation mask path: provide one or multiple volume files, or a dataset directory.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=lambda x: Path(x).resolve(),
        default=Path("./Output").resolve(),
        help="Output directory / file",
    )
    if with_uncertainty:
        parser.add_argument("-uncertainty", action="store_true", help="Run uncertainty workflow.")

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--gpu",
        type=int,
        nargs="+",
        default=[],
        help="GPU device ids to use, e.g. '0' or '0,1,2'. If omitted runs on CPU.",
    )

    def non_negative_int(value: str) -> int:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError("CPU value must be > 0")
        return ivalue

    device_group.add_argument(
        "--cpu",
        type=non_negative_int,
        default=None,
        help="Run on CPU using N worker processes/cores. If omitted, uses GPU when available.",
    )

    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress console output for a quieter execution")

    kwargs = vars(parser.parse_args())
    if kwargs["cpu"] is not None:
        kwargs["gpu"] = []
    if not with_uncertainty:
        kwargs["uncertainty"] = False
    return kwargs


def main_apps():
    """
    Entry point for the `konfai-apps` command-line interface.

    This CLI provides two execution modes:
    - Local mode (default): runs apps locally via `KonfAIApp`.
    - Remote mode: when `--host` is provided, submits jobs to an app server via
      `KonfAIAppClient(RemoteServer(...))`.

    Supported subcommands
    ---------------------
    - infer       : inference for one or more input groups
    - eval        : evaluation using ground-truth (and optional mask)
    - uncertainty : uncertainty estimation workflow
    - pipeline    : inference + evaluation + optional uncertainty
    - fine-tune   : fine-tuning on a dataset directory

    Parsing details
    ---------------
    - Input paths are resolved to absolute paths.
    - Device selection is mutually exclusive: either GPU ids or CPU workers.

    Execution details
    -----------------
    - For remote mode: the client calls `/apps/{app}/{command}` on the server and
      streams logs / downloads results.
    - For local mode: the app is executed in-process.

    Notes
    -----
    For the `fine-tune` command, this CLI sets:
        kwargs["tmp_dir"] = kwargs["output"]
    so that the app uses the output directory as its working directory.
    """
    parser = argparse.ArgumentParser(
        prog="konfai-apps", description="KonfAI Apps – Apps for Medical AI Models", allow_abbrev=False
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(parser: argparse.ArgumentParser, is_fine_tune: bool = False):
        parser.add_argument("app", type=str, help="KonfAI App name")

        parser.add_argument("--host", type=str, default=None, help="Server host")
        parser.add_argument("--port", type=int, default=8000, help="Server port")
        parser.add_argument(
            "--token",
            type=str,
            default=os.environ.get("KONFAI_API_TOKEN"),
            help="Bearer token (or use KONFAI_API_TOKEN env var)",
        )

        if not is_fine_tune:
            parser.add_argument(
                "-i",
                "--inputs",
                type=lambda x: Path(x).resolve(),
                nargs="+",
                action="append",
                required=True,
                help="Input path(s): provide one or multiple volume files, or a dataset directory.",
            )
        else:
            parser.add_argument(
                "-d",
                "--dataset",
                type=lambda x: Path(x).resolve(),
                required=True,
                help="dataset path(s): provide a dataset directory.",
            )
        parser.add_argument(
            "-o",
            "--output",
            type=lambda x: Path(x).resolve(),
            default=Path("./Output").resolve(),
            help="Output directory / file",
        )

        if not is_fine_tune:
            parser.add_argument(
                "--tmp-dir",
                "--tmp_dir",
                type=lambda x: Path(x).resolve(),
                default=None,
                help="Temporary directory (optional).",
            )
        device_group = parser.add_mutually_exclusive_group()
        device_group.add_argument(
            "--gpu",
            type=int,
            nargs="+",
            default=[],
            help="GPU device ids to use, e.g. '0' or '0,1,2'. If omitted runs on CPU.",
        )

        def non_negative_int(value: str) -> int:
            ivalue = int(value)
            if ivalue <= 0:
                raise argparse.ArgumentTypeError("CPU value must be > 0")
            return ivalue

        device_group.add_argument(
            "--cpu",
            type=non_negative_int,
            default=None,
            help="Run on CPU using N worker processes/cores. If omitted, uses GPU when available.",
        )

        parser.add_argument(
            "-q", "--quiet", action="store_true", help="Suppress console output for a quieter execution"
        )

    # -----------------
    # 1) INFERENCE
    # -----------------
    infer_p = subparsers.add_parser("infer", help="Run inference using a KonfAI App.")
    add_common_args(infer_p)
    group = infer_p.add_mutually_exclusive_group()
    group.add_argument("--ensemble", type=int, default=0, help="Number of models in the ensemble (auto-select).")
    group.add_argument(
        "--ensemble-models",
        "--ensemble_models",
        nargs="+",
        default=[],
        help="Explicit list of model identifiers/paths to use.",
    )

    infer_p.add_argument("--tta", type=int, default=0, help="Number of Test-Time Augmentations")
    infer_p.add_argument("--mc", type=int, default=0, help="Monte Carlo dropout samples")
    infer_p.add_argument(
        "--prediction-file",
        "--prediction_file",
        type=str,
        default="Prediction.yml",
        help="Optional prediction config filename",
    )

    # -----------------
    # 2) EVALUATION
    # -----------------
    eval_p = subparsers.add_parser("eval", help="Evaluate a KonfAI App using ground-truth labels.")
    add_common_args(eval_p)
    eval_p.add_argument(
        "--gt",
        type=lambda x: Path(x).resolve(),
        nargs="+",
        action="append",
        required=True,
        help="Ground-truth path(s): provide one or multiple data files, or a dataset directory.",
    )

    eval_p.add_argument(
        "--mask",
        type=lambda x: Path(x).resolve(),
        nargs="+",
        action="append",
        help="Optional evaluation mask path: provide one or multiple volume files, or a dataset directory.",
    )

    eval_p.add_argument(
        "--evaluation-file",
        "--evaluation_file",
        type=str,
        default="Evaluation.yml",
        help="Optional evaluation config filename",
    )

    # -----------------
    # 3) UNCERTAINTY
    # -----------------
    unc_p = subparsers.add_parser("uncertainty", help="Compute model uncertainty for a KonfAI App.")
    add_common_args(unc_p)
    unc_p.add_argument(
        "--uncertainty-file",
        "--uncertainty_file",
        type=str,
        default="Uncertainty.yml",
        help="Optional uncertainty config filename",
    )

    # -----------------
    # 4) Pipeline
    # -----------------
    pipe_p = subparsers.add_parser(
        "pipeline", help="Run inference and optionally evaluation and uncertainty in a single command."
    )
    add_common_args(pipe_p)

    group = pipe_p.add_mutually_exclusive_group()
    group.add_argument("--ensemble", type=int, default=0, help="Number of models in the ensemble (auto-select).")
    group.add_argument(
        "--ensemble-models",
        "--ensemble_models",
        nargs="+",
        default=[],
        help="Explicit list of model identifiers/paths to use.",
    )
    pipe_p.add_argument("--tta", type=int, default=0, help="Number of Test-Time Augmentations.")
    pipe_p.add_argument("--mc", type=int, default=0, help="Number of Monte Carlo dropout samples.")

    pipe_p.add_argument(
        "--prediction-file",
        "--prediction_file",
        type=str,
        default="Prediction.yml",
        help="Optional prediction config filename",
    )
    pipe_p.add_argument(
        "--gt",
        type=lambda x: Path(x).resolve(),
        nargs="+",
        action="append",
        required=True,
        help="Ground-truth path(s): provide one or multiple data files, or a dataset directory.",
    )

    pipe_p.add_argument(
        "--mask",
        type=lambda x: Path(x).resolve(),
        nargs="+",
        action="append",
        help="Optional evaluation mask path: provide one or multiple volume files, or a dataset directory.",
    )

    pipe_p.add_argument(
        "--evaluation-file",
        "--evaluation_file",
        type=str,
        default="Evaluation.yml",
        help="Optional evaluation config filename",
    )
    pipe_p.add_argument(
        "--uncertainty-file",
        "--uncertainty_file",
        type=str,
        default="Uncertainty.yml",
        help="Optional uncertainty config filename",
    )

    pipe_p.add_argument("-uncertainty", action="store_true", help="Run uncertainty workflow.")

    # -----------------
    # 5) FINE-TUNE
    # -----------------
    ft_p = subparsers.add_parser("fine-tune", help="Fine-tune a KonfAI App on a dataset.")
    add_common_args(ft_p, True)
    ft_p.add_argument("name", type=str, help="New KonfAI App display name")
    ft_p.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    ft_p.add_argument(
        "--it-validation",
        "--it_validation",
        type=int,
        default=1000,
        help="Number of training iterations between validation runs.",
    )

    parser.add_argument("--version", action="version", version=importlib.metadata.version("konfai"))

    kwargs = vars(parser.parse_args())

    from konfai.app import AbstractKonfAIApp, KonfAIApp, KonfAIAppClient

    host = kwargs.pop("host")
    port = kwargs.pop("port")
    token = kwargs.pop("token")

    konfai_app: AbstractKonfAIApp
    if host is not None:
        konfai_app = KonfAIAppClient(kwargs.pop("app"), RemoteServer(host, port, token))
    else:
        konfai_app = KonfAIApp(kwargs.pop("app"))
    command = kwargs.pop("command")
    if command == "infer":
        konfai_app.infer(**kwargs)
    elif command == "eval":
        konfai_app.evaluate(**kwargs)
    elif command == "uncertainty":
        konfai_app.uncertainty(**kwargs)
    elif command == "pipeline":
        konfai_app.pipeline(**kwargs)
    elif command == "fine-tune":
        kwargs["tmp_dir"] = kwargs["output"]
        konfai_app.fine_tune(**kwargs)


def main_apps_server():
    """
    Entry point for launching the KonfAI Apps FastAPI server (uvicorn).

    This command wraps uvicorn startup and optionally configures bearer-token
    authentication through environment variables.

    Auth modes
    ----------
    - off    : no token required (not recommended beyond trusted environments)
    - bearer : requires a token (default). The token is read from an environment
              variable (default: KONFAI_API_TOKEN) or can be overridden via
              `--token` (development convenience).

    Parameters (CLI)
    ----------------
    --host        : bind address (default: 127.0.0.1)
    --port        : bind port (default: 8000)
    --auth        : "off" | "bearer"
    --token-env   : environment variable name holding the bearer token
    --token       : token override (dev only; avoid in production)

    Raises
    ------
    SystemExit
        If auth is enabled but no token is provided via env or `--token`.
    """
    import uvicorn

    parser = argparse.ArgumentParser(description="KonfAI apps server", allow_abbrev=False)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument("--auth", choices=["off", "bearer"], default="bearer", help="Auth mode (default: bearer)")
    parser.add_argument(
        "--token-env", "--token_env", type=str, default="KONFAI_API_TOKEN", help="Env var name holding the bearer token"
    )
    parser.add_argument("--token", type=str, default=None, help="(dev) Bearer token override (NOT recommended in prod)")
    parser.add_argument(
        "--apps",
        type=Path,
        required=True,
        help="Config file listing available apps (json).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Pre-download all apps listed in --apps into the local cache before starting the server.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate all apps listed in --apps (no download).",
    )

    args = parser.parse_args()

    if args.auth == "bearer":
        if args.token:
            os.environ[args.token_env] = args.token
        if not os.environ.get(args.token_env):
            raise SystemExit(f"Auth is enabled but no token found. Set {args.token_env} or pass --token (dev).")

    if args.apps:
        if not args.apps.exists():
            raise SystemExit(f"Config file not found: {args.apps}")

        data = json.loads(args.apps.read_text(encoding="utf-8"))

        if "apps" not in data or not isinstance(data["apps"], list):
            raise SystemExit("Invalid config file: expected a JSON object with an 'apps' list.")

        os.environ["KONFAI_APPS_CONFIG"] = json.dumps(data)
    apps = []
    if args.check or args.download:
        from konfai.utils.utils import get_app_repository_info

        errors = []
        for app_id in data["apps"]:
            try:
                apps.append(get_app_repository_info(str(app_id)))
                print(f"[KonfAI-Apps] OK: {app_id}", flush=True)
            except Exception as e:
                errors.append((app_id, str(e)))
                print(f"[KonfAI-Apps] ERROR: {app_id} -> {e}", flush=True)

        if errors:
            raise SystemExit("One or more apps are invalid:\n" + "\n".join(f"  - {a}: {err}" for a, err in errors))

        print("[KonfAI-Apps] All apps validated successfully.")

    if args.download:
        from konfai.utils.utils import LocalAppRepository, get_app_repository_info

        for app in apps:
            try:
                if isinstance(app, LocalAppRepository):
                    # force download of configs/checkpoints so cache is warm
                    _ = app.download_train()
                    print(f"[KonfAI-Apps] Cached: {app_id}", flush=True)
            except Exception as e:
                print(f"[KonfAI-Apps] Failed to cache '{app_id}': {e}", flush=True)

    uvicorn.run("konfai.app_server:app", host=args.host, port=args.port, log_level="info", reload=False)


def _run(parser: argparse.ArgumentParser) -> None:
    """
    Shared CLI builder and dispatcher for the main KonfAI training/inference commands.

    This function:
    1) defines common arguments used by TRAIN / RESUME / PREDICTION / EVALUATION
       (config file, overwrite, device selection, quiet, tensorboard)
    2) defines subcommands and their command-specific arguments
    3) parses CLI args and dispatches to the correct implementation:
       - `konfai.trainer.train` for TRAIN and RESUME
       - `konfai.predictor.predict` for PREDICTION
       - `konfai.evaluator.evaluate` for EVALUATION

    Device selection
    ----------------
    GPU and CPU are mutually exclusive:
    - `--gpu` accepts one or more GPU ids, constrained to available devices
      returned by `cuda_visible_devices()`.
    - `--cpu` accepts a strictly positive integer (number of workers).

    Config handling
    ---------------
    If `--config` is omitted, the `config` key is removed from the argument dict,
    so downstream functions can use their own default config filename.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The top-level parser created by the caller.
    """

    def add_common_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default=None,
            help="Path to the configuration file (YAML). "
            "If omitted, a command-specific default is used (e.g. Train.yml, Prediction.yml, Evaluation.yml).",
        )
        parser.add_argument(
            "-y",
            "--overwrite",
            action="store_true",
            help="Overwrite existing outputs (checkpoints, logs, predictions) without prompting.",
        )

        device_group = parser.add_mutually_exclusive_group()
        devices = cuda_visible_devices()
        device_group.add_argument(
            "--gpu",
            type=int,
            nargs="+",
            choices=devices,
            default=[],
            help="GPU device ids to use, e.g. '0' or '0,1,2'. If omitted runs on CPU.",
        )

        def non_negative_int(value: str) -> int:
            ivalue = int(value)
            if ivalue <= 0:
                raise argparse.ArgumentTypeError("CPU value must be > 0")
            return ivalue

        device_group.add_argument(
            "--cpu",
            type=non_negative_int,
            default=None,
            help="Run on CPU using N worker processes/cores. If omitted, uses GPU when available.",
        )

        parser.add_argument(
            "-q", "--quiet", action="store_true", help="Suppress console output for a quieter execution"
        )
        parser.add_argument("-tb", "--tensorboard", action="store_true", help="Launch TensorBoard.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser(str(State.TRAIN), help="Train a model from scratch.")
    add_common_args(train_p)
    train_p.add_argument(
        "--checkpoints-dir",
        "--checkpoints_dir",
        type=str,
        default="./Checkpoints/",
        help="Directory where checkpoints are saved (default: ./Checkpoints/).",
    )

    train_p.add_argument(
        "--statistics-dir",
        "--statistics_dir",
        type=str,
        default="./Statistics/",
        help="Directory where training statistics/logs are saved (default: ./Statistics/).",
    )

    resume_p = subparsers.add_parser(str(State.RESUME), help="Resume training from existing checkpoints.")
    add_common_args(resume_p)
    resume_p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Checkpoint path to resume from",
    )

    resume_p.add_argument(
        "-checkpoints-dir",
        "-checkpoints_dir",
        type=str,
        default="./Checkpoints/",
        help="Directory where checkpoints are saved (default: ./Checkpoints/)",
    )

    resume_p.add_argument(
        "-statistics-dir",
        "-statistics_dir",
        type=str,
        default="./Statistics/",
        help="Directory where training statistics/logs are saved (default: ./Statistics/).",
    )

    predict_p = subparsers.add_parser(str(State.PREDICTION), help="Run inference using a trained model.")
    add_common_args(predict_p)

    predict_p.add_argument(
        "--models",
        type=str,
        nargs="+",
        metavar="PATH",
        required=True,
        help="One or more checkpoint/model paths to resume from.",
    )

    predict_p.add_argument(
        "--predictions-dir",
        "--predictions_dir",
        type=str,
        default="./Predictions/",
        help="Directory where predictions are written (default: ./Predictions/).",
    )

    eval_p = subparsers.add_parser(str(State.EVALUATION), help="Evaluate model.")
    add_common_args(eval_p)

    eval_p.add_argument(
        "--evaluations-dir",
        "--evaluations_dir",
        type=str,
        default="./Evaluations/",
        help="Directory where evaluation outputs are written (default: ./Evaluations/).",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=importlib.metadata.version("konfai"),
        help="Print KonfAI version and exit.",
    )

    args = vars(parser.parse_args())
    if args["config"] is None:
        del args["config"]
    if args["command"] == "PREDICTION":
        from konfai.predictor import predict

        predict(**args)
    elif args["command"] == "EVALUATION":
        from konfai.evaluator import evaluate

        evaluate(**args)
    else:
        from konfai.trainer import train

        train(**args)


def main():
    """
    Entry point for the `konfAI` command-line interface.

    This function builds the top-level CLI parser and delegates the full argument
    parsing and command dispatching to `_run(parser)`.

    Supported commands are:
    - TRAIN
    - RESUME
    - PREDICTION
    - EVALUATION

    Notes
    -----
    The actual execution logic is implemented in `konfai.trainer.train`,
    `konfai.predictor.predict`, and `konfai.evaluator.evaluate`.
    """
    parser = argparse.ArgumentParser(
        prog="konfAI", description="KonfAI – Deep learning framework for Medical AI Models", allow_abbrev=False
    )
    _run(parser)


def cluster():
    """
    Entry point for running KonfAI with cluster-oriented CLI arguments.

    This command extends the standard KonfAI CLI with a "Cluster manager arguments"
    group (job name, nodes, memory, time limit, resubmit), then delegates parsing
    and command dispatching to `_run(parser)`.

    Notes
    -----
    - This function only defines extra CLI arguments.
    """
    parser = argparse.ArgumentParser(
        prog="konfAI", description="KonfAI – Deep learning framework for Medical AI Models", allow_abbrev=False
    )

    # Cluster manager arguments
    cluster_args = parser.add_argument_group("Cluster manager arguments")
    cluster_args.add_argument("--name", type=str, help="Task name", required=True)
    cluster_args.add_argument("--num-nodes", "--num_nodes", default=1, type=int, help="Number of nodes")
    cluster_args.add_argument("--memory", type=int, default=16, help="Amount of memory per node")
    cluster_args.add_argument(
        "--time-limit",
        "--time_limit",
        type=int,
        default=1440,
        help="Job time limit in minute",
    )
    cluster_args.add_argument(
        "--resubmit",
        action="store_true",
        help="Automatically resubmit job just before timout",
    )
    _run(parser)
