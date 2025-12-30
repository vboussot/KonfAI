import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path

from konfai import cuda_visible_devices
from konfai.utils.utils import State

sys.path.insert(0, os.getcwd())


def main_apps():
    parser = argparse.ArgumentParser(prog="konfai-apps", description="KonfAI Apps – Apps for Medical AI Models")

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(parser: argparse.ArgumentParser, is_fine_tune: bool = False):
        parser.add_argument("app", type=str, help="KonfAI App name")
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
                "--tmp_dir",
                type=lambda x: Path(x).resolve(),
                default=Path(tempfile.mkdtemp(prefix="konfai_app_")),
                help="Temporary directory (optional).",
            )
        device_group = parser.add_mutually_exclusive_group()
        devices = cuda_visible_devices()
        device_group.add_argument(
            "--gpu",
            type=int,
            nargs="+",
            choices=devices,
            default=devices,
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
    infer_p.add_argument("--ensemble", type=int, default=0, help="Size of model ensemble")
    infer_p.add_argument("--tta", type=int, default=0, help="Number of Test-Time Augmentations")
    infer_p.add_argument("--mc_dropout", type=int, default=0, help="Monte Carlo dropout samples")
    infer_p.add_argument(
        "--prediction_file", type=str, default="Prediction.yml", help="Optional prediction config filename"
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
        "--evaluation_file", type=str, default="Evaluation.yml", help="Optional evaluation config filename"
    )

    # -----------------
    # 3) UNCERTAINTY
    # -----------------
    unc_p = subparsers.add_parser("uncertainty", help="Compute model uncertainty for a KonfAI App.")
    add_common_args(unc_p)
    unc_p.add_argument(
        "--uncertainty_file", type=str, default="Uncertainty.yml", help="Optional uncertainty config filename"
    )

    # -----------------
    # 4) Pipeline
    # -----------------
    pipe_p = subparsers.add_parser(
        "pipeline", help="Run inference and optionally evaluation and uncertainty in a single command."
    )
    add_common_args(pipe_p)

    pipe_p.add_argument("--mc_dropout", type=int, default=0, help="Number of Monte Carlo dropout samples.")
    pipe_p.add_argument("--tta", type=int, default=0, help="Number of Test-Time Augmentations.")
    pipe_p.add_argument("--ensemble", type=int, default=0, help="Number of models in ensemble.")
    pipe_p.add_argument(
        "--prediction_file", type=str, default="Prediction.yml", help="Optional prediction config filename"
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
        "--evaluation_file", type=str, default="Evaluation.yml", help="Optional evaluation config filename"
    )
    pipe_p.add_argument(
        "--uncertainty_file", type=str, default="Uncertainty.yml", help="Optional uncertainty config filename"
    )

    # -----------------
    # 5) FINE-TUNE
    # -----------------
    ft_p = subparsers.add_parser("fine-tune", help="Fine-tune a KonfAI App on a dataset.")
    add_common_args(ft_p, True)
    ft_p.add_argument("name", type=str, help="New KonfAI App display name")
    ft_p.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    ft_p.add_argument(
        "--it_validation", type=int, default=1000, help="Number of training iterations between validation runs."
    )

    parser.add_argument("--version", action="version", version=importlib.metadata.version("konfai"))

    kwargs = vars(parser.parse_args())
    if kwargs["cpu"] is not None:
        kwargs["gpu"] = []

    from konfai.app import KonfAIApp

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


def _run(parser: argparse.ArgumentParser) -> None:
    # KONFAI arguments

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
            default=devices[0],
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
        "--checkpoints_dir",
        type=str,
        default="./Checkpoints/",
        help="Directory where checkpoints are saved (default: ./Checkpoints/).",
    )

    train_p.add_argument(
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
        "-checkpoints_dir",
        type=str,
        default="./Checkpoints/",
        help="Directory where checkpoints are saved (default: ./Checkpoints/)",
    )

    resume_p.add_argument(
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
        "--predictions_dir",
        type=str,
        default="./Predictions/",
        help="Directory where predictions are written (default: ./Predictions/).",
    )

    eval_p = subparsers.add_parser(str(State.EVALUATION), help="Evaluate model.")
    add_common_args(eval_p)

    eval_p.add_argument(
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
    Entry point for launching KonfAI training locally.

    - Parses arguments (if any) via a setup parser.
    - Initializes distributed environment based on available CUDA devices or CPU cores.
    - Launches training via `mp.spawn`.
    - Manages logging and TensorBoard context.

    KeyboardInterrupt is caught to allow clean manual termination.
    """
    parser = argparse.ArgumentParser(
        prog="konfAI", description="KonfAI – Deep learning framework for Medical AI Models"
    )
    _run(parser)


def cluster():
    """
    Entry point for launching KonfAI on a cluster using Submitit.

    - Parses cluster-specific arguments: job name, nodes, memory, time limit, etc.
    - Sets up distributed environment based on number of nodes and GPUs.
    - Configures Submitit executor with job specs.
    - Submits the job to SLURM (or another Submitit-compatible backend).

    Environment variables:
        KONFAI_OVERWRITE: Set to force overwrite of previous training runs.
        KONFAI_CLUSTER: Mark this as a cluster job (used downstream).

    Raises:
        KeyboardInterrupt: On manual interruption.
        Exception: Any submission-related error is printed and causes exit.
    """
    parser = argparse.ArgumentParser(
        prog="konfAI", description="KonfAI – Deep learning framework for Medical AI Models"
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
