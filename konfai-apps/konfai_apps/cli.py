"""Command-line entrypoints for standalone KonfAI Apps workflows and services."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from konfai import RemoteServer

from . import app as app_module
from .app_repository import LocalAppRepository, get_app_repository_info

if TYPE_CHECKING:
    from .app import AbstractKonfAIApp


def _package_version() -> str:
    try:
        return importlib.metadata.version("konfai-apps")
    except importlib.metadata.PackageNotFoundError:
        return "0+local"


def add_common_konfai_apps(parser: argparse.ArgumentParser, with_uncertainty: bool = True) -> dict[str, Any]:
    """Add shared CLI arguments for app-focused commands and parse them."""
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
    parser.add_argument("--download", action="store_true", help="Download the full KonfAI app upfront")
    parser.add_argument(
        "--force_update",
        action="store_true",
        help="Ensure required files are updated to the latest version during execution",
    )

    kwargs = vars(parser.parse_args())
    if kwargs["cpu"] is not None:
        kwargs["gpu"] = []
    if not with_uncertainty:
        kwargs["uncertainty"] = False
    return kwargs


def main_apps() -> None:
    """Entry point for the `konfai-apps` command-line interface."""
    parser = argparse.ArgumentParser(
        prog="konfai-apps", description="KonfAI Apps - Apps for Medical AI Models", allow_abbrev=False
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(parser: argparse.ArgumentParser, is_fine_tune: bool = False) -> None:
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
        parser.add_argument("--download", action="store_true", help="Download the full KonfAI app upfront")
        parser.add_argument(
            "--force_update",
            action="store_true",
            help="Ensure required files are updated to the latest version during execution",
        )

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
    infer_p.add_argument("-uncertainty", action="store_true", help="If enabled, inference write the inference stack")
    infer_p.add_argument(
        "--prediction-file",
        "--prediction_file",
        type=str,
        default="Prediction.yml",
        help="Optional prediction config filename",
    )

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

    unc_p = subparsers.add_parser("uncertainty", help="Compute model uncertainty for a KonfAI App.")
    add_common_args(unc_p)
    unc_p.add_argument(
        "--uncertainty-file",
        "--uncertainty_file",
        type=str,
        default="Uncertainty.yml",
        help="Optional uncertainty config filename",
    )

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

    parser.add_argument("--version", action="version", version=_package_version())

    kwargs = vars(parser.parse_args())
    host = kwargs.pop("host")
    port = kwargs.pop("port")
    token = kwargs.pop("token")

    konfai_app: AbstractKonfAIApp
    if host is not None:
        konfai_app = app_module.KonfAIAppClient(kwargs.pop("app"), RemoteServer(host, port, token))
    else:
        konfai_app = app_module.KonfAIApp(kwargs.pop("app"), kwargs.pop("download"), kwargs.pop("force_update"))

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


def main_apps_server() -> None:
    """Entry point for launching the KonfAI Apps FastAPI server."""
    import uvicorn

    parser = argparse.ArgumentParser(description="KonfAI apps server", allow_abbrev=False)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--auth", choices=["off", "bearer"], default="bearer", help="Auth mode (default: bearer)")
    parser.add_argument(
        "--token-env", "--token_env", type=str, default="KONFAI_API_TOKEN", help="Env var name holding the bearer token"
    )
    parser.add_argument("--token", type=str, default=None, help="(dev) Bearer token override (NOT recommended in prod)")
    parser.add_argument("--apps", type=Path, required=True, help="Config file listing available apps (json).")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Pre-download all apps listed in --apps into the local cache before starting the server.",
    )
    parser.add_argument("--check", action="store_true", help="Validate all apps listed in --apps (no download).")

    args = parser.parse_args()

    if args.auth == "bearer":
        if args.token:
            os.environ[args.token_env] = args.token
        if not os.environ.get(args.token_env):
            raise SystemExit(f"Auth is enabled but no token found. Set {args.token_env} or pass --token (dev).")

    if not args.apps.exists():
        raise SystemExit(f"Config file not found: {args.apps}")

    data = json.loads(args.apps.read_text(encoding="utf-8"))
    if "apps" not in data or not isinstance(data["apps"], list):
        raise SystemExit("Invalid config file: expected a JSON object with an 'apps' list.")

    os.environ["KONFAI_APPS_CONFIG"] = json.dumps(data)
    apps = []
    if args.check or args.download:
        errors = []
        for app_id in data["apps"]:
            try:
                apps.append(get_app_repository_info(str(app_id), True))
                print(f"[KonfAI-Apps] OK: {app_id}", flush=True)
            except Exception as exc:
                errors.append((app_id, str(exc)))
                print(f"[KonfAI-Apps] ERROR: {app_id} -> {exc}", flush=True)

        if errors:
            raise SystemExit("One or more apps are invalid:\n" + "\n".join(f"  - {a}: {err}" for a, err in errors))

        print("[KonfAI-Apps] All apps validated successfully.")

    if args.download:
        for app in apps:
            try:
                if isinstance(app, LocalAppRepository):
                    _ = app.download_app()
                    print(f"[KonfAI-Apps] Cached: {app.get_name()}", flush=True)
            except Exception as exc:
                print(f"[KonfAI-Apps] Failed to cache '{app.get_name()}': {exc}", flush=True)

    uvicorn.run("konfai_apps.app_server:app", host=args.host, port=args.port, log_level="info", reload=False)
