import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from konfai import RemoteServer, check_server

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
APP_ASSETS_DIR = ASSETS_DIR / "AppClientRemote" / "TinySynthesisApp"
WORKFLOW_ASSETS_DIR = ASSETS_DIR / "Workflows"
SimpleITK = pytest.importorskip("SimpleITK")


@dataclass
class RunningAppServer:
    process: subprocess.Popen[str]
    port: int
    token: str
    env: dict[str, str]
    app_id: str


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_image(path: Path) -> None:
    image = SimpleITK.GetImageFromArray(torch.linspace(-1, 1, 16 * 16, dtype=torch.float32).reshape(1, 16, 16).numpy())
    image.SetSpacing((1.0, 1.0, 1.0))
    image = SimpleITK.Cast(image, SimpleITK.sitkFloat32)
    SimpleITK.WriteImage(image, str(path))


def _resolve_entrypoint(bin_dir: Path, name: str) -> Path:
    candidates = [bin_dir / name, bin_dir / f"{name}.cmd", bin_dir / f"{name}.exe"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise AssertionError(f"Missing CLI entrypoint in test bin dir: {name}")


def _write_cli_entrypoints(bin_dir: Path) -> None:
    def write_script(name: str, target: str) -> None:
        script = bin_dir / name
        script.write_text(
            f"#!{sys.executable}\n"
            "import runpy, sys\n"
            f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            f"from konfai.main import {target} as _entry\n"
            "if __name__ == '__main__':\n"
            "    _entry()\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        (bin_dir / f"{name}.cmd").write_text(
            f'@"{sys.executable}" "{script}" %*\r\n',
            encoding="utf-8",
        )

    write_script("konfai-apps", "main_apps")
    write_script("konfai-apps-server", "main_apps_server")


def _write_local_synthesis_app(app_dir: Path) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    for asset in APP_ASSETS_DIR.iterdir():
        if asset.is_file():
            shutil.copy2(asset, app_dir / asset.name)
    shutil.copy2(WORKFLOW_ASSETS_DIR / "TinySynth.py", app_dir / "TinySynth.py")

    spec = importlib.util.spec_from_file_location("TinySynth", app_dir / "TinySynth.py")
    if spec is None or spec.loader is None:
        raise AssertionError("Failed to load TinySynth test module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TinySynthNet()
    torch.save({"Model": model.state_dict()}, app_dir / "tiny.pt")


def _subprocess_env(bin_dir: Path, token: str) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not pythonpath else f"{REPO_ROOT}{os.pathsep}{pythonpath}"
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["KONFAI_API_TOKEN"] = token
    return env


@pytest.fixture
def running_app_server(tmp_path: Path) -> Generator[RunningAppServer]:
    token = "secret"
    port = _free_port()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_cli_entrypoints(bin_dir)

    app_dir = tmp_path / "TinySynthesisApp"
    _write_local_synthesis_app(app_dir)

    apps_config = tmp_path / "apps.json"
    apps_config.write_text(json.dumps({"apps": [str(app_dir)]}), encoding="utf-8")

    env = _subprocess_env(bin_dir, token)
    server_command = [
        str(_resolve_entrypoint(bin_dir, "konfai-apps-server")),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--token",
        token,
        "--apps",
        str(apps_config),
    ]
    process = subprocess.Popen(
        server_command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    remote = RemoteServer("127.0.0.1", port, token)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise AssertionError(f"konfai-apps-server exited early with code {process.returncode}\n{output}")
        ok, _ = check_server(remote, timeout_s=0.2)
        if ok:
            break
        time.sleep(0.1)
    else:
        process.terminate()
        output = process.stdout.read() if process.stdout is not None else ""
        process.wait(timeout=10)
        raise AssertionError(f"konfai-apps-server did not become ready\n{output}")

    try:
        yield RunningAppServer(
            process=process,
            port=port,
            token=token,
            env=env,
            app_id=str(app_dir),
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def test_main_apps_remote_infer_roundtrip_against_main_apps_server(
    running_app_server: RunningAppServer,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mha"
    _write_image(input_path)
    output_dir = tmp_path / "output"

    command = [
        str(_resolve_entrypoint(Path(running_app_server.env["PATH"].split(os.pathsep)[0]), "konfai-apps")),
        "infer",
        running_app_server.app_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(running_app_server.port),
        "--token",
        running_app_server.token,
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
        "--cpu",
        "1",
        "--prediction-file",
        "Prediction.yml",
    ]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=running_app_server.env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Remote CLI roundtrip failed.\n\n" f"STDOUT:\n{result.stdout}\n\n" f"STDERR:\n{result.stderr}"
    )
    assert "Remote job" in result.stdout
    assert "Result written to" in result.stdout
    predicted = sorted(output_dir.rglob("sCT.mha"))
    assert predicted, f"No synthesized output produced.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
