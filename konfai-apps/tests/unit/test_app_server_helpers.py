import asyncio
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import konfai_apps.app_server as app_server
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None,
    reason="fastapi is not installed",
)


def _make_job(job_id: str, status: str = "queued") -> app_server.Job:
    job = app_server.Job(
        job_id=job_id,
        app_name="demo",
        run_dir=Path(f"/tmp/{job_id}"),
        input_dir=Path(f"/tmp/{job_id}_in"),
        output_dir=Path(f"/tmp/{job_id}_out"),
        zip_path=Path(f"/tmp/{job_id}.zip"),
    )
    job.status = status
    return job


def test_require_token_accepts_missing_configured_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    assert app_server.require_token(None) is None


def test_require_token_rejects_invalid_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KONFAI_API_TOKEN", "secret")

    with pytest.raises(HTTPException, match="Invalid token"):
        app_server.require_token(
            HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="wrong",
            )
        )


def test_save_uploads_enforces_limits(tmp_path: Path) -> None:
    payload = io.BytesIO(b"abc")
    upload = SimpleNamespace(filename="sample.bin", file=payload)

    written = app_server.save_uploads(
        [upload],
        tmp_path,
        max_file_bytes=8,
        max_total_bytes=8,
    )
    assert written[0].read_bytes() == b"abc"

    too_large = SimpleNamespace(
        filename="large.bin",
        file=io.BytesIO(b"0123456789"),
    )
    with pytest.raises(HTTPException, match="File too large"):
        app_server.save_uploads(
            [too_large],
            tmp_path / "overflow",
            max_file_bytes=4,
            max_total_bytes=32,
        )


def test_save_uploads_cleans_previous_files_on_total_limit(tmp_path: Path) -> None:
    uploads = [
        SimpleNamespace(filename="first.bin", file=io.BytesIO(b"1234")),
        SimpleNamespace(filename="second.bin", file=io.BytesIO(b"5678")),
    ]

    with pytest.raises(HTTPException, match="Total upload too large"):
        app_server.save_uploads(
            uploads,
            tmp_path / "overflow",
            max_file_bytes=8,
            max_total_bytes=6,
        )

    assert list((tmp_path / "overflow").glob("*")) == []


def test_app_lifespan_initializes_gpu_semaphores(monkeypatch: pytest.MonkeyPatch) -> None:
    old = app_server.GPU_SEM.copy()
    app_server.GPU_SEM.clear()
    monkeypatch.setattr(app_server.konfai, "get_available_devices", lambda: ([0, 2], ["gpu0", "gpu2"]))

    async def scenario() -> None:
        async with app_server.lifespan(app_server.app):
            assert sorted(app_server.GPU_SEM) == [0, 2]

    try:
        asyncio.run(scenario())
        assert app_server.GPU_SEM == {}
    finally:
        app_server.GPU_SEM.clear()
        app_server.GPU_SEM.update(old)


def test_server_state_keeps_backward_compatible_aliases() -> None:
    assert app_server.GPU_SEM is app_server.SERVER_STATE.gpu_semaphores
    assert app_server.JOBS is app_server.SERVER_STATE.jobs


def test_active_job_count_ignores_finished_jobs() -> None:
    old_jobs = dict(app_server.JOBS)
    app_server.JOBS.clear()
    app_server.JOBS.update(
        {
            "queued": _make_job("queued", "queued"),
            "running": _make_job("running", "running"),
            "done": _make_job("done", "done"),
        }
    )
    try:
        assert app_server.active_job_count() == 2
    finally:
        app_server.JOBS.clear()
        app_server.JOBS.update(old_jobs)


def test_get_job_or_404_returns_registered_job() -> None:
    old_jobs = dict(app_server.JOBS)
    job = _make_job("known", "running")
    app_server.JOBS.clear()
    app_server.JOBS[job.job_id] = job
    try:
        assert app_server.get_job_or_404(job.job_id) is job
        with pytest.raises(HTTPException, match="Unknown job_id"):
            app_server.get_job_or_404("missing")
    finally:
        app_server.JOBS.clear()
        app_server.JOBS.update(old_jobs)


def test_acquire_and_release_gpus_in_auto_mode() -> None:
    async def scenario() -> None:
        old = app_server.GPU_SEM.copy()
        app_server.GPU_SEM.clear()
        app_server.GPU_SEM.update({0: asyncio.Semaphore(1)})
        try:
            job = app_server.Job(
                job_id="job",
                app_name="demo",
                run_dir=Path("/tmp/run"),
                input_dir=Path("/tmp/in"),
                output_dir=Path("/tmp/out"),
                zip_path=Path("/tmp/out.zip"),
            )
            acquired = await app_server.acquire_gpus(job, [])
            assert acquired == [0]
            assert job.status == "waiting"

            app_server.release_gpus(acquired)
            assert app_server.GPU_SEM[0].locked() is False
        finally:
            app_server.GPU_SEM.clear()
            app_server.GPU_SEM.update(old)

    asyncio.run(scenario())


def test_submit_job_cleans_workspace_when_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir()
    monkeypatch.setattr(app_server.tempfile, "mkdtemp", lambda prefix: str(run_dir))

    @app_server.submit_job()
    async def failing_job(*args, **kwargs):
        raise HTTPException(400, "bad request")

    async def scenario() -> None:
        with pytest.raises(HTTPException, match="bad request"):
            await failing_job(
                app_name="demo",
                inputs=None,
                gt=None,
                mask=None,
                gpu=None,
                cpu=1,
                quiet=False,
            )

    asyncio.run(scenario())
    assert app_server.JOBS == {}
    assert run_dir.exists() is False
