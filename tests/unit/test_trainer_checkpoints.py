from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch

import konfai.trainer as trainer_module
from konfai.trainer import _Trainer


class _DummySummaryWriter:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class _DummyModelModule:
    @staticmethod
    def state_dict() -> dict[str, torch.Tensor]:
        return {"weight": torch.tensor([1.0])}

    @staticmethod
    def get_networks() -> dict[str, object]:
        return {}


class _DummyModel:
    def __init__(self) -> None:
        self.module = _DummyModelModule()


def _date_sequence(values: list[str]) -> Iterator[str]:
    yield from values
    while True:
        yield values[-1]


def _build_trainer(tmp_path: Path, monkeypatch, date_values: list[str]) -> _Trainer:
    checkpoints_dir = tmp_path / "Checkpoints"
    statistics_dir = tmp_path / "Statistics"
    date_iter = _date_sequence(date_values)

    monkeypatch.setattr(trainer_module, "checkpoints_directory", lambda: checkpoints_dir)
    monkeypatch.setattr(trainer_module, "statistics_directory", lambda: statistics_dir)
    monkeypatch.setattr(trainer_module, "SummaryWriter", _DummySummaryWriter)
    monkeypatch.setattr(trainer_module, "current_date", lambda: next(date_iter))

    return _Trainer(
        world_size=1,
        global_rank=0,
        local_rank=0,
        size=1,
        train_name="RUN",
        early_stopping=None,
        data_log=None,
        save_checkpoint_mode="BEST",
        epochs=1,
        epoch=0,
        autocast=False,
        it_validation=1,
        it_lr_update=1,
        it=0,
        model=cast(Any, _DummyModel()),
        model_ema=None,
        dataloader_training=[object()],
        dataloader_validation=None,
    )


def test_best_checkpoint_save_keeps_only_best_without_rescanning(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["ckpt_a", "ckpt_b", "ckpt_c"])
    original_load = torch.load

    def fail_if_reloaded(*args, **kwargs):
        raise AssertionError("BEST checkpoint save unexpectedly rescanned saved checkpoints")

    monkeypatch.setattr(trainer_module.torch, "load", fail_if_reloaded)

    trainer.checkpoint_save(2.0)
    trainer.checkpoint_save(1.0)
    trainer.checkpoint_save(3.0)

    checkpoints = sorted((tmp_path / "Checkpoints" / "RUN").glob("*.pt"))
    assert [path.name for path in checkpoints] == ["ckpt_b.pt"]
    assert original_load(checkpoints[0], map_location="cpu", weights_only=False)["loss"] == 1.0


def test_best_checkpoint_bootstrap_scans_existing_files_once_and_prunes_stale_ones(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_dir = tmp_path / "Checkpoints" / "RUN"
    checkpoint_dir.mkdir(parents=True)
    torch.save({"loss": 5.0}, checkpoint_dir / "old_a.pt")
    torch.save({"loss": 3.0}, checkpoint_dir / "old_b.pt")

    original_load = trainer_module.torch.load
    load_calls: list[Path] = []

    def counted_load(path, *args, **kwargs):
        load_calls.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(trainer_module.torch, "load", counted_load)

    trainer = _build_trainer(tmp_path, monkeypatch, ["ckpt_new_worse", "ckpt_new_best"])

    assert [path.name for path in sorted(checkpoint_dir.glob("*.pt"))] == ["old_b.pt"]
    assert [path.name for path in load_calls] == ["old_a.pt", "old_b.pt"]

    trainer.checkpoint_save(4.0)
    trainer.checkpoint_save(2.0)

    assert [path.name for path in load_calls] == ["old_a.pt", "old_b.pt"]
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    assert [path.name for path in checkpoints] == ["ckpt_new_best.pt"]
    assert original_load(checkpoints[0], map_location="cpu", weights_only=False)["loss"] == 2.0
