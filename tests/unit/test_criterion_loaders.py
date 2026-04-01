from types import SimpleNamespace
from typing import cast

import pytest

import konfai.network.network as network_module


def test_network_criterion_loader_resets_scheduler_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMeasure:
        def __init__(self) -> None:
            pass

    class DummySchedulerLoader:
        def __init__(self) -> None:
            self.nb_step = 3

        def getschedulers(self, key: str, scheduler_classname: str):
            return f"{key}:{scheduler_classname}"

    monkeypatch.setattr(network_module, "apply_config", lambda *args, **kwargs: (lambda cls: cls))
    monkeypatch.setattr(network_module, "konfai_root", lambda: "Trainer")
    monkeypatch.setattr(
        network_module,
        "get_module",
        lambda classpath, default: (SimpleNamespace(Measure=DummyMeasure, __name__="torch.optim"), "Measure"),
    )

    attr = network_module.CriterionsAttr(
        schedulers=cast(
            dict[str, network_module.LossSchedulersLoader],
            {"Constant": DummySchedulerLoader()},
        )
    )
    loader = network_module.CriterionsLoader({"dummy:Measure": attr})

    loader.get_criterions("DemoModel", "Output", "Target")
    first_schedulers = dict(attr.schedulers)
    loader.get_criterions("DemoModel", "Output", "Target")

    assert attr.isTorchCriterion is True
    assert len(attr.schedulers) == 1
    assert attr.schedulers == first_schedulers
