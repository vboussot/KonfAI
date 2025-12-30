import argparse

from konfai.app import KonfAIApp
from konfai.main import add_common_konfai_apps
from konfai.utils.utils import get_available_models_on_hf_repo

IMPACT_SYNTH_KONFAI_REPO = "VBoussot/ImpactSynth"


def main():
    parser = argparse.ArgumentParser(
        prog="impact-synth-konfai",
        description="ImpactSynth (KonfAI app wrapper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        choices=list(get_available_models_on_hf_repo(IMPACT_SYNTH_KONFAI_REPO)),
        help="Select which model to use. This determines what is predicted.",
    )
    parser.add_argument("--ensemble", type=int, default=0, help="Size of model ensemble")
    parser.add_argument("--tta", type=int, default=0, help="Number of Test-Time Augmentations")
    parser.add_argument("--mc_dropout", type=int, default=0, help="Monte Carlo dropout samples")

    kwargs = add_common_konfai_apps(parser)

    konfai_app = KonfAIApp(f"{IMPACT_SYNTH_KONFAI_REPO}:{kwargs.pop("model")}")
    konfai_app.pipeline(**kwargs)
