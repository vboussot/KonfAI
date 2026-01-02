import argparse

from konfai.app import KonfAIApp
from konfai.main import add_common_konfai_apps
from konfai.utils.utils import get_available_models_on_hf_repo

TOTAL_SEGMENTATOR_KONFAI_REPO = "VBoussot/TotalSegmentator-KonfAI"


def main():
    parser = argparse.ArgumentParser(
        prog="totalsegmentator-konfai",
        description="TotalSegmentator (KonfAI app wrapper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "task",
        choices=list(get_available_models_on_hf_repo(TOTAL_SEGMENTATOR_KONFAI_REPO)),
        help="Select which model to use. This determines what is predicted.",
    )
    parser.add_argument("--models", nargs="+", default=[], help="Explicit list of model identifiers/paths to use.")
    kwargs = add_common_konfai_apps(parser, False)
    konfai_app = KonfAIApp(f"{TOTAL_SEGMENTATOR_KONFAI_REPO}:{kwargs.pop("task")}")
    kwargs["ensemble_models"] = kwargs.pop("models")
    konfai_app.pipeline(**kwargs)
