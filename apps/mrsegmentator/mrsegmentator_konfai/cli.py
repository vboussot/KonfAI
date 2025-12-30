import argparse

from konfai.app import KonfAIApp
from konfai.main import add_common_konfai_apps

MR_SEGMENTATOR_KONFAI_REPO = "VBoussot/MRSegmentator-KonfAI"


def main():
    parser = argparse.ArgumentParser(
        prog="mrsegmentator-konfai",
        description="MRSegmentator (KonfAI app wrapper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f", "--folds", choices=[1, 2, 3, 4, 5], help="Number of folds to ensemble.", default=2, type=int
    )
    kwargs = add_common_konfai_apps(parser)
    kwargs["ensemble"] = kwargs.pop("folds")
    konfai_app = KonfAIApp(f"{MR_SEGMENTATOR_KONFAI_REPO}:MRSegmentator")
    konfai_app.pipeline(**kwargs)
