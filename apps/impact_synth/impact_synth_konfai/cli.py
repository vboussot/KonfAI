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

from konfai.app import KonfAIApp
from konfai.main import add_common_konfai_apps
from konfai.utils.utils import get_available_apps_on_hf_repo

IMPACT_SYNTH_KONFAI_REPO = "VBoussot/ImpactSynth"


def main():
    parser = argparse.ArgumentParser(
        prog="impact-synth-konfai",
        description="ImpactSynth (KonfAI app wrapper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        choices=list(get_available_apps_on_hf_repo(IMPACT_SYNTH_KONFAI_REPO, False)),
        help="Select which model to use. This determines what is predicted.",
    )
    parser.add_argument("--ensemble", type=int, default=0, help="Size of model ensemble")
    parser.add_argument("--tta", type=int, default=0, help="Number of Test-Time Augmentations")
    parser.add_argument("--mc", type=int, default=0, help="Monte Carlo dropout samples")

    kwargs = add_common_konfai_apps(parser)

    konfai_app = KonfAIApp(
        f"{IMPACT_SYNTH_KONFAI_REPO}:{kwargs.pop("model")}", kwargs.pop("download"), kwargs.pop("force_update")
    )
    konfai_app.pipeline(**kwargs)
