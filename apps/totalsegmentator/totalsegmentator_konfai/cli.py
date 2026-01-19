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

TOTAL_SEGMENTATOR_KONFAI_REPO = "VBoussot/TotalSegmentator-KonfAI"


def main():
    parser = argparse.ArgumentParser(
        prog="totalsegmentator-konfai",
        description="TotalSegmentator (KonfAI app wrapper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "task",
        choices=list(get_available_apps_on_hf_repo(TOTAL_SEGMENTATOR_KONFAI_REPO, False)),
        help="Select which model to use. This determines what is predicted.",
    )
    parser.add_argument("--models", nargs="+", default=[], help="Explicit list of model identifiers/paths to use.")
    kwargs = add_common_konfai_apps(parser, False)
    konfai_app = KonfAIApp(
        f"{TOTAL_SEGMENTATOR_KONFAI_REPO}:{kwargs.pop("task")}", kwargs.pop("download"), kwargs.pop("force_update")
    )
    kwargs["ensemble_models"] = kwargs.pop("models")
    konfai_app.pipeline(**kwargs)
