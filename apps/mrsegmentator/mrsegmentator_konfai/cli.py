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
    konfai_app = KonfAIApp(
        f"{MR_SEGMENTATOR_KONFAI_REPO}:MRSegmentator", kwargs.pop("download"), kwargs.pop("force_update")
    )
    konfai_app.pipeline(**kwargs)
