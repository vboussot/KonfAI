"""Test bootstrap for the standalone `konfai-apps` package.

These tests live in a subpackage of the monorepo, so we add both the repo root
and the standalone package root to `sys.path` to support local `pytest`
invocations without requiring a prior editable install.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KONFAI_APPS_ROOT = REPO_ROOT / "konfai-apps"

for path in (REPO_ROOT, KONFAI_APPS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
