"""Make `latentmoe` importable without installing the package.

pytest imports this before collecting tests, so `pytest tests/` works from any
working directory and with any invocation (`pytest`, `python -m pytest`, CI).
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
