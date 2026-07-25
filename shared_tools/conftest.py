"""Add shared_tools/ to sys.path so bare imports (from pricing import ...) resolve."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
