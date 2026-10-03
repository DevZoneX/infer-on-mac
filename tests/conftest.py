from __future__ import annotations

import sys
from pathlib import Path

# Keep the tests runnable from a source checkout without requiring an editable
# install first.  This also makes it explicit that the suite exercises the code
# in this repository, rather than an older globally-installed mlx_chat package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
