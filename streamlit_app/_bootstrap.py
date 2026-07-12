"""
_bootstrap.py
=============
Import-path bootstrap for the Streamlit app.

`streamlit run streamlit_app/Home.py` executes the script from the
`streamlit_app/` directory's perspective, which means the project root
(the directory containing `src/`) is NOT automatically on `sys.path`.
Without this, every `from src.xxx import yyy` in `Home.py` and every page
under `streamlit_app/pages/` would raise `ModuleNotFoundError: No module
named 'src'`.

Every page imports this module FIRST (`import _bootstrap`) purely for its
side effect of inserting the project root onto `sys.path`, before doing
any `from src...` imports.

Usage (top of every Streamlit page):
    import _bootstrap  # noqa: F401  (side effect: fixes sys.path)
    from src.data.preprocess import preprocess_all
"""

import sys
from pathlib import Path

# streamlit_app/_bootstrap.py -> streamlit_app -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
