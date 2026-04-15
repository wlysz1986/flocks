"""Lazy tiktoken cache bootstrapper.

Ensures the cl100k_base encoding data is available before tiktoken is
first imported, without polluting the global environment at package
init time.  Call ``ensure()`` once before any ``import tiktoken``.
"""

import os
import shutil
from pathlib import Path

_CACHE_KEY = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
_done = False


def ensure() -> None:
    """Seed the user-level tiktoken cache from the bundled asset if needed.

    Safe to call repeatedly — only the first invocation does real work.
    """
    global _done
    if _done:
        return
    _done = True

    if "TIKTOKEN_CACHE_DIR" in os.environ:
        return

    cache_dir = Path.home() / ".flocks" / "data" / "tiktoken_cache"

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)

        if not (cache_dir / _CACHE_KEY).exists():
            bundled = Path(__file__).resolve().parent.parent.parent / ".flocks" / "data" / "tiktoken" / _CACHE_KEY
            if bundled.exists():
                shutil.copy2(bundled, cache_dir / _CACHE_KEY)
    except OSError:
        pass

    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
