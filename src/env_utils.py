"""Small shared utility — avoids duplicating .env-loading logic between
pipeline.py and extractor.py (and avoids a circular import between them)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def load_env_file(env_path: Path = DEFAULT_ENV_PATH) -> None:
    """Minimal .env loader (no python-dotenv dependency needed).
    Does not override variables already set in the real environment."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
