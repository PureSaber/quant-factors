"""Exercise real QDK stores on Windows without depending on the OS long-path flag."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path(tmp_path: Path) -> Path:
    if os.name == "nt":
        path = str(tmp_path.resolve())
        if not path.startswith("\\\\?\\"):
            return Path("\\\\?\\UNC\\" + path[2:] if path.startswith("\\\\") else "\\\\?\\" + path)
    return tmp_path
