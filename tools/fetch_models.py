"""One-time download of a vision encoder's ONNX assets.

Downloads with the standard library - this runs before onnxruntime is
necessarily importable, and three static URLs do not justify a huggingface_hub
dependency. The registry is the one non-stdlib import, for the file map.

    python -m tools.fetch_models siglip2-b16
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

# The app has a pre-existing import cycle: services/__init__ -> config_saver ->
# ui -> services.config_saver, which resolves only when `ui` is imported first.
# Primed here because this module is also run standalone, before anything else
# has imported ui. Optional: in-app, services is already initialised.
try:
    import ui  # noqa: F401
except ImportError:                                     # pragma: no cover
    pass

from services.vision_models import get  # noqa: E402

MODELS_ROOT = "models"


def model_dir(model_key: str) -> Path:
    return Path(MODELS_ROOT) / get(model_key).subdir


def total_files(model_key: str) -> int:
    return len(get(model_key).files)


def missing(model_key: str) -> list[str]:
    """Which of this model's assets are not fully on disk.

    A `.part` file does not count: an interrupted download must never read as
    a finished one.
    """
    d = model_dir(model_key)
    return [name for name in get(model_key).files if not (d / name).is_file()]


def is_present(model_key: str) -> bool:
    """True only when every asset is fully downloaded."""
    return not missing(model_key)


def fetch(model_key: str, progress=None) -> None:
    """Download any missing assets.

    Writes to .part then renames, so an interrupted download can never be
    mistaken for a finished one.
    """
    model = get(model_key)
    d = model_dir(model_key)
    d.mkdir(parents=True, exist_ok=True)
    for name, remote in model.files.items():
        final = d / name
        if final.is_file():
            continue
        part = d / (name + ".part")
        if progress:
            progress(name, 0.0)
        with urllib.request.urlopen(f"{model.repo}/{remote}") as resp, \
                open(part, "wb") as out:
            total = int(resp.headers.get("content-length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(name, done / total)
        os.replace(part, final)
        if progress:
            progress(name, 1.0)


if __name__ == "__main__":
    import sys

    from services.vision_models import DEFAULT_KEY, REGISTRY

    def _cli(name: str, frac: float) -> None:
        print(f"\r{name}: {frac * 100:5.1f}%", end="", flush=True)
        if frac >= 1.0:
            print()

    key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_KEY
    if key not in REGISTRY:
        raise SystemExit(f"unknown model {key!r}; "
                         f"known: {', '.join(sorted(REGISTRY))}")
    fetch(key, progress=_cli)
    print(f"{key} ready in {model_dir(key)}")
