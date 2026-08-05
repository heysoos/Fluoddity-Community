"""One-time download of the CLIP ViT-B/32 ONNX assets.

Stdlib only - this runs before onnxruntime is necessarily importable, and we do
not want a huggingface_hub dependency for three static URLs.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

MODEL_DIR = "models/clip-vit-b32"

_REPO = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"

# local filename -> path within the HF repo
FILES: dict[str, str] = {
    "vision_model_fp16.onnx": "onnx/vision_model_fp16.onnx",
    "text_model_fp16.onnx": "onnx/text_model_fp16.onnx",
    "tokenizer.json": "tokenizer.json",
}


def _target_url(name: str) -> str:
    return f"{_REPO}/{FILES[name]}"


def is_present(dest_dir: str | Path = MODEL_DIR) -> bool:
    """True only when every asset is fully downloaded."""
    d = Path(dest_dir)
    return all((d / name).is_file() for name in FILES)


def fetch(dest_dir: str | Path = MODEL_DIR, progress=None) -> None:
    """Download any missing assets.

    Writes to .part then renames, so an interrupted download can never be
    mistaken for a finished one.
    """
    d = Path(dest_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        final = d / name
        if final.is_file():
            continue
        part = d / (name + ".part")
        url = _target_url(name)
        if progress:
            progress(name, 0.0)
        with urllib.request.urlopen(url) as resp, open(part, "wb") as out:
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
    def _cli(name: str, frac: float) -> None:
        print(f"\r{name}: {frac * 100:5.1f}%", end="", flush=True)
        if frac >= 1.0:
            print()

    fetch(progress=_cli)
    print(f"CLIP assets ready in {MODEL_DIR}")
