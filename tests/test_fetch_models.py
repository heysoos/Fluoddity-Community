"""Download bookkeeping, without touching the network."""
import pytest

from tools.fetch_models import is_present, missing, model_dir, total_files


def test_a_model_dir_is_named_for_its_subdir():
    assert model_dir("siglip2-b16").name == "siglip2-b16-224"
    assert model_dir("clip-b32").name == "clip-vit-b32"


def test_an_empty_dir_is_not_present_and_lists_everything_missing(tmp_path,
                                                                 monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not is_present("clip-b32")
    assert len(missing("clip-b32")) == total_files("clip-b32") == 3


def test_a_partial_download_is_not_present(tmp_path, monkeypatch):
    """A .part file must never read as a finished asset."""
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "models" / "clip-vit-b32"
    d.mkdir(parents=True)
    (d / "vision_model_fp16.onnx").write_bytes(b"x")
    (d / "text_model_fp16.onnx.part").write_bytes(b"x")
    assert not is_present("clip-b32")
    assert "text_model_fp16.onnx" in missing("clip-b32")
    assert "vision_model_fp16.onnx" not in missing("clip-b32")


def test_a_complete_dir_is_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "models" / "clip-vit-b32"
    d.mkdir(parents=True)
    for name in ("vision_model_fp16.onnx", "text_model_fp16.onnx",
                 "tokenizer.json"):
        (d / name).write_bytes(b"x")
    assert is_present("clip-b32")
    assert missing("clip-b32") == []


def test_each_model_downloads_into_its_own_directory():
    """Two encoders sharing a directory would overwrite each other's weights."""
    dirs = {model_dir(k) for k in ("clip-b32", "clip-b16", "siglip2-b16",
                                   "clip-l14")}
    assert len(dirs) == 4


def test_every_asset_url_points_at_its_pinned_repo():
    """Carried over from the single-model fetcher: a wrong path here downloads
    a plausible file from the wrong revision."""
    from services.vision_models import get

    m = get("clip-b32")
    assert (f"{m.repo}/{m.files['tokenizer.json']}" ==
            "https://huggingface.co/Xenova/clip-vit-base-patch32"
            "/resolve/main/tokenizer.json")
    for key in ("clip-b32", "clip-b16", "siglip2-b16", "clip-l14"):
        model = get(key)
        assert model.files["vision_model_fp16.onnx"].startswith("onnx/")
        assert model.repo.endswith("/resolve/main")


def test_an_unknown_key_is_refused():
    with pytest.raises(KeyError):
        is_present("nope")
