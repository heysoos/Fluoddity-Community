from tools.fetch_clip_onnx import FILES, is_present, _target_url


def test_files_map_covers_three_assets():
    assert set(FILES) == {
        "vision_model_fp16.onnx",
        "text_model_fp16.onnx",
        "tokenizer.json",
    }


def test_target_url_points_at_the_pinned_repo():
    url = _target_url("tokenizer.json")
    assert url == (
        "https://huggingface.co/Xenova/clip-vit-base-patch32"
        "/resolve/main/tokenizer.json"
    )
    assert _target_url("vision_model_fp16.onnx").endswith("/onnx/vision_model_fp16.onnx")


def test_is_present_false_when_a_file_is_missing(tmp_path):
    assert is_present(tmp_path) is False
    for name in FILES:
        (tmp_path / name).write_bytes(b"x")
    assert is_present(tmp_path) is True


def test_is_present_ignores_partial_downloads(tmp_path):
    """A .part file must never be mistaken for a finished download."""
    for name in FILES:
        (tmp_path / (name + ".part")).write_bytes(b"x")
    assert is_present(tmp_path) is False
