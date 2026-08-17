"""Moving the encoder weights out of the launched folder.

They used to be read from the working directory, which made them a property of
whichever checkout was started rather than of the user: five worktrees wanted
five copies of several gigabytes, and a packaged build under Program Files
could not write the folder at all.
"""
from utilities.paths import get_models_root, migrate_models

ASSETS = ("vision_model_fp16.onnx", "text_model_fp16.onnx", "tokenizer.json")


def _weights(root, subdir="clip-vit-b32", body=b"weights"):
    d = root / "models" / subdir
    d.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        (d / name).write_bytes(body)
    return d


def test_the_root_is_user_data(tmp_path):
    assert get_models_root(tmp_path) == tmp_path / "models"


def test_weights_beside_the_app_are_adopted(tmp_path):
    app, user = tmp_path / "app", tmp_path / "user"
    _weights(app)
    user.mkdir()

    root = migrate_models(user_dir=user, app_dir=app)
    assert root == user / "models"
    for name in ASSETS:
        assert (root / "clip-vit-b32" / name).read_bytes() == b"weights"


def test_it_is_a_move_so_gigabytes_are_not_duplicated(tmp_path):
    app, user = tmp_path / "app", tmp_path / "user"
    _weights(app)
    user.mkdir()

    migrate_models(user_dir=user, app_dir=app)
    assert not (app / "models").exists()


def test_every_encoder_travels_together(tmp_path):
    app, user = tmp_path / "app", tmp_path / "user"
    for sub in ("clip-vit-b32", "clip-vit-b16", "siglip2-b16-224"):
        _weights(app, sub)
    user.mkdir()

    root = migrate_models(user_dir=user, app_dir=app)
    assert {p.name for p in root.iterdir()} == {
        "clip-vit-b32", "clip-vit-b16", "siglip2-b16-224"}


def test_a_second_checkout_cannot_donate_over_the_one_in_use(tmp_path):
    """The rule migrate_legacy_archive already follows: once the target
    exists, nothing is swallowed into it."""
    app, user = tmp_path / "app", tmp_path / "user"
    _weights(app, body=b"the other checkout")
    _weights(user, body=b"in use")

    migrate_models(user_dir=user, app_dir=app)
    assert (user / "models" / "clip-vit-b32"
            / "tokenizer.json").read_bytes() == b"in use"
    assert (app / "models").exists(), "the other copy is left alone, not eaten"


def test_running_twice_changes_nothing(tmp_path):
    """It runs at every launch."""
    app, user = tmp_path / "app", tmp_path / "user"
    _weights(app)
    user.mkdir()

    migrate_models(user_dir=user, app_dir=app)
    migrate_models(user_dir=user, app_dir=app)
    assert (user / "models" / "clip-vit-b32" / "tokenizer.json").is_file()


def test_nothing_to_move_creates_nothing(tmp_path):
    """An empty models/ here reads as a finished migration, so creating one
    would strand weights that are still beside the app."""
    app, user = tmp_path / "app", tmp_path / "user"
    app.mkdir()
    user.mkdir()

    root = migrate_models(user_dir=user, app_dir=app)
    assert not root.exists()


def test_a_failure_leaves_the_weights_where_they_are(tmp_path, monkeypatch):
    """Launching matters more than migrating, and a partial move would lose
    weights the app can no longer find."""
    import utilities.paths as paths

    app, user = tmp_path / "app", tmp_path / "user"
    _weights(app)
    user.mkdir()

    def boom(src, dst):
        raise OSError("different volume")

    monkeypatch.setattr(paths.shutil, "move", boom)
    migrate_models(user_dir=user, app_dir=app)
    assert (app / "models" / "clip-vit-b32" / "tokenizer.json").is_file()
