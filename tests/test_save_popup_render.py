"""Actually render the shared save dialog.

One dialog now serves File > Save and every tournament mode, so a begin/end
imbalance or a None dereference in it would take down saving everywhere at
once. ImGui asserts on an unbalanced stack inside EndFrame, so a test that
completes at all has proved the stack balances.
"""
from pathlib import Path

import pytest
from imgui_bundle import imgui

from services import save_targets
from ui.popup_modals import PopupModalsMixin


@pytest.fixture(scope="module")
def gui():
    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 900)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    for _ in range(2):
        imgui.new_frame()
        imgui.render()
    yield
    imgui.destroy_context()


class Harness(PopupModalsMixin):
    """Only the attributes render_popup_modals touches."""

    def __init__(self, tmp_path):
        self.user_configs_dir = Path(tmp_path)
        self.currently_open_project = "reef"
        self.save_popup_open = False
        self.save_filename_buffer = ""
        self.save_popup_kind = ""
        self.save_popup_arg = -1
        self.save_popup_tiles = ()
        self.overwrite_confirm_filename = None
        self.delete_confirm_filename = None
        self.delete_confirm_category = None
        self._save_filename = ""
        self._save_kind = ""
        self._save_arg = -1
        self._save_tiles = ()
        self._request_save_file = False
        self._delete_filename = ""
        self._delete_category = ""
        self._request_delete_file = False


def draw(h, n=2):
    for _ in range(n):
        imgui.new_frame()
        imgui.begin("host", True)
        h.render_popup_modals()
        imgui.end()
        imgui.render()


SUBJECTS = [
    (save_targets.CONFIG, -1, ()),
    (save_targets.TOURNAMENT_TILE, -1, (3,)),
    (save_targets.TOURNAMENT_TILE, -1, (1, 4, 9)),
    (save_targets.AUTO_BEST, -1, ()),
    (save_targets.AUTO_TILE, 7, ()),
    (save_targets.ARCHIVE_ENTRY, 123, ()),
]


@pytest.mark.parametrize("kind,arg,tiles", SUBJECTS)
def test_the_dialog_renders_for_every_subject(gui, tmp_path, kind, arg, tiles):
    h = Harness(tmp_path)
    h.open_save_popup(kind, arg=arg, tiles=tiles, generation=42)
    draw(h)


def test_the_dialog_renders_with_a_blank_name(gui, tmp_path):
    """Save is disabled here; begin_disabled/end_disabled must still balance."""
    h = Harness(tmp_path)
    h.open_save_popup(save_targets.AUTO_BEST)
    h.save_filename_buffer = ""
    draw(h)


def test_the_overwrite_dialog_renders_for_one_and_for_many(gui, tmp_path):
    (tmp_path / "reef.json").write_text("{}")
    h = Harness(tmp_path)
    h.open_save_popup(save_targets.CONFIG)
    h.overwrite_confirm_filename = "reef"
    draw(h)

    for t in (1, 4, 9):
        (tmp_path / f"many_tile{t}.json").write_text("{}")
    h2 = Harness(tmp_path)
    h2.open_save_popup(save_targets.TOURNAMENT_TILE, tiles=(1, 4, 9))
    h2.overwrite_confirm_filename = "many"
    draw(h2)


# ---- the state the dialog hands back ------------------------------------

def test_opening_prefills_the_suggested_name(tmp_path):
    h = Harness(tmp_path)
    h.open_save_popup(save_targets.AUTO_TILE, arg=7, generation=42)
    assert h.save_filename_buffer == "tile7_gen0042"


def test_the_live_config_prefills_the_open_project(tmp_path):
    h = Harness(tmp_path)
    h.open_save_popup()
    assert h.save_filename_buffer == "reef"


def test_committing_carries_the_subject_not_just_the_name(tmp_path):
    """The orchestrator dispatches on kind, so losing it here would save the
    live sliders under the name the user typed for a tile."""
    h = Harness(tmp_path)
    h.open_save_popup(save_targets.ARCHIVE_ENTRY, arg=123)
    h._commit_save("keeper")

    assert h._request_save_file is True
    assert h._save_filename == "keeper"
    assert h._save_kind == save_targets.ARCHIVE_ENTRY
    assert h._save_arg == 123
    assert h.save_popup_open is False


def test_the_selected_tiles_travel_with_the_request(tmp_path):
    h = Harness(tmp_path)
    h.open_save_popup(save_targets.TOURNAMENT_TILE, tiles=(1, 4))
    h._commit_save("reef")
    assert h._save_tiles == (1, 4)


def test_existing_files_are_detected_across_every_target(tmp_path):
    """With several tiles the check has to cover all of them - confirming only
    the first would silently clobber the rest."""
    (tmp_path / "reef_tile4.json").write_text("{}")
    h = Harness(tmp_path)
    h.open_save_popup(save_targets.TOURNAMENT_TILE, tiles=(1, 4))
    assert h._existing_stems("reef") == ["reef_tile4"]


def test_no_clash_when_nothing_is_on_disk(tmp_path):
    h = Harness(tmp_path)
    h.open_save_popup(save_targets.CONFIG)
    assert h._existing_stems("brand-new") == []
