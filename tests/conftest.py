import sys
from pathlib import Path

# Put the repo root on sys.path so `import services`, `import state`, etc. resolve
# when pytest is run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The app has a pre-existing import cycle: services/__init__ -> config_saver ->
# ui.physics_params -> ui/__init__ -> ui.core -> services.config_saver.
# It resolves only when `ui` is imported before `services` (which is what main.py
# happens to do). Prime that order here so a cold `import services.x` works in tests.
import ui  # noqa: E402,F401

# Dear ImGui persists every window's position and size to imgui.ini, loading it
# back in create_context() and writing it out in destroy_context(). The tests
# create their own windows, so without this they SAVE a layout and then CONSUME
# it on the next run: the archive gallery's window came back 382px tall and two
# of its eight entries no longer fit, failing a test that had passed all along.
# Running the app writes the same file, so the suite's result depended on
# whether anyone had resized a panel since.
from imgui_bundle import imgui  # noqa: E402

from ui import ini_path  # noqa: E402

# The app points ImGui at Documents/Fluoddity/imgui.ini during UI construction,
# which runs after create_context and would undo the null below.
ini_path.suppress()

_create_context = imgui.create_context


def _create_context_without_ini(*args, **kwargs):
    ctx = _create_context(*args, **kwargs)
    # Reading it back segfaults on the null; only ever set it.
    imgui.get_io().set_ini_filename(None)
    return ctx


imgui.create_context = _create_context_without_ini
