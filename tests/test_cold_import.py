"""main.py imports `sim` before `ui`. Anything sim.py pulls in at module scope
must survive that order.

The app has a pre-existing circular import: services/__init__ -> config_saver
-> ui.physics_params -> ui/__init__ -> ui.core -> services.config_saver. It
resolves only when `ui` is imported FIRST. tests/conftest.py primes that order
for the whole suite, and tools/shader_compile_check.py does the same - so a
top-level `from services...` in sim.py passes every check here and still kills
the real app on launch. That happened.

This test runs the import in a SUBPROCESS so conftest's priming cannot hide it.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_sim_imports_before_ui_like_main_does():
    code = "import sim; from sim import Sim, SIZE_OF_ENTITY_STRUCT"
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT,
        capture_output=True, text=True,
    )
    assert r.returncode == 0, (
        "sim.py cannot be imported before ui - a module-scope `from services...`"
        f" has re-triggered the circular import:\n{r.stderr}"
    )


def test_main_import_order_works():
    """main.py's actual prefix: camera, sim, ui, services, command_handler.

    `ui` on line 7 is what makes the rest resolve, so this mirrors that order
    exactly rather than an order main.py never uses.
    """
    code = (
        "from camera import Camera; "
        "from sim import Sim; "
        "from ui import UI; "
        "import command_handler; "
        "from services.brains import default_layout; "
        "assert default_layout().signature() == 'fourier-n10'"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT,
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"main.py's import order is broken:\n{r.stderr}"
