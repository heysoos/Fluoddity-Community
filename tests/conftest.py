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
