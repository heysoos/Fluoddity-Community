import sys
from pathlib import Path

# Put the repo root on sys.path so `import services`, `import state`, etc. resolve
# when pytest is run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
