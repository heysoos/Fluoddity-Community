# Swappable Brain Modalities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the particle brain a swappable component — Fourier (existing), Gabor, Lenia bumps, MLP — each with its own parameter layout, selectable from the UI, searchable by the existing optimizers, and inspectable as a rendered response field.

**Architecture:** A modality is two files (one GLSL function, one Python class) plus a registry entry. The GPU dispatches on a uniform int, so switching costs one uniform write and no recompile. All brains live in one flat `float` SSBO with an active length that bounds every loop. A `BrainLayout` (modality + structural shape + float count) is the key that the genome spec, checkpoints, and archive directory all derive from.

**Tech Stack:** Python 3.12, NumPy, ModernGL (OpenGL 4.3 compute), GLSL 450, imgui_bundle, pytest.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-08-brain-modalities-design.md`. Read it before Task 1.
- **`MAX_BRAIN_FLOATS = 512`.** Every layout must fit. Fourier ≤48 centers (384), Gabor ≤36 filters (504), Lenia ≤48 bumps (480), MLP ≤48 hidden (436).
- **Brain GLSL functions must be pure** — they may read only `brain_params[]`, their `base` offset, `x`, and `BRAIN_SHAPE`. No entity state, no trail texture. The Brain Inspector depends on this.
- **Fourier must stay bit-exact** at its default layout (10 centers). Existing genomes and archives must decode identically.
- **`sim.py` is user-owned.** Modify only the specific lines named in a task. Do not restructure, do not reorder its hardcoded parameter lists.
- **Naming:** shader uniforms `ALL_CAPS_UNDERSCORE`, UI labels Title Case, private UI state `_snake_case`.
- **Windows platform.** Use forward slashes or `os.path`. Use `rm`, not `del`, in bash.
- **Run tests with the venv interpreter:** `.venv/Scripts/python.exe -m pytest`. Bare `python` is 3.10 and has no pytest.
- **No GPU in tests.** Tests target the CPU half and assert on shader *source text*, following `tests/test_shader_source.py`.

---

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `services/brains/__init__.py` | `BrainLayout`, `BrainModality` protocol, `REGISTRY`, `get(name)`, `default_layout()` |
| `services/brains/fourier.py` | Fourier decode/encode/random/settings |
| `services/brains/gabor.py` | Gabor decode/encode/random/settings |
| `services/brains/lenia.py` | Lenia decode/encode/random/settings |
| `services/brains/mlp.py` | MLP decode/encode/random/settings |
| `shaders/brains/fourier.glsl` | `brain_fourier(uint, vec4) -> vec4` |
| `shaders/brains/gabor.glsl` | `brain_gabor(uint, vec4) -> vec4` |
| `shaders/brains/lenia.glsl` | `brain_lenia(uint, vec4) -> vec4` |
| `shaders/brains/mlp.glsl` | `brain_mlp(uint, vec4) -> vec4` |
| `shaders/brain_preview.frag` | Renders one unit's response into the inspector atlas |
| `services/brain_preview.py` | Owns the preview FBO and the atlas draw |
| `ui/brain_window.py` | `BrainWindowMixin` — dropdown, settings, status, saturation meter, inspector |
| `tests/test_brain_layout.py` | Layout arithmetic, signatures, registry completeness |
| `tests/test_brain_modalities.py` | Per-modality round-trip and range tests |
| `tests/test_brain_fourier_parity.py` | New decode == old decode, bit-exact |
| `tests/test_brain_shader_source.py` | GLSL contract assertions |
| `tests/test_archive_signature.py` | Signature-keyed dirs, width rejection, migration |

**Modified:**

| Path | Change |
|---|---|
| `services/genome_spec.py` | `decode`/`encode` delegate to the active modality; `spec_for(layout)` |
| `services/archive.py:123` | `_brain` becomes `(0, layout.length)` flat |
| `services/archive_io.py` | Store the layout signature in the archive |
| `services/optimizers.py:300-301` | `GAOptimizer` reshape uses layout, not hardcoded 8 |
| `shaders/entity_update.glsl:13-21,470-472` | `Brain` SSBO, dispatch uniforms, `black_box` branch |
| `sim.py:140` | Prepend the modality GLSL files |
| `sim.py` (`write_tournament_rules`) | Pack flat params |
| `utilities/gl_helpers.py:67` | Delete `set_rule_uniform`, add `pack_brains` |
| `ui/core.py:19-55` | Register `BrainWindowMixin` |
| `main.py:240` | Archive path from layout signature |

---

## Phase 1 — CPU foundation (no behaviour change)

### Task 1: BrainLayout and the registry

**Files:**
- Create: `services/brains/__init__.py`
- Test: `tests/test_brain_layout.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `BrainLayout(modality: str, shape: tuple[int,...], length: int)` with `.signature() -> str`; `Setting(key, label, kind, lo, hi, default, choices=())`; `MAX_BRAIN_FLOATS: int = 512`; `REGISTRY: dict[str, BrainModality]`; `get(name) -> BrainModality`; `default_layout() -> BrainLayout`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brain_layout.py
import pytest

from services.brains import MAX_BRAIN_FLOATS, BrainLayout


def test_signature_is_modality_and_shape():
    assert BrainLayout("fourier", (10,), 80).signature() == "fourier-n10"
    assert BrainLayout("mlp", (16, 0), 148).signature() == "mlp-n16-a0"


def test_signature_distinguishes_layouts():
    a = BrainLayout("gabor", (12,), 168)
    b = BrainLayout("gabor", (8,), 112)
    assert a.signature() != b.signature()


def test_layout_rejects_over_budget():
    with pytest.raises(ValueError, match="exceeds MAX_BRAIN_FLOATS"):
        BrainLayout("fourier", (100,), 800)


def test_max_brain_floats_is_512():
    assert MAX_BRAIN_FLOATS == 512
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_layout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.brains'`

- [ ] **Step 3: Write minimal implementation**

```python
# services/brains/__init__.py
"""Swappable particle brains.

A modality is one GLSL function plus one Python class. `BrainLayout` is the key
everything downstream derives from: the search dimension, the checkpoint
signature, and the archive directory all come from it.

See docs/superpowers/specs/2026-08-08-brain-modalities-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

# The flat parameter buffer's stride. Sized so every UI-reachable layout fits:
# Fourier 48*8=384, Gabor 36*14=504, Lenia 48*10=480, MLP H=48 -> 9*48+4=436.
MAX_BRAIN_FLOATS = 512


@dataclass(frozen=True)
class BrainLayout:
    modality: str
    shape: tuple[int, ...]
    length: int

    def __post_init__(self):
        if self.length > MAX_BRAIN_FLOATS:
            raise ValueError(
                f"{self.modality} layout needs {self.length} floats, which "
                f"exceeds MAX_BRAIN_FLOATS ({MAX_BRAIN_FLOATS})"
            )

    def signature(self) -> str:
        """Archive directory name and checkpoint guard. Must be stable across
        runs - a change here silently merges two archives."""
        parts = "-".join(f"{c}{v}" for c, v in zip("nabc", self.shape))
        return f"{self.modality}-{parts}"


@dataclass(frozen=True)
class Setting:
    """One UI knob a modality declares. Lives here, not in a modality module,
    so every modality imports it from the same place."""
    key: str
    label: str
    kind: str            # "int" | "float" | "choice"
    lo: float
    hi: float
    default: float
    choices: tuple = ()


REGISTRY: dict = {}


def register(modality) -> None:
    if modality.name in REGISTRY:
        raise ValueError(f"duplicate modality name {modality.name!r}")
    ids = {m.modality_id for m in REGISTRY.values()}
    if modality.modality_id in ids:
        raise ValueError(f"duplicate modality_id {modality.modality_id}")
    REGISTRY[modality.name] = modality


def get(name: str):
    """The named modality, or Fourier if the name is unknown.

    Never raises: an unknown name comes from a config file written by a newer
    build, and the app must keep running.
    """
    return REGISTRY.get(name) or REGISTRY["fourier"]


def default_layout() -> BrainLayout:
    return REGISTRY["fourier"].layout_from_settings({})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_layout.py -v`
Expected: PASS, 4 tests. (`default_layout` is untested here — Task 2 registers Fourier.)

- [ ] **Step 5: Commit**

```bash
git add services/brains/__init__.py tests/test_brain_layout.py
git commit -m "feat: add BrainLayout and the modality registry"
```

---

### Task 2: Fourier modality with bit-exact parity

**Files:**
- Create: `services/brains/fourier.py`
- Test: `tests/test_brain_fourier_parity.py`

**Interfaces:**
- Consumes: `BrainLayout`, `register` from Task 1.
- Produces: `FourierModality` with `name="fourier"`, `modality_id=0`, and methods `settings_schema() -> list[Setting]`, `layout_from_settings(dict) -> BrainLayout`, `decode(z, layout) -> np.ndarray` (flat, GPU order), `encode(params, layout) -> tuple[np.ndarray, int]`, `random(rng, layout) -> np.ndarray`. Also `Setting(key, label, kind, lo, hi, default)`.

Parameter order in the flat array is **GPU order**: `freq0(4), amp0(4), freq1(4), amp1(4), ...` — that is `(N, 8).reshape(-1)`, matching `struct FourierCenter { vec4 frequency; vec4 amplitude; }`. Note the search vector `z` uses a *different* order (all frequencies, then all amplitudes), preserved from `genome_spec.decode`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brain_fourier_parity.py
"""The new decode must reproduce the old one exactly, or every archived
genome silently becomes a different creature."""
import numpy as np

from services.brains import BrainLayout
from services.brains.fourier import FourierModality

# The legacy implementation, frozen here on purpose. Task 3 rewrites
# genome_spec.decode to delegate to the modality, so importing it would make
# this test compare the modality against itself and silently stop testing.
LEGACY_FREQ_SCALE = 3.0
LEGACY_AMP_SCALE = 1.0


def legacy_decode(z):
    z = np.asarray(z, dtype=np.float32).reshape(20, 4)
    freq = LEGACY_FREQ_SCALE * np.tanh(z[:10])
    amp = LEGACY_AMP_SCALE * np.tanh(z[10:])
    return np.concatenate([freq, amp], axis=1).astype(np.float32)


def test_decode_matches_legacy_bit_for_bit():
    rng = np.random.default_rng(0)
    m = FourierModality()
    layout = m.layout_from_settings({})
    for _ in range(20):
        z = rng.normal(0, 1.5, 80).astype(np.float32)
        got = m.decode(z, layout)
        want = legacy_decode(z).reshape(-1)
        assert np.array_equal(got, want)


def test_default_layout_is_ten_centers_eighty_floats():
    layout = FourierModality().layout_from_settings({})
    assert layout == BrainLayout("fourier", (10,), 80)
    assert layout.signature() == "fourier-n10"


def test_encode_decode_round_trips():
    rng = np.random.default_rng(1)
    m = FourierModality()
    layout = m.layout_from_settings({})
    z = rng.normal(0, 0.5, 80).astype(np.float32)
    params = m.decode(z, layout)
    back, clamped = m.encode(params, layout)
    assert clamped == 0
    assert np.allclose(back, z, atol=1e-4)


def test_random_matches_the_hand_tuned_prior_scale():
    """random_genome biases frequencies low on purpose; the modality's prior
    must keep that. Measured mean |freq| of the legacy generator is 0.832."""
    m = FourierModality()
    layout = m.layout_from_settings({})
    rng = np.random.default_rng(7)
    freqs = []
    for _ in range(500):
        p = m.random(rng, layout).reshape(10, 8)
        freqs.append(np.abs(p[:, :4]))
    assert 0.75 < float(np.mean(freqs)) < 0.92


def test_larger_layout_scales_length():
    m = FourierModality()
    assert m.layout_from_settings({"centers": 24}).length == 192
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_fourier_parity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.brains.fourier'`

- [ ] **Step 3: Write minimal implementation**

```python
# services/brains/fourier.py
"""The original brain: a random-Fourier-features net, 10 units, linear in the
amplitudes.

    out = sum_i amp_i * basis(dot(x, freq_i) + offset_i)

Global and periodic - every reading excites every unit in all directions, with
no 'off' state. Kept bit-exact at the default layout so archived genomes are
unchanged.
"""
from __future__ import annotations

import numpy as np

from services.brains import BrainLayout, Setting, register

FREQ_SCALE = 3.0
AMP_SCALE = 1.0
EPS = 1e-4


class FourierModality:
    name = "fourier"
    modality_id = 0
    glsl_file = "shaders/brains/fourier.glsl"

    def settings_schema(self) -> list[Setting]:
        return [
            Setting("centers", "Centers", "int", 4, 48, 10),
            # The legacy decode used a flat 3.0 with no low-frequency bias,
            # while random_genome biases low. Measured over five runs, evolved
            # frequencies drift to ~2x the generator's mean. Exposing both as
            # settings makes that a choice rather than an accident.
            Setting("freq_scale", "Freq Scale", "float", 0.5, 4.0, FREQ_SCALE),
            Setting("low_freq_bias", "Low-Freq Bias", "float", 0.0, 1.0, 0.0),
        ]

    def layout_from_settings(self, s: dict) -> BrainLayout:
        n = int(s.get("centers", 10))
        return BrainLayout("fourier", (n,), 8 * n)

    def decode(self, z: np.ndarray, layout: BrainLayout) -> np.ndarray:
        """(8N,) search vector -> (8N,) flat params in GPU order.

        z is ordered [all frequencies, all amplitudes]; the GPU wants them
        interleaved per center. Preserved from genome_spec.decode.
        """
        n = layout.shape[0]
        z = np.asarray(z, dtype=np.float32).reshape(n * 2, 4)
        freq = FREQ_SCALE * np.tanh(z[:n])
        amp = AMP_SCALE * np.tanh(z[n:])
        return np.concatenate([freq, amp], axis=1).astype(np.float32).reshape(-1)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        n = layout.shape[0]
        g = np.asarray(params, dtype=np.float32).reshape(n, 8)
        raw = np.concatenate([g[:, :4] / FREQ_SCALE, g[:, 4:] / AMP_SCALE], axis=0)
        n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
        z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
        return z.reshape(-1).astype(np.float32), n_clamped

    def random(self, rng, layout: BrainLayout) -> np.ndarray:
        """Mirrors services.genome.random_genome, which biases frequencies low
        'for smoother base behaviors'."""
        n = layout.shape[0]
        freq_scale = 1.0 + 2.0 * rng.random((n, 4)) ** 2
        freq = (rng.random((n, 4)) * 2.0 - 1.0) * freq_scale
        amp = rng.random((n, 4)) * 2.0 - 1.0
        return np.concatenate([freq, amp], axis=1).astype(np.float32).reshape(-1)


register(FourierModality())
```

Then add the import at the bottom of `services/brains/__init__.py` so registration happens on package import:

```python
# services/brains/__init__.py — append at end of file
from services.brains import fourier as _fourier  # noqa: E402,F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_fourier_parity.py tests/test_brain_layout.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Run the full suite to confirm nothing regressed**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: exit code 0.

- [ ] **Step 6: Commit**

```bash
git add services/brains/fourier.py services/brains/__init__.py tests/test_brain_fourier_parity.py
git commit -m "feat: add the Fourier modality with bit-exact legacy parity"
```

---

### Task 3: Delegate genome_spec to the active modality

**Files:**
- Modify: `services/genome_spec.py`
- Test: `tests/test_brain_layout.py` (extend)

**Interfaces:**
- Consumes: `BrainLayout`, `get`, `FourierModality` from Tasks 1–2.
- Produces: `spec_for(layout: BrainLayout) -> GenomeSpec` and `physics_spec_for(layout) -> GenomeSpec`. `BRAIN_SPEC` and `BRAIN_PHYSICS_SPEC` remain as the Fourier-default instances so every existing import keeps working.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brain_layout.py — append
import numpy as np

from services.brains import default_layout
from services.genome_spec import spec_for


def test_spec_for_layout_sizes_the_brain_block():
    layout = default_layout()
    spec = spec_for(layout)
    assert spec.dim == layout.length
    assert spec.signature() == "brain:80"


def test_spec_decode_returns_flat_params():
    layout = default_layout()
    spec = spec_for(layout)
    parts = spec.decode(np.zeros(layout.length, dtype=np.float32))
    assert parts["brain"].shape == (layout.length,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_layout.py -v`
Expected: FAIL with `ImportError: cannot import name 'spec_for'`

- [ ] **Step 3: Write minimal implementation**

Replace the module-level `decode`/`encode` in `services/genome_spec.py` with delegations, and add the factories. Keep the old names — `sim.py`, `command_handler.py` and eight test files import them.

```python
# services/genome_spec.py — replace decode() and encode(), keep everything else
from services.brains import BrainLayout, default_layout, get

def _active(layout: BrainLayout | None):
    layout = layout or default_layout()
    return get(layout.modality), layout


def decode(z: np.ndarray, layout: BrainLayout | None = None) -> np.ndarray:
    """(dim,) -> flat params. Shape (10, 8) is preserved for the Fourier
    default so legacy callers that reshape are unaffected."""
    m, layout = _active(layout)
    flat = m.decode(z, layout)
    if layout.modality == "fourier":
        return flat.reshape(layout.shape[0], 8)
    return flat


def encode(genome: np.ndarray, layout: BrainLayout | None = None):
    m, layout = _active(layout)
    return m.encode(np.asarray(genome, dtype=np.float32).reshape(-1), layout)


def spec_for(layout: BrainLayout) -> GenomeSpec:
    return GenomeSpec([Block("brain", layout.length)], layout)


def physics_spec_for(layout: BrainLayout) -> GenomeSpec:
    from services.physics_genome import PHYSICS_DIM
    return GenomeSpec(
        [Block("brain", layout.length), Block("physics", PHYSICS_DIM)], layout)
```

`GenomeSpec.decode` currently special-cases `b.name == "brain"` by calling the module `decode`. Change that line to pass the layout through:

```python
# services/genome_spec.py — inside GenomeSpec.decode
out[b.name] = (
    get(self.layout.modality).decode(chunk, self.layout)
    if b.name == "brain" else chunk.copy()
)
```

and give `GenomeSpec.__init__` an optional layout defaulting to `default_layout()`:

```python
def __init__(self, blocks: list[Block], layout: BrainLayout | None = None):
    self.blocks = list(blocks)
    self.layout = layout or default_layout()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_layout.py tests/test_brain_fourier_parity.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite — this is the parity gate**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: exit code 0. All 726 pre-existing tests must still pass unmodified. If any fail, the delegation changed behaviour — fix it rather than editing the test.

- [ ] **Step 6: Commit**

```bash
git add services/genome_spec.py tests/test_brain_layout.py
git commit -m "refactor: delegate genome_spec decode/encode to the active modality"
```

---

## Phase 2 — GPU plumbing (Fourier stays bit-exact)

### Task 4: Flat brain buffer and uniform dispatch

**Files:**
- Create: `shaders/brains/fourier.glsl`
- Modify: `shaders/entity_update.glsl` (lines 13–21 struct/SSBO, 470–472 `black_box`)
- Modify: `sim.py:140` (prepend), `write_tournament_rules`
- Modify: `utilities/gl_helpers.py` (replace `set_rule_uniform` with `pack_brains`)
- Test: `tests/test_brain_shader_source.py`

**Interfaces:**
- Consumes: `MAX_BRAIN_FLOATS`, `BrainLayout` from Task 1.
- Produces: `pack_brains(params_list, layout) -> bytes` in `utilities/gl_helpers.py`, producing `len(params_list) * MAX_BRAIN_FLOATS * 4` bytes with each brain zero-padded to the stride.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brain_shader_source.py
"""No GPU in CI, so the GLSL contract is asserted on source text."""
from pathlib import Path

import numpy as np

from services.brains import MAX_BRAIN_FLOATS, default_layout
from utilities.gl_helpers import pack_brains

ROOT = Path(__file__).resolve().parent.parent


def read(p):
    return (ROOT / p).read_text()


def test_entity_update_declares_the_flat_brain_buffer():
    src = read("shaders/entity_update.glsl")
    assert "buffer BrainBuffer" in src
    assert "float brain_params[]" in src
    assert "uniform int BRAIN_MODALITY;" in src
    assert "uniform int BRAIN_LEN;" in src
    assert "uniform ivec4 BRAIN_SHAPE;" in src


def test_black_box_dispatches_on_modality():
    src = read("shaders/entity_update.glsl")
    i = src.index("vec4 black_box")
    body = src[i:i + 600]
    assert "BRAIN_MODALITY == 0" in body
    assert "brain_fourier" in body


def test_fourier_glsl_has_the_contract_signature():
    src = read("shaders/brains/fourier.glsl")
    assert "vec4 brain_fourier(uint base, vec4 x)" in src


def test_fourier_glsl_is_pure():
    """It may read only brain_params, base, x and BRAIN_SHAPE. Touching entity
    state would break the Brain Inspector, which calls it from a fragment pass."""
    src = read("shaders/brains/fourier.glsl")
    for forbidden in ("entities[", "get_can(", "canvas", "e.pos", "e.vel"):
        assert forbidden not in src, f"brain function is not pure: {forbidden}"


def test_pack_brains_pads_each_brain_to_the_stride():
    layout = default_layout()
    a = np.arange(layout.length, dtype=np.float32)
    b = np.zeros(layout.length, dtype=np.float32)
    blob = pack_brains([a, b], layout)
    assert len(blob) == 2 * MAX_BRAIN_FLOATS * 4
    got = np.frombuffer(blob, dtype=np.float32)
    assert np.array_equal(got[:layout.length], a)
    assert np.all(got[layout.length:MAX_BRAIN_FLOATS] == 0)
    assert np.array_equal(got[MAX_BRAIN_FLOATS:MAX_BRAIN_FLOATS + layout.length], b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_shader_source.py -v`
Expected: FAIL — `ImportError: cannot import name 'pack_brains'` and missing shader file.

- [ ] **Step 3a: Create the Fourier GLSL**

```glsl
// shaders/brains/fourier.glsl
// Random-Fourier-features brain. Bit-exact with the original fourier_noise()
// at 10 centers: same basis, same index-derived phase offsets.
//
// PURE: reads only brain_params, base, x and BRAIN_SHAPE. The Brain Inspector
// calls this from a fragment pass, so it must not touch entity state.
vec4 brain_fourier(uint base, vec4 x) {
    vec4 result = vec4(0.0);
    int n = BRAIN_SHAPE.x;
    for (int i = 0; i < n; i++) {
        uint o = base + uint(i * 8);
        vec4 f = vec4(brain_params[o + 0u], brain_params[o + 1u],
                      brain_params[o + 2u], brain_params[o + 3u]);
        vec4 a = vec4(brain_params[o + 4u], brain_params[o + 5u],
                      brain_params[o + 6u], brain_params[o + 7u]);
        float phase = dot(x, f);
        float po = 2.0 * float(i) * 0.6283 + a.w * 3.14159;
        vec4 basis = vec4(
            sin(phase + po),
            cos(phase + po * 0.7),
            sin(phase * 2.0 + po * 1.3),
            cos(phase * 2.0 + po * 0.5)
        );
        result += a * basis;
    }
    return result;
}
```

- [ ] **Step 3b: Rewire entity_update.glsl**

Replace the `Rule` struct and `RuleBuffer` block (lines 13–21) with:

```glsl
#define MAX_BRAIN_FLOATS 512
layout(std430, binding = 4) buffer BrainBuffer {
    float brain_params[];
};
uniform int   BRAIN_MODALITY;   // which brain_* function to call
uniform int   BRAIN_LEN;        // active floats per brain; bounds every loop
uniform ivec4 BRAIN_SHAPE;      // structural ints (n_centers, hidden width, ...)
```

Replace `black_box` (lines 470–472) with:

```glsl
// Dispatch is a UNIFORM branch - every particle in the dispatch takes the same
// path, so there is no warp divergence and switching costs one uniform write.
vec4 black_box(vec2 L, vec2 R, uint base) {
    vec4 x = vec4(L, R);
    if (BRAIN_MODALITY == 0) return brain_fourier(base, x);
    if (BRAIN_MODALITY == 1) return brain_gabor(base, x);
    if (BRAIN_MODALITY == 2) return brain_lenia(base, x);
    return brain_mlp(base, x);
}
```

In `calculate_entity_behavior`, replace the `Rule rule` parameter with `uint base` and update the two calls:

```glsl
    vec4 baseterm   = black_box(L, R, base);
    vec4 mirrorterm = black_box(y_reflect(R), y_reflect(L), base);
```

In `main()`, replace the `current_rule` block (lines 529–548) with a base offset:

```glsl
    uint brain_slot = (TOURNAMENT_MODE == 1) ? uint(tournament_home_tile(index)) : 0u;
    uint base = brain_slot * uint(MAX_BRAIN_FLOATS);
```

and pass `base` to `calculate_entity_behavior`.

**Note:** this removes per-particle rule mutation and the `rules[]` readback
buffer. Task 5 restores both against the flat layout. Between these two tasks
`MUTATION_SCALE` has no effect and click-to-adopt is broken; do not ship the
intermediate state.

- [ ] **Step 3c: Add pack_brains and delete set_rule_uniform**

```python
# utilities/gl_helpers.py — replace set_rule_uniform
def pack_brains(params_list, layout) -> bytes:
    """Pack brains into the flat SSBO, each zero-padded to MAX_BRAIN_FLOATS.

    std430 gives a float array a 4-byte stride with no padding, so this is a
    straight memcpy - there is no struct alignment to get wrong.
    """
    import numpy as np

    from services.brains import MAX_BRAIN_FLOATS

    out = np.zeros((len(params_list), MAX_BRAIN_FLOATS), dtype=np.float32)
    for i, p in enumerate(params_list):
        flat = np.asarray(p, dtype=np.float32).reshape(-1)
        out[i, : layout.length] = flat[: layout.length]
    return out.tobytes()
```

- [ ] **Step 3d: Prepend the modality shaders in sim.py**

At `sim.py:140`, replace the single `fourier4_4.glsl` prepend with all brain
files. Order matters — brain functions must appear before `black_box` uses them,
and `shader_prepend` inserts after the `#version` line, so prepend in reverse.

```python
        self.entity_update_source = read_shader('shaders/entity_update.glsl')
        for _brain in ('mlp', 'lenia', 'gabor', 'fourier'):
            self.entity_update_source = shader_prepend(
                self.entity_update_source, read_shader(f'shaders/brains/{_brain}.glsl'))
        self.entity_update_source = shader_prepend(
            self.entity_update_source, read_shader('shaders/fourier4_4.glsl'))
        self.entity_update_source = prepend_defines(self.entity_update_source, self.entity_count)
```

**Note:** `fourier4_4.glsl` is still prepended — `hash()`, `pcg_hash()` and
`generate_random_centers()` are used elsewhere in `entity_update.glsl`. Only
`fourier_noise()` becomes dead.

- [ ] **Step 3e: Create stubs for the not-yet-written modalities**

`black_box` references `brain_gabor`, `brain_lenia` and `brain_mlp`, so the
shader will not compile without them. Create all three now; Tasks 6–8 replace
each in turn.

```glsl
// shaders/brains/gabor.glsl  (stub, replaced in Task 6)
vec4 brain_gabor(uint base, vec4 x) { return vec4(0.0); }
```

```glsl
// shaders/brains/lenia.glsl  (stub, replaced in Task 7)
vec4 brain_lenia(uint base, vec4 x) { return vec4(0.0); }
```

```glsl
// shaders/brains/mlp.glsl  (stub, replaced in Task 8)
vec4 brain_mlp(uint base, vec4 x) { return vec4(0.0); }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_shader_source.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Verify the app still runs and Fourier is unchanged**

Run the app, load a preset, confirm particles move exactly as before.
Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: exit code 0.

- [ ] **Step 6: Commit**

```bash
git add shaders/brains/ shaders/entity_update.glsl sim.py utilities/gl_helpers.py tests/test_brain_shader_source.py
git commit -m "feat: dispatch brains from a flat SSBO on a uniform branch"
```

---

### Task 5: Per-particle buffer sized to BRAIN_LEN

**Files:**
- Modify: `sim.py` (rule buffer allocation), `shaders/entity_update.glsl` (`mutate_brain`, `WRITE_RULES`)
- Test: `tests/test_brain_shader_source.py` (extend)

**Interfaces:**
- Consumes: `pack_brains`, `BrainLayout`.
- Produces: `Sim.realloc_brain_buffers(layout)`.

This restores per-particle mutation and click-to-adopt, which Task 4 removed.

**The trap:** the per-particle buffer must be `BRAIN_LEN` floats per particle,
never `MAX_BRAIN_FLOATS`. At 320 bytes it is already 192 MB; at 512 floats it
would be ~1 KB per particle, about 600 MB.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brain_shader_source.py — append
def test_per_particle_buffer_is_sized_by_brain_len_not_max():
    """MAX_BRAIN_FLOATS per particle would be ~600 MB. Must use BRAIN_LEN."""
    src = read("sim.py")
    i = src.index("def realloc_brain_buffers")
    body = src[i:i + 900]
    assert "BRAIN_LEN" in body or "layout.length" in body
    assert "MAX_BRAIN_FLOATS" not in body.split("multi_load")[0], (
        "per-particle buffer must not use the max stride"
    )


def test_mutate_brain_walks_the_active_length():
    src = read("shaders/entity_update.glsl")
    i = src.index("void mutate_brain")
    body = src[i:i + 400]
    assert "BRAIN_LEN" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_shader_source.py -v`
Expected: FAIL with `ValueError: substring not found`

- [ ] **Step 3a: Add the generic mutation to entity_update.glsl**

```glsl
// Per-particle mutation, applied to a copy of the tile's brain. Deterministic
// per cohort, so a cohort's variant is stable frame to frame.
//
// Generic over modality: it jitters every active float additively. The old
// mutate_rule() split additive (amplitude) from multiplicative (frequency),
// which cannot be expressed without knowing the layout. Additive-only is the
// honest generic choice; a modality that needs otherwise can scale its own
// decode ranges to compensate.
void mutate_brain(inout float p[MAX_BRAIN_FLOATS], float amount, float cohort) {
    if (amount == 0.0) return;
    for (int i = 0; i < BRAIN_LEN; i++) {
        p[i] += amount * (hash(vec2(cohort, float(i))) - 0.5);
    }
}
```

- [ ] **Step 3b: Add the reallocation to sim.py**

```python
    def realloc_brain_buffers(self, layout) -> None:
        """Resize the brain buffers for a new layout.

        The per-particle buffer is BRAIN_LEN floats per particle, NOT
        MAX_BRAIN_FLOATS: at the max stride it would be ~1 KB per particle,
        about 600 MB. Called on every layout change, which already resets the
        optimizer and switches archive, so the cost is invisible.
        """
        self._brain_layout = layout
        per_particle = self.entity_count * layout.length * 4
        if self.brain_readback_buffer is not None:
            self.brain_readback_buffer.release()
        self.brain_readback_buffer = self.ctx.buffer(reserve=per_particle)
        self.brain_readback_buffer.bind_to_storage_buffer(2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_shader_source.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Verify manually**

Run the app. Confirm: particles move, Mutation Scale still varies cohorts within
a tile, and clicking a particle adopts its brain.

- [ ] **Step 6: Commit**

```bash
git add sim.py shaders/entity_update.glsl tests/test_brain_shader_source.py
git commit -m "feat: size the per-particle brain buffer to the active length"
```

---

## Phase 3 — The three new modalities

Each task follows the same shape. The GLSL and the Python decode must agree on
parameter order, and the round-trip test is what enforces it.

### Task 6: Gabor modality

**Files:**
- Create: `services/brains/gabor.py`
- Replace stub: `shaders/brains/gabor.glsl`
- Test: `tests/test_brain_modalities.py`

**Interfaces:**
- Consumes: `BrainLayout`, `register`, `Setting`.
- Produces: `GaborModality`, `name="gabor"`, `modality_id=1`, 14 floats per filter ordered `center(4), freq(4), amp(4), sigma, phase`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brain_modalities.py
"""Round-trip and range tests for every modality.

The round-trip is what enforces agreement between a modality's Python decode
and its GLSL parameter order - they are two hand-written halves of one layout.
"""
import numpy as np
import pytest

from services.brains import REGISTRY, MAX_BRAIN_FLOATS


def _all_modalities():
    import services.brains.fourier  # noqa: F401
    import services.brains.gabor    # noqa: F401
    import services.brains.lenia    # noqa: F401
    import services.brains.mlp      # noqa: F401
    return sorted(REGISTRY.values(), key=lambda m: m.modality_id)


@pytest.mark.parametrize("m", _all_modalities(), ids=lambda m: m.name)
def test_decode_length_matches_layout(m):
    layout = m.layout_from_settings({})
    rng = np.random.default_rng(0)
    z = rng.normal(0, 0.5, layout.length).astype(np.float32)
    assert m.decode(z, layout).shape == (layout.length,)


@pytest.mark.parametrize("m", _all_modalities(), ids=lambda m: m.name)
def test_encode_decode_round_trips(m):
    layout = m.layout_from_settings({})
    rng = np.random.default_rng(1)
    z = rng.normal(0, 0.4, layout.length).astype(np.float32)
    params = m.decode(z, layout)
    back, clamped = m.encode(params, layout)
    assert clamped == 0
    assert np.allclose(back, z, atol=1e-3)


@pytest.mark.parametrize("m", _all_modalities(), ids=lambda m: m.name)
def test_every_reachable_setting_fits_the_budget(m):
    for s in m.settings_schema():
        if s.kind != "int":
            continue
        layout = m.layout_from_settings({s.key: int(s.hi)})
        assert layout.length <= MAX_BRAIN_FLOATS, (
            f"{m.name} at {s.key}={int(s.hi)} needs {layout.length}"
        )


@pytest.mark.parametrize("m", _all_modalities(), ids=lambda m: m.name)
def test_decode_is_finite_for_extreme_z(m):
    layout = m.layout_from_settings({})
    for scale in (5.0, 50.0):
        z = np.full(layout.length, scale, dtype=np.float32)
        assert np.all(np.isfinite(m.decode(z, layout)))


def test_gabor_sigma_is_strictly_positive():
    """A zero or negative envelope width divides by zero on the GPU."""
    from services.brains.gabor import GaborModality
    m = GaborModality()
    layout = m.layout_from_settings({})
    for scale in (-50.0, 0.0, 50.0):
        p = m.decode(np.full(layout.length, scale, np.float32), layout)
        sigmas = p.reshape(layout.shape[0], 14)[:, 12]
        assert np.all(sigmas > 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_modalities.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.brains.gabor'`

- [ ] **Step 3a: Write the Python side**

```python
# services/brains/gabor.py
"""Gabor brain: a Gaussian envelope wrapped around the oscillation.

    g_i(x) = exp(-||x - c_i||^2 / 2 sigma_i^2) * cos(dot(x, f_i) + phi_i)
    out    = sum_i a_i * g_i(x)

The envelope is the point. A unit responds near its centre and is silent
elsewhere, so a brain is a set of localised 'when I see roughly this, do that'
rules rather than one global interference pattern. It also damps the drift
measured on the Fourier runs: a high frequency inside a narrow envelope makes a
locally intricate response, not global chaos.

Layout, 14 floats per filter: centre(4), frequency(4), amplitude(4), sigma, phase.
"""
from __future__ import annotations

import numpy as np

from services.brains import BrainLayout, Setting, register

FLOATS_PER_FILTER = 14
CENTER_SCALE = 2.0
FREQ_SCALE = 3.0
AMP_SCALE = 1.0
SIGMA_MIN = 0.15
SIGMA_MAX = 2.0
EPS = 1e-4


def _squash(z, scale):
    return scale * np.tanh(z)


class GaborModality:
    name = "gabor"
    modality_id = 1
    glsl_file = "shaders/brains/gabor.glsl"

    def settings_schema(self) -> list[Setting]:
        return [
            Setting("filters", "Filters", "int", 4, 36, 12),
            Setting("freq_scale", "Freq Scale", "float", 0.5, 6.0, FREQ_SCALE),
            Setting("envelope_width", "Envelope Width", "float", 0.2, 2.0, 1.0),
            Setting("phase_spread", "Phase Spread", "float", 0.0, 3.1416, 3.1416),
        ]

    def layout_from_settings(self, s: dict) -> BrainLayout:
        n = int(s.get("filters", 12))
        return BrainLayout("gabor", (n,), FLOATS_PER_FILTER * n)

    def decode(self, z: np.ndarray, layout: BrainLayout) -> np.ndarray:
        n = layout.shape[0]
        z = np.asarray(z, dtype=np.float32).reshape(n, FLOATS_PER_FILTER)
        out = np.empty_like(z)
        out[:, 0:4] = _squash(z[:, 0:4], CENTER_SCALE)
        out[:, 4:8] = _squash(z[:, 4:8], FREQ_SCALE)
        out[:, 8:12] = _squash(z[:, 8:12], AMP_SCALE)
        # Strictly positive and bounded: a zero width divides by zero on the GPU.
        half = 0.5 * (SIGMA_MAX - SIGMA_MIN)
        out[:, 12] = SIGMA_MIN + half * (1.0 + np.tanh(z[:, 12]))
        out[:, 13] = _squash(z[:, 13], np.pi)
        return out.reshape(-1).astype(np.float32)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        n = layout.shape[0]
        p = np.asarray(params, dtype=np.float32).reshape(n, FLOATS_PER_FILTER)
        raw = np.empty_like(p)
        raw[:, 0:4] = p[:, 0:4] / CENTER_SCALE
        raw[:, 4:8] = p[:, 4:8] / FREQ_SCALE
        raw[:, 8:12] = p[:, 8:12] / AMP_SCALE
        half = 0.5 * (SIGMA_MAX - SIGMA_MIN)
        raw[:, 12] = (p[:, 12] - SIGMA_MIN) / half - 1.0
        raw[:, 13] = p[:, 13] / np.pi
        n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
        z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
        return z.reshape(-1).astype(np.float32), n_clamped

    def random(self, rng, layout: BrainLayout) -> np.ndarray:
        """No hand-tuned prior in this iteration (see the spec). A plain
        Gaussian in z is decoded through the same squash the search uses."""
        z = rng.normal(0.0, 0.5, layout.length).astype(np.float32)
        return self.decode(z, layout)


register(GaborModality())
```

Append the import to `services/brains/__init__.py`:

```python
from services.brains import gabor as _gabor  # noqa: E402,F401
```

- [ ] **Step 3b: Write the GLSL**

```glsl
// shaders/brains/gabor.glsl
// Gaussian envelope around an oscillation. Localised: silent away from centre.
// Layout, 14 floats: centre(4), frequency(4), amplitude(4), sigma, phase.
//
// PURE: reads only brain_params, base, x and BRAIN_SHAPE.
vec4 brain_gabor(uint base, vec4 x) {
    vec4 result = vec4(0.0);
    int n = BRAIN_SHAPE.x;
    for (int i = 0; i < n; i++) {
        uint o = base + uint(i * 14);
        vec4 c = vec4(brain_params[o + 0u], brain_params[o + 1u],
                      brain_params[o + 2u], brain_params[o + 3u]);
        vec4 f = vec4(brain_params[o + 4u], brain_params[o + 5u],
                      brain_params[o + 6u], brain_params[o + 7u]);
        vec4 a = vec4(brain_params[o + 8u], brain_params[o + 9u],
                      brain_params[o + 10u], brain_params[o + 11u]);
        float sg = brain_params[o + 12u];
        float ph = brain_params[o + 13u];
        vec4 d = x - c;
        float env = exp(-dot(d, d) / (2.0 * sg * sg));
        result += a * (env * cos(dot(x, f) + ph));
    }
    return result;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_modalities.py -v -k "gabor or fourier"`
Expected: PASS for gabor and fourier; lenia and mlp collection errors remain until Tasks 7–8.

- [ ] **Step 5: Commit**

```bash
git add services/brains/gabor.py services/brains/__init__.py shaders/brains/gabor.glsl tests/test_brain_modalities.py
git commit -m "feat: add the Gabor brain modality"
```

---

### Task 7: Lenia bump modality

**Files:**
- Create: `services/brains/lenia.py`
- Replace stub: `shaders/brains/lenia.glsl`
- Test: `tests/test_brain_modalities.py` (extend)

**Interfaces:**
- Consumes: `BrainLayout`, `register`, `Setting`.
- Produces: `LeniaModality`, `name="lenia"`, `modality_id=2`, 10 floats per bump ordered `w(4), amp(4), mu, sigma`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brain_modalities.py — append
def test_lenia_growth_is_negative_far_from_the_band():
    """The -1 offset is what makes Lenia Lenia: positive inside a narrow band
    of sensor values, negative everywhere else."""
    import numpy as np
    from services.brains.lenia import LeniaModality, growth
    m = LeniaModality()
    layout = m.layout_from_settings({})
    assert growth(np.array([0.0]), 0.0, 0.1)[0] == pytest.approx(1.0)
    assert growth(np.array([10.0]), 0.0, 0.1)[0] == pytest.approx(-1.0)


def test_lenia_sigma_is_strictly_positive():
    import numpy as np
    from services.brains.lenia import LeniaModality
    m = LeniaModality()
    layout = m.layout_from_settings({})
    for scale in (-50.0, 0.0, 50.0):
        p = m.decode(np.full(layout.length, scale, np.float32), layout)
        assert np.all(p.reshape(layout.shape[0], 10)[:, 9] > 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_modalities.py -v -k lenia`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.brains.lenia'`

- [ ] **Step 3a: Write the Python side**

```python
# services/brains/lenia.py
"""Lenia bumps: the growth function, a Gaussian band rather than an oscillation.

    u_i = dot(x, w_i)
    G_i = 2 * exp(-(u_i - mu_i)^2 / 2 sigma_i^2) - 1
    out = sum_i a_i * G_i

The -1 is essential. Response is positive inside a narrow band of sensor values
and negative everywhere else - the 'thrive at this density, die away from it'
rule that gives Lenia its membranes. Non-oscillatory and smooth; expected to be
the best-conditioned landscape of the four.

On priors: canonical Lenia parameters (orbium and friends) are tuned for a
continuous CA whose input is a kernel-weighted neighbourhood sum over a grid.
Here the input is two sensor taps scaled by sqrt(ws)*38.855*SENSOR_GAIN. Those
numbers DO NOT transfer and must not be copied in. What transfers is
scale-free: Lenia growth bands are consistently narrow, roughly sigma/mu ~ 0.1,
which is what SIGMA_MAX encodes relative to MU_SCALE.

Layout, 10 floats per bump: projection(4), amplitude(4), mu, sigma.
"""
from __future__ import annotations

import numpy as np

from services.brains import BrainLayout, Setting, register

FLOATS_PER_BUMP = 10
W_SCALE = 3.0
AMP_SCALE = 1.0
MU_SCALE = 2.0
SIGMA_MIN = 0.02
SIGMA_MAX = 0.6
EPS = 1e-4


def growth(u, mu, sigma):
    """Lenia's growth mapping, vectorised. Peaks at +1 on the band, -1 off it."""
    return 2.0 * np.exp(-((u - mu) ** 2) / (2.0 * sigma ** 2)) - 1.0


class LeniaModality:
    name = "lenia"
    modality_id = 2
    glsl_file = "shaders/brains/lenia.glsl"

    def settings_schema(self) -> list[Setting]:
        return [
            Setting("bumps", "Bumps", "int", 4, 48, 12),
            Setting("mu_scale", "Band Center", "float", 0.5, 4.0, MU_SCALE),
            Setting("sigma_max", "Band Width", "float", 0.05, 1.0, SIGMA_MAX),
        ]

    def layout_from_settings(self, s: dict) -> BrainLayout:
        n = int(s.get("bumps", 12))
        return BrainLayout("lenia", (n,), FLOATS_PER_BUMP * n)

    def decode(self, z: np.ndarray, layout: BrainLayout) -> np.ndarray:
        n = layout.shape[0]
        z = np.asarray(z, dtype=np.float32).reshape(n, FLOATS_PER_BUMP)
        out = np.empty_like(z)
        out[:, 0:4] = W_SCALE * np.tanh(z[:, 0:4])
        out[:, 4:8] = AMP_SCALE * np.tanh(z[:, 4:8])
        out[:, 8] = MU_SCALE * np.tanh(z[:, 8])
        half = 0.5 * (SIGMA_MAX - SIGMA_MIN)
        out[:, 9] = SIGMA_MIN + half * (1.0 + np.tanh(z[:, 9]))
        return out.reshape(-1).astype(np.float32)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        n = layout.shape[0]
        p = np.asarray(params, dtype=np.float32).reshape(n, FLOATS_PER_BUMP)
        raw = np.empty_like(p)
        raw[:, 0:4] = p[:, 0:4] / W_SCALE
        raw[:, 4:8] = p[:, 4:8] / AMP_SCALE
        raw[:, 8] = p[:, 8] / MU_SCALE
        half = 0.5 * (SIGMA_MAX - SIGMA_MIN)
        raw[:, 9] = (p[:, 9] - SIGMA_MIN) / half - 1.0
        n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
        z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
        return z.reshape(-1).astype(np.float32), n_clamped

    def random(self, rng, layout: BrainLayout) -> np.ndarray:
        z = rng.normal(0.0, 0.5, layout.length).astype(np.float32)
        return self.decode(z, layout)


register(LeniaModality())
```

Append to `services/brains/__init__.py`:

```python
from services.brains import lenia as _lenia  # noqa: E402,F401
```

- [ ] **Step 3b: Write the GLSL**

```glsl
// shaders/brains/lenia.glsl
// Lenia growth bands. The -1 gives negative growth away from the band, which
// is what produces membranes rather than a smooth blur.
// Layout, 10 floats: projection(4), amplitude(4), mu, sigma.
//
// PURE: reads only brain_params, base, x and BRAIN_SHAPE.
vec4 brain_lenia(uint base, vec4 x) {
    vec4 result = vec4(0.0);
    int n = BRAIN_SHAPE.x;
    for (int i = 0; i < n; i++) {
        uint o = base + uint(i * 10);
        vec4 w = vec4(brain_params[o + 0u], brain_params[o + 1u],
                      brain_params[o + 2u], brain_params[o + 3u]);
        vec4 a = vec4(brain_params[o + 4u], brain_params[o + 5u],
                      brain_params[o + 6u], brain_params[o + 7u]);
        float mu = brain_params[o + 8u];
        float sg = brain_params[o + 9u];
        float u = dot(x, w) - mu;
        float g = 2.0 * exp(-(u * u) / (2.0 * sg * sg)) - 1.0;
        result += a * g;
    }
    return result;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_modalities.py -v -k "lenia or gabor or fourier"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/brains/lenia.py services/brains/__init__.py shaders/brains/lenia.glsl tests/test_brain_modalities.py
git commit -m "feat: add the Lenia bump brain modality"
```

---

### Task 8: MLP modality

**Files:**
- Create: `services/brains/mlp.py`
- Replace stub: `shaders/brains/mlp.glsl`
- Test: `tests/test_brain_modalities.py` (extend)

**Interfaces:**
- Consumes: `BrainLayout`, `register`, `Setting`.
- Produces: `MLPModality`, `name="mlp"`, `modality_id=3`. Layout `shape = (hidden, act_index)`, length `9H + 4`. Flat order: `W1 (H rows of 4), b1 (H), W2 (4 rows of H), b2 (4)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brain_modalities.py — append
def test_mlp_length_is_nine_h_plus_four():
    from services.brains.mlp import MLPModality
    m = MLPModality()
    for h in (4, 16, 48):
        assert m.layout_from_settings({"hidden": h}).length == 9 * h + 4


def test_mlp_signature_includes_activation():
    from services.brains.mlp import MLPModality
    m = MLPModality()
    a = m.layout_from_settings({"hidden": 16, "activation": 0})
    b = m.layout_from_settings({"hidden": 16, "activation": 1})
    assert a.signature() != b.signature(), (
        "two activations are different brains and must not share an archive"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_modalities.py -v -k mlp`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.brains.mlp'`

- [ ] **Step 3a: Write the Python side**

```python
# services/brains/mlp.py
"""A one-hidden-layer MLP: out = W2 . act(W1 x + b1) + b2.

Activation is part of the layout, not a free parameter - tanh and sin are
different function families, and a genome evolved under one means nothing under
the other. That is why activation appears in the signature.

Layout, 9H + 4 floats: W1 (H rows of 4), b1 (H), W2 (4 rows of H), b2 (4).
"""
from __future__ import annotations

import numpy as np

from services.brains import BrainLayout, Setting, register

ACTIVATIONS = ("tanh", "sin", "gelu")
W_SCALE = 2.0
B_SCALE = 1.0
EPS = 1e-4


class MLPModality:
    name = "mlp"
    modality_id = 3
    glsl_file = "shaders/brains/mlp.glsl"

    def settings_schema(self) -> list[Setting]:
        return [
            Setting("hidden", "Hidden Width", "int", 4, 48, 16),
            Setting("activation", "Activation", "choice", 0, 2, 0,
                    choices=ACTIVATIONS),
        ]

    def layout_from_settings(self, s: dict) -> BrainLayout:
        h = int(s.get("hidden", 16))
        act = int(s.get("activation", 0))
        return BrainLayout("mlp", (h, act), 9 * h + 4)

    def decode(self, z: np.ndarray, layout: BrainLayout) -> np.ndarray:
        h = layout.shape[0]
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        out = np.empty_like(z)
        out[: 4 * h] = W_SCALE * np.tanh(z[: 4 * h])              # W1
        out[4 * h: 5 * h] = B_SCALE * np.tanh(z[4 * h: 5 * h])    # b1
        out[5 * h: 9 * h] = W_SCALE * np.tanh(z[5 * h: 9 * h])    # W2
        out[9 * h:] = B_SCALE * np.tanh(z[9 * h:])                # b2
        return out.astype(np.float32)

    def encode(self, params: np.ndarray, layout: BrainLayout):
        h = layout.shape[0]
        p = np.asarray(params, dtype=np.float32).reshape(-1)
        raw = np.empty_like(p)
        raw[: 4 * h] = p[: 4 * h] / W_SCALE
        raw[4 * h: 5 * h] = p[4 * h: 5 * h] / B_SCALE
        raw[5 * h: 9 * h] = p[5 * h: 9 * h] / W_SCALE
        raw[9 * h:] = p[9 * h:] / B_SCALE
        n_clamped = int(np.count_nonzero(np.abs(raw) >= 1.0 - EPS))
        z = np.arctanh(np.clip(raw, -1.0 + EPS, 1.0 - EPS))
        return z.astype(np.float32), n_clamped

    def random(self, rng, layout: BrainLayout) -> np.ndarray:
        z = rng.normal(0.0, 0.5, layout.length).astype(np.float32)
        return self.decode(z, layout)


register(MLPModality())
```

Append to `services/brains/__init__.py`:

```python
from services.brains import mlp as _mlp  # noqa: E402,F401
```

- [ ] **Step 3b: Write the GLSL**

```glsl
// shaders/brains/mlp.glsl
// out = W2 . act(W1 x + b1) + b2
// Layout: W1 (H rows of 4), b1 (H), W2 (4 rows of H), b2 (4).
// BRAIN_SHAPE.x = H, BRAIN_SHAPE.y = activation index (0 tanh, 1 sin, 2 gelu).
//
// The hidden vector is a fixed-size local array. At H=48 that is 192 bytes per
// invocation, which spills to local memory - measurably slower than the other
// modalities but correct, and the alternative (recomputing the hidden layer
// once per output channel) costs 4x the dot products.
//
// PURE: reads only brain_params, base, x and BRAIN_SHAPE.
#define MAX_HIDDEN 48

float brain_mlp_act(float s, int kind) {
    if (kind == 0) return tanh(s);
    if (kind == 1) return sin(s);
    return 0.5 * s * (1.0 + tanh(0.7978845608 * (s + 0.044715 * s * s * s)));
}

vec4 brain_mlp(uint base, vec4 x) {
    int H = min(BRAIN_SHAPE.x, MAX_HIDDEN);
    int kind = BRAIN_SHAPE.y;
    float h[MAX_HIDDEN];

    for (int j = 0; j < H; j++) {
        uint o = base + uint(j * 4);
        vec4 w = vec4(brain_params[o + 0u], brain_params[o + 1u],
                      brain_params[o + 2u], brain_params[o + 3u]);
        float s = dot(x, w) + brain_params[base + uint(4 * H + j)];
        h[j] = brain_mlp_act(s, kind);
    }

    uint w2 = base + uint(5 * H);
    uint b2 = base + uint(9 * H);
    vec4 result = vec4(0.0);
    for (int k = 0; k < 4; k++) {
        float s = brain_params[b2 + uint(k)];
        for (int j = 0; j < H; j++) {
            s += brain_params[w2 + uint(k * H + j)] * h[j];
        }
        result[k] = s;
    }
    return result;
}
```

- [ ] **Step 4: Run the full modality suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_modalities.py -v`
Expected: PASS, all four modalities parametrized.

- [ ] **Step 5: Verify each modality runs on the GPU**

Run the app. Switch through all four modalities (the UI arrives in Task 11; for
now set the uniform from a scratch script or temporarily hardcode
`BRAIN_MODALITY`). Confirm each compiles and particles move.

- [ ] **Step 6: Commit**

```bash
git add services/brains/mlp.py services/brains/__init__.py shaders/brains/mlp.glsl tests/test_brain_modalities.py
git commit -m "feat: add the MLP brain modality"
```

---

## Phase 4 — Search and archive integration

### Task 9: Archive stores flat brains keyed by signature

**Files:**
- Modify: `services/archive.py:123` and `load_from_store`
- Modify: `services/archive_io.py`
- Modify: `main.py:240`
- Test: `tests/test_archive_signature.py`

**Interfaces:**
- Consumes: `BrainLayout`.
- Produces: `Archive(store, layout=...)` with `_brain` shaped `(capacity, layout.length)`; `ArchiveStore(root, layout)` writing `archive/<signature>/`; `migrate_legacy_archive(root) -> Path | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_archive_signature.py
import json

import numpy as np
import pytest

from services.archive import Archive
from services.archive_io import ArchiveStore, migrate_legacy_archive
from services.brains import BrainLayout, default_layout


def test_store_root_is_the_layout_signature(tmp_path):
    layout = default_layout()
    store = ArchiveStore(tmp_path, layout)
    assert store.root.name == "fourier-n10"


def test_two_layouts_get_separate_directories(tmp_path):
    a = ArchiveStore(tmp_path, BrainLayout("gabor", (12,), 168))
    b = ArchiveStore(tmp_path, BrainLayout("gabor", (8,), 112))
    assert a.root != b.root


def test_archive_brain_width_follows_the_layout(tmp_path):
    layout = BrainLayout("gabor", (12,), 168)
    arc = Archive(store=ArchiveStore(tmp_path, layout), layout=layout)
    assert arc._brain.shape[1] == 168


def test_legacy_archive_is_moved_into_the_fourier_signature(tmp_path):
    """`migrate_legacy_archive` takes the ARCHIVE directory - the one holding
    index.jsonl - and moves its contents down into <dir>/fourier-n10/."""
    legacy = tmp_path / "archive"
    legacy.mkdir()
    (legacy / "index.jsonl").write_text('{"id": 0}\n')
    np.savez(legacy / "vectors.npz",
             ids=np.array([0]),
             embeddings=np.zeros((1, 512), np.float32),
             brains=np.zeros((1, 10, 8), np.float32),
             physics=np.zeros((1, 8), np.float32))

    moved = migrate_legacy_archive(legacy)

    assert moved == legacy / "fourier-n10"
    assert moved.is_dir()
    assert (moved / "index.jsonl").read_text() == '{"id": 0}\n'
    assert (moved / "vectors.npz").is_file()
    # The old files must not be left behind at the top level.
    assert not (legacy / "index.jsonl").exists()


def test_migration_is_idempotent(tmp_path):
    """Called on an already-migrated or empty directory, it does nothing."""
    empty = tmp_path / "archive"
    empty.mkdir()
    assert migrate_legacy_archive(empty) is None
    assert migrate_legacy_archive(empty) is None


def test_legacy_brains_load_as_flat_vectors(tmp_path):
    """Stored (N, 10, 8) must read back as 80-float rows without reshaping
    every call site."""
    layout = default_layout()
    store = ArchiveStore(tmp_path, layout)
    brains = np.arange(80, dtype=np.float32).reshape(1, 10, 8)
    store.flush_vectors(np.array([0]), np.zeros((1, 512), np.float32),
                        brains, np.zeros((1, 8), np.float32))
    arrays = store.load_vectors()
    assert arrays["brains"].reshape(1, -1).shape == (1, 80)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_signature.py -v`
Expected: FAIL with `ImportError: cannot import name 'migrate_legacy_archive'`

- [ ] **Step 3a: Add signature routing and migration to archive_io.py**

```python
# services/archive_io.py — modify __init__ and add migrate_legacy_archive
class ArchiveStore:
    def __init__(self, root, layout=None):
        """`root` is the archives parent directory; the store lives in a
        subdirectory named for the layout signature, so two layouts can never
        share entries whose floats mean different things."""
        from services.brains import default_layout

        layout = layout or default_layout()
        self.layout = layout
        self.root = Path(root) / layout.signature()
        self.enabled = True
        self._fh = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "thumbs").mkdir(exist_ok=True)
            self._fh = open(self.index_path, "a", encoding="utf-8")
        except OSError as exc:
            self.enabled = False
            print(f"[Archive] persistence disabled ({exc}); the run continues")


def migrate_legacy_archive(archive_dir) -> Path | None:
    """Move a pre-modality archive down into `<archive_dir>/fourier-n10/`.

    `archive_dir` is the directory holding `index.jsonl` — the same path
    `ArchiveStore` used to be given directly. Entry contents are untouched;
    only the path changes, and the stored (N, 10, 8) brains are read back as
    flat 80-float rows. Returns the new path, or None if there was nothing to
    migrate (already migrated, or empty).
    """
    archive_dir = Path(archive_dir)
    if not (archive_dir / "index.jsonl").is_file():
        return None
    target = archive_dir / "fourier-n10"
    if target.exists():
        return None

    # Move into a sibling staging dir first, then rename it into place. Moving
    # items one at a time into a subdirectory of the directory being iterated
    # is the kind of thing that half-completes on an error and leaves an
    # archive that loads as empty.
    staging = archive_dir.parent / f".{archive_dir.name}-migrating"
    if staging.exists():
        raise OSError(f"stale migration staging dir at {staging}; "
                      "move it aside before starting the app")
    staging.mkdir(parents=True)
    for item in list(archive_dir.iterdir()):
        item.rename(staging / item.name)
    staging.rename(target)
    return target
```

- [ ] **Step 3b: Make the Archive brain width follow the layout**

```python
# services/archive.py — __init__ signature and line 123
    def __init__(self, store=None, capacity: int = 20000, k: int = 10,
                 liveness_min: float = 0.002, target_rate: float = 0.15,
                 seed_n: int = 256, dim: int = 512, layout=None):
        from services.brains import default_layout

        self.layout = layout or default_layout()
        ...
        # Flat, because brain width is now a per-layout property. An archive
        # only ever holds one layout - the directory is keyed by its signature.
        self._brain = np.zeros((0, self.layout.length), dtype=np.float32)
```

In `load_from_store`, reshape legacy `(N, 10, 8)` to `(N, 80)`:

```python
        brains = np.asarray(arrays["brains"], dtype=np.float32)
        if brains.ndim == 3:            # legacy (N, 10, 8)
            brains = brains.reshape(len(brains), -1)
```

- [ ] **Step 3c: Wire main.py**

Add the layout to `App.__init__`, next to `self.archive = None` (~`main.py:106`):

```python
        from services.brains import default_layout
        self.brain_layout = default_layout()
```

Then in `_build_archive_set`, migrate before constructing, and pass the layout
through (`main.py:235-241`):

```python
        from services.archive_io import ArchiveStore, migrate_legacy_archive

        migrate_legacy_archive(path)
        store = ArchiveStore(path, self.brain_layout)
        archive = Archive(store=store, layout=self.brain_layout)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_signature.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: exit code 0. `tests/test_archive_projection.py`, `test_archive_entry_actions.py`
and `test_archive_window_render.py` all construct archives — if any fail, they
were relying on the `(N, 10, 8)` shape and need the flat form.

- [ ] **Step 6: Verify migration on real data**

Back up `archive/` first. Launch the app, confirm the directory became
`archive/fourier-n10/` and the entry count in the Archive window is unchanged.

```bash
cp -r archive archive-backup
```

- [ ] **Step 7: Commit**

```bash
git add services/archive.py services/archive_io.py main.py tests/test_archive_signature.py
git commit -m "feat: key archives by brain layout signature with legacy migration"
```

---

### Task 10: Optimizer and checkpoint layout awareness

**Files:**
- Modify: `services/optimizers.py:300-301`
- Modify: `services/prompt_driver.py`, `services/imgep_driver.py` (checkpoint signature)
- Test: `tests/test_brain_layout.py` (extend)

**Interfaces:**
- Consumes: `BrainLayout`, `spec_for`.
- Produces: `GAOptimizer(dim, popsize, sigma0, seed, x0, layout=None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brain_layout.py — append
def test_ga_optimizer_handles_a_non_multiple_of_eight():
    """MLP at H=16 is 148 floats, which is not divisible by 8. The old
    reshape(-1, 8) would raise."""
    import numpy as np
    from services.brains.mlp import MLPModality
    from services.optimizers import GAOptimizer

    layout = MLPModality().layout_from_settings({"hidden": 16})
    assert layout.length % 8 != 0
    opt = GAOptimizer(layout.length, 8, 0.5, 0, None, layout=layout)
    z = opt.ask(8)
    opt.tell(z, np.arange(8, dtype=np.float32))
    assert opt.ask(8).shape == (8, layout.length)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_layout.py -v -k ga_optimizer`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'layout'`, or `ValueError: cannot reshape`.

- [ ] **Step 3: Fix the reshape**

`GAOptimizer.tell` reshapes to `(-1, 8)` because `services.genome.crossover`
and `mutate` expect the Fourier `(N, 8)` layout. For other modalities, operate
on the flat vector.

```python
# services/optimizers.py — GAOptimizer.__init__ and tell
    def __init__(self, dim, popsize, sigma0=0.5, seed=0, x0=None, layout=None):
        super().__init__(dim, popsize, sigma0, seed, x0)
        self._layout = layout
        self._rng = np.random.default_rng(seed)
        self._pop = self._fresh(popsize)
        self._told = False

    def _breed(self, a, b):
        """Fourier keeps per-center crossover, which is meaningful because a
        center is a unit. For other layouts a row is not a unit, so uniform
        per-gene crossover is the honest generic operator."""
        if self._layout is None or self._layout.modality == "fourier":
            child = crossover(a.reshape(-1, 8), b.reshape(-1, 8), self._rng)
            return mutate(child, self.MUT, self._rng).reshape(-1)
        mask = self._rng.random(a.shape) < 0.5
        child = np.where(mask, a, b)
        jitter = self.MUT * self._rng.normal(0, 1, a.shape)
        return (child + jitter).astype(np.float32)
```

and in `tell`, replace the crossover/mutate block with `self._breed(a, b)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_layout.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: exit code 0.

- [ ] **Step 6: Commit**

```bash
git add services/optimizers.py tests/test_brain_layout.py
git commit -m "feat: make the GA operator layout-aware"
```

---

## Phase 5 — UI and inspector

### Task 11: Brain window

**Files:**
- Create: `ui/brain_window.py`
- Modify: `ui/core.py:19-55`, `state/ui_state.py`
- Test: `tests/test_brain_window_render.py`

**Interfaces:**
- Consumes: `REGISTRY`, `BrainLayout`, `Setting`.
- Produces: `BrainWindowMixin.render_brain_window()`; UI state `_brain_modality: str`, `_brain_settings: dict`, `_request_brain_layout_change: bool`.

Follow the pattern in `ui/tournament_window.py`; the render test follows
`tests/test_tournament_window_render.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brain_window_render.py
"""Follows tests/test_tournament_window_render.py - drives the mixin with a
stub imgui to assert on structure, not pixels."""
from services.brains import REGISTRY


def test_every_modality_is_offered():
    import services.brains.fourier, services.brains.gabor  # noqa: F401
    import services.brains.lenia, services.brains.mlp      # noqa: F401
    assert {"fourier", "gabor", "lenia", "mlp"} <= set(REGISTRY)


def test_changing_a_count_setting_flags_a_layout_change():
    from ui.brain_window import layout_change_needed
    assert layout_change_needed({"centers": 10}, {"centers": 16}) is True
    assert layout_change_needed({"centers": 10}, {"freq_scale": 2.0}) is False


def test_saturation_fraction_counts_params_near_the_rails():
    import numpy as np
    from ui.brain_window import saturation_fraction
    z = np.array([0.0, 0.1, 5.0, -5.0], dtype=np.float32)
    assert saturation_fraction(z) == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_window_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui.brain_window'`

- [ ] **Step 3: Write the mixin**

```python
# ui/brain_window.py
"""Brain modality selection and settings.

Passive, like every other UI mixin: it renders widgets and exposes state. The
orchestrator reads _request_brain_layout_change and performs the switch.
"""
from __future__ import annotations

import numpy as np
from imgui_bundle import imgui

from services.brains import REGISTRY, get

# tanh(2.65) ~ 0.99, so |z| above this is within 1% of the rail.
SATURATION_Z = 2.65


def saturation_fraction(z) -> float:
    """Fraction of search coordinates within 1% of their decoded limit.

    Measured over five runs, evolved Fourier genomes drift from ~0% to 10%
    hard-saturated while the reported sigma barely moves, so this number is
    not visible from anything else on screen.
    """
    a = np.abs(np.asarray(z, dtype=np.float32).reshape(-1))
    return float(np.mean(a >= SATURATION_Z)) if a.size else 0.0


def layout_change_needed(old: dict, new: dict) -> bool:
    """True when a setting that changes the parameter COUNT was edited.
    Those reset the optimizer and switch archive; the rest are free."""
    for key in ("centers", "filters", "bumps", "hidden", "activation"):
        if key in new and old.get(key) != new[key]:
            return True
    return False


class BrainWindowMixin:
    def render_brain_window(self) -> None:
        if not self._show_brain_window:
            return
        _, self._show_brain_window = imgui.begin("Brain", True)

        names = sorted(REGISTRY, key=lambda n: REGISTRY[n].modality_id)
        idx = names.index(self._brain_modality)
        changed, idx = imgui.combo("Modality", idx, names)
        if changed:
            self._brain_modality = names[idx]
            self._brain_settings = {}
            self._request_brain_layout_change = True

        modality = get(self._brain_modality)
        pending = dict(self._brain_settings)
        for s in modality.settings_schema():
            cur = pending.get(s.key, s.default)
            if s.kind == "int":
                ch, v = imgui.slider_int(s.label, int(cur), int(s.lo), int(s.hi))
            elif s.kind == "choice":
                ch, v = imgui.combo(s.label, int(cur), list(s.choices))
            else:
                ch, v = imgui.slider_float(s.label, float(cur), s.lo, s.hi)
            if ch:
                pending[s.key] = v

        if pending != self._brain_settings:
            if layout_change_needed(self._brain_settings, pending):
                self._pending_brain_settings = pending
                imgui.open_popup("Change brain layout?")
            else:
                self._brain_settings = pending

        self._render_layout_change_popup()

        layout = modality.layout_from_settings(self._brain_settings)
        imgui.separator()
        imgui.text(f"search dim: {layout.length} · archive: {layout.signature()}"
                   f" ({self._archive_size} entries)")
        frac = saturation_fraction(self._current_best_z)
        imgui.text(f"saturated params: {frac:.0%}")
        imgui.progress_bar(frac, (-1, 0))
        imgui.end()

    def _render_layout_change_popup(self) -> None:
        if not imgui.begin_popup_modal("Change brain layout?")[0]:
            return
        imgui.text("This changes the genome layout.\n"
                   "The optimizer will reset and searching switches archive.")
        if imgui.button("Apply"):
            self._brain_settings = self._pending_brain_settings
            self._request_brain_layout_change = True
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Cancel"):
            self._pending_brain_settings = {}
            imgui.close_current_popup()
        imgui.end_popup()
```

Register in `ui/core.py`: add `from .brain_window import BrainWindowMixin` to the
imports and `BrainWindowMixin,` to the `UI` base list. Initialise these in
`UI.__init__`:

```python
        self._show_brain_window = False
        self._brain_modality = "fourier"
        self._brain_settings = {}
        self._pending_brain_settings = {}
        self._request_brain_layout_change = False
        self._archive_size = 0
        self._current_best_z = np.zeros(80, dtype=np.float32)
```

Expose `brain_modality`, `brain_settings` and `request_brain_layout_change` in
`get_state()`, following the one-shot-flag convention the orchestrator already
uses for `request_reset`.

- [ ] **Step 4: Handle the layout change in the orchestrator**

The flag is inert without this. The handler lives on `App` in `main.py`, not on
`CommandHandler`, because it needs `_build_archive_set` and the same teardown
`_switch_archive` performs. `CommandHandler` only reads and clears the flag,
like every other one-shot flag.

First, initialise the layout in `App.__init__` next to `self.archive = None`
(around `main.py:106`):

```python
        from services.brains import default_layout
        self.brain_layout = default_layout()
```

Then add the handler, mirroring `_switch_archive`'s ordering:

```python
# main.py — new method on App
    def _apply_brain_layout(self, layout) -> None:
        """Switch brain layout. A hard reset of the search, never partial.

        The teardown is _switch_archive's, for the same reasons: flush before
        closing the store or the entries since the last 200-admission vector
        flush are lost, and release the thumbnail cache before rebuilding or
        the new archive shows the old one's pictures - entry ids restart at 0
        in every archive.

        The archive changes because its directory is keyed by the layout
        signature, so a layout change IS an archive switch.
        """
        from services.archive_io import migrate_legacy_archive
        from services.genome_spec import physics_spec_for, spec_for
        from services.archive_library import resolve
        from utilities.paths import get_archives_root

        if layout == self.brain_layout:
            return

        if self.auto_service is not None:
            self.auto_service.pause()
        if self.imgep_driver is not None:
            self.imgep_driver.end_expedition()
        if self.archive is not None:
            self.archive.maybe_flush(force=True)
        if self.goal_list is not None:
            self.goal_list.save()
        if self.archive_store is not None:
            self.archive_store.close()
        if self.thumb_cache is not None:
            self.thumb_cache.release()

        self.brain_layout = layout
        self.sim.realloc_brain_buffers(layout)

        if self.auto_service is not None:
            spec = (physics_spec_for(layout) if self.auto_service.physics_enabled
                    else spec_for(layout))
            self.auto_service.driver.set_spec(spec)
            self.auto_service.driver.reset()

        path = resolve(get_archives_root(), self.ui_state.preferences.archive_name)
        migrate_legacy_archive(path)
        self._build_archive_set(path)
```

`_build_archive_set` also needs to pass the layout through — update its two
constructor calls at `main.py:240-241`:

```python
        store = ArchiveStore(path, self.brain_layout)
        archive = Archive(store=store, layout=self.brain_layout)
```

And in `command_handler.py`, read and clear the flag:

```python
        if state.request_brain_layout_change:
            from services.brains import get
            layout = get(state.brain_modality).layout_from_settings(
                state.brain_settings)
            self.app._apply_brain_layout(layout)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_window_render.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 6: Verify manually**

Run the app. Open the Brain window and confirm:
- switching through all four modalities changes the status line's search dim
- changing a count raises the warning; Cancel leaves the layout untouched
- Apply switches the archive directory and the entry count changes accordingly
- a non-count setting (Freq Scale) applies with no warning

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: exit code 0.

- [ ] **Step 8: Commit**

```bash
git add ui/brain_window.py ui/core.py state/ui_state.py command_handler.py tests/test_brain_window_render.py
git commit -m "feat: add the Brain window with modality selection"
```

---

### Task 12: Brain Inspector

**Files:**
- Create: `shaders/brain_preview.frag`, `services/brain_preview.py`
- Modify: `ui/brain_window.py`
- Test: `tests/test_brain_shader_source.py` (extend)

**Interfaces:**
- Consumes: the pure `brain_*` GLSL functions, `BrainLayout`.
- Produces: `BrainPreview(ctx, size=512)` with `render(layout, slice_axes, channel, unit_index, value_range) -> moderngl.Texture` and `unit_count(layout) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brain_shader_source.py — append
def test_preview_shader_calls_the_same_brain_functions():
    """The inspector must not reimplement the brain - that is the whole reason
    the brain functions are pure."""
    src = read("shaders/brain_preview.frag")
    for fn in ("brain_fourier", "brain_gabor", "brain_lenia", "brain_mlp"):
        assert fn in src


def test_preview_shader_binds_the_same_brain_buffer():
    src = read("shaders/brain_preview.frag")
    assert "buffer BrainBuffer" in src
    assert "float brain_params[]" in src


def test_preview_isolates_a_single_unit():
    """Per-unit thumbnails need one unit's contribution, which means the
    preview passes a base offset pointing at that unit alone."""
    src = read("shaders/brain_preview.frag")
    assert "PREVIEW_UNIT" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_shader_source.py -v -k preview`
Expected: FAIL with `FileNotFoundError: shaders/brain_preview.frag`

- [ ] **Step 3a: Write the preview fragment shader**

```glsl
// shaders/brain_preview.frag
// Renders one brain unit's response over a 2D slice of the 4D sensor space.
//
// This includes the SAME brain_*() functions the compute shader uses and binds
// the SAME parameter buffer, so what it draws is exactly what the particles
// compute. That is only possible because those functions are pure.
#version 430

in vec2 uv;
out vec4 frag;

layout(std430, binding = 4) buffer BrainBuffer {
    float brain_params[];
};

uniform int   BRAIN_MODALITY;
uniform int   BRAIN_LEN;
uniform ivec4 BRAIN_SHAPE;

uniform int   PREVIEW_UNIT;     // -1 = whole brain, >=0 = that unit alone
uniform ivec2 PREVIEW_AXES;     // which two of the 4 inputs to sweep
uniform int   PREVIEW_CHANNEL;  // 0..3, or 4 for all as RGBA
uniform float PREVIEW_RANGE;    // half-extent of the swept axes
uniform int   PREVIEW_STRIDE;   // floats per unit, for the base offset

vec4 eval_brain(uint base, vec4 x) {
    if (BRAIN_MODALITY == 0) return brain_fourier(base, x);
    if (BRAIN_MODALITY == 1) return brain_gabor(base, x);
    if (BRAIN_MODALITY == 2) return brain_lenia(base, x);
    return brain_mlp(base, x);
}

void main() {
    vec4 x = vec4(0.0);
    x[PREVIEW_AXES.x] = (uv.x * 2.0 - 1.0) * PREVIEW_RANGE;
    x[PREVIEW_AXES.y] = (uv.y * 2.0 - 1.0) * PREVIEW_RANGE;

    uint base = (PREVIEW_UNIT < 0) ? 0u : uint(PREVIEW_UNIT * PREVIEW_STRIDE);
    vec4 r = eval_brain(base, x);

    if (PREVIEW_CHANNEL == 4) {
        frag = vec4(r.xyz * 0.5 + 0.5, 1.0);
    } else {
        float v = r[PREVIEW_CHANNEL];
        // Diverging map: blue negative, black zero, orange positive.
        frag = vec4(max(-v, 0.0), max(v, 0.0) * 0.6, max(v, 0.0), 1.0);
    }
}
```

**Note:** rendering one unit alone requires `BRAIN_SHAPE.x = 1` for that draw,
so the loop runs once starting at the unit's own offset. `BrainPreview.render`
sets that.

- [ ] **Step 3b: Write the preview service**

```python
# services/brain_preview.py
"""Offscreen render of brain response fields for the Brain Inspector.

One 512x512 texture holds every unit's thumbnail as an atlas, filled in a
single draw per unit. 48 units is 48 small viewport draws into one target,
which is cheaper than 48 textures and lets ImGui blit sub-rects.
"""
from __future__ import annotations

import moderngl

from services.brains import get
from utilities.gl_helpers import read_shader, shader_prepend

ATLAS = 512
CELL = 64
UNITS_PER_ROW = ATLAS // CELL

# Floats per unit, per modality. Must match each modality's decode layout.
# MLP is 0 because its hidden neurons are not localised units worth drawing
# separately; it renders the aggregate only.
STRIDE = {"fourier": 8, "gabor": 14, "lenia": 10, "mlp": 0}

# A fullscreen triangle, defined here rather than reusing shaders/canvas.vert:
# that shader carries camera uniforms and attributes this pass does not have,
# and binding it would couple the inspector to the renderer's vertex format.
PREVIEW_VERT = """
#version 430
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""


def unit_count(layout) -> int:
    """MLP has no per-unit decomposition worth drawing - its hidden neurons are
    not localised - so it shows only the aggregate."""
    return 0 if layout.modality == "mlp" else layout.shape[0]


class BrainPreview:
    def __init__(self, ctx: moderngl.Context, size: int = ATLAS):
        self.ctx = ctx
        self.size = size
        src = read_shader("shaders/brain_preview.frag")
        for brain in ("mlp", "lenia", "gabor", "fourier"):
            src = shader_prepend(src, read_shader(f"shaders/brains/{brain}.glsl"))
        self.program = ctx.program(vertex_shader=PREVIEW_VERT, fragment_shader=src)
        self.texture = ctx.texture((size, size), 4, dtype="f1")
        self.fbo = ctx.framebuffer(color_attachments=[self.texture])
        self.quad = ctx.vertex_array(self.program, [])

    def render(self, layout, slice_axes=(0, 2), channel=0, value_range=1.0):
        """Fill the atlas: the aggregate in cell 0, then one cell per unit."""
        # BrainLayout carries the modality NAME; the numeric id lives on the
        # modality object.
        modality_id = get(layout.modality).modality_id
        act = layout.shape[1] if len(layout.shape) > 1 else 0

        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.program["BRAIN_MODALITY"].value = modality_id
        self.program["BRAIN_LEN"].value = layout.length
        self.program["PREVIEW_AXES"].value = slice_axes
        self.program["PREVIEW_CHANNEL"].value = channel
        self.program["PREVIEW_RANGE"].value = value_range
        self.program["PREVIEW_STRIDE"].value = STRIDE[layout.modality]

        # Cell 0 is the aggregate - the actual force field particles feel, and
        # the view that says whether a creature will be smooth or frantic.
        # Per-unit cells re-run the same function with BRAIN_SHAPE.x = 1 so the
        # loop executes once, starting at that unit's own offset.
        cells = [(-1, layout.shape[0])] + [(i, 1) for i in range(unit_count(layout))]
        for slot, (unit, shape_x) in enumerate(cells):
            col, row = slot % UNITS_PER_ROW, slot // UNITS_PER_ROW
            if row >= UNITS_PER_ROW:
                break
            self.ctx.viewport = (col * CELL, row * CELL, CELL, CELL)
            self.program["PREVIEW_UNIT"].value = unit
            self.program["BRAIN_SHAPE"].value = (shape_x, act, 0, 0)
            self.quad.render(moderngl.TRIANGLE_STRIP, vertices=4)
        self.ctx.viewport = (0, 0, self.size, self.size)
        return self.texture

    def release(self) -> None:
        self.fbo.release()
        self.texture.release()
        self.quad.release()
        self.program.release()
```

- [ ] **Step 3c: Draw the atlas in the Brain window**

Add three more fields to `UI.__init__` alongside the ones from Task 11:

```python
        self._brain_preview_texture = None   # set by the orchestrator each frame
        self._preview_channel = 0
        self._preview_range = 1.0
```

`main.py` owns the `BrainPreview` instance, calls `render(...)` when the layout
or parameters changed, and assigns `ui._brain_preview_texture = tex.glo`.

Append to `render_brain_window`, before `imgui.end()`:

```python
        if imgui.collapsing_header("Inspector"):
            _, self._preview_channel = imgui.combo(
                "Channel", self._preview_channel,
                ["force axial", "force lateral", "strafe axial",
                 "strafe lateral", "all (RGBA)"])
            _, self._preview_range = imgui.slider_float(
                "Range", self._preview_range, 0.1, 10.0)
            tex = self._brain_preview_texture
            if tex is not None:
                n = 1 + (0 if layout.modality == "mlp" else layout.shape[0])
                per_row = 512 // 64
                for slot in range(min(n, per_row * per_row)):
                    col, row = slot % per_row, slot // per_row
                    u0, v0 = col / per_row, row / per_row
                    imgui.image(tex, (64, 64), (u0, v0),
                                (u0 + 1 / per_row, v0 + 1 / per_row))
                    if (slot + 1) % 6:
                        imgui.same_line()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_shader_source.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Verify manually — this is the real gate**

Run the app. For each modality:
- Confirm the thumbnails render and change when you reroll the seed.
- Confirm a Gabor unit shows a **localised blob**, not a full-field pattern —
  if it fills the cell, the envelope is not being applied and the parameter
  order is wrong.
- Confirm a Lenia unit shows a **band**, positive inside and negative outside.
- Confirm the aggregate tile changes as a tournament runs.

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: exit code 0.

- [ ] **Step 7: Commit**

```bash
git add shaders/brain_preview.frag services/brain_preview.py ui/brain_window.py tests/test_brain_shader_source.py
git commit -m "feat: add the Brain Inspector response-field preview"
```

---

## Final verification

- [ ] Full suite green: `.venv/Scripts/python.exe -m pytest -q` → exit 0
- [ ] Fourier at 10 centers is visually identical to `main` on the same preset and seed
- [ ] All four modalities produce motion
- [ ] Switching modality mid-tournament does not crash
- [ ] `archive/` migrated to `archive/fourier-n10/` with the entry count unchanged
- [ ] Saturation meter moves during a tournament
- [ ] Inspector thumbnails match observed behaviour per Task 12 Step 5
- [ ] Update `CLAUDE.md` with the two non-obvious constraints: the per-particle
      buffer must use `BRAIN_LEN` not `MAX_BRAIN_FLOATS`, and brain GLSL
      functions must stay pure or the Inspector breaks
