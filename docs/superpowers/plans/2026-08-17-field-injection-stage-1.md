# Field Injection Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-writer "Shader Driven Field" override with an ordered stack of layers that composite procedural, image and brush sources into the force and strafe fields.

**Architecture:** A `FieldBus` owns the destination texture and rebuilds it from the stack every frame, so a disabled layer contributes nothing by definition. Sources return a texture handle; one composite shader applies the mapping while native GL blend state and a colour mask do the combining. Any source needing memory (the brush) owns its own buffer — the bus itself is stateless per frame.

**Tech Stack:** Python 3.12, moderngl (OpenGL 4.3), imgui_bundle, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-17-field-injection-design.md`. This plan covers Stage 1 only. Video, webcam and optical flow are Stage 2; the `canvas` and `spawn` destinations are Stage 3; `param:NAME` is Stage 4.

## Global Constraints

- **Test interpreter is `.venv/Scripts/python.exe`.** Bare `python` is 3.10 with no pytest.
- **Stage commits by filename.** This tree is shared with concurrent sessions; `git add -A` and `git add <directory>` will commit another agent's work.
- **Commit trailer:** end every commit message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- **Comments state the rule, never the evidence.** No percentages, timings, dates, or before/after comparisons in comments, docstrings or tooltips. Measured facts live in the spec or CLAUDE.md; everywhere else points at that home.
- **A tooltip is ONE sentence** naming what the control does.
- **Every new `PreferencesState` field must be added to `UNDOABLE_FIELDS` or to `NOT_UNDOABLE` with a reason string**, or `tests/test_undo_fields.py` fails. It derives its cases from `__dataclass_fields__`.
- **An ImGui widget's identity is its label.** Two visible items with the same label put up a "conflicting ID" dialog and one stops responding. Use `##suffix` to disambiguate.
- **Settings panels push `layout.push_settings_width()`.** A label wider than `layout.WIDEST_LABEL` ("Autosave every N gens") fails `tests/test_label_widths.py`.
- **`sim.py` is user-owned.** Do not restructure it. Stage 1 touches it not at all.
- **Shaders live under `shaders/`** relative to the executable, or builds break.
- **Platform is Windows.** Use forward slashes or `os.path`; use `rm` not `del` in bash.

---

## File Structure

| file | responsibility |
|---|---|
| `state/field_stack.py` | `FieldLayer` / `FieldStack` dataclasses, JSON round-trip. No GL, no imgui. |
| `services/shader_params.py` | `parse_shader_params(src)` — pure text → parameter descriptors. |
| `services/field_sources.py` | Source registry: descriptors and `evaluate()` per source kind. |
| `utilities/field_bus.py` | Owns the destination texture and scratch; runs the rebuild loop. |
| `shaders/field/composite.frag` | The single composite shader: mapping, strength, NaN clamp. |
| `shaders/field/noise.frag` | Built-in fbm/worley source. |
| `shaders/field/gradient.frag` | Built-in linear/radial ramp source. |
| `ui/field_stack_window.py` | The layer list and the inspect panel. |
| `services/config_saver.py` | Gains a `field_stack` block and the legacy-PNG migration. |
| `simulation_runner.py` | Calls `bus.rebuild()` where `process_override` was called. |

Deleted at Task 9: `AdvancedDrawingProcessor.process_override`, `_ensure_override_resources`, `_cleanup_override`, `get_available_override_shaders`, `resolve_override_shader_path`, and the Shader Driven Field block in `ui/advanced_drawing_window.py`.

---

### Task 1: The layer data model

**Files:**
- Create: `state/field_stack.py`
- Test: `tests/test_field_stack.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FieldLayer` dataclass with fields `uid: str`, `enabled: bool`, `source: str`, `params: dict`, `mapping: str`, `destination: str`, `blend: str`, `strength: float`, `blur: float`, `sign: float`, `error: str | None`. `FieldStack` with `layers: list[FieldLayer]`. Module functions `stack_to_dict(stack) -> dict`, `stack_from_dict(d) -> FieldStack`, `new_uid() -> str`. Constants `MAPPINGS`, `DESTINATIONS`, `BLENDS`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_stack.py`:

```python
"""The layer stack is pure data: it round-trips through JSON unchanged.

`error` is runtime state, so it must never survive serialization - a layer
whose shader failed last session must not open pre-broken.
"""
import json

from state.field_stack import (
    FieldLayer, FieldStack, new_uid, stack_to_dict, stack_from_dict,
    MAPPINGS, DESTINATIONS, BLENDS,
)


def test_uids_are_unique():
    assert new_uid() != new_uid()


def test_a_new_layer_gets_a_uid():
    assert FieldLayer().uid


def test_round_trip_preserves_every_field():
    stack = FieldStack(layers=[
        FieldLayer(source="noise", mapping="curl", destination="force",
                   blend="add", strength=0.8, blur=2.0, sign=-1.0,
                   params={"speed": 1.5, "octaves": 3}),
        FieldLayer(source="brush", mapping="rg_direct", destination="strafe",
                   blend="replace", strength=1.0, enabled=False),
    ])
    restored = stack_from_dict(json.loads(json.dumps(stack_to_dict(stack))))

    assert len(restored.layers) == 2
    for before, after in zip(stack.layers, restored.layers):
        assert after.uid == before.uid
        assert after.enabled == before.enabled
        assert after.source == before.source
        assert after.params == before.params
        assert after.mapping == before.mapping
        assert after.destination == before.destination
        assert after.blend == before.blend
        assert after.strength == before.strength
        assert after.blur == before.blur
        assert after.sign == before.sign


def test_error_is_not_serialized():
    stack = FieldStack(layers=[FieldLayer(error="0:31 no matching function")])
    assert "error" not in stack_to_dict(stack)["layers"][0]
    assert stack_from_dict(stack_to_dict(stack)).layers[0].error is None


def test_an_empty_dict_is_an_empty_stack():
    assert stack_from_dict({}).layers == []


def test_an_unknown_enum_value_falls_back_to_the_default():
    d = {"layers": [{"uid": "a", "mapping": "nonsense", "blend": "nonsense",
                     "destination": "nonsense"}]}
    layer = stack_from_dict(d).layers[0]
    assert layer.mapping in MAPPINGS
    assert layer.blend in BLENDS
    assert layer.destination in DESTINATIONS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_stack.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'state.field_stack'`

- [ ] **Step 3: Write the implementation**

Create `state/field_stack.py`:

```python
"""The field injection layer stack: pure data, no GL and no imgui.

A layer binds a source to a destination through a mapping. The bus rebuilds
every destination from this list each frame, so a disabled layer contributes
nothing without anything having to undo it.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

MAPPINGS = ("rg_direct", "polar", "gradient", "curl", "luminance")
DESTINATIONS = ("force", "strafe")
BLENDS = ("replace", "add", "multiply", "max")


def new_uid() -> str:
    """A layer's stable identity, used to key its GPU state across reorders."""
    return uuid.uuid4().hex[:12]


def _pick(value, allowed, default):
    return value if value in allowed else default


@dataclass
class FieldLayer:
    uid: str = field(default_factory=new_uid)
    enabled: bool = True
    source: str = "noise"
    params: dict = field(default_factory=dict)
    mapping: str = "curl"
    destination: str = "force"
    blend: str = "add"
    strength: float = 1.0
    blur: float = 0.0
    sign: float = 1.0
    # Runtime only. Never serialized: a layer whose shader failed last session
    # must not open pre-broken.
    error: str | None = None


@dataclass
class FieldStack:
    layers: list[FieldLayer] = field(default_factory=list)


def stack_to_dict(stack: FieldStack) -> dict:
    return {"layers": [
        {
            "uid": l.uid,
            "enabled": l.enabled,
            "source": l.source,
            "params": dict(l.params),
            "mapping": l.mapping,
            "destination": l.destination,
            "blend": l.blend,
            "strength": l.strength,
            "blur": l.blur,
            "sign": l.sign,
        }
        for l in stack.layers
    ]}


def stack_from_dict(d: dict) -> FieldStack:
    """Rebuild a stack. An unknown enum value falls back to the default rather
    than raising, so a file from a build with more sources still opens."""
    out = []
    for raw in (d or {}).get("layers", []):
        out.append(FieldLayer(
            uid=str(raw.get("uid") or new_uid()),
            enabled=bool(raw.get("enabled", True)),
            source=str(raw.get("source", "noise")),
            params=dict(raw.get("params", {})),
            mapping=_pick(raw.get("mapping"), MAPPINGS, "curl"),
            destination=_pick(raw.get("destination"), DESTINATIONS, "force"),
            blend=_pick(raw.get("blend"), BLENDS, "add"),
            strength=float(raw.get("strength", 1.0)),
            blur=float(raw.get("blur", 0.0)),
            sign=float(raw.get("sign", 1.0)),
        ))
    return FieldStack(layers=out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_stack.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add state/field_stack.py tests/test_field_stack.py
git commit -m "feat: the field injection layer stack, as pure data

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: The shader uniform parser

**Files:**
- Create: `services/shader_params.py`
- Test: `tests/test_shader_params.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ShaderParam` dataclass with `name: str`, `kind: str` (one of `"float"`, `"int"`, `"bool"`, `"color"`, `"vec"`), `lo: float`, `hi: float`, `default: tuple[float, ...]`, `label: str`, `components: int`. Function `parse_shader_params(src: str) -> list[ShaderParam]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_shader_params.py`:

```python
"""Sliders are discovered from a .frag's own uniform declarations.

Annotation is opt-in: an unannotated uniform yields no ShaderParam, so a
shader written before this existed still compiles and still runs, just with
no UI. That property is what keeps march.frag working.
"""
from services.shader_params import parse_shader_params, ShaderParam


def names(src):
    return [p.name for p in parse_shader_params(src)]


def one(src) -> ShaderParam:
    params = parse_shader_params(src)
    assert len(params) == 1, params
    return params[0]


def test_an_unannotated_uniform_yields_nothing():
    assert names("uniform float speed;") == []


def test_a_shader_with_no_uniforms_yields_nothing():
    assert names("void main(){ fragColor = vec4(0.0); }") == []


def test_a_ranged_float():
    p = one('uniform float speed;   // 0..5 = 1.5   "Speed"')
    assert (p.name, p.kind, p.lo, p.hi) == ("speed", "float", 0.0, 5.0)
    assert p.default == (1.5,)
    assert p.label == "Speed"
    assert p.components == 1


def test_a_ranged_int():
    p = one('uniform int octaves; // 1..8 = 4 "Octaves"')
    assert (p.name, p.kind, p.lo, p.hi, p.default) == ("octaves", "int", 1.0, 8.0, (4.0,))


def test_a_bool_has_no_range():
    p = one('uniform bool invert; // = false "Invert"')
    assert (p.name, p.kind, p.default) == ("invert", "bool", (0.0,))


def test_a_bool_defaulting_true():
    assert one('uniform bool on; // = true "On"').default == (1.0,)


def test_a_colour_vec3():
    p = one('uniform vec3 tint; // color = 1,.5,0 "Tint"')
    assert (p.name, p.kind, p.components) == ("tint", "color", 3)
    assert p.default == (1.0, 0.5, 0.0)


def test_a_ranged_vec2_is_a_multi_slider():
    p = one('uniform vec2 centre; // 0..1 = .5,.5 "Centre"')
    assert (p.name, p.kind, p.components, p.default) == ("centre", "vec", 2, (0.5, 0.5))


def test_the_label_defaults_to_a_titled_name():
    assert one("uniform float draw_speed; // 0..1 = 0.5").label == "Draw Speed"


def test_declarations_are_found_among_real_shader_text():
    src = (
        "#version 430\n"
        "in vec2 texcoord;\n"
        "out vec4 fragColor;\n"
        "uniform vec2 canvas_resolution;\n"
        'uniform float speed;  // 0..5 = 1.0 "Speed"\n'
        'uniform int   steps;  // 1..64 = 8 "Steps"\n'
        "void main(){ fragColor = vec4(speed); }\n"
    )
    assert names(src) == ["speed", "steps"]


def test_a_commented_out_declaration_is_ignored():
    assert names('// uniform float speed; // 0..5 = 1.0 "Speed"') == []


def test_a_malformed_annotation_is_ignored_rather_than_raising():
    assert names('uniform float speed; // nonsense here') == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_shader_params.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.shader_params'`

- [ ] **Step 3: Write the implementation**

Create `services/shader_params.py`:

```python
"""Discover a field source shader's sliders from its own uniform declarations.

Annotation is opt-in. A uniform with no trailing annotation yields no
parameter, so a shader written before this existed keeps working with its
uniforms left at their GLSL defaults.

    uniform float speed;   // 0..5 = 1.0       "Speed"
    uniform int   octaves; // 1..8 = 4         "Octaves"
    uniform vec3  tint;    // color = 1,.5,0   "Tint"
    uniform bool  invert;  // = false          "Invert"
"""
from __future__ import annotations

import re
from dataclasses import dataclass

COMPONENTS = {"float": 1, "int": 1, "bool": 1, "vec2": 2, "vec3": 3, "vec4": 4}

_DECL = re.compile(
    r"^[ \t]*uniform[ \t]+(?P<type>float|int|bool|vec2|vec3|vec4)[ \t]+"
    r"(?P<name>[A-Za-z_]\w*)[ \t]*;[ \t]*//[ \t]*(?P<note>.*)$",
    re.MULTILINE,
)
_NOTE = re.compile(
    r"^(?:(?P<color>color)|(?P<lo>-?[\d.]+)\.\.(?P<hi>-?[\d.]+))?[ \t]*"
    r"=[ \t]*(?P<default>[^\"]+?)[ \t]*(?:\"(?P<label>[^\"]*)\")?[ \t]*$"
)


@dataclass
class ShaderParam:
    name: str
    kind: str          # float | int | bool | color | vec
    lo: float
    hi: float
    default: tuple[float, ...]
    label: str
    components: int


def _numbers(text: str, want: int) -> tuple[float, ...] | None:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != want:
        return None
    out = []
    for p in parts:
        low = p.lower()
        if low in ("true", "false"):
            out.append(1.0 if low == "true" else 0.0)
            continue
        try:
            out.append(float(p))
        except ValueError:
            return None
    return tuple(out)


def parse_shader_params(src: str) -> list[ShaderParam]:
    """Return one ShaderParam per annotated uniform, in declaration order.

    A declaration whose annotation does not parse is skipped rather than
    raising: a shader is a user's text file and must never fail to load
    because of a comment.
    """
    found: list[ShaderParam] = []
    for m in _DECL.finditer(src):
        note = _NOTE.match(m.group("note").strip())
        if note is None:
            continue
        gl_type = m.group("type")
        comps = COMPONENTS[gl_type]
        default = _numbers(note.group("default"), comps)
        if default is None:
            continue

        if gl_type == "bool":
            kind, lo, hi = "bool", 0.0, 1.0
        elif note.group("color"):
            kind, lo, hi = "color", 0.0, 1.0
        elif note.group("lo") is None:
            continue
        elif gl_type == "int":
            kind, lo, hi = "int", float(note.group("lo")), float(note.group("hi"))
        elif comps == 1:
            kind, lo, hi = "float", float(note.group("lo")), float(note.group("hi"))
        else:
            kind, lo, hi = "vec", float(note.group("lo")), float(note.group("hi"))

        found.append(ShaderParam(
            name=m.group("name"),
            kind=kind,
            lo=lo,
            hi=hi,
            default=default,
            label=note.group("label") or m.group("name").replace("_", " ").title(),
            components=comps,
        ))
    return found
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_shader_params.py -q`
Expected: PASS, 12 passed

- [ ] **Step 5: Commit**

```bash
git add services/shader_params.py tests/test_shader_params.py
git commit -m "feat: discover a field shader's sliders from its own uniforms

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The composite shader

**Files:**
- Create: `shaders/field/composite.frag`
- Create: `utilities/field_bus.py` (partial — construction, destination texture, `composite_one`)
- Test: `tests/test_field_composite_gl.py`

**Interfaces:**
- Consumes: `FieldLayer` from Task 1.
- Produces: `FieldBus(ctx)` with `.field_texture` (property, `moderngl.Texture` or `None`), `.resolution` (property, `(w, h)`), `.ensure(width, height, scale)`, `.clear()`, `.composite_one(src_tex, layer)`, `.cleanup()`. Class constant `FieldBus.DEST_CHANNELS: dict[str, tuple[bool, bool, bool, bool]]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_composite_gl.py`:

```python
"""The composite pass, on a real GPU.

Blending is native GL state rather than a shader branch, and a destination is
restricted to its own channel pair by a colour mask. Both are easy to get
subtly wrong and invisible on screen: a force layer that also writes strafe
looks like a preset that strafes more than it used to.

Skipped when no GL 4.3 context is available.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from state.field_stack import FieldLayer  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402

RES = 32


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


@pytest.fixture
def bus(ctx):
    b = FieldBus(ctx)
    b.ensure(RES, RES, 1.0)
    yield b
    b.cleanup()


def flat(ctx, value):
    """A RES x RES RGBA32F texture with every texel set to `value`."""
    data = np.tile(np.array(value, dtype="f4"), (RES, RES, 1))
    return ctx.texture((RES, RES), 4, data.tobytes(), dtype="f4")


def read(bus):
    tex = bus.field_texture
    return np.frombuffer(tex.read(), dtype="f4").reshape(RES, RES, 4)


def layer(**kw):
    kw.setdefault("mapping", "rg_direct")
    kw.setdefault("destination", "force")
    kw.setdefault("blend", "replace")
    return FieldLayer(**kw)


def test_replace_writes_the_value(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.25, 0.5, 0.0, 0.0)), layer(strength=1.0))
    out = read(bus)
    assert np.allclose(out[..., 0], 0.25)
    assert np.allclose(out[..., 1], 0.5)


def test_strength_scales_the_value(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.4, 0.0, 0.0, 0.0)), layer(strength=0.5))
    assert np.allclose(read(bus)[..., 0], 0.2)


def test_add_accumulates(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.3, 0.0, 0.0, 0.0)), layer(blend="add"))
    bus.composite_one(flat(ctx, (0.2, 0.0, 0.0, 0.0)), layer(blend="add"))
    assert np.allclose(read(bus)[..., 0], 0.5)


def test_multiply_multiplies(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.5, 0.0, 0.0, 0.0)), layer(blend="replace"))
    bus.composite_one(flat(ctx, (0.5, 0.0, 0.0, 0.0)), layer(blend="multiply"))
    assert np.allclose(read(bus)[..., 0], 0.25)


def test_max_takes_the_larger(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.7, 0.0, 0.0, 0.0)), layer(blend="replace"))
    bus.composite_one(flat(ctx, (0.2, 0.0, 0.0, 0.0)), layer(blend="max"))
    assert np.allclose(read(bus)[..., 0], 0.7)


def test_a_force_layer_leaves_strafe_untouched(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.9, 0.9, 0.0, 0.0)),
                      layer(destination="strafe"))
    bus.composite_one(flat(ctx, (0.1, 0.1, 0.0, 0.0)),
                      layer(destination="force"))
    out = read(bus)
    assert np.allclose(out[..., 2], 0.9), "strafe was clobbered by a force layer"
    assert np.allclose(out[..., 3], 0.9)
    assert np.allclose(out[..., 0], 0.1)


def test_a_strafe_layer_leaves_force_untouched(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (0.9, 0.9, 0.0, 0.0)),
                      layer(destination="force"))
    bus.composite_one(flat(ctx, (0.1, 0.1, 0.0, 0.0)),
                      layer(destination="strafe"))
    out = read(bus)
    assert np.allclose(out[..., 0], 0.9), "force was clobbered by a strafe layer"
    assert np.allclose(out[..., 2], 0.1)


def test_a_nan_source_reaches_the_destination_as_zero(ctx, bus):
    bus.clear()
    bus.composite_one(flat(ctx, (np.nan, np.inf, 0.0, 0.0)), layer())
    out = read(bus)
    assert np.all(np.isfinite(out)), "a non-finite value escaped the composite"
    assert np.allclose(out[..., 0:2], 0.0)


def test_clear_zeroes_every_channel(ctx, bus):
    bus.composite_one(flat(ctx, (0.5, 0.5, 0.0, 0.0)), layer(destination="force"))
    bus.composite_one(flat(ctx, (0.5, 0.5, 0.0, 0.0)), layer(destination="strafe"))
    bus.clear()
    assert np.allclose(read(bus), 0.0)


def test_the_bus_resolution_scale_shrinks_the_texture(ctx):
    b = FieldBus(ctx)
    b.ensure(64, 64, 0.5)
    assert b.resolution == (32, 32)
    b.cleanup()


def test_a_scale_can_never_produce_a_zero_sized_texture(ctx):
    b = FieldBus(ctx)
    b.ensure(2, 2, 0.25)
    assert b.resolution == (1, 1)
    b.cleanup()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_composite_gl.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'utilities.field_bus'`

- [ ] **Step 3: Write the composite shader**

Create `shaders/field/composite.frag`:

```glsl
#version 430

// Maps a source texture into a destination's channel pair. The one composite
// shader in the system: blending and channel masking are GL state, not
// branches here.
//
// Both channel pairs are written with the same value; the colour mask decides
// which pair lands.

in vec2 texcoord;
out vec4 fragColor;

uniform sampler2D src;
uniform int   mapping;    // 0 rg_direct, 1 polar, 2 gradient, 3 curl, 4 luminance
uniform float strength;
uniform float sign_mul;   // +1 attract, -1 repel
uniform float blur_lod;   // mip level of the pre-filter
uniform vec2  texel;
uniform bool  scalar_out; // destination is scalar: contribute the magnitude

float scalar_at(vec2 uv){
    vec3 c = textureLod(src, uv, blur_lod).rgb;
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

void main(){
    vec2 v;
    if (mapping == 0) {
        v = texture(src, texcoord).rg;
    } else if (mapping == 1) {
        // Hue as angle, value as magnitude. The historical PNG field encoding.
        vec4 c = texture(src, texcoord);
        float a = c.r * 6.28318530718;
        v = vec2(cos(a), sin(a)) * c.b;
    } else if (mapping == 4) {
        v = vec2(scalar_at(texcoord), 0.0);
    } else {
        float l = scalar_at(texcoord - vec2(texel.x, 0.0));
        float r = scalar_at(texcoord + vec2(texel.x, 0.0));
        float d = scalar_at(texcoord - vec2(0.0, texel.y));
        float u = scalar_at(texcoord + vec2(0.0, texel.y));
        vec2 g = 0.5 * vec2(r - l, u - d);
        // curl is the gradient turned a quarter turn, so it is divergence-free
        // and particles circulate along contours instead of piling into peaks.
        v = (mapping == 2) ? g : vec2(-g.y, g.x);
        v *= sign_mul;
    }

    v *= strength;
    if (scalar_out) v = vec2(length(v), 0.0);
    // A source may produce a non-finite value; the particles must never see one.
    if (any(isnan(v)) || any(isinf(v))) v = vec2(0.0);

    fragColor = vec4(v, v);
}
```

- [ ] **Step 4: Write the bus**

Create `utilities/field_bus.py`:

```python
"""The field injection bus: owns the destination texture and composites into it.

The bus is stateless per frame. Every destination is cleared and rebuilt from
the stack, so a disabled layer contributes nothing without anything having to
undo it. A source that needs memory owns its own buffer.
"""
from __future__ import annotations

import moderngl

from state.field_stack import MAPPINGS
from utilities.gl_helpers import read_shader, tryset

BLEND_STATE = {
    "replace":  None,
    "add":      (moderngl.ONE, moderngl.ONE, moderngl.FUNC_ADD),
    "multiply": (moderngl.DST_COLOR, moderngl.ZERO, moderngl.FUNC_ADD),
    "max":      (moderngl.ONE, moderngl.ONE, moderngl.MAX),
}


class FieldBus:
    """Owns the force/strafe destination texture and the composite pass."""

    # force is .xy, strafe is .zw of one RGBA32F texture, so entity_update's
    # get_field() is unchanged by this feature.
    DEST_CHANNELS = {
        "force":  (True, True, False, False),
        "strafe": (False, False, True, True),
    }
    SCALAR_DESTINATIONS = ()

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._res = (0, 0)
        self._tex = None
        self._fbo = None
        self._composite = None
        self._composite_vao = None

    # -- public ---------------------------------------------------------

    @property
    def field_texture(self):
        return self._tex

    @property
    def resolution(self) -> tuple[int, int]:
        return self._res

    def ensure(self, canvas_width: int, canvas_height: int, scale: float) -> None:
        """Create or resize the destination texture. Never zero-sized."""
        w = max(1, int(canvas_width * scale))
        h = max(1, int(canvas_height * scale))
        if self._tex is not None and self._res == (w, h):
            return
        self._release_target()
        self._res = (w, h)
        tex = self.ctx.texture((w, h), 4, dtype="f4")
        # LINEAR because the bus runs below canvas resolution by default and
        # nearest sampling is visibly blocky there.
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        self._tex = tex
        self._fbo = self.ctx.framebuffer(color_attachments=[tex])
        self._fbo.clear()
        self._ensure_composite()

    def clear(self) -> None:
        if self._fbo is None:
            return
        self._fbo.color_mask = (True, True, True, True)
        self._fbo.clear()

    def composite_one(self, src_tex, layer) -> None:
        """Map `src_tex` through `layer` and blend it into the destination."""
        if self._fbo is None or src_tex is None:
            return
        self._ensure_composite()

        if layer.blur > 0.0:
            src_tex.build_mipmaps()
        src_tex.use(location=0)

        prog = self._composite
        tryset(prog, "src", 0)
        tryset(prog, "mapping", MAPPINGS.index(layer.mapping))
        tryset(prog, "strength", float(layer.strength))
        tryset(prog, "sign_mul", float(layer.sign))
        tryset(prog, "blur_lod", float(layer.blur))
        tryset(prog, "texel", (1.0 / self._res[0], 1.0 / self._res[1]))
        tryset(prog, "scalar_out", layer.destination in self.SCALAR_DESTINATIONS)

        self._fbo.use()
        self._fbo.color_mask = self.DEST_CHANNELS[layer.destination]

        state = BLEND_STATE[layer.blend]
        if state is None:
            self.ctx.disable(moderngl.BLEND)
        else:
            src_factor, dst_factor, equation = state
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = src_factor, dst_factor
            self.ctx.blend_equation = equation

        self._composite_vao.render(mode=moderngl.TRIANGLE_FAN, vertices=4)

        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self._fbo.color_mask = (True, True, True, True)

    def cleanup(self) -> None:
        self._release_target()
        if self._composite_vao is not None:
            self._composite_vao.release()
            self._composite_vao = None
        if self._composite is not None:
            self._composite.release()
            self._composite = None

    # -- internal -------------------------------------------------------

    def _ensure_composite(self) -> None:
        if self._composite is not None:
            return
        self._composite = self.ctx.program(
            vertex_shader=read_shader("shaders/canvas.vert"),
            fragment_shader=read_shader("shaders/field/composite.frag"),
        )
        self._composite_vao = self.ctx.vertex_array(self._composite, [])

    def _release_target(self) -> None:
        if self._fbo is not None:
            self._fbo.release()
            self._fbo = None
        if self._tex is not None:
            self._tex.release()
            self._tex = None
        self._res = (0, 0)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_composite_gl.py -q`
Expected: PASS, 11 passed (or all skipped if the machine has no GL 4.3 context — if skipped, say so rather than reporting a pass)

- [ ] **Step 6: Commit**

```bash
git add shaders/field/composite.frag utilities/field_bus.py tests/test_field_composite_gl.py
git commit -m "feat: the field composite pass, with blending as GL state

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The source registry and the procedural sources

**Files:**
- Create: `services/field_sources.py`
- Create: `shaders/field/noise.frag`
- Create: `shaders/field/gradient.frag`
- Modify: `utilities/field_bus.py` (add scratch texture + `scratch_fbo`)
- Test: `tests/test_field_sources.py`

**Interfaces:**
- Consumes: `FieldBus` from Task 3, `parse_shader_params` from Task 2, `FieldLayer` from Task 1.
- Produces: `FrameContext` dataclass with `time: float`, `frame_count: int`, `mouse: tuple[float, float]`, `prev_mouse: tuple[float, float]`, `canvas_texture`. `SourceDescriptor` dataclass with `key: str`, `label: str`, `keeps_history: bool`, `builtin_params: list[ShaderParam]`. Functions `descriptors() -> list[SourceDescriptor]`, `get(key) -> SourceDescriptor`, `params_for(layer) -> list[ShaderParam]`, `make_source(key, ctx) -> Source`. `Source` objects expose `evaluate(bus, layer, frame) -> moderngl.Texture | None`, `error: str | None`, and `release()`. New on `FieldBus`: `.scratch_texture` property and `.render_into_scratch(program, vao)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_sources.py`:

```python
"""The source registry: descriptors are data, evaluation returns a texture.

Two properties matter beyond "it renders". A source returns a HANDLE rather
than filling a caller's buffer, so a texture-owning source costs no GPU pass.
And params_for() merges a shader's discovered uniforms with the built-ins, so
a .frag layer's UI comes from the file rather than from a hardcoded list.
"""
from __future__ import annotations

import pytest

moderngl = pytest.importorskip("moderngl")

from state.field_stack import FieldLayer  # noqa: E402
from services import field_sources  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402

RES = 32


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


@pytest.fixture
def bus(ctx):
    b = FieldBus(ctx)
    b.ensure(RES, RES, 1.0)
    yield b
    b.cleanup()


@pytest.fixture
def frame():
    return field_sources.FrameContext(
        time=1.0, frame_count=60, mouse=(0.5, 0.5),
        prev_mouse=(0.5, 0.5), canvas_texture=None)


def test_every_descriptor_has_a_unique_key():
    keys = [d.key for d in field_sources.descriptors()]
    assert len(keys) == len(set(keys))


def test_the_stage_one_sources_are_registered():
    keys = {d.key for d in field_sources.descriptors()}
    assert {"noise", "image", "gradient", "shader", "brush", "feedback"} <= keys


def test_only_some_sources_keep_history():
    assert field_sources.get("shader").keeps_history
    assert field_sources.get("feedback").keeps_history
    assert not field_sources.get("image").keeps_history
    assert not field_sources.get("gradient").keeps_history


@pytest.mark.parametrize("key", ["noise", "gradient"])
def test_a_procedural_source_returns_a_texture_at_bus_resolution(ctx, bus, frame, key):
    src = field_sources.make_source(key, ctx)
    tex = src.evaluate(bus, FieldLayer(source=key), frame)
    assert tex is not None
    assert tex.size == (RES, RES)
    src.release()


def test_a_procedural_source_reports_no_error(ctx, bus, frame):
    src = field_sources.make_source("noise", ctx)
    src.evaluate(bus, FieldLayer(source="noise"), frame)
    assert src.error is None
    src.release()


def test_builtin_params_are_exposed_for_the_ui(ctx):
    names = [p.name for p in field_sources.params_for(FieldLayer(source="noise"))]
    assert "scale" in names
    assert "speed" in names


def test_a_shader_layer_with_no_file_has_no_params():
    layer = FieldLayer(source="shader", params={})
    assert field_sources.params_for(layer) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_sources.py -q`
Expected: FAIL — `ImportError: cannot import name 'field_sources' from 'services'`

- [ ] **Step 3: Add scratch to the bus**

In `utilities/field_bus.py`, add to `__init__` after `self._composite_vao = None`:

```python
        self._scratch = None
        self._scratch_fbo = None
```

Add these methods to `FieldBus`, immediately after `composite_one`:

```python
    @property
    def scratch_texture(self):
        """The shared target procedural sources render into.

        Safe to share because a layer's composite consumes it immediately
        after that layer's own evaluation.
        """
        return self._scratch

    def render_into_scratch(self, vao) -> moderngl.Texture:
        """Render a source's fullscreen quad into scratch and return it."""
        self._ensure_scratch()
        self._scratch_fbo.use()
        self.ctx.disable(moderngl.BLEND)
        vao.render(mode=moderngl.TRIANGLE_FAN, vertices=4)
        return self._scratch
```

Add to `_release_target`, before `self._res = (0, 0)`:

```python
        if self._scratch_fbo is not None:
            self._scratch_fbo.release()
            self._scratch_fbo = None
        if self._scratch is not None:
            self._scratch.release()
            self._scratch = None
```

And add the builder, after `_ensure_composite`:

```python
    def _ensure_scratch(self) -> None:
        if self._scratch is not None:
            return
        w, h = self._res
        tex = self.ctx.texture((w, h), 4, dtype="f4")
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        self._scratch = tex
        self._scratch_fbo = self.ctx.framebuffer(color_attachments=[tex])
```

- [ ] **Step 4: Write the two built-in source shaders**

Create `shaders/field/noise.frag`:

```glsl
#version 430

in vec2 texcoord;
out vec4 fragColor;

uniform vec2  canvas_resolution;
uniform float time;

uniform float scale;    // 0.5..32 = 4.0   "Scale"
uniform int   octaves;  // 1..8 = 4        "Octaves"
uniform float speed;    // 0..4 = 0.4      "Speed"
uniform float warp;     // 0..2 = 0.0      "Domain Warp"

float hash(vec2 p){
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float vnoise(vec2 p){
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}

float fbm(vec2 p){
    float sum = 0.0, amp = 0.5;
    for (int i = 0; i < octaves; i++){
        sum += amp * vnoise(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return sum;
}

void main(){
    vec2 p = texcoord * scale + vec2(0.0, time * speed);
    if (warp > 0.0){
        p += warp * vec2(fbm(p + 7.3), fbm(p - 3.1));
    }
    float v = fbm(p);
    fragColor = vec4(v, v, v, 1.0);
}
```

Create `shaders/field/gradient.frag`:

```glsl
#version 430

in vec2 texcoord;
out vec4 fragColor;

uniform vec2 canvas_resolution;

uniform int   shape;   // 0..1 = 0        "Radial"
uniform float angle;   // -3.15..3.15 = 0 "Angle"
uniform vec2  centre;  // 0..1 = .5,.5    "Centre"
uniform float falloff; // 0.1..4 = 1.0    "Falloff"

void main(){
    float v;
    if (shape == 0){
        v = dot(texcoord - centre, vec2(sin(angle), cos(angle))) + 0.5;
    } else {
        v = 1.0 - clamp(length(texcoord - centre) * 2.0, 0.0, 1.0);
    }
    v = pow(clamp(v, 0.0, 1.0), falloff);
    fragColor = vec4(v, v, v, 1.0);
}
```

- [ ] **Step 5: Write the registry**

Create `services/field_sources.py`:

```python
"""The field source registry.

A source produces an RGBA texture and returns a HANDLE. Procedural sources
render into the bus's shared scratch; sources that own a texture return their
own, so they cost no GPU pass at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import moderngl

from services.shader_params import ShaderParam, parse_shader_params
from utilities.gl_helpers import read_shader, tryset
from utilities.paths import get_app_dir, get_user_data_dir


@dataclass
class FrameContext:
    """Everything a source may read about the frame it is being drawn for."""
    time: float
    frame_count: int
    mouse: tuple[float, float]
    prev_mouse: tuple[float, float]
    canvas_texture: object | None


@dataclass
class SourceDescriptor:
    key: str
    label: str
    keeps_history: bool = False
    shader: str = ""              # bundled .frag, for the procedural sources
    builtin_params: list[ShaderParam] = field(default_factory=list)


def _params_of(shader_relpath: str) -> list[ShaderParam]:
    return parse_shader_params(read_shader(shader_relpath))


_REGISTRY: dict[str, SourceDescriptor] = {}


def _register(desc: SourceDescriptor) -> None:
    _REGISTRY[desc.key] = desc


def _build_registry() -> None:
    if _REGISTRY:
        return
    _register(SourceDescriptor("noise", "Noise", shader="shaders/field/noise.frag",
                               builtin_params=_params_of("shaders/field/noise.frag")))
    _register(SourceDescriptor("gradient", "Gradient",
                               shader="shaders/field/gradient.frag",
                               builtin_params=_params_of("shaders/field/gradient.frag")))
    _register(SourceDescriptor("image", "Image"))
    _register(SourceDescriptor("shader", "Shader", keeps_history=True))
    _register(SourceDescriptor("brush", "Brush"))
    _register(SourceDescriptor("feedback", "Feedback", keeps_history=True))


def descriptors() -> list[SourceDescriptor]:
    _build_registry()
    return list(_REGISTRY.values())


def get(key: str) -> SourceDescriptor:
    _build_registry()
    return _REGISTRY[key]


def available_shader_files() -> list[str]:
    """User .frag files, then bundled ones. Filenames only."""
    names = []
    for folder in (get_user_data_dir(), get_user_data_dir() / "shaders",
                   get_app_dir() / "shaders" / "field_override"):
        if folder.exists():
            names += [f.name for f in sorted(folder.glob("*.frag"))]
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def resolve_shader_path(name: str) -> Path | None:
    for folder in (get_user_data_dir(), get_user_data_dir() / "shaders",
                   get_app_dir() / "shaders" / "field_override"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def params_for(layer) -> list[ShaderParam]:
    """The parameter list a layer's UI should draw."""
    desc = get(layer.source)
    if layer.source != "shader":
        return desc.builtin_params
    path = resolve_shader_path(layer.params.get("_file", ""))
    if path is None:
        return []
    try:
        return parse_shader_params(path.read_text())
    except OSError:
        return []


# -- sources ------------------------------------------------------------


class _ProgramSource:
    """A source backed by a fragment shader rendering into the bus scratch."""

    def __init__(self, ctx: moderngl.Context, frag_source: str):
        self.ctx = ctx
        self.error: str | None = None
        self._program = None
        self._vao = None
        self._compile(frag_source)

    def _compile(self, frag_source: str) -> None:
        try:
            program = self.ctx.program(
                vertex_shader=read_shader("shaders/canvas.vert"),
                fragment_shader=frag_source)
        except Exception as exc:
            self.error = str(exc).strip().splitlines()[0] if str(exc).strip() else "compile failed"
            return
        self.release()
        self._program = program
        self._vao = self.ctx.vertex_array(program, [])
        self.error = None

    def evaluate(self, bus, layer, frame):
        if self._program is None:
            return None
        prog = self._program
        tryset(prog, "canvas_resolution", tuple(float(v) for v in bus.resolution))
        tryset(prog, "time", float(frame.time))
        tryset(prog, "frame_count", int(frame.frame_count))
        tryset(prog, "mouse", tuple(float(v) for v in frame.mouse))
        tryset(prog, "prev_mouse", tuple(float(v) for v in frame.prev_mouse))
        for param in params_for(layer):
            value = layer.params.get(param.name)
            if value is None:
                value = param.default[0] if param.components == 1 else tuple(param.default)
            if param.kind == "int":
                value = int(value)
            elif param.kind == "bool":
                value = bool(value)
            elif param.components == 1:
                value = float(value)
            else:
                value = tuple(float(v) for v in value)
            tryset(prog, param.name, value)
        return bus.render_into_scratch(self._vao)

    def release(self) -> None:
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        if self._program is not None:
            self._program.release()
            self._program = None


class _BuiltinSource(_ProgramSource):
    def __init__(self, ctx, key):
        super().__init__(ctx, read_shader(get(key).shader))


def make_source(key: str, ctx: moderngl.Context):
    """Instantiate the GPU-side object for a source kind."""
    _build_registry()
    if key in ("noise", "gradient"):
        return _BuiltinSource(ctx, key)
    raise KeyError(f"source '{key}' is not implemented yet")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_sources.py -q`
Expected: PASS, 9 passed

- [ ] **Step 7: Commit**

```bash
git add services/field_sources.py shaders/field/noise.frag shaders/field/gradient.frag utilities/field_bus.py tests/test_field_sources.py
git commit -m "feat: the field source registry and the procedural sources

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: The texture-owning sources — image, feedback, shader, brush

**Files:**
- Modify: `services/field_sources.py`
- Test: `tests/test_field_sources_owned.py`

**Interfaces:**
- Consumes: everything from Task 4.
- Produces: `make_source` additionally handles `"image"`, `"feedback"`, `"shader"`, `"brush"`. `_BrushSource` exposes `snapshot() -> np.ndarray | None` and `write(data: np.ndarray) -> None` so `FieldHandler` can retarget onto it in Task 10.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_sources_owned.py`:

```python
"""Sources that own a texture return their own handle, costing no GPU pass.

The brush is the one source with memory, and that is the whole reason the bus
can be stateless: persistence belongs to a source, never to the bus.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from state.field_stack import FieldLayer  # noqa: E402
from services import field_sources  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402

RES = 32


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


@pytest.fixture
def bus(ctx):
    b = FieldBus(ctx)
    b.ensure(RES, RES, 1.0)
    yield b
    b.cleanup()


def frame_with(canvas=None):
    return field_sources.FrameContext(
        time=0.0, frame_count=0, mouse=(0.5, 0.5),
        prev_mouse=(0.5, 0.5), canvas_texture=canvas)


def test_feedback_returns_the_canvas_texture_itself(ctx, bus):
    canvas = ctx.texture((8, 8), 2, dtype="f4")
    src = field_sources.make_source("feedback", ctx)
    got = src.evaluate(bus, FieldLayer(source="feedback"), frame_with(canvas))
    assert got is canvas, "feedback must borrow the canvas, not copy it"
    src.release()
    canvas.release()


def test_feedback_with_no_canvas_contributes_nothing(ctx, bus):
    src = field_sources.make_source("feedback", ctx)
    assert src.evaluate(bus, FieldLayer(source="feedback"), frame_with(None)) is None
    src.release()


def test_a_missing_image_file_sets_an_error_and_returns_none(ctx, bus):
    src = field_sources.make_source("image", ctx)
    layer = FieldLayer(source="image", params={"_file": "does_not_exist.png"})
    assert src.evaluate(bus, layer, frame_with()) is None
    assert src.error and "not found" in src.error.lower()
    src.release()


def test_a_shader_layer_with_no_file_sets_an_error(ctx, bus):
    src = field_sources.make_source("shader", ctx)
    assert src.evaluate(bus, FieldLayer(source="shader"), frame_with()) is None
    assert src.error
    src.release()


def test_a_shader_that_does_not_compile_reports_the_log(ctx, bus, tmp_path, monkeypatch):
    bad = tmp_path / "bad.frag"
    bad.write_text("#version 430\nout vec4 fragColor;\n"
                   "void main(){ fragColor = nope(1.0); }\n")
    monkeypatch.setattr(field_sources, "resolve_shader_path", lambda name: bad)
    src = field_sources.make_source("shader", ctx)
    layer = FieldLayer(source="shader", params={"_file": "bad.frag"})
    assert src.evaluate(bus, layer, frame_with()) is None
    assert src.error, "a compile failure must be reported, not swallowed"
    src.release()


def test_the_brush_owns_a_persistent_buffer(ctx, bus):
    src = field_sources.make_source("brush", ctx)
    data = np.full((RES, RES, 4), 0.25, dtype="f4")
    src.write(data)
    got = src.evaluate(bus, FieldLayer(source="brush"), frame_with())
    assert got is not None
    assert np.allclose(src.snapshot(), 0.25)
    src.release()


def test_the_brush_survives_being_evaluated_twice(ctx, bus):
    src = field_sources.make_source("brush", ctx)
    src.write(np.full((RES, RES, 4), 0.5, dtype="f4"))
    src.evaluate(bus, FieldLayer(source="brush"), frame_with())
    src.evaluate(bus, FieldLayer(source="brush"), frame_with())
    assert np.allclose(src.snapshot(), 0.5), "the bus must not clear a source's memory"
    src.release()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_sources_owned.py -q`
Expected: FAIL — `KeyError: "source 'feedback' is not implemented yet"`

- [ ] **Step 3: Write the implementation**

In `services/field_sources.py`, add these imports at the top of the import block:

```python
import numpy as np
from PIL import Image
```

Append these classes above `make_source`:

```python
class _FeedbackSource:
    """The sim's own canvas, borrowed. Costs no pass and no copy."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.error: str | None = None

    def evaluate(self, bus, layer, frame):
        self.error = None
        return frame.canvas_texture

    def release(self) -> None:
        pass


class _ImageSource:
    """A file on disk, uploaded once and reused until the path changes."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.error: str | None = None
        self._tex = None
        self._loaded = None

    def evaluate(self, bus, layer, frame):
        name = layer.params.get("_file", "")
        if name != self._loaded:
            self._load(name)
        return self._tex

    def _load(self, name: str) -> None:
        self.release()
        self._loaded = name
        if not name:
            self.error = "no image selected"
            return
        path = Path(name)
        if not path.is_absolute():
            path = get_user_data_dir() / name
        if not path.exists():
            self.error = f"image not found: {name}"
            return
        try:
            img = Image.open(path).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
        except OSError as exc:
            self.error = f"could not read {name}: {exc}"
            return
        data = (np.asarray(img, dtype=np.float32) / 255.0).astype("f4")
        tex = self.ctx.texture(img.size, 4, data.tobytes(), dtype="f4")
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        tex.build_mipmaps()
        self._tex = tex
        self.error = None

    def release(self) -> None:
        if self._tex is not None:
            self._tex.release()
            self._tex = None


class _UserShaderSource(_ProgramSource):
    """A user .frag, recompiled whenever the chosen file changes."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.error = "no shader selected"
        self._program = None
        self._vao = None
        self._loaded = None

    def evaluate(self, bus, layer, frame):
        name = layer.params.get("_file", "")
        if name != self._loaded:
            self._loaded = name
            path = resolve_shader_path(name) if name else None
            if path is None:
                self.release()
                self.error = f"shader not found: {name}" if name else "no shader selected"
            else:
                try:
                    self._compile(path.read_text())
                except OSError as exc:
                    self.release()
                    self.error = f"could not read {name}: {exc}"
        if self._program is None:
            return None
        return super().evaluate(bus, layer, frame)

    def reload(self) -> None:
        """Force a recompile on the next evaluate. Bound to the V key."""
        self._loaded = None


class _BrushSource:
    """The mouse brush's accumulation buffer.

    The one source with memory, which is what lets the bus stay stateless:
    persistence is a property of a source, never of the bus.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.error: str | None = None
        self._tex = None
        self._fbo = None

    def ensure(self, width: int, height: int):
        if self._tex is not None and self._tex.size == (width, height):
            return self._tex
        self.release()
        tex = self.ctx.texture((width, height), 4, dtype="f4")
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        self._tex = tex
        self._fbo = self.ctx.framebuffer(color_attachments=[tex])
        self._fbo.clear()
        return tex

    @property
    def framebuffer(self):
        return self._fbo

    def evaluate(self, bus, layer, frame):
        return self.ensure(*bus.resolution)

    def snapshot(self):
        if self._tex is None:
            return None
        h, w = self._tex.height, self._tex.width
        return np.frombuffer(self._tex.read(), dtype=np.float32).reshape(h, w, 4).copy()

    def write(self, data) -> None:
        tex = self.ensure(int(data.shape[1]), int(data.shape[0]))
        tex.write(np.ascontiguousarray(data, dtype="f4").tobytes())

    def clear(self) -> None:
        if self._fbo is not None:
            self._fbo.clear()

    def release(self) -> None:
        if self._fbo is not None:
            self._fbo.release()
            self._fbo = None
        if self._tex is not None:
            self._tex.release()
            self._tex = None
```

Replace the body of `make_source` with:

```python
def make_source(key: str, ctx: moderngl.Context):
    """Instantiate the GPU-side object for a source kind."""
    _build_registry()
    if key in ("noise", "gradient"):
        return _BuiltinSource(ctx, key)
    if key == "feedback":
        return _FeedbackSource(ctx)
    if key == "image":
        return _ImageSource(ctx)
    if key == "shader":
        return _UserShaderSource(ctx)
    if key == "brush":
        return _BrushSource(ctx)
    raise KeyError(f"unknown source '{key}'")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_sources_owned.py tests/test_field_sources.py -q`
Expected: PASS, 16 passed

- [ ] **Step 5: Commit**

```bash
git add services/field_sources.py tests/test_field_sources_owned.py
git commit -m "feat: image, feedback, shader and brush field sources

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The rebuild loop, the dirty flag, and per-layer errors

**Files:**
- Modify: `utilities/field_bus.py`
- Test: `tests/test_field_bus_rebuild.py`

**Interfaces:**
- Consumes: everything from Tasks 1, 3, 4, 5.
- Produces: `FieldBus.rebuild(stack, canvas_width, canvas_height, scale, frame) -> bool` (True if it rebuilt), `FieldBus.mark_dirty()`, `FieldBus.dirty` (property), `FieldBus.pass_count` (property, int — passes run on the last rebuild), `FieldBus.source_for(layer)`, `FieldBus.reload_shaders()`, `FieldBus.brush_source()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_bus_rebuild.py`:

```python
"""The rebuild loop: order, the disabled-layer guarantee, and the dirty flag.

The disabled-layer test is the regression test for the reported defect - a
shader-driven field that could not be turned off, because nothing cleared the
texture it had painted.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from state.field_stack import FieldLayer, FieldStack  # noqa: E402
from services.field_sources import FrameContext  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402

RES = 32


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


@pytest.fixture
def bus(ctx):
    b = FieldBus(ctx)
    yield b
    b.cleanup()


FRAME = FrameContext(time=0.0, frame_count=0, mouse=(0.5, 0.5),
                     prev_mouse=(0.5, 0.5), canvas_texture=None)


def rebuild(bus, stack):
    return bus.rebuild(stack, RES, RES, 1.0, FRAME)


def read(bus):
    tex = bus.field_texture
    return np.frombuffer(tex.read(), dtype="f4").reshape(RES, RES, 4)


def gradient_layer(**kw):
    kw.setdefault("source", "gradient")
    kw.setdefault("mapping", "luminance")
    kw.setdefault("destination", "force")
    kw.setdefault("blend", "replace")
    kw.setdefault("params", {"shape": 1, "falloff": 1.0, "centre": (0.5, 0.5)})
    return FieldLayer(**kw)


def test_an_empty_stack_allocates_nothing(bus):
    assert rebuild(bus, FieldStack()) is False
    assert bus.field_texture is None
    assert bus.pass_count == 0


def test_a_disabled_layer_contributes_exactly_zero(bus):
    rebuild(bus, FieldStack(layers=[gradient_layer()]))
    assert read(bus)[..., 0].max() > 0.0

    layer = gradient_layer(enabled=False)
    bus.mark_dirty()
    rebuild(bus, FieldStack(layers=[layer]))
    assert np.allclose(read(bus), 0.0), "a disabled layer left its contribution behind"


def test_removing_the_last_layer_clears_the_field(bus):
    rebuild(bus, FieldStack(layers=[gradient_layer()]))
    bus.mark_dirty()
    rebuild(bus, FieldStack(layers=[]))
    assert np.allclose(read(bus), 0.0)


def test_stack_order_decides_the_winner(bus):
    first = gradient_layer(strength=1.0)
    second = gradient_layer(strength=0.25)
    rebuild(bus, FieldStack(layers=[first, second]))
    late = read(bus)[..., 0].max()

    bus.mark_dirty()
    rebuild(bus, FieldStack(layers=[second, first]))
    early = read(bus)[..., 0].max()

    assert late < early, "the later replace layer must win"


def test_a_clean_bus_does_no_work(bus):
    stack = FieldStack(layers=[gradient_layer()])
    assert rebuild(bus, stack) is True
    assert bus.dirty is False
    assert rebuild(bus, stack) is False
    assert bus.pass_count == 0


def test_marking_dirty_makes_it_rebuild(bus):
    stack = FieldStack(layers=[gradient_layer()])
    rebuild(bus, stack)
    bus.mark_dirty()
    assert rebuild(bus, stack) is True


def test_a_resolution_change_forces_a_rebuild(bus):
    stack = FieldStack(layers=[gradient_layer()])
    rebuild(bus, stack)
    assert bus.rebuild(stack, RES, RES, 0.5, FRAME) is True
    assert bus.resolution == (RES // 2, RES // 2)


def test_a_broken_layer_does_not_stop_the_others(bus):
    broken = FieldLayer(source="shader", params={"_file": "nope.frag"},
                        destination="force")
    good = gradient_layer()
    rebuild(bus, FieldStack(layers=[broken, good]))
    assert broken.error, "the broken layer must carry its own error"
    assert good.error is None
    assert read(bus)[..., 0].max() > 0.0, "the healthy layer stopped rendering"


def test_an_error_clears_when_the_layer_is_fixed(bus):
    layer = FieldLayer(source="shader", params={"_file": "nope.frag"})
    rebuild(bus, FieldStack(layers=[layer]))
    assert layer.error

    fixed = gradient_layer(uid=layer.uid)
    bus.mark_dirty()
    rebuild(bus, FieldStack(layers=[fixed]))
    assert fixed.error is None


def test_gpu_state_is_keyed_by_uid_not_by_position(bus):
    a = gradient_layer()
    b = gradient_layer()
    rebuild(bus, FieldStack(layers=[a, b]))
    source_a = bus.source_for(a)

    bus.mark_dirty()
    rebuild(bus, FieldStack(layers=[b, a]))
    assert bus.source_for(a) is source_a, "reordering rebuilt a source from scratch"


def test_a_source_dropped_from_the_stack_is_released(bus):
    layer = gradient_layer()
    rebuild(bus, FieldStack(layers=[layer]))
    bus.mark_dirty()
    rebuild(bus, FieldStack(layers=[]))
    assert bus.source_for(layer) is None


def test_pass_count_reports_what_ran(bus):
    rebuild(bus, FieldStack(layers=[gradient_layer(), gradient_layer()]))
    # Two procedural layers: a source pass and a composite pass each.
    assert bus.pass_count == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_bus_rebuild.py -q`
Expected: FAIL — `AttributeError: 'FieldBus' object has no attribute 'rebuild'`

- [ ] **Step 3: Write the implementation**

In `utilities/field_bus.py`, add to the import block:

```python
from services import field_sources
```

Add to `__init__` after `self._scratch_fbo = None`:

```python
        self._sources = {}      # layer uid -> source object
        self._dirty = True
        self._pass_count = 0
        self._scale = 0.0
```

Append these methods to `FieldBus`, after `composite_one`:

```python
    # -- the rebuild loop -----------------------------------------------

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def pass_count(self) -> int:
        """Passes run on the last rebuild. Zero when the bus was clean."""
        return self._pass_count

    def mark_dirty(self) -> None:
        self._dirty = True

    def source_for(self, layer):
        """The GPU-side source object for a layer, or None if it has none."""
        return self._sources.get(layer.uid)

    def brush_source(self):
        """The brush layer's accumulation buffer, or None if no brush layer."""
        for uid, source in self._sources.items():
            if source.__class__.__name__ == "_BrushSource":
                return source
        return None

    def reload_shaders(self) -> None:
        """Recompile the composite and every user shader. Bound to the V key."""
        if self._composite_vao is not None:
            self._composite_vao.release()
            self._composite_vao = None
        if self._composite is not None:
            self._composite.release()
            self._composite = None
        for source in self._sources.values():
            reload_fn = getattr(source, "reload", None)
            if reload_fn is not None:
                reload_fn()
        self.mark_dirty()

    def rebuild(self, stack, canvas_width: int, canvas_height: int,
                scale: float, frame) -> bool:
        """Rebuild every destination from the stack. Returns True if it ran.

        Clean and unchanged means no passes at all, which is what makes a
        static layer free in steady state.
        """
        layers = list(getattr(stack, "layers", []))
        self._prune_sources(layers)

        if not layers:
            self._pass_count = 0
            if self._tex is not None:
                self.clear()
                self._release_target()
            self._dirty = False
            return False

        resized = (scale != self._scale) or self._tex is None
        if resized:
            self._scale = scale
            self.ensure(canvas_width, canvas_height, scale)
            self._dirty = True
        else:
            self.ensure(canvas_width, canvas_height, scale)

        if not self._dirty:
            self._pass_count = 0
            return False

        self._pass_count = 0
        self.clear()
        for layer in layers:
            if not layer.enabled:
                layer.error = None
                continue
            source = self._sources.get(layer.uid)
            if source is None or source.__class__ is not self._class_for(layer):
                if source is not None:
                    source.release()
                source = field_sources.make_source(layer.source, self.ctx)
                self._sources[layer.uid] = source
            tex = source.evaluate(self, layer, frame)
            layer.error = getattr(source, "error", None)
            if tex is None:
                continue
            if tex is self._scratch:
                self._pass_count += 1
            self.composite_one(tex, layer)
            self._pass_count += 1

        self._dirty = False
        return True

    def _class_for(self, layer):
        probe = self._sources.get(layer.uid)
        if probe is None:
            return None
        # A layer whose source kind changed needs a different object.
        expected = field_sources.make_source(layer.source, self.ctx)
        cls = expected.__class__
        expected.release()
        return cls if probe.__class__ is cls else None

    def _prune_sources(self, layers) -> None:
        live = {l.uid for l in layers}
        for uid in [u for u in self._sources if u not in live]:
            self._sources.pop(uid).release()
```

Add to `cleanup`, before `self._release_target()`:

```python
        for source in self._sources.values():
            source.release()
        self._sources.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_bus_rebuild.py -q`
Expected: PASS, 12 passed

If `test_gpu_state_is_keyed_by_uid_not_by_position` fails, `_class_for` is building a throwaway source every frame — replace it with a plain kind check by storing the layer's source key alongside the object in `self._sources` as a `(key, source)` tuple, and compare keys instead.

- [ ] **Step 5: Run the whole suite to check nothing regressed**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, no new failures

- [ ] **Step 6: Commit**

```bash
git add utilities/field_bus.py tests/test_field_bus_rebuild.py
git commit -m "feat: the field bus rebuild loop, with a dirty flag and per-layer errors

Turning a layer off removes its contribution because the destination is
rebuilt from the stack rather than painted into.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Persistence in the physics config

**Files:**
- Modify: `services/config_saver.py`
- Test: `tests/test_field_stack_persistence.py`

**Interfaces:**
- Consumes: `stack_to_dict` / `stack_from_dict` from Task 1.
- Produces: `PhysicsConfig.field_stack: dict` (default `{}`), serialized under the `field_stack` key of `to_dict()` and read back by `from_dict()`. Function `legacy_brush_stack() -> dict` returning a one-layer stack dict for a config that has a `_fields.png` and no `field_stack`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_stack_persistence.py`:

```python
"""A stack travels with a preset, and a preset written before it still opens.

The legacy path matters most: every config on disk predates this feature, and
one with a companion _fields.png must come back as a brush layer holding that
texture rather than as an empty stack.
"""
from services.config_saver import PhysicsConfig, legacy_brush_stack
from state.field_stack import stack_from_dict


def test_a_new_config_has_an_empty_stack():
    assert PhysicsConfig().field_stack == {}


def test_the_stack_round_trips_through_the_config():
    stack = {"layers": [{"uid": "abc", "source": "noise", "mapping": "curl",
                         "destination": "force", "blend": "add",
                         "strength": 0.5, "blur": 1.0, "sign": 1.0,
                         "enabled": True, "params": {"scale": 8.0}}]}
    config = PhysicsConfig(field_stack=stack)
    restored = PhysicsConfig.from_dict(config.to_dict())
    assert restored.field_stack == stack


def test_a_config_written_before_this_feature_opens_with_an_empty_stack():
    d = PhysicsConfig().to_dict()
    d.pop("field_stack", None)
    assert PhysicsConfig.from_dict(d).field_stack == {}


def test_the_legacy_migration_is_one_brush_layer():
    stack = stack_from_dict(legacy_brush_stack())
    assert len(stack.layers) == 2
    assert [l.source for l in stack.layers] == ["brush", "brush"]
    assert {l.destination for l in stack.layers} == {"force", "strafe"}
    for layer in stack.layers:
        assert layer.mapping == "rg_direct"
        assert layer.blend == "replace"


def test_the_legacy_migration_reads_the_channels_the_old_field_used():
    stack = stack_from_dict(legacy_brush_stack())
    force = next(l for l in stack.layers if l.destination == "force")
    strafe = next(l for l in stack.layers if l.destination == "strafe")
    assert force.params.get("_channels") == "xy"
    assert strafe.params.get("_channels") == "zw"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_stack_persistence.py -q`
Expected: FAIL — `ImportError: cannot import name 'legacy_brush_stack'`

- [ ] **Step 3: Write the implementation**

In `services/config_saver.py`, add this field to `PhysicsConfig` immediately after `brain_settings`:

```python
    # The field injection layer stack. Empty on every file written before it
    # existed; a config with a companion _fields.png migrates to a brush layer.
    field_stack: dict = field(default_factory=dict)
```

Add this module-level function just above `class PhysicsConfig`:

```python
def legacy_brush_stack() -> dict:
    """The stack a pre-stack config with a _fields.png loads as.

    The old field texture packed force in .xy and strafe in .zw of one image,
    so it becomes two brush layers reading one buffer through different
    channels - which is what the historical behaviour was.
    """
    from state.field_stack import new_uid
    return {"layers": [
        {"uid": new_uid(), "enabled": True, "source": "brush",
         "params": {"_channels": "xy"}, "mapping": "rg_direct",
         "destination": "force", "blend": "replace",
         "strength": 1.0, "blur": 0.0, "sign": 1.0},
        {"uid": new_uid(), "enabled": True, "source": "brush",
         "params": {"_channels": "zw"}, "mapping": "rg_direct",
         "destination": "strafe", "blend": "replace",
         "strength": 1.0, "blur": 0.0, "sign": 1.0},
    ]}
```

In `PhysicsConfig.to_dict()`, add to the returned dict alongside `brain_settings`:

```python
            'field_stack': self.field_stack,
```

In `PhysicsConfig.from_dict()`, add alongside the `brain_settings` read:

```python
            field_stack=d.get('field_stack', {}),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_stack_persistence.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Run the config tests to check nothing regressed**

Run: `.venv/Scripts/python.exe -m pytest -q -k "config or physics_origin"`
Expected: PASS, no new failures

- [ ] **Step 6: Commit**

```bash
git add services/config_saver.py tests/test_field_stack_persistence.py
git commit -m "feat: the field stack travels with a preset

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: The `_channels` read in the composite

**Files:**
- Modify: `shaders/field/composite.frag`
- Modify: `utilities/field_bus.py`
- Test: `tests/test_field_channel_swizzle_gl.py`

**Interfaces:**
- Consumes: `composite_one` from Task 3.
- Produces: `composite_one` honours `layer.params["_channels"]`, one of `"xy"` (default) or `"zw"`, selecting which pair of the SOURCE texture the mapping reads. Adds uniform `int src_channels` to `composite.frag`.

**Why this is its own task:** the legacy migration in Task 7 produces two brush layers reading different channel pairs of one buffer. Without this, both read `.xy` and every pre-existing preset loses its strafe field silently.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_channel_swizzle_gl.py`:

```python
"""A layer may read either channel pair of its source.

The legacy field texture packed force in .xy and strafe in .zw of one image.
Both migrated layers read one brush buffer, so without a source swizzle the
strafe half of every pre-existing preset silently becomes a copy of its force
half.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from state.field_stack import FieldLayer  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402

RES = 16


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


@pytest.fixture
def bus(ctx):
    b = FieldBus(ctx)
    b.ensure(RES, RES, 1.0)
    yield b
    b.cleanup()


def packed(ctx):
    """force=(0.1,0.2) in .xy, strafe=(0.7,0.8) in .zw - the legacy layout."""
    data = np.tile(np.array([0.1, 0.2, 0.7, 0.8], dtype="f4"), (RES, RES, 1))
    return ctx.texture((RES, RES), 4, data.tobytes(), dtype="f4")


def read(bus):
    return np.frombuffer(bus.field_texture.read(), dtype="f4").reshape(RES, RES, 4)


def test_the_default_reads_xy(ctx, bus):
    bus.clear()
    bus.composite_one(packed(ctx), FieldLayer(
        mapping="rg_direct", destination="force", blend="replace"))
    out = read(bus)
    assert np.allclose(out[..., 0], 0.1)
    assert np.allclose(out[..., 1], 0.2)


def test_zw_reads_the_second_pair(ctx, bus):
    bus.clear()
    bus.composite_one(packed(ctx), FieldLayer(
        mapping="rg_direct", destination="strafe", blend="replace",
        params={"_channels": "zw"}))
    out = read(bus)
    assert np.allclose(out[..., 2], 0.7)
    assert np.allclose(out[..., 3], 0.8)


def test_the_legacy_pair_reconstructs_the_old_field(ctx, bus):
    src = packed(ctx)
    bus.clear()
    bus.composite_one(src, FieldLayer(
        mapping="rg_direct", destination="force", blend="replace",
        params={"_channels": "xy"}))
    bus.composite_one(src, FieldLayer(
        mapping="rg_direct", destination="strafe", blend="replace",
        params={"_channels": "zw"}))
    assert np.allclose(read(bus)[0, 0], [0.1, 0.2, 0.7, 0.8])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_channel_swizzle_gl.py -q`
Expected: FAIL — `test_zw_reads_the_second_pair` asserts 0.7 and finds 0.1

- [ ] **Step 3: Add the uniform to the composite shader**

In `shaders/field/composite.frag`, add after the `scalar_out` uniform:

```glsl
uniform int src_channels; // 0 = read .xy / .rgb, 1 = read .zw
```

Replace `scalar_at` with:

```glsl
vec4 fetch(vec2 uv){
    vec4 c = textureLod(src, uv, blur_lod);
    return (src_channels == 1) ? vec4(c.zw, 0.0, 1.0) : c;
}

float scalar_at(vec2 uv){
    return dot(fetch(uv).rgb, vec3(0.2126, 0.7152, 0.0722));
}
```

Replace the two `texture(src, texcoord)` reads in `main()` with `fetch(texcoord)`:

```glsl
    if (mapping == 0) {
        v = fetch(texcoord).rg;
    } else if (mapping == 1) {
        vec4 c = fetch(texcoord);
```

- [ ] **Step 4: Push the uniform from the bus**

In `utilities/field_bus.py`, inside `composite_one`, add after the `scalar_out` tryset:

```python
        tryset(prog, "src_channels",
               1 if layer.params.get("_channels") == "zw" else 0)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_channel_swizzle_gl.py tests/test_field_composite_gl.py -q`
Expected: PASS, 14 passed

- [ ] **Step 6: Commit**

```bash
git add shaders/field/composite.frag utilities/field_bus.py tests/test_field_channel_swizzle_gl.py
git commit -m "feat: a layer may read either channel pair of its source

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Wire the bus into the app and retire the override path

**Files:**
- Modify: `main.py`
- Modify: `simulation_runner.py:54-108`
- Modify: `state/preferences_state.py`
- Modify: `utilities/advanced_drawing.py` (delete the override methods)
- Modify: `ui/advanced_drawing_window.py:207-244` (delete the Shader Driven Field block)
- Modify: `command_handler.py` (V key reloads the bus)
- Test: `tests/test_field_bus_wiring.py`

**Interfaces:**
- Consumes: `FieldBus.rebuild` from Task 6.
- Produces: `App.field_bus: FieldBus`. `PreferencesState` gains `field_stack_window_open: bool = False` and `field_bus_scale: float = 0.5`. `SimulationRunner.__init__` gains `field_bus=None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_bus_wiring.py`:

```python
"""The bus is wired to the sim, and the override path is gone.

The old feature was gated on "is the Drawing Controls window open", which is
why closing the window silently stopped it. The bus must have no such gate.
"""
import inspect

import simulation_runner
from state import preferences_state
from utilities import advanced_drawing


def test_the_override_shader_path_is_gone():
    processor = advanced_drawing.AdvancedDrawingProcessor
    for name in ("process_override", "_ensure_override_resources",
                 "_cleanup_override", "get_available_override_shaders",
                 "resolve_override_shader_path"):
        assert not hasattr(processor, name), f"{name} should have been removed"


def test_the_override_preferences_are_gone():
    fields = preferences_state.PreferencesState.__dataclass_fields__
    assert "shader_driven_field" not in fields
    assert "field_override_shader" not in fields


def test_the_new_preferences_are_classified():
    fields = preferences_state.PreferencesState.__dataclass_fields__
    assert "field_bus_scale" in fields
    assert "field_stack_window_open" in fields
    classified = (set(preferences_state.UNDOABLE_FIELDS)
                  | set(preferences_state.NOT_UNDOABLE))
    assert {"field_bus_scale", "field_stack_window_open"} <= classified


def test_the_runner_takes_a_field_bus():
    params = inspect.signature(simulation_runner.SimulationRunner.__init__).parameters
    assert "field_bus" in params


def test_the_bus_rebuild_is_not_gated_on_a_window():
    src = inspect.getsource(simulation_runner.SimulationRunner.run_simulation_frame)
    assert "field_bus.rebuild" in src
    rebuild_line = next(l for l in src.splitlines() if "field_bus.rebuild" in l)
    assert "advanced_drawing_enabled" not in rebuild_line
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_bus_wiring.py -q`
Expected: FAIL — `assert not hasattr(processor, 'process_override')`

- [ ] **Step 3: Update preferences**

In `state/preferences_state.py`, delete these two lines:

```python
    shader_driven_field: bool = False  # Use a frag shader to override the field texture
    field_override_shader: str = "march.frag"  # Currently selected field override shader filename
```

Add in their place:

```python
    field_bus_scale: float = 0.5  # Field bus resolution as a fraction of the canvas
    field_stack_window_open: bool = False  # Whether the Field Stack window is shown
```

In the same file, remove any `shader_driven_field` or `field_override_shader` entries from `UNDOABLE_FIELDS` and `NOT_UNDOABLE`, and add:

```python
    "field_stack_window_open": "window visibility or panel collapse state",
```

to `NOT_UNDOABLE`, and add `"field_bus_scale"` to `UNDOABLE_FIELDS`.

- [ ] **Step 4: Delete the override path**

In `utilities/advanced_drawing.py`, delete the methods `process_override`, `get_available_override_shaders`, `resolve_override_shader_path`, `_ensure_override_resources`, `_cleanup_override`; delete `self._override_resources` and `self._override_shader_name` from `__init__`; delete the override branch at the end of `reload()`; and delete the `self._cleanup_override()` call at the top of `cleanup()`.

In `ui/advanced_drawing_window.py`, delete the block from `# === 11. Shader Driven Field ===` to the end of the `if prefs.shader_driven_field:` body, and delete the now-unused `from utilities.advanced_drawing import AdvancedDrawingProcessor` import.

- [ ] **Step 5: Wire the bus into the runner**

In `simulation_runner.py`, add `field_bus=None` to `__init__`'s parameters and `self.field_bus = field_bus` to its body.

Replace the whole `if adv_prefs.advanced_drawing_enabled and adv_prefs.shader_driven_field:` branch (the `process_override` call) with nothing, and immediately before the `if self.advanced_drawing_processor is not None:` block insert:

```python
        # The field bus rebuilds every frame from its stack, with no gate on any
        # window being open. Disabled while a tournament grid runs: tiles are
        # isolated worlds and one shared field would be scored instead of the
        # genome.
        if self.field_bus is not None:
            if ui_state.tournament.enabled:
                self.field_bus.rebuild(None, 0, 0, 1.0, None)
            else:
                self.field_bus.rebuild(
                    ui_state.field_stack,
                    self.sim.can.size[0], self.sim.can.size[1],
                    ui_state.preferences.field_bus_scale,
                    field_sources.FrameContext(
                        time=self.sim.time,
                        frame_count=self.sim.frame_count,
                        mouse=mouse_tex_coords,
                        prev_mouse=self.prev_mouse_tex_coords,
                        canvas_texture=self.sim.can,
                    ),
                )
```

Add at the top of `simulation_runner.py`:

```python
from services import field_sources
```

In `_run_physics_step`, replace the `field_texture` argument:

```python
            field_texture=(self.field_bus.field_texture
                           if self.field_bus is not None else None),
```

and in `_build_assemble_kwargs`, replace both `self.advanced_drawing_processor.field_texture` reads the same way.

- [ ] **Step 6: Wire the bus into the app**

In `main.py`, add the import beside the advanced drawing one:

```python
from utilities.field_bus import FieldBus
```

After `self.advanced_drawing_processor = AdvancedDrawingProcessor(self.ctx)`, add:

```python
        self.field_bus = FieldBus(self.ctx)
        self.ui.field_bus = self.field_bus
```

Pass it to the runner: add `field_bus=self.field_bus,` to the `SimulationRunner(...)` call.

In `cleanup`, add before the advanced drawing step:

```python
        self._step("field bus", self.field_bus.cleanup)
```

In the view-option branch that reads `self.advanced_drawing_processor.field_texture` for view options 4 and 5, read `self.field_bus.field_texture` instead.

In `command_handler.py`, find the `V` key handler that calls `adv_draw.reload()` and add beside it:

```python
        if self.field_bus is not None:
            self.field_bus.reload_shaders()
```

adding `field_bus=None` to `CommandHandler.__init__` and `self.field_bus = field_bus`, and passing `field_bus=self.field_bus` from `main.py`.

- [ ] **Step 7: Add the stack to UI state**

In the state container that `ui.get_state()` returns (the same object carrying `.preferences` and `.tournament`), add a `field_stack: FieldStack` field defaulting to `field_stack.FieldStack()`, so `ui_state.field_stack` resolves.

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_bus_wiring.py tests/test_undo_fields.py -q`
Expected: PASS

- [ ] **Step 9: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, no new failures. `tests/test_cold_import.py` must still pass — the bus must not import onnxruntime, tokenizers or cmaes.

- [ ] **Step 10: Commit**

```bash
git add main.py simulation_runner.py command_handler.py state/preferences_state.py utilities/advanced_drawing.py ui/advanced_drawing_window.py tests/test_field_bus_wiring.py
git commit -m "feat: wire the field bus in and retire the override path

The override was gated on whether the Drawing Controls window was open, so
closing it silently stopped the shader. The bus has no such gate, and is off
under a tournament because tiles are isolated worlds.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Retarget the brush and the field handler onto the brush source

**Files:**
- Modify: `services/field_handler.py`
- Modify: `simulation_runner.py`
- Test: `tests/test_field_handler_brush.py`

**Interfaces:**
- Consumes: `_BrushSource.snapshot()` / `.write()` from Task 5, `FieldBus.brush_source()` from Task 6.
- Produces: `FieldHandler.__init__` takes `field_bus` instead of `adv_draw`; `FieldHandler.apply_for_config(config, json_filepath, ui_state)` writes a legacy `_fields.png` into the brush source and installs `legacy_brush_stack()` when the config carries no `field_stack`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_handler_brush.py`:

```python
"""The brush's buffer is the field handler's target, and legacy PNGs still load.

Every preset on disk predates the stack. One with a companion _fields.png must
come back as two brush layers over that texture, or its field silently
vanishes.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from services.field_handler import FieldHandler
from state.field_stack import FieldStack, stack_from_dict
from services.config_saver import legacy_brush_stack


class FakeBrush:
    def __init__(self):
        self.data = None

    def snapshot(self):
        return self.data

    def write(self, data):
        self.data = np.array(data, dtype="f4")

    def clear(self):
        self.data = None


class FakeBus:
    def __init__(self, brush):
        self._brush = brush
        self.dirtied = False

    def brush_source(self):
        return self._brush

    def mark_dirty(self):
        self.dirtied = True


@pytest.fixture
def handler():
    brush = FakeBrush()
    bus = FakeBus(brush)
    sim = SimpleNamespace(get_canvas_dimensions=lambda: (8, 8))
    return FieldHandler(bus, sim), brush, bus


def test_a_snapshot_reads_the_brush_buffer(handler):
    h, brush, _ = handler
    brush.write(np.full((8, 8, 4), 0.3, dtype="f4"))
    ui = SimpleNamespace(preferences=SimpleNamespace(
        force_field_strength=1.0, strafe_field_strength=1.0))
    data, strengths = h.snapshot_with_strengths(ui)
    assert data is not None
    assert np.allclose(data, 0.3)
    assert strengths == (1.0, 1.0)


def test_a_zero_buffer_snapshots_as_none(handler):
    h, brush, _ = handler
    brush.write(np.zeros((8, 8, 4), dtype="f4"))
    ui = SimpleNamespace(preferences=SimpleNamespace(
        force_field_strength=1.0, strafe_field_strength=1.0))
    data, _ = h.snapshot_with_strengths(ui)
    assert data is None


def test_a_legacy_config_installs_two_brush_layers(handler):
    h, brush, bus = handler
    ui = SimpleNamespace(
        preferences=SimpleNamespace(force_field_strength=1.0,
                                    strafe_field_strength=1.0),
        field_stack=FieldStack())
    config = SimpleNamespace(field_stack={}, force_field_strength=1.0,
                             strafe_field_strength=1.0)
    h.cache.get = lambda *a, **k: np.full((8, 8, 4), 0.4, dtype="f4")

    h.apply_for_config(config, "whatever.json", ui)

    assert [l.source for l in ui.field_stack.layers] == ["brush", "brush"]
    assert np.allclose(brush.data, 0.4)
    assert bus.dirtied


def test_a_config_with_its_own_stack_is_used_verbatim(handler):
    h, _, _ = handler
    ui = SimpleNamespace(
        preferences=SimpleNamespace(force_field_strength=1.0,
                                    strafe_field_strength=1.0),
        field_stack=FieldStack())
    own = {"layers": [{"uid": "x", "source": "noise", "destination": "force"}]}
    config = SimpleNamespace(field_stack=own, force_field_strength=1.0,
                             strafe_field_strength=1.0)
    h.cache.get = lambda *a, **k: None

    h.apply_for_config(config, "whatever.json", ui)

    assert [l.source for l in ui.field_stack.layers] == ["noise"]


def test_a_config_with_neither_leaves_an_empty_stack(handler):
    h, _, _ = handler
    ui = SimpleNamespace(
        preferences=SimpleNamespace(force_field_strength=1.0,
                                    strafe_field_strength=1.0),
        field_stack=FieldStack(layers=list(stack_from_dict(legacy_brush_stack()).layers)))
    config = SimpleNamespace(field_stack={}, force_field_strength=1.0,
                             strafe_field_strength=1.0)
    h.cache.get = lambda *a, **k: None

    h.apply_for_config(config, "whatever.json", ui)

    assert ui.field_stack.layers == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_handler_brush.py -q`
Expected: FAIL — `TypeError` on `FieldHandler(bus, sim)`, or `AttributeError: 'FakeBus' object has no attribute 'field_texture'`

- [ ] **Step 3: Write the implementation**

In `services/field_handler.py`, change the constructor signature and body:

```python
    def __init__(self, field_bus, sim, param_lock_service=None):
        self.field_bus = field_bus
        self.sim = sim
        self.param_lock_service = param_lock_service
        self.cache = FieldTextureCache(max_size=20)
        ...
```

Replace the `_has_field_tex` property and every `self.adv_draw.<x>` call with brush-source equivalents:

```python
    @property
    def _brush(self):
        """The brush source's buffer, or None when no brush layer exists."""
        return self.field_bus.brush_source() if self.field_bus is not None else None

    @property
    def _has_field_tex(self):
        return self._brush is not None and self._brush.snapshot() is not None
```

- `self.adv_draw.snapshot_field_data()` becomes `self._brush.snapshot()`
- `self.adv_draw.write_field_data(x)` becomes `self._brush.write(x); self.field_bus.mark_dirty()`
- `self.adv_draw.clear_fields()` becomes `self._brush.clear(); self.field_bus.mark_dirty()`
- `self.adv_draw.ensure_initialized(...)` calls are deleted; `_BrushSource.write` sizes itself.

Replace `apply_for_config` with:

```python
    def apply_for_config(self, config, json_filepath, ui_state):
        """Install the config's layer stack and its brush buffer.

        A config with no stack but a companion _fields.png predates the stack
        and becomes two brush layers over that texture.
        """
        from state.field_stack import stack_from_dict
        from services.config_saver import legacy_brush_stack

        canvas_x, canvas_y = self.sim.get_canvas_dimensions()
        field_data = self.cache.get(json_filepath, canvas_y, canvas_x)

        stack_dict = getattr(config, "field_stack", {}) or {}
        if not stack_dict and field_data is not None:
            stack_dict = legacy_brush_stack()
        ui_state.field_stack.layers = list(stack_from_dict(stack_dict).layers)

        if field_data is not None:
            self._write_field_with_locks(field_data)
        elif self._brush is not None and not self._should_skip_clear():
            self._brush.clear()

        if self.field_bus is not None:
            self.field_bus.mark_dirty()

        if config.force_field_strength is not None:
            self._write_field_strengths(
                ui_state, config.force_field_strength, config.strafe_field_strength)
        elif field_data is None:
            self._write_field_strengths(ui_state, 1.0, 1.0)
```

In `_write_field_with_locks`, replace the `adv_draw` calls the same way, guarding on `self._brush is None` with an early return.

In `main.py`, change the `FieldHandler(...)` construction to pass `self.field_bus` in place of `self.advanced_drawing_processor`.

In `simulation_runner.py`, point the brush's draw pass at the brush source's framebuffer: replace `self.advanced_drawing_processor.process(...)`'s FBO target by passing `self.field_bus.brush_source()` in, and mark the bus dirty whenever a stroke was drawn.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_handler_brush.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, no new failures

- [ ] **Step 6: Commit**

```bash
git add services/field_handler.py simulation_runner.py main.py tests/test_field_handler_brush.py
git commit -m "feat: the brush owns its buffer, and legacy field PNGs migrate to it

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: The layer list window

**Files:**
- Create: `ui/field_stack_window.py`
- Modify: `ui/core.py`
- Modify: `ui/menu_bar.py`
- Test: `tests/test_field_stack_window_render.py`

**Interfaces:**
- Consumes: `field_sources.descriptors()` / `params_for()` from Task 4, `FieldLayer` from Task 1.
- Produces: `FieldStackWindowMixin.render_field_stack_window()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_stack_window_render.py`:

```python
"""The layer list draws, and its controls are actually reachable.

Two traps this observes, both documented in CLAUDE.md. ImGui clips to the
WINDOW, not the display, so the host must be sized taller than the panel or
half the controls draw no vertices and the test passes on nothing. And a
duplicate ImGui label kills one of the two widgets silently, so ids are
asserted to be unique.
"""
from __future__ import annotations

import pytest

imgui = pytest.importorskip("imgui_bundle").imgui
from imgui_bundle import immapp  # noqa: E402

from state.field_stack import FieldLayer, FieldStack  # noqa: E402


@pytest.fixture
def drawn():
    """Render the window once offscreen and return the labels it drew."""
    from tests.helpers.imgui_harness import capture_labels  # see step 3
    return capture_labels


def test_an_empty_stack_still_offers_add_layer(drawn):
    labels = drawn(FieldStack())
    assert any("Add layer" in l for l in labels)


def test_a_layer_draws_its_source_and_destination(drawn):
    labels = drawn(FieldStack(layers=[FieldLayer(source="noise",
                                                 destination="force")]))
    assert any("noise" in l for l in labels)
    assert any("force" in l for l in labels)


def test_a_layer_error_is_drawn(drawn):
    layer = FieldLayer(source="shader")
    layer.error = "0:31 'noize' : no matching function"
    labels = drawn(FieldStack(layers=[layer]))
    assert any("noize" in l for l in labels), "the compile log was not shown"


def test_no_two_widgets_share_an_id(drawn):
    labels = drawn(FieldStack(layers=[FieldLayer(), FieldLayer(), FieldLayer()]))
    assert len(labels) == len(set(labels)), "duplicate ImGui id kills a widget"
```

- [ ] **Step 2: Write the render harness**

Create `tests/helpers/__init__.py` (empty file) and `tests/helpers/imgui_harness.py`:

```python
"""Draw one offscreen ImGui frame and collect the labels it produced.

The host window is sized well past the panel because ImGui clips a window's
contents to the WINDOW rather than to the display: at a default size the
settings column falls outside it and draws no vertices at all, which turns
"the control rendered" into a test of nothing.

`imgui.ini` is disabled because ImGui restores each window's saved size and
position from it - the file is gitignored, so a test that reads it passes on a
fresh clone and fails on a machine that has run the app.
"""
from __future__ import annotations

from imgui_bundle import imgui, immapp

HOST_SIZE = (1400, 1800)


def capture_labels(stack):
    from ui.field_stack_window import FieldStackWindowMixin

    collected: list[str] = []

    class Harness(FieldStackWindowMixin):
        def __init__(self):
            self.state = _State(stack)
            self.field_bus = None

        def _delayed_tooltip(self, _text):
            pass

    def gui():
        imgui.get_io().set_ini_filename("")
        imgui.set_next_window_size(imgui.ImVec2(*HOST_SIZE))
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        Harness().render_field_stack_window(collect=collected)
        immapp.manual_render.get_runner_params().app_shall_exit = True

    immapp.run(gui_function=gui, window_size=HOST_SIZE,
               window_title="harness", with_implot=False)
    return collected


class _State:
    def __init__(self, stack):
        self.field_stack = stack
        self.preferences = _Prefs()


class _Prefs:
    field_bus_scale = 0.5
    field_stack_window_open = True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_stack_window_render.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.field_stack_window'`

- [ ] **Step 4: Write the window**

Create `ui/field_stack_window.py`:

```python
"""Field Stack window: the ordered list of injection layers."""
from imgui_bundle import imgui

from services import field_sources
from state.field_stack import (
    BLENDS, DESTINATIONS, MAPPINGS, FieldLayer,
)
from ui import layout
from ui.notices import BAD

SCALES = [("1/1", 1.0), ("1/2", 0.5), ("1/4", 0.25)]


class FieldStackWindowMixin:
    """Mixin for the field injection layer list. Combined into UI."""

    def render_field_stack_window(self, collect=None):
        expanded, opened = imgui.begin("Field Stack", True)
        if not opened:
            self.state.preferences.field_stack_window_open = False
            imgui.end()
            return
        if expanded:
            self._draw_field_stack(collect)
        imgui.end()

    # -- internals ------------------------------------------------------

    def _label(self, text, collect):
        if collect is not None:
            collect.append(text)
        return text

    def _draw_field_stack(self, collect):
        prefs = self.state.preferences
        stack = self.state.field_stack
        layout.push_settings_width()

        names = [n for n, _ in SCALES]
        current = min(range(len(SCALES)),
                      key=lambda i: abs(SCALES[i][1] - prefs.field_bus_scale))
        changed, current = imgui.combo(
            self._label("Bus Resolution##fieldbus", collect), current, names)
        if changed:
            prefs.field_bus_scale = SCALES[current][1]
            if self.field_bus is not None:
                self.field_bus.mark_dirty()
        self._delayed_tooltip(
            "Resolution the injection layers are composited at, "
            "as a fraction of the canvas.")

        imgui.separator()

        remove_index = None
        for index, layer in enumerate(stack.layers):
            if self._draw_layer_row(index, layer, collect):
                remove_index = index
        if remove_index is not None:
            stack.layers.pop(remove_index)
            self._mark_dirty()

        imgui.separator()
        if imgui.button(self._label("+ Add layer##fieldstack", collect)):
            stack.layers.append(FieldLayer())
            self._mark_dirty()

        imgui.same_line()
        passes = self.field_bus.pass_count if self.field_bus is not None else 0
        res = self.field_bus.resolution if self.field_bus is not None else (0, 0)
        imgui.text_disabled(self._label(
            f"{passes} passes | {res[0]}x{res[1]}", collect))

        imgui.pop_item_width()

    def _draw_layer_row(self, index, layer, collect) -> bool:
        uid = layer.uid
        changed_any = False

        changed, layer.enabled = imgui.checkbox(f"##en{uid}", layer.enabled)
        changed_any |= changed
        imgui.same_line()

        keys = [d.key for d in field_sources.descriptors()]
        pos = keys.index(layer.source) if layer.source in keys else 0
        changed, pos = imgui.combo(
            self._label(f"{layer.source}##src{uid}", collect), pos, keys)
        if changed:
            layer.source = keys[pos]
            layer.params = {}
            changed_any = True

        imgui.same_line()
        changed, layer.mapping = self._enum(
            layer.mapping, MAPPINGS, f"map{uid}", collect)
        changed_any |= changed

        imgui.same_line()
        changed, layer.destination = self._enum(
            layer.destination, DESTINATIONS, f"dst{uid}", collect)
        changed_any |= changed

        imgui.same_line()
        changed, layer.blend = self._enum(
            layer.blend, BLENDS, f"bl{uid}", collect)
        changed_any |= changed

        changed, layer.strength = imgui.slider_float(
            self._label(f"Strength##{uid}", collect), layer.strength, 0.0, 4.0)
        changed_any |= changed

        if layer.error:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*BAD))
            imgui.text_wrapped(self._label(layer.error, collect))
            imgui.pop_style_color()

        changed_any |= self._draw_layer_params(layer, collect)

        remove = imgui.button(self._label(f"Remove##{uid}", collect))
        imgui.separator()

        if changed_any:
            self._mark_dirty()
        return remove

    def _draw_layer_params(self, layer, collect) -> bool:
        changed_any = False
        for param in field_sources.params_for(layer):
            key = param.name
            value = layer.params.get(key, param.default[0]
                                     if param.components == 1
                                     else list(param.default))
            tag = self._label(f"{param.label}##{layer.uid}{key}", collect)
            if param.kind == "bool":
                changed, value = imgui.checkbox(tag, bool(value))
            elif param.kind == "int":
                changed, value = imgui.slider_int(
                    tag, int(value), int(param.lo), int(param.hi))
            elif param.components == 1:
                changed, value = imgui.slider_float(
                    tag, float(value), param.lo, param.hi)
            elif param.kind == "color" and param.components == 3:
                changed, value = imgui.color_edit3(tag, list(value))
            else:
                changed, value = imgui.slider_float2(
                    tag, list(value)[:2], param.lo, param.hi)
            if changed:
                layer.params[key] = value
                changed_any = True
        return changed_any

    def _enum(self, current, options, tag, collect):
        pos = options.index(current) if current in options else 0
        changed, pos = imgui.combo(
            self._label(f"{current}##{tag}", collect), pos, list(options))
        return changed, options[pos]

    def _mark_dirty(self):
        if getattr(self, "field_bus", None) is not None:
            self.field_bus.mark_dirty()
```

- [ ] **Step 5: Register the mixin and the menu entry**

In `ui/core.py`, add `from .field_stack_window import FieldStackWindowMixin` beside the other window imports, add `FieldStackWindowMixin,` to the `class UI(...)` base list beside `AdvancedDrawingWindowMixin`, and add beside the advanced drawing dispatch at line 719:

```python
        if self.show_sidebar and self.state.preferences.field_stack_window_open:
            self.render_field_stack_window()
```

In `ui/menu_bar.py`, beside the Advanced Drawing checkbox at line 376, add:

```python
                _, self.state.preferences.field_stack_window_open = imgui.checkbox(
                    "Field Stack",
                    self.state.preferences.field_stack_window_open
                )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_stack_window_render.py -q`
Expected: PASS, 4 passed

- [ ] **Step 7: Check label widths**

Run: `.venv/Scripts/python.exe -m pytest tests/test_label_widths.py -q`
Expected: PASS. If a label exceeds `layout.WIDEST_LABEL`, shorten it — do not widen the constant.

- [ ] **Step 8: Commit**

```bash
git add ui/field_stack_window.py ui/core.py ui/menu_bar.py tests/helpers/__init__.py tests/helpers/imgui_harness.py tests/test_field_stack_window_render.py
git commit -m "feat: the field stack layer list window

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Thumbnails and the inspect panel

**Files:**
- Modify: `utilities/field_bus.py`
- Modify: `ui/field_stack_window.py`
- Test: `tests/test_field_inspect.py`

**Interfaces:**
- Consumes: `FieldBus.rebuild` from Task 6.
- Produces: `FieldBus.thumbnail_for(layer)` returning a `moderngl.Texture` or `None`; `FieldBus.set_thumbnails_enabled(bool)`; `FieldBus.inspect(layer, view)` where `view` is `"source" | "mapped" | "destination"`, returning a texture. `FieldStackWindowMixin` draws a 96px `imgui.image` per row and an Inspect child window.

- [ ] **Step 1: Write the failing test**

Create `tests/test_field_inspect.py`:

```python
"""Previews cost nothing when nobody is looking.

The row thumbnails only fill while the window is open, and inspect only
renders the one layer being looked at - a preview that ran unconditionally
would put the feature's cost back exactly where the design removed it.
"""
from __future__ import annotations

import pytest

moderngl = pytest.importorskip("moderngl")

from state.field_stack import FieldLayer, FieldStack  # noqa: E402
from services.field_sources import FrameContext  # noqa: E402
from utilities.field_bus import FieldBus  # noqa: E402

RES = 32
FRAME = FrameContext(time=0.0, frame_count=0, mouse=(0.5, 0.5),
                     prev_mouse=(0.5, 0.5), canvas_texture=None)


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


@pytest.fixture
def bus(ctx):
    b = FieldBus(ctx)
    yield b
    b.cleanup()


def a_layer():
    return FieldLayer(source="gradient", mapping="luminance",
                      destination="force", blend="replace",
                      params={"shape": 1, "falloff": 1.0, "centre": (0.5, 0.5)})


def test_no_thumbnail_when_previews_are_off(bus):
    layer = a_layer()
    bus.set_thumbnails_enabled(False)
    bus.rebuild(FieldStack(layers=[layer]), RES, RES, 1.0, FRAME)
    assert bus.thumbnail_for(layer) is None


def test_a_thumbnail_appears_when_previews_are_on(bus):
    layer = a_layer()
    bus.set_thumbnails_enabled(True)
    bus.rebuild(FieldStack(layers=[layer]), RES, RES, 1.0, FRAME)
    thumb = bus.thumbnail_for(layer)
    assert thumb is not None
    assert max(thumb.size) <= 96


def test_turning_previews_off_releases_the_thumbnails(bus):
    layer = a_layer()
    bus.set_thumbnails_enabled(True)
    bus.rebuild(FieldStack(layers=[layer]), RES, RES, 1.0, FRAME)
    bus.set_thumbnails_enabled(False)
    assert bus.thumbnail_for(layer) is None


def test_inspect_source_returns_the_raw_source(bus):
    layer = a_layer()
    bus.set_thumbnails_enabled(True)
    bus.rebuild(FieldStack(layers=[layer]), RES, RES, 1.0, FRAME)
    assert bus.inspect(layer, "source") is not None


def test_inspect_destination_returns_the_field(bus):
    layer = a_layer()
    bus.rebuild(FieldStack(layers=[layer]), RES, RES, 1.0, FRAME)
    assert bus.inspect(layer, "destination") is bus.field_texture


def test_inspect_on_a_broken_layer_returns_none(bus):
    layer = FieldLayer(source="shader", params={"_file": "nope.frag"})
    bus.rebuild(FieldStack(layers=[layer]), RES, RES, 1.0, FRAME)
    assert bus.inspect(layer, "source") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_inspect.py -q`
Expected: FAIL — `AttributeError: 'FieldBus' object has no attribute 'set_thumbnails_enabled'`

- [ ] **Step 3: Add previews to the bus**

In `utilities/field_bus.py`, add to `__init__`:

```python
        self._thumbs = {}           # layer uid -> (texture, framebuffer)
        self._thumbs_on = False
        self._inspect_view = {}     # layer uid -> last source texture handle
```

Add these methods after `rebuild`:

```python
    THUMB_MAX = 96

    def set_thumbnails_enabled(self, on: bool) -> None:
        """Row previews only fill while the layer list is open."""
        if on == self._thumbs_on:
            return
        self._thumbs_on = on
        if not on:
            for tex, fbo in self._thumbs.values():
                fbo.release()
                tex.release()
            self._thumbs.clear()
        self.mark_dirty()

    def thumbnail_for(self, layer):
        entry = self._thumbs.get(layer.uid)
        return entry[0] if entry else None

    def inspect(self, layer, view: str):
        """The texture the Inspect panel should draw for `layer`."""
        if view == "destination":
            return self._tex
        if layer.error:
            return None
        return self._inspect_view.get(layer.uid)

    def _capture_thumbnail(self, uid, src_tex) -> None:
        if not self._thumbs_on:
            return
        w, h = src_tex.size
        longest = max(w, h) or 1
        scale = min(1.0, self.THUMB_MAX / longest)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        entry = self._thumbs.get(uid)
        if entry is None or entry[0].size != size:
            if entry is not None:
                entry[1].release()
                entry[0].release()
            tex = self.ctx.texture(size, 4, dtype="f4")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._thumbs[uid] = (tex, self.ctx.framebuffer(color_attachments=[tex]))
        tex, fbo = self._thumbs[uid]
        self._ensure_blit()
        src_tex.use(location=0)
        tryset(self._blit, "src", 0)
        fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self._blit_vao.render(mode=moderngl.TRIANGLE_FAN, vertices=4)

    def _ensure_blit(self) -> None:
        if getattr(self, "_blit", None) is not None:
            return
        self._blit = self.ctx.program(
            vertex_shader=read_shader("shaders/canvas.vert"),
            fragment_shader=(
                "#version 430\n"
                "in vec2 texcoord;\nout vec4 fragColor;\n"
                "uniform sampler2D src;\n"
                "void main(){ fragColor = texture(src, texcoord); }\n"),
        )
        self._blit_vao = self.ctx.vertex_array(self._blit, [])
```

In `rebuild`, immediately after `tex = source.evaluate(self, layer, frame)` and the error assignment, add:

```python
            if tex is not None:
                self._inspect_view[layer.uid] = tex
                self._capture_thumbnail(layer.uid, tex)
```

Add to `_prune_sources`, inside the removal loop:

```python
            entry = self._thumbs.pop(uid, None)
            if entry is not None:
                entry[1].release()
                entry[0].release()
            self._inspect_view.pop(uid, None)
```

Add to `cleanup`, before the sources loop:

```python
        self.set_thumbnails_enabled(False)
        if getattr(self, "_blit_vao", None) is not None:
            self._blit_vao.release()
            self._blit_vao = None
        if getattr(self, "_blit", None) is not None:
            self._blit.release()
            self._blit = None
```

- [ ] **Step 4: Draw them in the window**

In `ui/field_stack_window.py`, add at the top of `_draw_field_stack`:

```python
        if self.field_bus is not None:
            self.field_bus.set_thumbnails_enabled(True)
```

and in `render_field_stack_window`'s `not opened` branch, before `imgui.end()`:

```python
            if getattr(self, "field_bus", None) is not None:
                self.field_bus.set_thumbnails_enabled(False)
```

In `_draw_layer_row`, before the enable checkbox:

```python
        thumb = (self.field_bus.thumbnail_for(layer)
                 if getattr(self, "field_bus", None) is not None else None)
        if thumb is not None:
            imgui.image(thumb.glo, imgui.ImVec2(48, 48))
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.image(thumb.glo, imgui.ImVec2(256, 256))
                imgui.end_tooltip()
            if imgui.is_item_clicked():
                self._inspect_uid = uid
            imgui.same_line()
```

And append to `_draw_field_stack`, before `imgui.pop_item_width()`:

```python
        inspect_uid = getattr(self, "_inspect_uid", None)
        if inspect_uid is not None and self.field_bus is not None:
            match = next((l for l in stack.layers if l.uid == inspect_uid), None)
            if match is None:
                self._inspect_uid = None
            else:
                views = ["source", "mapped", "destination"]
                view = getattr(self, "_inspect_view_mode", 0)
                changed, view = imgui.combo(
                    self._label("Inspect##fieldinspect", collect), view, views)
                self._inspect_view_mode = view
                tex = self.field_bus.inspect(match, views[view])
                if tex is not None:
                    imgui.image(tex.glo, imgui.ImVec2(384, 384))
                if imgui.button(self._label("Close##fieldinspect", collect)):
                    self._inspect_uid = None
```

Note: `"mapped"` falls back to the source texture in Stage 1. The arrow-overlay rendering of a vector field arrives with Stage 2, where optical flow makes it necessary.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_field_inspect.py tests/test_field_stack_window_render.py -q`
Expected: PASS, 10 passed

- [ ] **Step 6: Commit**

```bash
git add utilities/field_bus.py ui/field_stack_window.py tests/test_field_inspect.py
git commit -m "feat: row thumbnails and an inspect panel for field layers

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 13: Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Test: `.venv/Scripts/python.exe -m pytest -q` (the full suite)

- [ ] **Step 1: Add the caveats**

In `CLAUDE.md`, add a new subsection under "Important Caveats", after "The encoder and the capture":

```markdown
### Field injection

- **The field bus is STATELESS PER FRAME, and any source that needs memory owns
  its own buffer.** Every destination is cleared and rebuilt from the layer
  stack, which is the only reason turning a layer off removes its
  contribution. The previous override shader painted one persistent texture and
  nothing cleared it, so unticking the box left the last frame it drew on
  screen forever — and enabling it had force-switched the draw target, so the
  Clear button no longer named the thing that had been turned on. The brush is
  not special-cased: it is a source with a private accumulation buffer.

- **`force` and `strafe` are CHANNEL PAIRS of one RGBA32F texture, written
  under a colour mask, and the four blend modes are native GL state.**
  `replace` is blending off, `add` is `ONE,ONE`, `multiply` is
  `DST_COLOR,ZERO`, `max` is `blend_equation=MAX`. So there is exactly one
  composite shader, and `get_field()` in `entity_update.glsl` never changed.
  A `multiply` layer FIRST in a stack composites against the zero clear and
  yields zero; multiply is for layer 2 and after.

- **The bus runs BELOW canvas resolution by default and its texture filter is
  therefore `LINEAR`, not `NEAREST`.** A forcing field is smooth. Nothing else
  needs to know: `get_field()` takes its aspect correction from
  `textureSize()`, so a uniformly scaled texture reads correctly.

- **The bus is OFF under a tournament grid.** Tiles are isolated small worlds,
  so one field across the canvas is shared by every tile — the optimizer would
  score the injected texture rather than the genome, and the entries it
  admitted would be unreproducible. Same discipline as `color_by_cohort` being
  forced off.

- **Source shader annotation is OPT-IN, and that is what keeps old files
  working.** `parse_shader_params` reads `uniform float x; // 0..1 = 0.5 "X"`;
  an unannotated uniform yields no UI and keeps its GLSL default, so a `.frag`
  written before the parser existed still runs. A malformed annotation is
  skipped rather than raised — a shader is a user's text file and must never
  fail to load over a comment.

- **A layer's GPU state is keyed by `FieldLayer.uid`, never by its position.**
  Reordering the stack must not recompile a shader or discard the brush's
  paint.

- **A config with a `_fields.png` and no `field_stack` migrates to TWO brush
  layers**, reading `.xy` and `.zw` of one buffer through the composite's
  `src_channels` swizzle. Without the swizzle both read `.xy` and every
  pre-existing preset silently loses its strafe field.
```

- [ ] **Step 2: Document the feature for users**

In `README.md`, add a section after the tournament walkthrough describing: opening Extras > Field Stack, adding a layer, what each of source / mapping / destination / blend does in one line each, the `.frag` annotation syntax with the four-line example from Task 2's docstring, where to put a `.frag` (`Documents/Fluoddity/shaders/`), and that `V` reloads them.

- [ ] **Step 3: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, no failures

- [ ] **Step 4: Launch the app and confirm by hand**

Run: `.venv/Scripts/python.exe main.py`

Confirm, and report what you saw rather than that you ran it:
1. Extras > Field Stack opens the window; `+ Add layer` adds a noise layer and the particles react.
2. Unticking the layer's checkbox returns the sim to unforced motion **immediately** — this is the reported defect.
3. Closing the Field Stack window leaves the injection running (the old gate is gone).
4. A layer set to `shader` with a nonexistent file draws a red row, and other layers keep rendering.
5. Loading a preset that has a `_fields.png` shows two brush layers and the field looks as it did before.
6. Bus Resolution 1/4 visibly costs less and the field still looks smooth.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: the field injection bus and its caveats

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review Notes

**Spec coverage.** Every Stage 1 item in the spec maps to a task: data model (1), `.frag` contract (2), composite and blend modes (3), source registry and procedural sources (4), the six sources (4–5), rebuild loop and dirty flag (6), persistence and legacy migration (7, 10), the channel swizzle the migration needs (8), wiring and the tournament gate (9), the layer list (11), thumbnails and inspect (12), documentation (13). Bus resolution is covered in 3 and surfaced in 11. `python -m tools.measure_field_bus` is deliberately **not** in Stage 1 — the spec attaches it to the Stage 4 cost gate, and there is nothing to measure until a consumer exists.

**Known gaps, called out rather than hidden.**
- The `mapped` inspect view falls back to the source texture in Stage 1. The arrow-overlay version arrives with Stage 2, where optical flow makes it load-bearing.
- Task 10's brush-draw retarget in `simulation_runner.py` is described rather than shown, because it depends on how `AdvancedDrawingProcessor.process` is left after Task 9's deletions. Read that method before starting Task 10.
- `tests/helpers/imgui_harness.py` uses `immapp.run` with a one-shot exit. If the installed `imgui_bundle` exposes a different offscreen entry point, adapt the harness — the assertions are what matter, not the runner.
