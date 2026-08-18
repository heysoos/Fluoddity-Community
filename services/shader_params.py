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
