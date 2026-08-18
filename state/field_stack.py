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
    """Rebuild a stack.

    An unknown enum value falls back to the default rather than raising, so a
    file written by a build with more sources than this one still opens.
    """
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
