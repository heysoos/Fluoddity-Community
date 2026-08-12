"""UI state for audio-driven modulation.

Named audio_in_state so it cannot be confused with the MIDI-out side. The
persisted set is an explicit allowlist: most of this dataclass is one-shot
commands and view buffers, and persisting one would replay a command on load.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from services.audio_mapping import MODES, Mapping
from services.audio_shapers import SHAPER_KINDS, ShaperParams

# `enabled` is deliberately absent: opening the app must never start capturing.
PERSISTED_FIELDS: tuple[str, ...] = (
    "mappings", "brain_mappings", "strengths", "global_strength",
    "auto_gain", "device_name",
)

_STRENGTH_MAX = 2.0


@dataclass
class AudioInState:
    enabled: bool = False               # is capture running
    show_window: bool = False           # is the panel open

    device_name: str = ""               # last chosen input, by name
    auto_gain: bool = True

    # Physics mappings, and brain mappings keyed by modality name so a rig
    # built for one brain is waiting when you switch back to it.
    mappings: list[Mapping] = field(default_factory=list)
    brain_mappings: dict[str, list[Mapping]] = field(default_factory=dict)

    strengths: dict[str, float] = field(default_factory=dict)
    global_strength: float = 1.0

    # One-shot commands, read and cleared by the orchestrator.
    request_start: bool = False
    request_stop: bool = False
    # Set by the physics slider's context menu; opens the panel on this row.
    open_target: str = ""

    # Live view state, written by the orchestrator for the panel to draw.
    status: str = "idle"                # "idle" | "active" | "error"
    last_error: str = ""


def _mapping_to_dict(m: Mapping) -> dict:
    return {
        "signal": m.signal, "target": m.target, "mode": m.mode,
        "depth": float(m.depth), "gain": float(m.gain),
        "enabled": bool(m.enabled),
        "shaper": {
            "kind": m.shaper.kind, "attack": m.shaper.attack,
            "release": m.shaper.release, "threshold": m.shaper.threshold,
            "hold": m.shaper.hold, "rate_min": m.shaper.rate_min,
            "rate_max": m.shaper.rate_max, "wave": m.shaper.wave,
        },
    }


def _mapping_from_dict(d) -> Mapping | None:
    """None for anything malformed, so one bad row cannot lose the rig."""
    if not isinstance(d, dict):
        return None
    signal, target = d.get("signal"), d.get("target")
    if not isinstance(signal, str) or not isinstance(target, str):
        return None
    mode = d.get("mode", "add")
    if mode not in MODES:
        return None
    sd = d.get("shaper") or {}
    shaper = ShaperParams()
    if isinstance(sd, dict):
        kind = sd.get("kind", "none")
        shaper = replace(
            shaper,
            kind=kind if kind in SHAPER_KINDS else "none",
            **{k: float(sd[k]) for k in
               ("attack", "release", "threshold", "hold", "rate_min", "rate_max")
               if isinstance(sd.get(k), (int, float))},
        )
        if isinstance(sd.get("wave"), str):
            shaper.wave = sd["wave"]
    try:
        return Mapping(
            signal=signal, target=target, mode=mode,
            depth=float(d.get("depth", 0.5)), gain=float(d.get("gain", 1.0)),
            shaper=shaper, enabled=bool(d.get("enabled", True)),
        )
    except (TypeError, ValueError):
        return None


def to_dict(state: AudioInState) -> dict:
    return {
        "mappings": [_mapping_to_dict(m) for m in state.mappings],
        "brain_mappings": {k: [_mapping_to_dict(m) for m in v]
                           for k, v in state.brain_mappings.items()},
        "strengths": {k: float(v) for k, v in state.strengths.items()},
        "global_strength": float(state.global_strength),
        "auto_gain": bool(state.auto_gain),
        "device_name": str(state.device_name),
    }


def _clamp_strength(v) -> float:
    return min(_STRENGTH_MAX, max(0.0, float(v)))


def apply_dict(state: AudioInState, data: dict) -> None:
    """Apply a stored rig. A missing key keeps the current value."""
    if not isinstance(data, dict):
        return

    if isinstance(data.get("mappings"), list):
        state.mappings = [m for m in
                          (_mapping_from_dict(d) for d in data["mappings"])
                          if m is not None]

    if isinstance(data.get("brain_mappings"), dict):
        state.brain_mappings = {
            k: [m for m in (_mapping_from_dict(d) for d in v) if m is not None]
            for k, v in data["brain_mappings"].items() if isinstance(v, list)
        }

    if isinstance(data.get("strengths"), dict):
        state.strengths = {k: _clamp_strength(v)
                           for k, v in data["strengths"].items()
                           if isinstance(v, (int, float))}

    if isinstance(data.get("global_strength"), (int, float)):
        state.global_strength = _clamp_strength(data["global_strength"])
    if isinstance(data.get("auto_gain"), bool):
        state.auto_gain = data["auto_gain"]
    if isinstance(data.get("device_name"), str):
        state.device_name = data["device_name"]
