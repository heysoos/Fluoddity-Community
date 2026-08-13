"""UI state for audio-driven modulation.

Named audio_in_state so it cannot be confused with the MIDI-out side. The
persisted set is an explicit allowlist: most of this dataclass is one-shot
commands and view buffers, and persisting one would replay a command on load.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # annotations only - see below
    from services.audio_mapping import Mapping

# `services` is NOT imported at module scope. Importing it runs
# services/__init__, which reaches ui, which imports back from state - and this
# module is reached while state/__init__ is still part-built. The two functions
# that need Mapping and ShaperParams import them when called.

# `enabled` is deliberately absent: opening the app must never start capturing.
PERSISTED_FIELDS: tuple[str, ...] = (
    "mappings", "brain_mappings", "strengths", "global_strength",
    "auto_gain", "device_name", "modulate", "muted",
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

    # The master bypass, and the per-target one. Both silence the modulation
    # while leaving capture running, so the traces keep moving and you can see
    # what you would be turning back on.
    #
    # Separate from each Mapping's own `enabled`, deliberately: muting a row by
    # clearing its mappings' flags would resurrect the bands the user had
    # switched off individually when the row came back.
    modulate: bool = True
    muted: dict[str, bool] = field(default_factory=dict)

    # One-shot commands, read and cleared by the orchestrator.
    request_start: bool = False
    request_stop: bool = False

    # Which row's drawer is open, by target key, and which band tab it shows
    # ("" is the total). One drawer at a time, as in the boids panel, so only
    # one row's traces are ever drawn. The physics slider's Audio... context
    # item writes open_target to jump straight to a row.
    open_target: str = ""
    open_band: str = ""

    # Live view state, written by the orchestrator for the panel to draw.
    status: str = "idle"                # "idle" | "active" | "error" | "waiting"
    last_error: str = ""
    # The newest SignalSnapshot. The UI is passive and owns no service, so the
    # orchestrator hands it the frame's analysis rather than the panel reaching
    # into the capture thread.
    snapshot: object = None
    # Each applied mapping's post-shaper signal, by id(mapping). Only the
    # modulation maths sees it, and a drawer that drew the raw band instead
    # would show nothing of what its shaper does.
    shaped: dict = field(default_factory=dict)


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
    from services.audio_mapping import MODES, Mapping
    from services.audio_shapers import SHAPER_KINDS, ShaperParams

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
        "modulate": bool(state.modulate),
        # Only the muted ones, so a rig does not carry a row per parameter.
        "muted": sorted(k for k, v in state.muted.items() if v),
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
    if isinstance(data.get("modulate"), bool):
        state.modulate = data["modulate"]
    if isinstance(data.get("muted"), list):
        state.muted = {k: True for k in data["muted"] if isinstance(k, str)}
