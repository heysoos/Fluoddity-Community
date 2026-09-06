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
    "auto_gain", "device_name", "modulate", "muted", "bands",
    "release_seconds", "rate_scale",
    "channels", "channel_mappings", "feed",
)

_STRENGTH_MAX = 2.0


@dataclass
class AudioInState:
    enabled: bool = False               # is capture running
    show_window: bool = False           # is the panel open

    device_name: str = ""               # last chosen input, by name
    # Off by default: it normalises each band against its own running peak,
    # which any steady input reaches, so a quiet room reads near full scale on
    # every band. See the audio caveats in CLAUDE.md.
    auto_gain: bool = False

    # Physics mappings, and brain mappings keyed by modality name so a rig
    # built for one brain is waiting when you switch back to it.
    mappings: list[Mapping] = field(default_factory=list)
    brain_mappings: dict[str, list[Mapping]] = field(default_factory=dict)

    # The brain's audio inputs: a label and range per channel, the rows that
    # sum into them (target AUDIO_IN_<k>), and the tab's master switch. Feed
    # off is exact silence, which is the deaf brain.
    channels: list = field(default_factory=list)
    channel_mappings: list[Mapping] = field(default_factory=list)
    feed: bool = True

    strengths: dict[str, float] = field(default_factory=dict)
    global_strength: float = 1.0

    # How each band turns its bins into one number, and over what dB window:
    # {"bass": {"measure": "power", "floor": -60.0, "ceiling": -5.0}, ...}.
    # Empty means the analyser's defaults, so a rig saved before these existed
    # opens on them. See services/audio_analysis.effective_bands.
    bands: dict[str, dict] = field(default_factory=dict)
    # The band smoother's release. 0 hands every shaper the raw per-block
    # measurement; the rise is never smoothed at any setting.
    release_seconds: float = 0.075
    # Multiplies every phase shaper's rate at once, so a rig tuned to one
    # tempo can be slid onto another without touching each row. It scales
    # FREQUENCY only - an envelope's attack and release are durations, and
    # dragging this must not stretch them.
    rate_scale: float = 1.0

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
    request_load_preset: bool = False

    # Which named rig the preset row shows. Not persisted: the rig itself is
    # what comes back at launch, and a name would claim the live rig is still
    # that preset after an edit.
    preset_name: str = ""
    # Result of the last rig save or load, shown in the panel and dismissed.
    notice: str = ""
    warning: str = ""

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
    # Each applied mapping's post-shaper signal, by mapping uid. Only the
    # modulation maths sees it, and a drawer that drew the raw band instead
    # would show nothing of what its shaper does.
    shaped: dict = field(default_factory=dict)


def _mapping_to_dict(m: Mapping) -> dict:
    """`uid` is in-session only, so a loaded rig mints fresh ones."""
    from services.cohort_audio import is_full, mask_to_text

    out = {
        "signal": m.signal, "target": m.target, "mode": m.mode,
        "depth": float(m.depth), "gain": float(m.gain),
        "enabled": bool(m.enabled),
        "shaper": {
            "kind": m.shaper.kind, "attack": m.shaper.attack,
            "release": m.shaper.release, "threshold": m.shaper.threshold,
            "hold": m.shaper.hold, "rate": m.shaper.rate,
            "wave": m.shaper.wave,
        },
    }
    # Only when something is painted out, so an unmasked row adds nothing.
    if not is_full(m.cohorts):
        out["cohorts"] = mask_to_text(m.cohorts)
    return out


# `rate_max` was already the Hz a full-scale band asks for, so it becomes
# `rate`; `rate_min`, the silence floor, has no counterpart.
_RENAMED_KINDS = {"lfo": "phase"}
_RENAMED_FLOATS = {"rate_max": "rate"}
_SHAPER_FLOATS = ("attack", "release", "threshold", "hold", "rate")


def _read_mask(m, raw) -> None:
    """A malformed mask leaves the mapping on every cohort, never drops it."""
    from services.cohort_audio import read_mask

    read_mask(m.cohorts, raw)


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
        kind = _RENAMED_KINDS.get(sd.get("kind"), sd.get("kind", "none"))
        sd = {_RENAMED_FLOATS.get(k, k): v for k, v in sd.items()}
        shaper = replace(
            shaper,
            kind=kind if kind in SHAPER_KINDS else "none",
            **{k: float(sd[k]) for k in _SHAPER_FLOATS
               if isinstance(sd.get(k), (int, float))},
        )
        if isinstance(sd.get("wave"), str):
            shaper.wave = sd["wave"]
    try:
        m = Mapping(
            signal=signal, target=target, mode=mode,
            depth=float(d.get("depth", 0.5)), gain=float(d.get("gain", 1.0)),
            shaper=shaper, enabled=bool(d.get("enabled", True)),
        )
    except (TypeError, ValueError):
        return None
    _read_mask(m, d.get("cohorts"))
    return m


def _channel_from_dict(d):
    """A malformed entry is a default channel, never a dropped one: the
    index is the identity and dropping one would renumber the rest."""
    from services.audio_mapping import Channel

    c = Channel()
    if not isinstance(d, dict):
        return c
    if isinstance(d.get("name"), str):
        c.name = d["name"]
    r = d.get("range")
    if isinstance(r, (int, float)) and not isinstance(r, bool) and r > 0:
        c.range = float(r)
    return c


def to_dict(state: AudioInState) -> dict:
    return {
        "mappings": [_mapping_to_dict(m) for m in state.mappings],
        "channels": [{"name": str(c.name), "range": float(c.range)}
                     for c in state.channels],
        "channel_mappings": [_mapping_to_dict(m)
                             for m in state.channel_mappings],
        "feed": bool(state.feed),
        "brain_mappings": {k: [_mapping_to_dict(m) for m in v]
                           for k, v in state.brain_mappings.items()},
        "strengths": {k: float(v) for k, v in state.strengths.items()},
        "global_strength": float(state.global_strength),
        "auto_gain": bool(state.auto_gain),
        "device_name": str(state.device_name),
        "modulate": bool(state.modulate),
        # Only the muted ones, so a rig does not carry a row per parameter.
        "muted": sorted(k for k, v in state.muted.items() if v),
        "bands": {k: {"measure": str(v.get("measure", "")),
                      "floor": float(v.get("floor", 0.0)),
                      "ceiling": float(v.get("ceiling", 0.0))}
                  for k, v in state.bands.items() if isinstance(v, dict)},
        "release_seconds": float(state.release_seconds),
        "rate_scale": float(state.rate_scale),
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

    if isinstance(data.get("channels"), list):
        state.channels = [_channel_from_dict(d) for d in data["channels"]]
    if isinstance(data.get("channel_mappings"), list):
        state.channel_mappings = [
            m for m in (_mapping_from_dict(d) for d in data["channel_mappings"])
            if m is not None]
    if isinstance(data.get("feed"), bool):
        state.feed = data["feed"]

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
    if isinstance(data.get("bands"), dict):
        # Validated in the analyser, which owns what a measure and a window
        # mean; anything it does not recognise falls back to the default.
        state.bands = {k: dict(v) for k, v in data["bands"].items()
                       if isinstance(k, str) and isinstance(v, dict)}
    if isinstance(data.get("release_seconds"), (int, float)):
        state.release_seconds = max(0.0, min(2.0,
                                             float(data["release_seconds"])))
    if isinstance(data.get("rate_scale"), (int, float)):
        state.rate_scale = max(0.05, min(8.0, float(data["rate_scale"])))
