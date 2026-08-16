"""Holds the capture, the shaper states and the brain vector between frames.

The orchestrator calls update() once per frame and hands what it returns to the
sim. Nothing here writes the user's state.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from services import audio_capture
from services.audio_brain import BrainModulator
from services.audio_capture import AudioCapture
from services.audio_mapping import (brain_targets, deaf_targets, modulate,
                                    physics_targets)


def muted_targets(ast, prefix: str = "") -> set[str]:
    """Targets the user has switched off, by key.

    Fed to modulate() as deafness, which is what it already means: the target
    keeps its value and no mapping may touch it.

    Brain rows are stored under `<modality>:<key>`, because two modalities may
    declare the same scale name - mlp and lenia both have `w_scale` - and their
    mappings are already kept apart that way.
    """
    out = set()
    for key, value in (getattr(ast, "muted", None) or {}).items():
        if not value:
            continue
        if prefix:
            if key.startswith(prefix):
                out.add(key[len(prefix):])
        elif ":" not in key:
            out.add(key)
    return out


class AudioRuntime:
    def __init__(self) -> None:
        self.capture = AudioCapture()
        self._brain = BrainModulator()
        self._states: dict[int, object] = {}
        self._base_brain_id = None

    def close(self) -> None:
        self.capture.stop()
        self._brain.clear()

    def _prune_states(self, ast) -> None:
        """Drop the shaper state of every mapping the rig no longer holds.

        The whole rig, not the modality on screen: switching brains and back
        must not restart the shapers that were waiting there.
        """
        live = {m.uid for m in ast.mappings}
        for rows in ast.brain_mappings.values():
            live.update(m.uid for m in rows)
        for uid in [u for u in self._states if u not in live]:
            del self._states[uid]

    def _sync_capture(self, ast) -> None:
        if ast.request_start:
            ast.request_start = False
            index = None
            for d in audio_capture.list_devices():
                if d["name"] == ast.device_name:
                    index = d["index"]
                    break
            ast.enabled = self.capture.start(index, ast.auto_gain)
        if ast.request_stop:
            ast.request_stop = False
            self.capture.stop()
            ast.enabled = False
        # Every frame, not just at Start: these are the only escape from a
        # setting that is wrong for the material, so they must take effect
        # while it plays rather than after a Stop/Start.
        self.capture.set_auto_gain(ast.auto_gain)
        self.capture.set_bands(ast.bands)
        self.capture.set_release(ast.release_seconds)
        ast.status = self.capture.status
        # The panel is passive and owns no service, so the snapshot is handed
        # to it here rather than reached for through the capture thread.
        ast.snapshot = self.capture.snapshot()
        # A loopback endpoint renders nothing while the machine is silent, so a
        # perfectly healthy capture can sit with no snapshot for as long as
        # nothing is playing. Saying "active" there makes a working panel look
        # broken; the panel names what it is waiting for instead.
        if ast.status == "active" and ast.snapshot is None:
            ast.status = "waiting"
        ast.last_error = self.capture.last_error

    def update(self, ui_state, dt: float, brain_layout, current_rule):
        """Returns (sim_state_for_the_sim, modulated_brain_or_None).

        The first is `ui_state.sim` itself when nothing is modulated, and a
        copy otherwise - so the caller never has to know which.
        """
        ast = ui_state.audio
        self._sync_capture(ast)
        # Cleared before any early return, so a bypassed or stopped rig empties
        # the drawer traces rather than freezing them on their last value.
        ast.shaped = {}
        # Likewise before any early return: nothing tells the runtime a row was
        # deleted, so the table is cut back to the rig every frame.
        self._prune_states(ast)

        # Auto and Explore rank tiles against each other. Modulating physics
        # mid-comparison would move what is being compared.
        if ui_state.auto_tournament.enabled or ui_state.archive.enabled:
            return ui_state.sim, None
        if not ast.enabled:
            return ui_state.sim, None
        # The master bypass stops the modulation, NOT the capture: the panel
        # keeps drawing so you can see what turning it back on would do.
        if not ast.modulate:
            return ui_state.sim, None

        snap = ast.snapshot
        if snap is None:
            return ui_state.sim, None
        signals = snap.signals
        # Which of them are remembered rather than measured this block.
        held = getattr(snap, "held", frozenset())

        sim_out = ui_state.sim
        p_targets = physics_targets(ui_state.sim)
        deaf = deaf_targets(ui_state.sim) | muted_targets(ast)
        bases = {t.key: float(getattr(ui_state.sim, t.key, 0.0))
                 for t in p_targets}
        moved = modulate(bases, p_targets, ast.mappings, signals, self._states,
                         ast.strengths, ast.global_strength, dt, deaf,
                         ast.shaped, held=held, rate_scale=ast.rate_scale)
        if moved:
            sim_out = replace(ui_state.sim, **moved)

        brain_out = self._update_brain(ui_state, ast, signals, dt,
                                       brain_layout, current_rule, held)
        return sim_out, brain_out

    def overlays(self, ui_state, modulated_sim) -> dict:
        """Per-target drawing data for the physics sliders.

        `reach` is where a full-scale signal would land, so the hatching shows
        the modulation's size rather than its current value.
        """
        ast = ui_state.audio
        if not ast.enabled or modulated_sim is ui_state.sim:
            return {}
        from ui.audio_reactive_window import SIGNAL_COLORS

        deaf = deaf_targets(ui_state.sim) | muted_targets(ast)
        targets = {t.key: t for t in physics_targets(ui_state.sim)}
        signals = {n: 1.0 for n in SIGNAL_COLORS}
        bases = {k: float(getattr(ui_state.sim, k, 0.0)) for k in targets}
        # The shapers are skipped rather than run from a throwaway state: a
        # fresh one is mid-attack, or a phase that has not travelled yet, which
        # reported a fraction of the swing the mapping really has. Face value
        # is the right answer for every kind, because the hatching asks how far
        # a mapping could reach and each of them reaches full scale eventually.
        full = modulate(bases, list(targets.values()), ast.mappings, signals,
                        dict(), ast.strengths, ast.global_strength, 1 / 60.0,
                        deaf, apply_shapers=False)

        out = {}
        for key, target in targets.items():
            bound = [m for m in ast.mappings
                     if m.target == key and m.enabled and key not in deaf]
            if not bound:
                continue
            out[key] = {
                "lo": target.lo, "hi": target.hi,
                "base": bases[key],
                "live": float(getattr(modulated_sim, key, bases[key])),
                "reach": full.get(key, bases[key]),
                "color": SIGNAL_COLORS[bound[0].signal],
            }
        return out

    def _update_brain(self, ui_state, ast, signals, dt, layout, current_rule,
                      held=()):
        if layout is None or current_rule is None:
            return None
        mappings = ast.brain_mappings.get(ui_state.brain.modality, ())
        if not mappings:
            return None

        from services import brains
        modality = brains.get(ui_state.brain.modality)

        # Re-encode only when the base brain actually changes; encode() clips
        # at the rails, so a round trip per frame would drift. A SCALE change
        # is deliberately NOT a change of base - see CLAUDE.md.
        arr = np.asarray(current_rule, dtype=np.float32).reshape(-1)
        ident = (id(current_rule), arr.size, layout.length,
                 float(arr[:8].sum()) if arr.size else 0.0)
        if ident != self._base_brain_id:
            self._brain.set_base(arr, modality, layout)
            self._base_brain_id = ident

        targets = brain_targets(modality, layout)
        if not targets:
            return None
        # The scales come from the layout that is live NOW, never from the one
        # captured when audio adopted this brain - that is what keeps the Brain
        # window's scale sliders working while a rig is running.
        bases = {k: float(v) for k, v in layout.scales}
        moved = modulate(bases, targets, mappings, signals, self._states,
                         ast.strengths, ast.global_strength, dt,
                         muted_targets(ast, f"{ui_state.brain.modality}:"),
                         ast.shaped, held=held, rate_scale=ast.rate_scale)
        if not moved:
            return None
        return self._brain.modulated({**bases, **moved})
