"""Small value objects carrying appearance settings down to the GPU.

`sim.update()` and `sim.entity_update()` already take a long keyword list.
Palette, mask and datamosh add two dozen more values between them, so they
travel as frozen bundles instead: `simulation_runner` builds them from
preferences, and the simulation and the post-process read them without knowing
`PreferencesState` exists.

All are frozen because they are per-frame snapshots -- nothing downstream
should be mutating them.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PaletteParams:
    """Remaps particle hue into a band around an externally supplied colour.

    In the VJ setup the colour is the rig's global `edgecol`, so the particles
    follow whatever the rest of the show is doing. `mix = 0` leaves the
    simulation's own hues untouched, which is the default.

    Attributes:
        mix: 0 = simulation hues, 1 = fully palette-driven.
        hue: target hue, 0..1 around the wheel.
        sat, value: target saturation and brightness.
        spread: hue band half-width in turns. 0.5 spans the whole wheel;
            small values give a tight single-hue family. Keeping the band
            narrow is what stops additive overlap washing out to white --
            overlapping particles stay in the same colour family.
        stops: 0 or 1 = continuous. 2+ quantizes the band into that many
            discrete hues, so `spread=0.5, stops=2` gives a complementary pair.
    """
    mix: float = 0.0
    hue: float = 0.0
    sat: float = 0.8
    value: float = 1.0
    spread: float = 0.15
    stops: int = 0

    @classmethod
    def from_preferences(cls, prefs) -> "PaletteParams":
        return cls(
            mix=float(prefs.palette_mix),
            hue=float(prefs.palette_hue),
            sat=float(prefs.palette_sat),
            value=float(prefs.palette_value),
            spread=float(prefs.palette_spread),
            stops=int(prefs.palette_stops),
        )

    @property
    def active(self) -> bool:
        return self.mix > 0.0


@dataclass(frozen=True)
class MaskParams:
    """Weights particle activity by region. A bias, not a stencil.

    The mask itself is built by `utilities.particle_mask.ParticleMask` from the
    incoming Spout texture and/or a vignette; these are the knobs for building
    it and for how strongly the simulation obeys it.

    Attributes:
        ink: how much particle alpha follows the mask. This is the important
            one: alpha gates trail deposition as well as display, so lowering
            it outside the mask weakens the trails there, which weakens the
            sensor attraction there, which draws particles back toward the
            active regions on their own. Self-reinforcing, no hard boundary.
        force: how much particle force follows the mask -- agitation rather
            than presence.
        pull: containment strength up the mask gradient.
        floor: activity level where the mask is dark. 1.0 is a no-op, 0.0 kills
            the empty areas entirely; in between is "quiet but alive".
        gamma: contrast applied to the mask.
        blur: blur radius in mask texels. The "enrich, don't trace" knob --
            enough blur turns a hard-edged source into soft regions rather than
            outlines to follow.
        vignette, vignette_softness: built-in radial falloff, usable with no
            incoming texture at all.
        source: 0=feed, 1=vignette, 2=max(feed, vignette), 3=feed*vignette.
    """
    ink: float = 0.0
    force: float = 0.0
    pull: float = 0.0
    floor: float = 0.3
    gamma: float = 1.0
    blur: float = 2.0
    vignette: float = 0.0
    vignette_softness: float = 0.5
    source: int = 0

    @classmethod
    def from_preferences(cls, prefs) -> "MaskParams":
        return cls(
            ink=float(prefs.mask_ink),
            force=float(prefs.mask_force),
            pull=float(prefs.mask_pull),
            floor=float(prefs.mask_floor),
            gamma=float(prefs.mask_gamma),
            blur=float(prefs.mask_blur),
            vignette=float(prefs.mask_vignette),
            vignette_softness=float(prefs.mask_vignette_softness),
            source=int(prefs.mask_source),
        )

    @property
    def active(self) -> bool:
        """True if the mask would change anything.

        Checked before generating the mask texture at all, so the default
        configuration costs nothing -- not a pass, not a bind, not a uniform.
        """
        return self.ink > 0.0 or self.force > 0.0 or self.pull > 0.0


@dataclass(frozen=True)
class MoshParams:
    """Datamosh: particle motion displaces the incoming texture's pixels.

    The inverse of every other mode -- nothing is drawn over the feed, the feed
    itself is torn along the flow the particles generate. The flow is the
    velocity map the simulation already writes (brush.frag splats it, the
    canvas accumulates it), so this reads the simulation and never touches it.

    Attributes:
        enabled: master switch. Off is a true no-op: the pass never runs and
            its buffers are released.
        source: what gets moshed. 0=Spout feed, 1=the particle frame,
            2=feed with particles added before moshing, so the particles are
            smeared into the image too. With no sender connected, 0 and 2 fall
            back to the particle frame rather than moshing black.
        amount: displacement per displayed frame, in output uv. This is the
            ceiling -- the actual shift is scaled by the local flow.
        contrast: how selective the response is. The flow magnitude is first
            normalized against the frame's own average, so 1.0 is always
            "average activity" whatever the world size and particle count;
            the curve is `x^c / (1 + x^c)`, which passes through half strength
            at that average for every setting. Below 1 it flattens and the
            whole frame drifts together in broad strokes; above 1 it sharpens
            toward a threshold where only the busiest streaks move at all.
        scale: stroke size, as a mip level of the flow field. Averaging the
            field lets opposing directions cancel, so raising this leaves only
            the large-scale coherent motion -- regions instead of individual
            streaks. It also softens the shift, so `amount` usually wants to
            come up with it.
        flow_mix: 0 = the persistent trail map (dense, smooth, long smears),
            1 = this frame's splat only (sparse, sharp, twitchy tearing).
        swirl: rotates the shift, up to +-90 degrees at +-1. Turns a push into
            a vortex without any extra field.
        refresh: how much of the live source returns each frame. 1 = no
            accumulation, just the current frame warped once. ~0.05 = the
            classic melt, artefacts surviving about twenty frames. 0 = nothing
            resets and the image is consumed entirely; a reseed is the way
            back from there.
        block: macroblock size in pixels for the flow lookup, 0 = off. Whole
            blocks then shift together, which is what reads as a codec
            artefact rather than a smooth displacement map.
        chroma: per-channel displacement spread. Colour fringing on the fast
            tears only.
        ink: crisp particles added back on top *after* moshing. 0 by default --
            the point of the mode is that the particles move the image instead
            of being drawn on it.
    """
    enabled: bool = False
    source: int = 0
    amount: float = 0.02
    contrast: float = 1.0
    scale: float = 0.0
    flow_mix: float = 0.0
    swirl: float = 0.0
    refresh: float = 0.08
    block: float = 0.0
    chroma: float = 0.0
    ink: float = 0.0

    @classmethod
    def from_preferences(cls, prefs) -> "MoshParams":
        return cls(
            enabled=bool(prefs.mosh_enabled),
            source=int(prefs.mosh_source),
            amount=float(prefs.mosh_amount),
            contrast=float(prefs.mosh_contrast),
            scale=float(prefs.mosh_scale),
            flow_mix=float(prefs.mosh_flow_mix),
            swirl=float(prefs.mosh_swirl),
            refresh=float(prefs.mosh_refresh),
            block=float(prefs.mosh_block),
            chroma=float(prefs.mosh_chroma),
            ink=float(prefs.mosh_ink),
        )

    @property
    def active(self) -> bool:
        """True if the pass would change anything.

        Checked before allocating the feedback buffers at all, so the default
        configuration costs nothing. `amount` is not part of the test: at zero
        displacement the pass still owns the frame, and refresh below 1 still
        leaves a fading ghost -- switching the mode on must always take effect.
        """
        return self.enabled
