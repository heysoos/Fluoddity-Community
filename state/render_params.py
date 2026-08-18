"""Small value objects carrying appearance settings down to the GPU.

`sim.update()` and `sim.entity_update()` already take a long keyword list.
Palette and mask add a dozen more values between them, so they travel as two
frozen bundles instead: `simulation_runner` builds them from preferences and
the simulation reads them without knowing `PreferencesState` exists.

Both are frozen because they are per-frame snapshots -- nothing downstream
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
