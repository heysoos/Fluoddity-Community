"""Canonical Fluoddity physics parameters as OSC specs.

The full app should call :func:`specs_from_registry` and pass its own
``ui.physics_params.PHYSICS_PARAMS`` — that registry is the single source of
truth there, so there is nothing to keep in sync.

:data:`CANONICAL` exists for Fluoddity-Core, which is an independent
reimplementation with no such registry. Its numbers mirror the full app's
registry as of writing; if you widen a slider range there, this table does not
follow. It only affects the 0..1 normalized mapping, never a hard limit.
"""
from .osc_control import ParamSpec

# (name, default_min, default_max, hard_min, hard_max, power)
CANONICAL = [
    ("SENSOR_GAIN",       0.0, 10.0,  None, None, None),
    ("SENSOR_ANGLE",     -1.0,  1.0,  -1.0,  1.0, None),
    ("SENSOR_DISTANCE",   0.0,  3.0,  None, None, None),
    ("MUTATION_SCALE",    0.0,  1.0,  None, None, None),
    ("GLOBAL_FORCE_MULT", 0.0,  2.0,  None, None, None),
    ("DRAG",             -1.0,  1.0,  -1.0,  1.0, None),
    ("AXIAL_FORCE",      -1.0,  1.0,  None, None, None),
    ("LATERAL_FORCE",    -1.0,  1.0,  None, None, None),
    ("STRAFE_POWER",      0.0,  0.5,  None, None, None),
    ("TRAIL_PERSISTENCE", 0.0,  1.0,   0.0,  1.0, None),
    ("TRAIL_DIFFUSION",   0.0,  1.0,   0.0,  1.0, None),
    ("HAZARD_RATE",       0.0,  0.05,  0.0, 0.05, 3.0),
]


# --- Extra groups, each with its own bulk OSC address ---
#
# These are *not* appended to CANONICAL. The physics registry is the one thing
# that might gain a parameter later, and appending would silently shift every
# slice index below it in the vvvv patch.

# Structure: already plain uniforms in sim.py, no shader change needed.
# rule_seed is re-read by the shader every frame (entity_update.glsl:476), so
# stepping it swaps the whole rule set instantly -- the strongest
# change-on-the-downbeat control the simulation has.
#
# Appended entries are safe: the bulk path zips specs against however many
# values arrive, so a patch still sending the original four keeps working and
# simply leaves the later ones alone. Never insert in the middle.
#
# color_by_cohort is here because it is what decouples hue variety from cohort
# count. With it on, hue is hash(floor(cohort)), so one cohort means one colour
# -- and one cohort is also what makes a reseed replace the whole behaviour
# rather than reshuffle an ensemble. Turning it off moves hue onto
# hue_sensitivity * behaviour, so a single cohort can still show a full
# spectrum. Both belong in vvvv's hands for that reason.
#
# rule_seed sits LAST on purpose. It has its own address, /fluoddity/rule_seed,
# because a new seed is a per-beat event while the rest of this group is
# set-and-forget -- firing them together meant every reseed also re-sent the
# cohort count. Being last means a 5-value bulk send covers the group without
# touching the seed, and it stays registered as a per-parameter address.
STRUCTURE = [
    ("num_cohorts",         1.0, 144.0, 1.0, 144.0, None, True),
    ("initial_conditions",  0.0,   2.0, 0.0,   2.0, None, True),
    ("boundary_conditions", 0.0,   2.0, 0.0,   2.0, None, True),
    ("color_by_cohort",     0.0,   1.0, 0.0,   1.0, None, True, True),
    ("hue_sensitivity",    -1.0,   1.0, -1.0,  1.0, None, False),
    ("rule_seed",           0.0,   1.0, None, None, None, False),
]

# Palette: bends particle hue toward the show's `edgecol`. palette_mix 0 is a
# no-op, so the default is exactly the pre-existing colouring.
PALETTE = [
    ("palette_mix",    0.0, 1.0, 0.0, 1.0, None, False),
    ("palette_hue",    0.0, 1.0, None, None, None, False),
    ("palette_sat",    0.0, 1.0, 0.0, 1.0, None, False),
    ("palette_value",  0.0, 1.0, 0.0, 1.0, None, False),
    # Hue band half-width in turns. 0.5 spans the whole wheel; 0.075 is a
    # tight single-hue family.
    ("palette_spread", 0.0, 0.5, 0.0, 0.5, None, False),
    # 0 or 1 = continuous. 2+ quantizes the band into that many hues.
    ("palette_stops",  0.0, 8.0, 0.0, 8.0, None, True),
]

# Mask: how much activity each region gets. A bias, not a stencil --
# mask_floor sets the activity level outside the masked area.
MASK = [
    ("mask_ink",       0.0, 1.0, 0.0, 1.0, None, False),
    ("mask_force",     0.0, 1.0, 0.0, 1.0, None, False),
    ("mask_pull",      0.0, 2.0, 0.0, 2.0, None, False),
    ("mask_floor",     0.0, 1.0, 0.0, 1.0, None, False),
    ("mask_gamma",     0.2, 4.0, 0.2, 8.0, None, False),
    ("mask_blur",      0.0, 8.0, 0.0, 8.0, None, False),
    ("mask_vignette",  0.0, 1.0, 0.0, 1.0, None, False),
    ("mask_vignette_softness", 0.0, 1.0, 0.01, 1.0, None, False),
    # 0=feed, 1=vignette, 2=max(feed,vignette), 3=feed*vignette
    ("mask_source",    0.0, 3.0, 0.0, 3.0, None, True),
]


# Spout field source (shaders/field_override/spout.frag) plus the two switches
# that arm it. The switches are here because closing Fluoddity's Drawing
# Controls window sets advanced_drawing_enabled False, which silently stops the
# incoming texture driving anything -- indistinguishable from a broken bridge
# unless you can see the flag. Better in vvvv's hands than hidden in a window.
FIELD = [
    # 0=channels (rg/ba), 1=luminance gradient, 2=gradient perpendicular
    ("spout_field_mode",         0.0, 2.0, 0.0, 2.0, None, True),
    ("spout_field_scale",        0.0, 4.0, 0.0, 8.0, None, False),
    ("spout_strafe_scale",       0.0, 4.0, 0.0, 8.0, None, False),
    ("shader_driven_field",      0.0, 1.0, 0.0, 1.0, None, True, True),
    ("advanced_drawing_enabled", 0.0, 1.0, 0.0, 1.0, None, True, True),
]


def _specs(table, target: str) -> list[ParamSpec]:
    """Build specs from a table of 7-tuples, or 8-tuples to mark a bool."""
    out = []
    for row in table:
        name, lo, hi, hard_min, hard_max, power, is_int = row[:7]
        is_bool = row[7] if len(row) > 7 else False
        out.append(ParamSpec(name=name, lo=lo, hi=hi, hard_min=hard_min,
                             hard_max=hard_max, power=power, is_int=is_int,
                             target=target, is_bool=is_bool))
    return out


def structure_specs() -> list[ParamSpec]:
    """Simulation structure controls, applied to SimState."""
    return _specs(STRUCTURE, "sim")


def palette_specs() -> list[ParamSpec]:
    """Palette controls.

    On preferences, not SimState: the palette is a show-level setting, and
    loading a physics config overwrites SimState's appearance fields
    (config_saver.py:348). You do not want a preset recall to change the colour
    the whole show is running in.
    """
    return _specs(PALETTE, "preferences")


def mask_specs() -> list[ParamSpec]:
    """Activity-mask controls. On preferences, same reasoning as the palette."""
    return _specs(MASK, "preferences")


def field_specs() -> list[ParamSpec]:
    """Spout field source and its arm switches. All on preferences."""
    return _specs(FIELD, "preferences")


def extra_groups() -> dict[str, list[ParamSpec]]:
    """The non-physics groups, keyed by their bulk address name."""
    return {
        "structure": structure_specs(),
        "palette": palette_specs(),
        "mask": mask_specs(),
        "field": field_specs(),
    }


def make_specs(lower: bool = False) -> list[ParamSpec]:
    """Build specs from :data:`CANONICAL`.

    Args:
        lower: lowercase the parameter names, matching Fluoddity-Core's
            config dict keys. The full app uses ALL_CAPS SimState fields.
    """
    return [
        ParamSpec(name=name.lower() if lower else name,
                  lo=lo, hi=hi, hard_min=hard_min, hard_max=hard_max, power=power)
        for name, lo, hi, hard_min, hard_max, power in CANONICAL
    ]


def specs_from_registry(physics_params) -> list[ParamSpec]:
    """Build specs from the full app's ``PHYSICS_PARAMS`` registry.

    Keeps ranges, hard limits and power scaling in step with the sliders
    automatically, including any parameter added later.
    """
    return [
        ParamSpec(
            name=p.name,
            lo=p.default_min,
            hi=p.default_max,
            hard_min=p.hard_min,
            hard_max=p.hard_max,
            power=p.power_exponent if p.is_power_scaled else None,
        )
        for p in physics_params
    ]
