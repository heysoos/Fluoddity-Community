"""The view indices, in one place.

The number IS the `view_mode` uniform `frame_assembly.frag` branches on, so the
two must agree; `tests/test_view_modes.py` checks that they do.

Nothing on disk carries one - `current_view_option` is in neither
`PhysicsConfig` nor `PreferencesState` - so renumbering costs no migration.
What it does cost is a silent behaviour change at every hardcoded literal, and
removing the brush view moved all six of them at once. These names exist so the
next removal is a one-line edit rather than a sweep across seven files.
"""

CANVAS = 0          # the trail texture itself
CAMERA = 1          # particles rendered as dots
CAMERA_TILED = 2    # the same, replicated infinitely
FORCE_FIELD = 3     # advanced drawing's force field, raw
STRAFE_FIELD = 4    # advanced drawing's strafe field, raw
CAMERA_TRAILS = 5   # particles over the trail

#: Views the camera renders, i.e. the ones that set cam_brush_mode.
CAMERA_VIEWS = (CAMERA, CAMERA_TILED, CAMERA_TRAILS)

#: Views watercolour applies to. NOT the same set: it has no meaning over the
#: trail overlay, so CAMERA_TRAILS is deliberately absent.
WATERCOLOR_VIEWS = (CAMERA, CAMERA_TILED)

#: The advanced-drawing field views, which show a texture rather than the sim.
FIELD_VIEWS = (FORCE_FIELD, STRAFE_FIELD)
