"""Run the real app and drive the Field Stack through every combination.

    python -m tools.drive_field_stack

A render test drives a bare mixin and a GL test drives a bare bus; neither
runs the assembled app, so a window body that names a missing attribute or
hands imgui the wrong type passes the whole suite and crashes on the first
frame. Everything this checks was found that way.

It measures the FIELD TEXTURE, never `pass_count` - the count belongs to the
last rebuild and reads zero on the clean frames in between, so a layer that is
quietly forcing looks idle.

Two things it must not disturb: `imgui.ini`, suppressed before the app is
built, and the user's archive, which a restored browser would open and flush
on close.
"""
from __future__ import annotations

import sys
import traceback

from ui import ini_path

ini_path.suppress()

import glfw                                          # noqa: E402
import numpy as np                                   # noqa: E402

import main as app_main                              # noqa: E402
from services import field_sources                   # noqa: E402
from state.field_stack import (BLENDS, DESTINATIONS,  # noqa: E402
                               MAPPINGS, FieldLayer)

FAILS: list[str] = []


def fail(what: str) -> None:
    FAILS.append(what)
    print(f"  FAIL: {what}")


def step(app, n, what) -> bool:
    """Run n real frames. -> False if one raised."""
    for i in range(n):
        try:
            glfw.poll_events()
            app.orchestrate_frame()
            glfw.swap_buffers(app.window)
        except Exception:
            fail(f"{what} raised on frame {i}")
            print(traceback.format_exc())
            return False
    return True


def field(app):
    """(max |force|, max |strafe|) in the bus texture, or None if released."""
    tex = app.field_bus.field_texture
    if tex is None:
        return None
    data = np.frombuffer(tex.read(), dtype=np.float32).reshape(-1, 4)
    return (float(np.abs(data[:, :2]).max()), float(np.abs(data[:, 2:]).max()))


def put(app, layers):
    app.ui.state.field_stack.layers[:] = layers
    app.field_bus.mark_dirty()


def main() -> int:
    app = app_main.App()
    # A restored browser opens the user's archive and flushes it on close.
    app.ui.state.archive.show_browser = False
    app.ui.state.archive.open_browser_requested = False
    app.ui.state.preferences.show_field_stack = True
    stack = app.ui.state.field_stack

    if not step(app, 20, "boot with the layer list open"):
        return finish(app)

    keys = [d.key for d in field_sources.descriptors()]
    for key in keys:
        layer = FieldLayer(source=key)
        if key == "shader":
            files = field_sources.available_shader_files()
            if files:
                layer.params["_file"] = files[0]
        put(app, [layer])
        if not step(app, 12, f"source={key}"):
            return finish(app)
        print(f"  source={key:9s} field={field(app)} err={layer.error!r}")

    for mapping in MAPPINGS:
        for dest in DESTINATIONS:
            put(app, [FieldLayer(source="noise", mapping=mapping,
                                 destination=dest)])
            if not step(app, 5, f"map={mapping} dest={dest}"):
                return finish(app)
    print("  ok every mapping x destination")

    for blend in BLENDS:
        put(app, [FieldLayer(source="noise"),
                  FieldLayer(source="gradient", blend=blend)])
        if not step(app, 5, f"blend={blend}"):
            return finish(app)
    # A multiply layer FIRST composites against the zero clear.
    put(app, [FieldLayer(source="noise", blend="multiply")])
    if not step(app, 5, "multiply first in the stack"):
        return finish(app)
    print("  ok every blend, and multiply first")

    # The defect the whole redesign exists to remove.
    put(app, [FieldLayer(source="noise")])
    step(app, 10, "layer on")
    on = field(app)
    stack.layers[0].enabled = False
    app.field_bus.mark_dirty()
    step(app, 10, "layer off")
    off = field(app)
    print(f"  enabled={on} disabled={off}")
    if on is None or on[0] == 0.0:
        fail("an enabled noise layer produced a zero field")
    if off is not None and off[0] != 0.0:
        fail("a disabled layer still contributes")

    stack.layers[0].enabled = True
    app.ui.state.preferences.show_field_stack = False
    app.field_bus.mark_dirty()
    step(app, 10, "window closed")
    closed = field(app)
    print(f"  window closed -> {closed}")
    if closed is None or closed[0] == 0.0:
        fail("closing the window stopped the injection")
    app.ui.state.preferences.show_field_stack = True

    for scale in (1.0, 0.5, 0.25):
        app.ui.state.preferences.field_bus_scale = scale
        app.field_bus.mark_dirty()
        if not step(app, 6, f"bus scale {scale}"):
            return finish(app)
        print(f"  scale {scale} -> {app.field_bus.resolution} {field(app)}")

    # The colour mask, which decides this and is applied at bind time.
    for dest in ("force", "strafe"):
        put(app, [FieldLayer(source="noise", destination=dest)])
        step(app, 8, f"isolation {dest}")
        f, s = field(app)
        print(f"  dest={dest}: force={f:.4f} strafe={s:.4f}")
        if dest == "force" and s != 0.0:
            fail("a force layer wrote strafe")
        if dest == "strafe" and f != 0.0:
            fail("a strafe layer wrote force")

    # A broken layer must be reported and must not stop the others.
    broken = FieldLayer(source="shader")
    broken.params["_file"] = "definitely_not_here.frag"
    put(app, [broken, FieldLayer(source="noise")])
    if not step(app, 8, "missing shader file"):
        return finish(app)
    print(f"  missing file -> {broken.error!r} {field(app)}")
    if not broken.error:
        fail("a layer pointed at a missing file drew no error")

    put(app, [FieldLayer(source=k) for k in keys if k != "shader"])
    if not step(app, 12, "every source stacked at once"):
        return finish(app)
    print(f"  stacked -> {field(app)}")
    stack.layers.reverse()
    app.field_bus.mark_dirty()
    if not step(app, 8, "stack reversed"):
        return finish(app)

    if not _roundtrip(app, stack):
        return finish(app)
    _tournament(app)

    put(app, [])
    step(app, 10, "empty stack")
    print(f"  emptied -> {field(app)}")
    return finish(app)


def _roundtrip(app, stack) -> bool:
    """A preset must carry its stack, through the real save and load."""
    import os
    import tempfile

    from services.config_saver import ConfigSaver, PhysicsConfig
    from state.field_stack import stack_from_dict, stack_to_dict

    put(app, [FieldLayer(source="noise", destination="strafe", blend="max",
                         strength=2.5, blur=3.0),
              FieldLayer(source="gradient", mapping="curl")])
    stack.layers[0].params["scale"] = 9.0
    if not step(app, 8, "the stack to save"):
        return False

    path = os.path.join(tempfile.gettempdir(), "_field_roundtrip.json")
    saver = ConfigSaver()
    saver.save_to_file(PhysicsConfig(field_stack=stack_to_dict(stack)), path)
    reloaded = stack_from_dict(saver.load_from_file(path).field_stack)
    os.remove(path)

    def shape(s):
        return [(l.source, l.destination, l.blend, l.strength, l.blur,
                 l.mapping, l.params) for l in s.layers]

    same = shape(stack) == shape(reloaded)
    print(f"  preset round-trip: {len(reloaded.layers)} layers, identical={same}")
    if not same:
        fail("a saved stack came back different")

    put(app, list(reloaded.layers))
    if not step(app, 10, "the reloaded stack"):
        return False
    print(f"  reloaded field={field(app)}")
    return True


def _tournament(app) -> None:
    """Tiles are isolated small worlds, so the bus is off under a grid."""
    app.ui.state.tournament.enabled = True
    app.field_bus.mark_dirty()
    step(app, 10, "tournament on")
    during = field(app)
    app.ui.state.tournament.enabled = False
    app.field_bus.mark_dirty()
    step(app, 10, "tournament off")
    after = field(app)
    print(f"  tournament: during={during} after={after}")
    if during is not None and during[1] != 0.0:
        fail("the bus still injected under a grid")
    if after is None or after[1] == 0.0:
        fail("the bus did not come back after the grid")


def finish(app) -> int:
    try:
        app._cleanup_safely()
    except Exception:
        print(traceback.format_exc())
    print("\n===== SUMMARY =====")
    for what in FAILS:
        print(f"FAIL: {what}")
    if not FAILS:
        print("no failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
