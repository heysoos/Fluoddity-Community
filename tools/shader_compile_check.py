"""Compile the assembled compute shader without launching the app.

There is no GPU in the test suite, so tests/test_brain_shader_source.py can
only assert on source TEXT. This catches what that cannot: undeclared
identifiers, wrong prepend order, type errors - anything the GLSL compiler
knows and grep does not.

Assembles exactly as sim.setup_shaders does. If the two ever diverge this tool
is lying, so keep the prepend order here identical to sim.py.

    python -m tools.shader_compile_check
"""
from __future__ import annotations

import sys

# The app has a pre-existing import cycle: services/__init__ -> config_saver ->
# ui.physics_params -> ui/__init__ -> ui.core -> services.config_saver. It
# resolves only when `ui` is imported first, which is what main.py happens to
# do. Prime that order, as tests/conftest.py does.
import ui  # noqa: F401

from utilities.gl_helpers import prepend_defines, read_shader, shader_prepend

BRAINS = ("mlp", "lenia", "gabor", "fourier")


def assemble(entity_count: int = 600000) -> str:
    """Mirror of sim.setup_shaders. shader_prepend inserts after the #version
    line, so the LAST prepend lands FIRST in the compiled source."""
    src = read_shader("shaders/entity_update.glsl")
    src = shader_prepend(src, read_shader("shaders/cohort_audio.glsl"))
    src = shader_prepend(src, read_shader("shaders/brains/_dispatch.glsl"))
    for brain in BRAINS:
        src = shader_prepend(src, read_shader(f"shaders/brains/{brain}.glsl"))
    src = shader_prepend(src, read_shader("shaders/brains/_header.glsl"))
    src = shader_prepend(src, read_shader("shaders/fourier4_4.glsl"))
    return prepend_defines(src, entity_count)


def main(argv: list[str]) -> int:
    src = assemble()
    try:
        import moderngl

        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        print(f"SKIP: no standalone GL context available ({exc})")
        return 0

    print(f"GL {ctx.info['GL_VERSION']}")
    try:
        ctx.compute_shader(src)
    except Exception as exc:
        print("COMPILE FAILED\n")
        print(exc)
        if "--source" in argv:
            for n, line in enumerate(src.splitlines(), 1):
                print(f"{n:5} {line}")
        else:
            print("\nRe-run with --source to dump the numbered assembled source.")
        return 1

    print("COMPILE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
