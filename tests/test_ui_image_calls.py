"""Every imgui image call must be handed an ImTextureRef, not a raw handle.

imgui_bundle raises TypeError on a bare int, but only on the frame the call
actually RUNS - and a thumbnail path is unreachable in a render test whose
harness has no texture source, so the window passes its own suite and crashes
the app. That shipped from `ui/field_stack_window.py`, where the harness set
`field_bus = None` and every image call returned before drawing.

Static, so it needs no GL context and no window.
"""
import ast
from pathlib import Path

IMAGE_CALLS = ("image", "image_button", "add_image", "add_image_quad")


def _image_calls():
    """Every imgui image call under ui/, as (file, line, texture-arg node)."""
    out = []
    for path in sorted(Path("ui").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name not in IMAGE_CALLS:
                continue
            # image_button takes a str_id first; everything else leads with it.
            arg = node.args[1] if name == "image_button" else node.args[0]
            if arg is not None:
                out.append((path.name, node.lineno, arg))
    return out


def test_no_image_call_is_handed_a_raw_texture_handle():
    offences = [f"{f}:{line}" for f, line, arg in _image_calls()
                if isinstance(arg, ast.Attribute) and arg.attr == "glo"]
    assert not offences, (
        "these pass a raw GL handle to an imgui image call:\n  "
        + "\n  ".join(offences)
        + "\nWrap it: imgui.ImTextureRef(tex.glo)")


def test_the_guard_can_see_the_calls_it_is_meant_to_police():
    """A guard that found nothing to check is imaginary coverage."""
    files = {f for f, _, _ in _image_calls()}
    for expected in ("archive_window.py", "brain_window.py",
                     "field_stack_window.py"):
        assert expected in files, f"{expected} draws images and was not read"


def test_every_field_preview_flips_v():
    """The bus renders GL textures - v=0 at the BOTTOM - and imgui draws uv0
    at the top-left, so an unflipped preview is a mirror of what the particles
    read. That reads as the sim being upside down rather than the picture."""
    missing = [f"{f}:{line}" for f, line, _ in _image_calls()
               if f == "field_stack_window.py"
               and not _has_uv(f, line)]
    assert not missing, (
        "these field previews draw without a uv flip:\n  " + "\n  ".join(missing))


def _has_uv(filename, lineno):
    for path in Path("ui").glob(filename):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and node.lineno == lineno
                    and len(node.args) >= 4):
                return True
    return False
