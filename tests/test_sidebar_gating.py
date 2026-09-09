"""Show/Hide Windows covers every control panel.

The gate is one `and` per dispatch line, so a new panel joins the group by
someone remembering to write it - the Perform Mode panel did not, and stayed on
screen after the key that exists to clear the screen. That is a fact about the
dispatch itself, which is why this reads it rather than driving a frame: the
render tests drive bare mixins and never see `UI.render` at all.

A window is exempt only for a reason named here.
"""
import ast
import inspect
from pathlib import Path

import pytest

from ui.core import UI

# Not control panels. Help and reference windows are opened deliberately from
# a menu and are not part of the group the key clears.
EXEMPT = {
    "render_controls_window": "help",
    "render_parameter_sweeps_window": "help",
    "render_tutorial_window": "help",
    "render_performance_window": "help",
    "render_undo_window": "help",
    "render_popup_modals": "a modal is answered, not hidden",
    "render_field_loader_window": "transient, owned by the field stack",
    "render_main_menu_bar": "the menu bar is how the key is pressed back",
}


def _render_body():
    source = Path(inspect.getfile(UI)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "UI":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "render":
                    return item
    pytest.fail("could not find UI.render in ui/core.py")


def _gates():
    """{method name: True if the sidebar guards its call}."""
    found = {}

    def walk(node, guarded):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                names = {n.id for n in ast.walk(child.test)
                         if isinstance(n, ast.Name)}
                names |= {a.attr for a in ast.walk(child.test)
                          if isinstance(a, ast.Attribute)}
                inner = guarded or "show_sidebar" in names
                for stmt in child.body:
                    walk(stmt, inner)
                for stmt in child.orelse:
                    walk(stmt, guarded)
                continue
            if (isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "self"
                    and child.func.attr.startswith("render_")):
                name = child.func.attr
                # Called from more than one branch: gated only if every call is.
                found[name] = found.get(name, True) and guarded
            walk(child, guarded)

    walk(_render_body(), False)
    return found


def test_the_dispatch_was_found_at_all():
    """A parser that matches nothing passes every assertion below."""
    gates = _gates()
    assert len(gates) > 8, f"only found {sorted(gates)}"
    assert "render_physics_settings_window" in gates


def test_every_control_panel_is_hidden_by_show_hide_windows():
    ungated = sorted(name for name, gated in _gates().items()
                     if not gated and name not in EXEMPT)
    assert not ungated, (
        f"these panels ignore Show/Hide Windows: {ungated}. Gate them on "
        "show_sidebar, or name them in EXEMPT with a reason.")


def test_the_perform_panel_is_one_of_them():
    """Named outright: it is the one that shipped ungated."""
    assert _gates().get("render_perform_window") is True
