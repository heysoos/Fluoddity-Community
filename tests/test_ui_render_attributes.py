"""Every attribute UI.render() reads off self must actually exist.

The render tests all drive a bare MIXIN, never the real UI, so a dispatch line
naming an attribute that does not exist compiles, passes the whole suite, and
crashes on the first frame the app draws. That is how `self.show_sidebar`
shipped where every other line reads `self.state.preferences.show_sidebar`.

Static, so it needs no GL context and no window: the names come from the AST
of ui/core.py, checked against the class, everything assigned anywhere in it,
and the handful of attributes the orchestrator injects from outside.
"""
import ast
import inspect
from pathlib import Path

import pytest

from ui.core import UI

# Assigned onto the UI instance by App, from outside the class.
EXTERNALLY_INJECTED = {
    "multi_load_service", "tournament_service", "advanced_drawing_processor",
    "param_lock_service", "field_bus", "brain_preview", "thumb_cache",
    "map_thumb_cache", "audio_runtime", "archive", "archive_store",
    "goal_list", "imgep_driver", "auto_service",
}


def _class_node():
    source = Path(inspect.getfile(UI)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "UI":
            return node
    pytest.fail("could not find the UI class in ui/core.py")


def _assigned_anywhere(class_node):
    """Every name assigned as self.<name> anywhere in the class body."""
    names = set()
    for node in ast.walk(class_node):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if (isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self"):
                    names.add(sub.attr)
        if isinstance(node, ast.For):
            for sub in ast.walk(node.target):
                if (isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self"):
                    names.add(sub.attr)
    return names


def _read_in(method_node):
    """Every name READ as self.<name> in a method (stores excluded)."""
    names = set()
    for node in ast.walk(method_node):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Load)):
            names.add(node.attr)
    return names


def test_render_reads_no_attribute_that_does_not_exist():
    class_node = _class_node()
    render = next((n for n in class_node.body
                   if isinstance(n, ast.FunctionDef) and n.name == "render"), None)
    assert render is not None, "UI.render disappeared"

    known = set(dir(UI)) | _assigned_anywhere(class_node) | EXTERNALLY_INJECTED
    missing = sorted(_read_in(render) - known)
    assert not missing, (
        f"UI.render() reads attribute(s) that exist nowhere: {missing}. "
        "A window dispatch gate is the usual culprit - visibility flags live "
        "on self.state.preferences, not on self.")


def test_the_guard_would_catch_a_bad_dispatch():
    """The check is only worth having if it actually rejects the mistake."""
    class_node = _class_node()
    known = set(dir(UI)) | _assigned_anywhere(class_node) | EXTERNALLY_INJECTED
    assert "show_sidebar" not in known, (
        "show_sidebar is a preference, so a bare self.show_sidebar must fail")
