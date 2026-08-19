"""Every attribute the UI reads off self must actually exist ON the UI.

The render tests drive a bare MIXIN through a stub harness, never the real UI,
so a method naming something that exists nowhere compiles, passes the whole
suite, and crashes on the first frame the app draws. Two shipped that way: a
dispatch reading `self.show_sidebar` where the flag lives on
`self.state.preferences`, and a call to `self._delayed_tooltip`, a helper
deleted when tooltips moved to `ui.hints` - which a harness had stubbed, so
the mixin's own test could never see it.

Covering `render()` alone was not enough: the second one was a level deeper,
inside a window body. So this walks EVERY method of the UI class and of every
mixin it inherits.

Static, so it needs no GL context and no window.
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


def _ui_classes():
    """The UI class and every mixin it inherits, with their source files."""
    return [c for c in UI.__mro__ if c is not object]


def _known_attributes():
    """Everything the assembled UI can legitimately answer to."""
    known = set(dir(UI)) | EXTERNALLY_INJECTED
    for cls in _ui_classes():
        node = _class_node_for(cls)
        if node is not None:
            known |= _assigned_anywhere(node)
    return known


def _class_node_for(cls):
    try:
        source = Path(inspect.getfile(cls)).read_text(encoding="utf-8")
    except (OSError, TypeError):
        return None
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            return node
    return None


def _methods_of(class_node):
    return [n for n in class_node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def test_no_ui_method_reads_an_attribute_that_does_not_exist():
    known = _known_attributes()
    offences = []
    for cls in _ui_classes():
        node = _class_node_for(cls)
        if node is None:
            continue
        for method in _methods_of(node):
            for name in sorted(_read_in(method) - known):
                offences.append(f"{cls.__name__}.{method.name} -> self.{name}")
    assert not offences, (
        "these read attributes that exist nowhere on the assembled UI:\n  "
        + "\n  ".join(offences)
        + "\nA visibility flag lives on self.state.preferences; a tooltip "
          "goes through ui.hints.tip.")


def test_the_guard_rejects_both_shipped_mistakes():
    """A guard that has never been shown to fail is imaginary coverage."""
    known = _known_attributes()
    assert "show_sidebar" not in known, (
        "show_sidebar is a preference, so a bare self.show_sidebar must fail")
    assert "_delayed_tooltip" not in known, (
        "the tooltip helper is ui.hints.tip; self._delayed_tooltip must fail")


def test_it_actually_looks_at_the_window_bodies():
    """Covering render() alone missed the second bug, so prove the reach."""
    names = set()
    for cls in _ui_classes():
        node = _class_node_for(cls)
        if node is not None:
            names |= {m.name for m in _methods_of(node)}
    assert "render_field_stack_window" in names
    assert "render_archive_window" in names
