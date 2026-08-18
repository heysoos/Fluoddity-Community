"""The encoder combos, in one place.

Three call sites want the same list with the same per-option explanation: the
Auto tab's picker, the New Archive modal's choice, and the Explore tab's
readout. Built from begin_combo rather than combo() because a plain combo has
no per-item hover, and which encoder to use is exactly the decision that needs
one while the list is open.
"""
from imgui_bundle import imgui

from ui import hints


def encoder_combo(label: str, current: str) -> tuple[bool, str]:
    """Pick an encoder. -> (changed, key)."""
    from services.vision_models import REGISTRY

    keys = sorted(REGISTRY)
    if current not in keys:
        current = keys[0]
    chosen = current
    if imgui.begin_combo(label, REGISTRY[current].label):
        for key in keys:
            model = REGISTRY[key]
            if imgui.selectable(model.label, key == current)[0]:
                chosen = key
            hints.tip(model.blurb)
        imgui.end_combo()
    return chosen != current, chosen


def encoder_readout(label: str, key: str) -> None:
    """Show an encoder that cannot be changed here."""
    from services.vision_models import REGISTRY, get

    model = REGISTRY.get(key) or get("clip-b32")
    imgui.begin_disabled()
    imgui.begin_combo(label, model.label)
    imgui.end_disabled()
    hints.tip(model.blurb)
