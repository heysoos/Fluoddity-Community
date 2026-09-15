"""A label for an action the user's file predates comes from the default."""
import json

from utilities.keybinding_management import KeybindingManager


def test_a_missing_action_is_labelled_from_the_default(tmp_path):
    user = tmp_path / "keys.json"
    user.write_text(json.dumps({"toggle_pause": "SPACE"}), encoding="utf-8")
    default = tmp_path / "default.json"
    default.write_text(json.dumps({"toggle_pause": "SPACE",
                                   "toggle_perform_mode": "F11"}),
                       encoding="utf-8")
    kb = KeybindingManager(config_path=user, default_config_path=default)
    assert kb.get_key_display_name("toggle_perform_mode") == "F11"
    assert kb.get_key_display_name("toggle_pause") == "SPACE"
    assert kb.get_key_display_name("no_such_action") == "?"
