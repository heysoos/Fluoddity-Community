"""A stack travels with a preset, and a preset written before it still opens.

The legacy path matters most: every config on disk predates this feature, and
one with a companion _fields.png must come back as a brush layer holding that
texture rather than as an empty stack.
"""
from services.config_saver import PhysicsConfig, legacy_brush_stack
from state.field_stack import stack_from_dict


def test_a_new_config_has_an_empty_stack():
    assert PhysicsConfig().field_stack == {}


def test_the_stack_round_trips_through_the_config():
    stack = {"layers": [{"uid": "abc", "source": "noise", "mapping": "curl",
                         "destination": "force", "blend": "add",
                         "strength": 0.5, "blur": 1.0, "sign": 1.0,
                         "enabled": True, "params": {"scale": 8.0}}]}
    config = PhysicsConfig(field_stack=stack)
    restored = PhysicsConfig.from_dict(config.to_dict())
    assert restored.field_stack == stack


def test_a_config_written_before_this_feature_opens_with_an_empty_stack():
    d = PhysicsConfig().to_dict()
    d.pop("field_stack", None)
    assert PhysicsConfig.from_dict(d).field_stack == {}


def test_the_legacy_migration_is_one_brush_layer():
    stack = stack_from_dict(legacy_brush_stack())
    assert len(stack.layers) == 2
    assert [l.source for l in stack.layers] == ["brush", "brush"]
    assert {l.destination for l in stack.layers} == {"force", "strafe"}
    for layer in stack.layers:
        assert layer.mapping == "rg_direct"
        assert layer.blend == "replace"


def test_the_legacy_migration_reads_the_channels_the_old_field_used():
    stack = stack_from_dict(legacy_brush_stack())
    force = next(l for l in stack.layers if l.destination == "force")
    strafe = next(l for l in stack.layers if l.destination == "strafe")
    assert force.params.get("_channels") == "xy"
    assert strafe.params.get("_channels") == "zw"
