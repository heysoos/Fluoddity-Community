"""The layer stack is pure data: it round-trips through JSON unchanged.

`error` is runtime state, so it must never survive serialization - a layer
whose shader failed last session must not open pre-broken.
"""
import json

from state.field_stack import (
    FieldLayer, FieldStack, new_uid, stack_to_dict, stack_from_dict,
    MAPPINGS, DESTINATIONS, BLENDS,
)


def test_uids_are_unique():
    assert new_uid() != new_uid()


def test_a_new_layer_gets_a_uid():
    assert FieldLayer().uid


def test_round_trip_preserves_every_field():
    stack = FieldStack(layers=[
        FieldLayer(source="noise", mapping="curl", destination="force",
                   blend="add", strength=0.8, blur=2.0, sign=-1.0,
                   params={"speed": 1.5, "octaves": 3}),
        FieldLayer(source="brush", mapping="rg_direct", destination="strafe",
                   blend="replace", strength=1.0, enabled=False),
    ])
    restored = stack_from_dict(json.loads(json.dumps(stack_to_dict(stack))))

    assert len(restored.layers) == 2
    for before, after in zip(stack.layers, restored.layers):
        assert after.uid == before.uid
        assert after.enabled == before.enabled
        assert after.source == before.source
        assert after.params == before.params
        assert after.mapping == before.mapping
        assert after.destination == before.destination
        assert after.blend == before.blend
        assert after.strength == before.strength
        assert after.blur == before.blur
        assert after.sign == before.sign


def test_error_is_not_serialized():
    stack = FieldStack(layers=[FieldLayer(error="0:31 no matching function")])
    assert "error" not in stack_to_dict(stack)["layers"][0]
    assert stack_from_dict(stack_to_dict(stack)).layers[0].error is None


def test_an_empty_dict_is_an_empty_stack():
    assert stack_from_dict({}).layers == []


def test_an_unknown_enum_value_falls_back_to_the_default():
    d = {"layers": [{"uid": "a", "mapping": "nonsense", "blend": "nonsense",
                     "destination": "nonsense"}]}
    layer = stack_from_dict(d).layers[0]
    assert layer.mapping in MAPPINGS
    assert layer.blend in BLENDS
    assert layer.destination in DESTINATIONS
