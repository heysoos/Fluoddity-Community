"""A set list round-trips through disk and refuses to escape its folder."""
from services import set_list_io
from state.set_list import DEFAULT_SLOT_KEYS, KnobBinding, SetList, Slot


def test_the_default_set_list_offers_the_number_row_and_no_presets():
    sl = SetList.default("show")
    assert [s.key for s in sl.slots] == list(DEFAULT_SLOT_KEYS)
    assert all(s.filename == "" for s in sl.slots)
    assert sl.slot_for_key("1") is None


def test_a_slot_is_found_by_key_in_either_case():
    sl = SetList(slots=[Slot("Q", "Core", "reef")])
    assert sl.slot_for_key("q").filename == "reef"
    assert sl.knob_for_cc(3) is None


def test_an_empty_slot_does_not_shadow_a_filled_one_on_the_same_key():
    sl = SetList(slots=[Slot("1"), Slot("1", "Custom", "karst")])
    assert sl.slot_for_key("1").filename == "karst"


def test_a_set_list_round_trips(tmp_path):
    sl = SetList(name="open house",
                 slots=[Slot("1", "Exhibition", "reef", rig="loud"), Slot("2")],
                 knobs=[KnobBinding(cc=4, param="SENSOR_GAIN", step=0.02,
                                    rest_colour=10, active_colour=100)])
    set_list_io.save_set_list(sl, tmp_path)
    assert set_list_io.list_set_lists(tmp_path) == ["open house"]
    back = set_list_io.load_set_list("open house", tmp_path)
    assert back == sl


def test_a_typed_name_cannot_write_outside_the_folder(tmp_path):
    sl = SetList(name="../escape")
    path = set_list_io.save_set_list(sl, tmp_path)
    assert path.parent == tmp_path


def test_a_missing_or_broken_file_reads_as_none(tmp_path):
    assert set_list_io.load_set_list("nope", tmp_path) is None
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert set_list_io.load_set_list("bad", tmp_path) is None
    assert set_list_io.list_set_lists(tmp_path / "absent") == []


def test_a_file_written_by_hand_is_tolerated():
    sl = SetList.from_dict({"slots": [{"key": "q"}, "junk"],
                            "knobs": [{"cc": 300, "param": "DRAG"}]})
    assert sl.slots[0].key == "Q"
    assert len(sl.slots) == 1
    assert sl.knobs[0].cc == 300 & 127


def test_a_slot_naming_only_a_rig_is_still_a_slot():
    sl = SetList(slots=[Slot("3", rig="quiet")])
    assert sl.slot_for_key("3").rig == "quiet"
