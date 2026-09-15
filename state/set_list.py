"""A set list: which key loads which preset, and which knob nudges which slider.

One file per show, under Documents/Fluoddity/setlists. Keys are the bare names
the keybinding file uses ("1", "Q"); a knob is a MIDI CC number.
"""
from dataclasses import dataclass, field

DEFAULT_SLOT_KEYS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")

# The Twister's colour wheel runs 1..126; 0 hands the encoder its own default.
DEFAULT_REST_COLOUR = 41
DEFAULT_ACTIVE_COLOUR = 86
DEFAULT_STEP = 0.01


@dataclass
class Slot:
    key: str = ""        # a KEY_NAME_TO_GLFW name, upper case
    category: str = ""   # a File > Load category ("Core", "Custom", a folder)
    filename: str = ""   # the config's stem, "" for an empty slot
    rig: str = ""        # a named audio rig to load with it, "" leaves the rig alone

    def to_dict(self) -> dict:
        return {"key": self.key, "category": self.category,
                "filename": self.filename, "rig": self.rig}

    @classmethod
    def from_dict(cls, d: dict) -> "Slot":
        return cls(key=str(d.get("key", "")).upper(),
                   category=str(d.get("category", "")),
                   filename=str(d.get("filename", "")),
                   rig=str(d.get("rig", "")))


@dataclass
class KnobBinding:
    cc: int = 0                      # encoder CC number, 0..127
    param: str = ""                  # a SimState field name from PHYSICS_PARAMS
    step: float = DEFAULT_STEP       # fraction of the slider range per detent
    rest_colour: int = DEFAULT_REST_COLOUR
    active_colour: int = DEFAULT_ACTIVE_COLOUR

    def to_dict(self) -> dict:
        return {"cc": self.cc, "param": self.param, "step": self.step,
                "rest_colour": self.rest_colour,
                "active_colour": self.active_colour}

    @classmethod
    def from_dict(cls, d: dict) -> "KnobBinding":
        return cls(cc=int(d.get("cc", 0)) & 127,
                   param=str(d.get("param", "")),
                   step=float(d.get("step", DEFAULT_STEP)),
                   rest_colour=int(d.get("rest_colour", DEFAULT_REST_COLOUR)) & 127,
                   active_colour=int(d.get("active_colour", DEFAULT_ACTIVE_COLOUR)) & 127)


@dataclass
class SetList:
    name: str = ""
    slots: list = field(default_factory=list)   # list[Slot]
    knobs: list = field(default_factory=list)   # list[KnobBinding]

    @classmethod
    def default(cls, name: str = "") -> "SetList":
        return cls(name=name, slots=[Slot(key=k) for k in DEFAULT_SLOT_KEYS])

    def slot_for_key(self, key: str) -> Slot | None:
        """The first slot bound to `key` that names a preset or a rig."""
        key = key.upper()
        for s in self.slots:
            if s.key == key and (s.filename or s.rig):
                return s
        return None

    def knob_for_cc(self, cc: int) -> KnobBinding | None:
        for k in self.knobs:
            if k.cc == cc and k.param:
                return k
        return None

    def to_dict(self) -> dict:
        return {"name": self.name,
                "slots": [s.to_dict() for s in self.slots],
                "knobs": [k.to_dict() for k in self.knobs]}

    @classmethod
    def from_dict(cls, d: dict) -> "SetList":
        slots = [Slot.from_dict(s) for s in d.get("slots", []) if isinstance(s, dict)]
        knobs = [KnobBinding.from_dict(k) for k in d.get("knobs", []) if isinstance(k, dict)]
        return cls(name=str(d.get("name", "")), slots=slots, knobs=knobs)
