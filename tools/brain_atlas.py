"""Render the Brain Inspector atlas to a PNG, without launching the app.

Same BrainPreview the Inspector uses, so a tile here is the same evidence a tile
there is: the shipped brain GLSL, prepended the same way, over the same
parameter buffer.

usage: brain_atlas.py <tree-root> <modality> <out.png> [key=value ...]
       brain_atlas.py . gabor gabor.png input_scale=0.5 filters=12

Settings are the modality's own settings_schema() keys; ints and floats are both
written as plain numbers. --range sets the half-width of the swept input box and
defaults to the modality's own input scale where it has one, so a filter is
drawn over the region it was designed for rather than a fixed window.
"""
import os
import sys
from pathlib import Path

TREE = sys.argv[1]
MODALITY = sys.argv[2]
OUT = sys.argv[3]

os.chdir(TREE)
sys.path.insert(0, TREE)

import numpy as np
import ui  # noqa: F401
import moderngl

from services.brain_preview import BrainPreview
from services.brains import get
from utilities.gl_helpers import pack_brains


def parse_overrides(argv):
    out, box = {}, None
    for a in argv:
        k, _, v = a.partition("=")
        if k == "--range":
            box = float(v)
        else:
            out[k] = float(v)
    return out, box


def main():
    overrides, box = parse_overrides(sys.argv[4:])
    modality = get(MODALITY)
    settings = {s.key: s.default for s in modality.settings_schema()}
    settings.update(overrides)
    layout = modality.layout_from_settings(settings)
    if box is None:
        # Draw a unit over the region it was built for. 3x the input scale
        # reaches well past the outermost centre.
        box = 3.0 * layout.scale("input_scale", 2.0)

    ctx = moderngl.create_standalone_context(require=430)
    preview = BrainPreview(ctx)

    # A brain drawn from the modality's own generator, so the picture is of a
    # typical member of the family the search actually explores.
    rng = np.random.default_rng(0)
    params = np.asarray(modality.random(rng, layout), np.float32).reshape(-1)
    buf = ctx.buffer(pack_brains([params], layout))

    tex = preview.render(layout, buf, axes=(0, 2), channel=0,
                         value_range=box, gain=1.0)
    w, h = tex.size
    img = np.frombuffer(tex.read(), dtype=np.uint8).reshape(h, w, 4)[::-1]

    try:
        from PIL import Image
        Image.fromarray(img, "RGBA").save(OUT)
    except ImportError:
        # No Pillow: a minimal uncompressed-deflate PNG, so the tool still works
        import struct
        import zlib
        raw = b"".join(b"\0" + img[y].tobytes() for y in range(h))
        def chunk(tag, data):
            c = tag + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
        Path(OUT).write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))

    print(f"{OUT}  {MODALITY} {layout.signature()}  "
          f"{preview.unit_count(layout)} units + total, box +/-{box:g}")
    print(f"  settings: {settings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
