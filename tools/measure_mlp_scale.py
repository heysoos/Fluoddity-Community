"""Where the MLP's decode scales stop doing anything.

A scale so small the hidden layer is effectively linear, or so large every unit
sits railed, is a slider position that changes nothing you can see - and both
ends of `w_scale`/`b_scale` are also the ends of what an audio mapping can
sweep. Evaluated as a PURE function over a fixed input grid, one invocation per
point: the sim does not reproduce itself run to run, so a trajectory tells you
nothing about a decode change.

    python -m tools.measure_mlp_scale
"""
from __future__ import annotations

import numpy as np
import ui  # noqa: F401  - imported first, or services/ hits a circular import

from services import brains
from services.brains.mlp import IN_DIM, OUT_DIM, layer_spans

# The brain's input scale is a property of the PRESET; these bracket the
# library's median and p90. See the brain input scale caveat in CLAUDE.md.
GRIDS = {"typical |x|~0.06": 0.06, "lively |x|~0.36": 0.36}
W_SCALES = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
B_SCALES = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0)
LAYERS = [[16, 0]]


def _forward(p, layout, x):
    """The MLP as a pure function, mirroring mlp.glsl's accumulation order."""
    hidden, out_w, out_b, _n = layer_spans(layout.shape)
    h = x
    for w_off, b_off, fan_in, w in hidden:
        h = np.tanh(h @ p[w_off:b_off].reshape(w, fan_in).T
                    + p[b_off:b_off + w])
    fan = hidden[-1][3]
    return (h @ p[out_w:out_b].reshape(OUT_DIM, fan).T
            + p[out_b:out_b + OUT_DIM]), h


def main() -> None:
    m = brains.get("mlp")
    rng = np.random.default_rng(7)
    base = m.layout_from_settings({"layers": LAYERS})
    z = rng.normal(0.0, 0.5, base.length).astype(np.float32)

    for label, amp in GRIDS.items():
        x = (rng.standard_normal((2048, IN_DIM)) * amp).astype(np.float32)
        print(f"\n=== weight scale, {label} ===")
        print(f"{'w_scale':>8} {'|out| p50':>10} {'|out| p90':>10} "
              f"{'railed':>8}")
        for ws in W_SCALES:
            layout = m.layout_from_settings({"layers": LAYERS, "w_scale": ws})
            out, h = _forward(m.decode(z, layout), layout, x)
            print(f"{ws:8.2f} {np.percentile(np.abs(out), 50):10.4f} "
                  f"{np.percentile(np.abs(out), 90):10.4f} "
                  f"{np.mean(np.abs(h) > 0.99):8.3f}")

    x = (rng.standard_normal((2048, IN_DIM)) * GRIDS["lively |x|~0.36"]
         ).astype(np.float32)
    print("\n=== bias scale, at the default weight scale ===")
    print(f"{'b_scale':>8} {'|out| p50':>10} {'railed':>8} {'spread':>8}")
    for bs in B_SCALES:
        layout = m.layout_from_settings({"layers": LAYERS, "b_scale": bs})
        out, h = _forward(m.decode(z, layout), layout, x)
        print(f"{bs:8.2f} {np.percentile(np.abs(out), 50):10.4f} "
              f"{np.mean(np.abs(h) > 0.99):8.3f} {out.std():8.4f}")


if __name__ == "__main__":
    main()
