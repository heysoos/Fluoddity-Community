"""Labelled figures from a phase-diagram sweep.

    python -m tools.phase_figure <sweep dir> --feature participation_ratio
    python -m tools.phase_figure <sweep dir> --contact

`tools/phase_view.py` writes the bare raster - the pixels and nothing else,
which is what a large print wants. This writes the READABLE version: axes with
real parameter values, a colourbar that says which end is which, and a header
naming the preset, the brain and what was run.

Drawn with PIL rather than matplotlib, which is not a dependency here and would
follow the app into a PyInstaller build for the sake of a few tools.

The colour ramps are lightness-monotonic (viridis, magma), so magnitude reads as
brightness even in greyscale or under colour-vision deficiency; the hue is
decoration on top of a channel that already carries the value. A rainbow ramp
would not have that property and is never used.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ui  # noqa: F401,E402  prime the services/ui import cycle

from tools.phase_view import (fill_holes, load, normalise,
                              recompute_series_columns)

# Dark surface: the subject is a glowing particle field, and a light ground
# washes out the low end of every ramp used here.
INK = (236, 239, 244)
INK_DIM = (150, 158, 172)
INK_FAINT = (96, 103, 118)
SURFACE = (14, 16, 21)
PANEL = (20, 23, 30)
RULE = (52, 58, 72)

FONTS = Path("C:/Windows/Fonts")


def font(size: int, weight: str = "regular"):
    names = {"regular": "segoeui.ttf", "semibold": "segoeuisl.ttf",
             "bold": "segoeuib.ttf", "light": "segoeuil.ttf",
             "mono": "consola.ttf"}
    try:
        return ImageFont.truetype(str(FONTS / names[weight]), size)
    except OSError:
        return ImageFont.load_default()


class Channel(NamedTuple):
    """What a feature is called, which end is which, and how it is computed.

    The formula is on the figure because a phase diagram is an argument about a
    quantity, and a reader cannot check the argument against a name. Kept in one
    table so a channel cannot acquire a plot without acquiring a definition -
    `tests/test_phase_figure.py` fails when one does.
    """
    title: str
    low: str
    high: str
    formula: str
    note: str


# rho is the per-texel trail magnitude ||canvas.xy||; N is the texel count.
CHANNELS: dict[str, Channel] = {
    "participation_ratio": Channel(
        "Trail concentration", "one blob", "spread evenly",
        "PR  =  (Σρ)²  /  (N · Σρ²)",
        "mass sitting in m of N texels scores m/N. Scale-free: it sees shape, "
        "never amount"),
    "coverage": Channel(
        "Fraction of canvas visited", "empty", "full",
        "C  =  fraction of texels with  ρ > τ,   τ = 0.01",
        "τ is one particle's deposit. Absolute, so cells stay comparable"),
    "change_rate": Channel(
        "Field turnover per probe", "frozen", "churning",
        "R  =  mean[ Σ|ρ(t) − ρ(t−k)|  /  Σρ(t) ]  over the last ¼ of probes",
        "k = 50 steps. Normalised by its own mass, so a faded field still "
        "scores high"),
    "structure": Channel(
        "Spatial self-similarity", "white noise", "coherent",
        "S  =  max | mean(g · g+l) / mean(g²) |   for lag l in "
        "{1,2,3,4,6,8,12,16}, both axes",
        "g = ρ − mean(ρ), and g+l is g shifted by l texels. Lag 1 alone scores a "
        "3px lattice exactly what noise scores, hence the several lags"),
    "field_order": Channel(
        "Flow alignment", "disordered", "one coherent flow",
        "Φ  =  ‖ Σ v ‖  /  Σ‖v‖      (v = trail vector per texel)",
        "the ρ-weighted mean unit vector — the Vicsek order parameter, read off "
        "the field"),
    "polar_order": Channel(
        "Particle alignment", "disordered", "one flock",
        "Ψ  =  ‖ (1/n) Σ v/‖v‖ ‖      over particles",
        "direction only; a fast swarm and a slow one going the same way score "
        "alike"),
    "particle_pr": Channel(
        "Particle clustering", "clumped", "uniform",
        "PR of a 64×64 histogram of particle positions",
        "the same participation ratio, asked of where particles ARE rather than "
        "of the trail they left"),
    "spec_entropy": Channel(
        "Spectral breadth", "one sharp scale", "broadband",
        "H  =  −Σ q ln q  /  ln B,    q = P(k) / ΣP",
        "P(k) is the 2-D FFT power AVERAGED over each annulus; a sum would make "
        "white noise slope upward"),
    "spec_peak_wavelen": Channel(
        "Characteristic scale", "fine", "canvas-sized",
        "λ  =  W / argmax P(k),   sub-bin by a parabola through ln P",
        "in texels; W is the canvas width. The interpolation is what stops λ "
        "terracing at W, W/2, W/3"),
    "speed_p50": Channel(
        "Median particle speed", "still", "fast",
        "median ‖v‖  over the particle subsample", "canvas units per step"),
    "speed_p90": Channel(
        "90th pct particle speed", "still", "fast",
        "90th percentile ‖v‖  over the particle subsample",
        "canvas units per step"),
    "alive_steps": Channel(
        "Steps before dying", "never lived", "survived",
        "last t with  mean(ρ) >= 1e-4  and  0.02 <= PR <= 0.98",
        "the mass floor is load-bearing: PR is scale-free and reads a faded "
        "canvas as healthy"),
    "rho_mean": Channel(
        "Mean trail mass", "faded", "dense", "mean(ρ)  over all texels",
        "the free variable here — the canvas is a velocity field, so a slow "
        "particle deposits almost nothing"),
}

# Kept as the (title, low, high) view the colourbar and header already use.
MEANING = {k: (c.title, c.low, c.high) for k, c in CHANNELS.items()}

RAMPS = {
    "viridis": np.array([
        (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
        (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
        (180, 222, 44), (253, 231, 37)], dtype=np.float64),
    "magma": np.array([
        (0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129),
        (181, 54, 122), (229, 80, 100), (251, 135, 97), (254, 194, 135),
        (252, 253, 191)], dtype=np.float64),
}


def ramp_image(norm: np.ndarray, ramp: str) -> np.ndarray:
    table = RAMPS[ramp]
    x = np.nan_to_num(norm, nan=0.0) * (len(table) - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(table) - 1)
    t = (x - lo)[..., None]
    rgb = table[lo] * (1 - t) + table[hi] * t
    rgb[~np.isfinite(norm)] = PANEL
    return rgb.astype(np.uint8)


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Round values inside [lo, hi]; a phase boundary usually sits on one."""
    span = hi - lo
    raw = span / max(1, count - 1)
    mag = 10.0 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10)),
               key=lambda s: abs(s - raw))
    first = np.ceil(lo / step) * step
    out = []
    v = first
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def fmt(v: float) -> str:
    if abs(v) >= 1000 or (v != 0 and abs(v) < 0.01):
        return f"{v:.3g}"
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


class Figure:
    """One panel, its axes, its colourbar and its header."""

    PAD_L, PAD_R = 96, 128
    PAD_T, PAD_B = 160, 122   # PAD_T carries title, formula and gloss

    def __init__(self, plot_px: int = 620):
        self.plot = plot_px
        self.w = self.PAD_L + plot_px + self.PAD_R
        self.h = self.PAD_T + plot_px + self.PAD_B
        self.img = Image.new("RGB", (self.w, self.h), SURFACE)
        self.d = ImageDraw.Draw(self.img)

    def _text(self, xy, s, f, fill, anchor="la"):
        self.d.text(xy, s, font=f, fill=fill, anchor=anchor)

    def panel(self, rgb: np.ndarray) -> None:
        im = Image.fromarray(rgb).resize((self.plot, self.plot), Image.NEAREST)
        self.img.paste(im, (self.PAD_L, self.PAD_T))
        self.d.rectangle(
            [self.PAD_L - 1, self.PAD_T - 1,
             self.PAD_L + self.plot, self.PAD_T + self.plot],
            outline=RULE, width=1)

    def axes(self, meta) -> None:
        (x0, x1), (y0, y1) = meta["x_range"], meta["y_range"]
        L, T, P = self.PAD_L, self.PAD_T, self.plot
        tick, lab = font(13), font(15)

        for v in nice_ticks(x0, x1):
            px = L + (v - x0) / (x1 - x0) * (P - 1)
            self.d.line([px, T + P, px, T + P + 6], fill=RULE, width=1)
            self._text((px, T + P + 12), fmt(v), tick, INK_DIM, "ma")
        for v in nice_ticks(y0, y1):
            py = T + (1 - (v - y0) / (y1 - y0)) * (P - 1)
            self.d.line([L - 6, py, L, py], fill=RULE, width=1)
            self._text((L - 12, py), fmt(v), tick, INK_DIM, "rm")

        self._text((L + P / 2, T + P + 42), meta["x_param"], lab, INK, "ma")
        # Rotated y label, drawn on its own layer because PIL cannot rotate text
        # in place.
        strip = Image.new("RGB", (P, 26), SURFACE)
        ImageDraw.Draw(strip).text((P / 2, 13), meta["y_param"], font=lab,
                                   fill=INK, anchor="mm")
        self.img.paste(strip.rotate(90, expand=True), (L - 78, T))

    def zero_guides(self, meta) -> None:
        """Faint lines on the axes' zeros.

        Both phase boundaries here sit on a sign change, so the reader is
        constantly asking where zero is; a dashed rule answers it without
        competing with the data.
        """
        (x0, x1), (y0, y1) = meta["x_range"], meta["y_range"]
        L, T, P = self.PAD_L, self.PAD_T, self.plot
        if x0 < 0 < x1:
            px = L + (0 - x0) / (x1 - x0) * (P - 1)
            for y in range(T, T + P, 10):
                self.d.line([px, y, px, min(y + 4, T + P)], fill=(255, 255, 255),
                            width=1)
        if y0 < 0 < y1:
            py = T + (1 - (0 - y0) / (y1 - y0)) * (P - 1)
            for x in range(L, L + P, 10):
                self.d.line([x, py, min(x + 4, L + P), py], fill=(255, 255, 255),
                            width=1)

    def marker(self, meta, label: str) -> None:
        phys = (meta.get("config") or {}).get("physics") or {}
        xv = phys.get(meta["x_param"].lower())
        yv = phys.get(meta["y_param"].lower())
        if xv is None or yv is None:
            return
        (x0, x1), (y0, y1) = meta["x_range"], meta["y_range"]
        L, T, P = self.PAD_L, self.PAD_T, self.plot
        cx = L + (xv - x0) / (x1 - x0) * (P - 1)
        cy = T + (1 - (yv - y0) / (y1 - y0)) * (P - 1)

        self.d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=(0, 0, 0), width=4)
        self.d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=(255, 255, 255),
                       width=2)
        f = font(13, "semibold")
        tx, anchor = (cx + 14, "lm") if cx < L + P * 0.7 else (cx - 14, "rm")
        text = f"{label}  ({fmt(xv)}, {fmt(yv)})"
        box = self.d.textbbox((tx, cy), text, font=f, anchor=anchor)
        self.d.rectangle([box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4],
                         fill=(0, 0, 0))
        self._text((tx, cy), text, f, (255, 255, 255), anchor)

    def colourbar(self, lo: float, hi: float, ramp: str, feature: str) -> None:
        L, T, P = self.PAD_L, self.PAD_T, self.plot
        x, w, h = L + P + 26, 16, int(P * 0.62)
        y = T + (P - h) // 2
        grad = np.linspace(1, 0, h)[:, None].repeat(w, axis=1)
        self.img.paste(Image.fromarray(ramp_image(grad, ramp)), (x, y))
        self.d.rectangle([x - 1, y - 1, x + w, y + h], outline=RULE, width=1)

        tick = font(12, "mono")
        self._text((x + w + 8, y), fmt(hi), tick, INK_DIM, "lm")
        self._text((x + w + 8, y + h), fmt(lo), tick, INK_DIM, "lm")

        _, low_word, high_word = MEANING.get(feature, ("", "low", "high"))
        small = font(12)
        for yy, word, anch in ((y - 14, high_word, "ls"), (y + h + 18, low_word, "ls")):
            self._text((x - 2, yy), word, small, INK_FAINT, anch)

    def caption(self, feature: str) -> int:
        """The formula and its one-line gloss. Returns the y it ended at.

        A phase diagram is an argument about a quantity, and a name is not
        enough to check the argument against - so the definition travels with
        the picture rather than living only in the source.
        """
        ch = CHANNELS.get(feature)
        if ch is None:
            return 62
        right = self.w - self.PAD_R + 100
        width_px = right - self.PAD_L

        y = 66
        self._text((self.PAD_L, y), ch.formula, font(13, "mono"), INK_DIM, "la")
        y += 22

        # Wrap on measured pixels: the note is proportional text and a character
        # count mispredicts its width by enough to overrun the panel.
        note, f = ch.note, font(12)
        avg = max(1.0, self.d.textlength("abcdefghij", font=f) / 10.0)
        for line in textwrap.wrap(note, width=max(20, int(width_px / avg))):
            self._text((self.PAD_L, y), line, f, INK_FAINT, "la")
            y += 15
        return y

    def header(self, meta, feature: str, done: int, total: int,
               compact: bool = False) -> None:
        title, _, _ = MEANING.get(feature, (feature.replace("_", " ").title(),
                                            "", ""))
        self._text((self.PAD_L, 30), title, font(27, "semibold"), INK, "ls")
        self._text((self.PAD_L, 40), feature, font(12, "mono"), INK_FAINT, "la")

        # On a sheet every panel carries the same run, so stating it per panel
        # is six copies of one fact - and at panel width the two collide.
        if compact:
            self._rule(self.caption(feature) + 8)
            return

        cfg = meta.get("config") or {}
        cohorts = (cfg.get("settings") or {}).get("num_cohorts")
        bits = [f"{meta['preset_name']}", meta["brain_layout"],
                f"{meta['steps']} steps", f"world {meta['world_size']}",
                f"{meta['canvas'][0]}px canvas"]
        if cohorts:
            bits.append(f"{cohorts} cohorts")
        self._text((self.w - self.PAD_R + 100, 30), " · ".join(bits),
                   font(13), INK_DIM, "rs")
        pct = 100.0 * done / max(1, total)
        note = (f"{meta['grid']}×{meta['grid']} grid · {done}/{total} cells"
                + ("" if done == total else f" ({pct:.0f}%, coarse cells filled)"))
        self._text((self.w - self.PAD_R + 100, 48), note, font(12), INK_FAINT,
                   "ra")
        # Below the caption, not at a fixed 68 - the formula lives there now and
        # a fixed rule struck straight through it.
        self._rule(self.caption(feature) + 8)

    def _rule(self, y: float) -> None:
        self.d.line([self.PAD_L, y, self.w - self.PAD_R + 100, y], fill=RULE,
                    width=1)

    def footer(self, text: str) -> None:
        self._rule(self.h - 46)
        self._text((self.PAD_L, self.h - 22), text, font(11), INK_FAINT, "ls")


def make(path: Path, feature: str, ramp: str, lo_pct: float, hi_pct: float,
         plot_px: int, label: str, compact: bool = False
         ) -> tuple[Image.Image, str]:
    data, meta = load(path)
    feats = recompute_series_columns(data, meta)
    names = [str(n) for n in data["cell_names"]]
    if feature not in names:
        raise SystemExit(f"no feature {feature!r}; have {names}")

    raw = feats[:, :, names.index(feature)]
    plane = np.flipud(fill_holes(raw, data["done"]))
    finite = raw[data["done"]]
    finite = finite[np.isfinite(finite)]
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])

    fig = Figure(plot_px)
    fig.panel(ramp_image(normalise(plane, lo_pct, hi_pct), ramp))
    fig.zero_guides(meta)
    fig.axes(meta)
    fig.marker(meta, label)
    fig.colourbar(lo, hi, ramp, feature)
    fig.header(meta, feature, int(data["done"].sum()), data["done"].size,
               compact)
    if not compact:
        fig.footer(f"colour spans the {lo_pct:g}–{hi_pct:g} percentile · "
                   f"frame features averaged over "
                   f"{meta.get('measure_frames', 1)} late probes · "
                   f"seed fixed across all cells")
    return fig.img, feature


def run_line(meta) -> str:
    cohorts = (meta.get("config") or {}).get("settings", {}).get("num_cohorts")
    bits = [meta["brain_layout"], f"{meta['steps']} steps",
            f"world {meta['world_size']}", f"{meta['canvas'][0]}px canvas"]
    if cohorts:
        bits.append(f"{cohorts} cohorts")
    return " · ".join(bits)


def contact(path: Path, features: list[str], ramp: str, cell_px: int,
            label: str) -> Image.Image:
    """Several channels side by side, under ONE statement of the run.

    Each panel keeps its own title, colourbar and axes - they encode different
    quantities and cannot share a scale - but everything true of the whole
    sweep is said once, at the top.
    """
    _, meta = load(path)
    tiles = [make(path, f, ramp, 2.0, 98.0, cell_px, label, compact=True)[0]
             for f in features]
    cols = 3 if len(tiles) > 4 else 2
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    head, foot = 118, 54

    sheet = Image.new("RGB", (cols * tw, head + rows * th + foot), SURFACE)
    d = ImageDraw.Draw(sheet)
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, head + (i // cols) * th))

    pad = Figure.PAD_L
    right = cols * tw - pad
    data, _ = load(path)
    done, total = int(data["done"].sum()), data["done"].size

    d.text((pad, 40), f"{meta['preset_name']}", font=font(34, "semibold"),
           fill=INK, anchor="ls")
    d.text((pad, 52), f"{meta['x_param']} × {meta['y_param']} phase diagram",
           font=font(15), fill=INK_DIM, anchor="la")
    d.text((right, 40), run_line(meta), font=font(14), fill=INK_DIM, anchor="rs")
    pct = 100.0 * done / max(1, total)
    d.text((right, 58), f"{meta['grid']}×{meta['grid']} grid · {done}/{total} "
                        f"cells" + ("" if done == total
                                    else f" ({pct:.0f}%, coarse cells filled)"),
           font=font(13), fill=INK_FAINT, anchor="ra")
    d.line([pad, 96, right, 96], fill=RULE, width=1)

    d.line([pad, head + rows * th + 14, right, head + rows * th + 14],
           fill=RULE, width=1)
    d.text((pad, head + rows * th + 38),
           "every panel: colour spans the 2–98 percentile · dashed rules mark "
           "zero on both axes · marker is the preset's own values · "
           "seed fixed across all cells",
           font=font(12), fill=INK_FAINT, anchor="ls")
    return sheet


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--feature", default="participation_ratio")
    ap.add_argument("--ramp", choices=sorted(RAMPS), default="viridis")
    ap.add_argument("--lo-pct", type=float, default=2.0)
    ap.add_argument("--hi-pct", type=float, default=98.0)
    ap.add_argument("--plot-px", type=int, default=620)
    ap.add_argument("--label", default="", help="name for the preset marker")
    ap.add_argument("--contact", action="store_true",
                    help="one sheet of the channels worth looking at")
    ap.add_argument("--all", action="store_true", help="every channel, separately")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    path = Path(args.path)
    data, meta = load(path)
    label = args.label or meta["preset_name"]
    dest = Path(args.out) if args.out else path / "figures"
    dest.mkdir(parents=True, exist_ok=True)

    if args.contact:
        picks = ["field_order", "change_rate", "participation_ratio",
                 "structure", "spec_entropy", "particle_pr"]
        img = contact(path, picks, args.ramp, args.plot_px, label)
        out = dest / f"{meta['preset_name']}_contact.png"
        img.save(out)
        print(f"wrote {out}  ({img.size[0]}x{img.size[1]})")
        return 0

    names = ([str(n) for n in data["cell_names"]] if args.all else [args.feature])
    for f in names:
        img, _ = make(path, f, args.ramp, args.lo_pct, args.hi_pct,
                      args.plot_px, label)
        out = dest / f"{f}.png"
        img.save(out)
        print(f"wrote {out}  ({img.size[0]}x{img.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
