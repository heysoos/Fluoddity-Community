# Brain colour outputs — design

Every brain gains two outputs, **hue** and **saturation**, so a particle's colour
is its own heritable gene instead of a readout of the axial force. The brain goes
from 4 inputs → 4 outputs to 4 inputs → 6 outputs, under every modality.

## What changes on screen

Today (`entity_update.glsl`):

```glsl
color = baseterm.xy + mirrorterm.xy;           // .x IS the axial force term
e.hue = HUE_SENSITIVITY * color.x;
e.sat = 0.8;
```

After:

```glsl
e.hue = HUE_SENSITIVITY * base.colour.x;                 // base term only
e.sat = 0.5 + 0.5 * tanh(base.colour.y + 0.693147);      // atanh(0.6): 0 -> 0.8
```

- Hue and saturation are **fully independent** of force and strafe.
- They take the **base** evaluation only — the mirror term is not added — so a
  creature may be coloured asymmetrically. Force and strafe keep their mirror
  combination unchanged.
- Saturation's offset makes a zero output land on today's constant 0.8.
- `color_by_cohort` still replaces hue only; saturation comes from the brain.
- `GRAYSCALE` capture still zeroes saturation; unchanged.
- `HUE_SENSITIVITY` stays as a **manual per-preset slider**. It is never added
  to `PHYSICS_PARAMS` — the search must never tune it.

## The brain output

GLSL has no `vec6`, so one evaluation returns both halves:

```glsl
struct BrainOut { vec4 motion; vec2 colour; };
```

`eval_brain`, `eval_brain_unit` and every `brain_*` / `*_unit` return it, and
`black_box` returns it. One evaluation, not two passes. The mirror call's colour
half is computed and discarded.

Per modality, the colour outputs are two more readouts of the SAME per-unit
feature the motion outputs already read:

| modality | unit today | colour floats | new unit stride |
|---|---|---|---|
| Fourier | freq(4) amp(4) audio(K) | 2 amplitudes after amp | 10 + K |
| Gabor | c(4) f(4) a(4) sigma phase audio(K) | 2 amplitudes after phase | 16 + K |
| Lenia | w(4) a(4) mu sigma audio(K) | 2 amplitudes after sigma | 12 + K |
| MLP | W_out 4 rows, b_out 4 | W_out rows 5–6, b_out 5–6 | +2·w_k + 2 floats |

Audio weights stay at the END of each unit. Fourier is the one modality whose
four outputs use different basis functions: the hue amplitude multiplies the
axial basis (`sin(phase + po)`), the saturation amplitude the lateral basis
(`cos(phase + po*0.7)`). That choice is what makes the legacy copy below exact.

Colour floats are OFFSET-type like the amplitudes they sit beside: zero encodes
to zero. `AMPLITUDE_SLICE` widens to cover them, so a grown unit is silenced on
all six outputs and a GROW move stays phenotype-preserving. MLP's `OUT_DIM`
becomes 6.

### Mutation

The four motion floats mutate exactly as they do today, bit for bit. The two
colour floats draw from their own hash stream. The generic per-float jitter
(`brain_jit`) hashes the **colourless, deaf** index, the same discipline the
audio inputs already follow, so an old preset with `MUTATION_SCALE > 0` mutates
its motion identically.

### Generated brains

"No rule loaded" draws its colourless deaf brain first, then colour, then ears,
so a no-rule preset's motion is unchanged. Its colour becomes an independent
draw rather than a copy of its force.

## Legacy genomes

A genome written before this feature has 4 outputs. Loading one widens it:

- **hue weights = 2 × the axial-force weights.** Today's hue is base + mirror,
  about twice the base term alone; doubling reproduces today's colour range.
  Exact for symmetric creatures; strongly asymmetric ones shift somewhat.
  Accepted.
- **saturation weights = 0**, which maps to exactly 0.8.
- **motion floats untouched**, so motion is bit-identical.

One function per modality: `widen_legacy(params, legacy_layout) -> params`.

## Layout names

Every new layout's signature carries `+c2`: `fourier-n10` → `fourier-n10+c2`,
`mlp-n16-a0+a3` → `mlp-n16-a0+c2+a3`. A signature without `+c2` is LEGACY and
names the 4-output layout it always did.

`upgrade_signature(sig)` is the ONE function that maps a legacy name to the
current one. Everything that reads a stored signature passes it through, so a
legacy name reaching any reader lands on the right layout.

## Presets

Presets are widened **in memory** on load and never rewritten on disk. A preset
naming a legacy signature is upgraded; an unsigned preset is legacy
`fourier-n10` when it carries 80 floats, as today. Saving writes the new format.
The all-zero "no brain" marker still means no brain at any width.

Old Auto **checkpoints** are refused with a message naming the reason: their
optimizer state lives at the old width.

## Archives

Once, at launch:

1. **Back up.** If `Documents/Fluoddity/archives_legacy/` does not exist, copy
   `archives/` there whole (882 MB today). It sits beside `archives/`, so the
   program never lists it. If it exists, it is never touched again.
2. **Migrate each archive**, per layout directory without `+c2`:
   - widen `vectors.npz`'s brains with `widen_legacy` (written to a temp file,
     then renamed);
   - rename the directory to `upgrade_signature(name)`;
   - rewrite the root files that name layouts: `settings.json`
     (`layout_signature`), `layouts.jsonl` (`parent`, `child`) and
     `map_layout.npz` (`sigs`).
3. **Skip, never merge.** If the upgraded directory already exists, the legacy
   one is left alone and a warning is printed. Ids restart per directory, so two
   directories cannot be merged without reissuing ids.

A directory whose signature cannot be parsed is left alone. Embeddings are kept
as stored: an asymmetric creature's thumbnail and embedding reflect its old
colouring until it is re-admitted. Accepted.

Because the migrated archive uses new names, an older build opening it sees
layouts it cannot rebuild and leaves them alone, instead of zeroing genomes of a
width it does not expect.

## Removed

`AUTO_HUE_MAX` and everything that serves it: `clamp_auto_hue` and its per-frame
call, the `prev_hue` parameter of `put_back_auto_overrides`, `_auto_prev_hue`,
and `tests/test_auto_hue_clamp.py`. With hue its own gene, the archive is left to
decide whether a recolour is novel.

## Brain Inspector

Gains **Hue** and **Saturation** channels. The projection channels (`PREVIEW_OUT`,
`PREVIEW_RGB`) stay over the four motion outputs.

## Docs

- CLAUDE.md: rewrite the "Hue IS the axial force term" caveat; add a caveat
  for the colour outputs covering the `+c2` rule, `upgrade_signature`, and the
  2× legacy copy.
- `tools/hue_nuisance.py` stays as a measuring tool.

## Testing

- **Legacy parity (GPU, pure function over a fixed input grid):** a widened
  legacy genome's motion outputs are bit-identical to the legacy brain's, and
  its hue output equals 2 × the legacy base axial output, per modality.
- **Mutation parity:** under `MUTATION_SCALE > 0`, a widened legacy genome's
  motion is bit-identical, per modality.
- **Saturation:** a zero colour output yields exactly 0.8.
- **Signatures:** `upgrade_signature` round-trips with `+aK`; legacy and current
  names parse to the right layouts.
- **Archive migration:** backup created once and never overwritten; directories
  renamed and widened; root files rewritten; a second run is a no-op; an
  existing target is skipped, not merged.
- **Layout moves:** a grown unit is silent on all six outputs.
- Existing parity references in `tests/test_brain_modalities_gpu.py` and the
  width assertions listed in the blast-radius sweep are updated to six outputs.
- Sim step time measured before and after in one process against one entity
  snapshot.

## Future

**The saturation offset may bias colour search.** A zero output maps to 0.8, not
the middle of the range, so a small push makes particles more vivid and a larger
one is needed to grey them out. It was chosen so old archives look the same. If
colour evolution later looks stuck on vivid, revisit it: centre the mapping on
0.5 and give legacy genomes a saturation bias instead (MLP has an output bias;
the unit modalities would need one).
