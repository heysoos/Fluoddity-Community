"""The field injection bus: owns the destination texture and composites into it.

The bus is stateless per frame. Every destination is cleared and rebuilt from
the stack, so a disabled layer contributes nothing without anything having to
undo it. A source that needs memory owns its own buffer.
"""
from __future__ import annotations

import moderngl

from services import field_sources
from state.field_stack import MAPPINGS
from utilities.gl_helpers import read_shader, tryset

BLEND_STATE = {
    "replace":  None,
    "add":      (moderngl.ONE, moderngl.ONE, moderngl.FUNC_ADD),
    "multiply": (moderngl.DST_COLOR, moderngl.ZERO, moderngl.FUNC_ADD),
    "max":      (moderngl.ONE, moderngl.ONE, moderngl.MAX),
}


class FieldBus:
    """Owns the force/strafe destination texture and the composite pass."""

    # force is .xy, strafe is .zw of one RGBA32F texture, so entity_update's
    # get_field() is unchanged by this feature.
    DEST_CHANNELS = {
        "force":  (True, True, False, False),
        "strafe": (False, False, True, True),
    }
    SCALAR_DESTINATIONS = ()

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._res = (0, 0)
        self._tex = None
        self._fbo = None
        self._composite = None
        self._composite_vao = None
        self._scratch = None
        self._scratch_fbo = None
        self._sources = {}      # layer uid -> (source key, source object)
        self._dirty = True
        self._pass_count = 0
        self._scale = 0.0
        self._thumbs = {}       # layer uid -> (texture, framebuffer)
        self._thumbs_on = False
        self._inspect_src = {}  # layer uid -> the texture its source produced
        self._blit = None
        self._blit_vao = None
        # The Inspect panel's own buffers: one layer composited alone, and the
        # readable rendering of a vector field.
        self._preview_tex = None
        self._preview_fbo = None
        self._vview_tex = None
        self._vview_fbo = None
        self._vview = None
        self._vview_vao = None
        self._devices = {}           # exclusive device name -> owning source
        self._inspect_want = None    # (uid, view) the panel is asking for
        self._inspect_ready = None   # (uid, view) currently in _vview_tex

    # -- public ---------------------------------------------------------

    @property
    def field_texture(self):
        return self._tex

    @property
    def resolution(self) -> tuple[int, int]:
        return self._res

    def ensure(self, canvas_width: int, canvas_height: int, scale: float) -> None:
        """Create or resize the destination texture. Never zero-sized."""
        w = max(1, int(canvas_width * scale))
        h = max(1, int(canvas_height * scale))
        if self._tex is not None and self._res == (w, h):
            return
        self._release_target()
        self._res = (w, h)
        tex = self.ctx.texture((w, h), 4, dtype="f4")
        # LINEAR because the bus runs below canvas resolution by default and
        # nearest sampling is visibly blocky there.
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        self._tex = tex
        self._fbo = self.ctx.framebuffer(color_attachments=[tex])
        self._fbo.clear()
        self._ensure_composite()

    def clear(self) -> None:
        if self._fbo is None:
            return
        self._fbo.color_mask = (True, True, True, True)
        self._fbo.clear()

    def composite_one(self, src_tex, layer, target=None, alone=False) -> None:
        """Map `src_tex` through `layer` and blend it into the destination.

        `target`/`alone` are for the Inspect panel: one layer, into a buffer of
        its own, with its blend and channel mask set aside so what is shown is
        the layer's own contribution rather than its share of the stack.
        """
        fbo = self._fbo if target is None else target
        if fbo is None or src_tex is None:
            return
        self._ensure_composite()

        # A mipmap min-filter over a texture with no mip chain is INCOMPLETE and
        # samples as black. Blur needs the chain, so it is built on demand and
        # the filter put back afterwards: `feedback` hands back the sim's own
        # canvas, and leaving that on a mipmap filter would change how the sim
        # samples its own trails.
        previous_filter = src_tex.filter
        if layer.blur > 0.0:
            src_tex.build_mipmaps()
            src_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        else:
            src_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        src_tex.use(location=0)

        prog = self._composite
        tryset(prog, "src", 0)
        tryset(prog, "mapping", MAPPINGS.index(layer.mapping))
        tryset(prog, "strength", float(layer.strength))
        tryset(prog, "sign_mul", float(layer.sign))
        tryset(prog, "blur_lod", float(layer.blur))
        tryset(prog, "texel", (1.0 / self._res[0], 1.0 / self._res[1]))
        tryset(prog, "scalar_out", layer.destination in self.SCALAR_DESTINATIONS)
        tryset(prog, "src_channels",
               1 if layer.params.get("_channels") == "zw" else 0)
        tryset(prog, "aspect_ratio", self._aspect_ratio(src_tex))

        # The mask is set BEFORE use(): moderngl applies a framebuffer's stored
        # state when it is bound, so a mask set afterwards misses this pass.
        fbo.color_mask = ((True, True, True, True) if alone
                          else self.DEST_CHANNELS[layer.destination])
        fbo.use()

        state = None if alone else BLEND_STATE[layer.blend]
        if state is None:
            self.ctx.disable(moderngl.BLEND)
        else:
            src_factor, dst_factor, equation = state
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = src_factor, dst_factor
            self.ctx.blend_equation = equation

        self._composite_vao.render(mode=moderngl.TRIANGLE_FAN, vertices=4)

        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_equation = moderngl.FUNC_ADD
        fbo.color_mask = (True, True, True, True)
        src_tex.filter = previous_filter

    def _aspect_ratio(self, src_tex) -> float:
        """Source aspect over destination aspect, for the cover crop.

        A 4:3 camera squeezed into a square bus is the visible symptom; the
        procedural sources render at the bus size and land on exactly 1.0,
        which the shader takes as its no-op.
        """
        sw, sh = src_tex.size
        dw, dh = self._res
        if not (sh and dh and dw):
            return 1.0
        return (sw / sh) / (dw / dh)

    @property
    def scratch_texture(self):
        """The shared target procedural sources render into.

        Safe to share because a layer's composite consumes it immediately
        after that layer's own evaluation.
        """
        return self._scratch

    def render_into_scratch(self, vao) -> moderngl.Texture:
        """Render a source's fullscreen quad into scratch and return it."""
        self._ensure_scratch()
        self._scratch_fbo.use()
        self.ctx.disable(moderngl.BLEND)
        vao.render(mode=moderngl.TRIANGLE_FAN, vertices=4)
        return self._scratch

    # -- the rebuild loop -----------------------------------------------

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def pass_count(self) -> int:
        """Passes run on the last rebuild. Zero when the bus was clean."""
        return self._pass_count

    def mark_dirty(self) -> None:
        self._dirty = True

    def _is_animated(self, layer) -> bool:
        """Whether this layer's source has to be redrawn every frame."""
        return bool(getattr(self._source_for_layer(layer), "animated", False))

    def claim_device(self, name: str, source):
        """Which source owns an exclusive device this frame. First one wins.

        Cleared at the top of every rebuild, so the claim follows the stack
        rather than outliving a layer that has gone.
        """
        return self._devices.setdefault(name, source)

    def source_for(self, layer):
        """The GPU-side source object for a layer, or None if it has none."""
        entry = self._sources.get(layer.uid)
        return entry[1] if entry else None

    def brush_source(self):
        """The brush layer's accumulation buffer, or None if no brush layer."""
        for key, source in self._sources.values():
            if key == "brush":
                return source
        return None

    def ensure_brush_source(self, stack):
        """Materialise the brush layer's buffer without waiting for a rebuild.

        Loading a config installs a stack and writes its saved paint in the
        same call, which is before the next rebuild would have created the
        source to write into.
        """
        resident = self.brush_source()
        if resident is not None:
            return resident
        for layer in getattr(stack, "layers", []):
            if layer.source == "brush":
                return self._source_for_layer(layer)
        return None

    def reload_shaders(self) -> None:
        """Recompile the composite and every user shader. Bound to the V key."""
        if self._composite_vao is not None:
            self._composite_vao.release()
            self._composite_vao = None
        if self._composite is not None:
            self._composite.release()
            self._composite = None
        for _key, source in self._sources.values():
            reload_fn = getattr(source, "reload", None)
            if reload_fn is not None:
                reload_fn()
        self.mark_dirty()

    def rebuild(self, stack, canvas_width: int, canvas_height: int,
                scale: float, frame) -> bool:
        """Rebuild every destination from the stack. Returns True if it ran.

        Clean and unchanged means no passes at all, which is what makes a
        static layer free in steady state.
        """
        entry_target = self.ctx.fbo
        layers = list(getattr(stack, "layers", []))
        # The global switch reads as an empty stack: the field is released and
        # contributes exactly zero, which is the rule that makes a disabled
        # layer free rather than leaving its last frame standing.
        if not getattr(stack, "enabled", True):
            layers = []
        self._prune_sources(layers)

        if not layers:
            self._pass_count = 0
            if self._tex is not None:
                self._release_target()
            self._dirty = False
            self._inspect_ready = None
            return False

        if scale != self._scale:
            self._scale = scale
            self._dirty = True
        had_target = self._tex is not None
        self.ensure(canvas_width, canvas_height, scale)
        if not had_target:
            self._dirty = True

        # A source that varies with time is redrawn on every frame, or it is
        # frozen on whichever frame a slider was last touched - which is what
        # made noise look like a still image and feedback look inert.
        if any(layer.enabled and self._is_animated(layer) for layer in layers):
            self._dirty = True

        # The count belongs to the last REBUILD, so a clean frame leaves it
        # alone: zeroing it made the readout say "0 passes" for every frame a
        # layer was quietly forcing, which reads as a stack doing nothing.
        if not self._dirty:
            self._draw_inspect_view(layers)
            self._restore_target(entry_target)
            return False

        self._pass_count = 0
        self._devices.clear()
        self.clear()
        for layer in layers:
            if not layer.enabled:
                layer.error = None
                continue
            source = self._source_for_layer(layer)
            tex = source.evaluate(self, layer, frame)
            layer.error = getattr(source, "error", None)
            if tex is None:
                self._inspect_src.pop(layer.uid, None)
                continue
            self._inspect_src[layer.uid] = tex
            if tex is self._scratch:
                self._pass_count += 1
            self.composite_one(tex, layer)
            self._pass_count += 1
            self._capture_thumbnail(layer.uid, tex)

        self._draw_inspect_view(layers)
        self._restore_target(entry_target)
        self._dirty = False
        return True

    def _restore_target(self, previous) -> None:
        """Put back whatever was bound before we started.

        Everything downstream - the sim's own passes, and imgui - inherits the
        target the last pass left. A rebuild that leaves its own bound sends
        the whole UI into an offscreen buffer, which reads as every window
        vanishing at once with the close button unable to bring them back.
        Restoring what was there beats binding the screen, which a standalone
        context does not have.
        """
        # A released framebuffer keeps its wrapper and swaps its `mglo` for an
        # InvalidObject, so the wrapper's own type says nothing. Binding one
        # raises; having nothing to put back is not an error.
        if previous is None:
            return
        if isinstance(getattr(previous, "mglo", None), moderngl.InvalidObject):
            return
        previous.use()

    # -- previews --------------------------------------------------------

    THUMB_MAX = 96

    def set_thumbnails_enabled(self, on: bool) -> None:
        """Row previews only fill while the layer list is open."""
        if on == self._thumbs_on:
            return
        self._thumbs_on = on
        if not on:
            for tex, fbo in self._thumbs.values():
                fbo.release()
                tex.release()
            self._thumbs.clear()
        self.mark_dirty()

    def thumbnail_for(self, layer):
        entry = self._thumbs.get(layer.uid)
        return entry[0] if entry else None

    def request_inspect(self, uid, view: str) -> None:
        """Ask for a view. Drawn by the next rebuild, never here.

        The UI calls this from inside a window body, where a GL pass would
        leave a framebuffer bound that is not the one imgui is about to draw
        into - every window vanishes and the close button cannot bring them
        back.
        """
        self._inspect_want = (uid, view) if uid is not None else None

    def inspect(self, layer, view: str):
        """The finished texture for `layer`'s current view, or None.

        The three views are three different computations. "source" is the
        picture the layer produced; "mapped" is that picture put through this
        layer's mapping ALONE, so what is shown is the layer's own
        contribution rather than its share of the stack; "destination" is the
        whole composited field. Both vector views are rendered as hue and
        brightness - raw RG shows opposite directions as much the same colour.
        """
        if view == "source":
            return None if layer.error else self._inspect_src.get(layer.uid)
        if self._inspect_ready == (layer.uid, view):
            return self._vview_tex
        return None

    def _draw_inspect_view(self, layers) -> None:
        """Render the requested view. Called by rebuild, on the frame loop."""
        self._inspect_ready = None
        want = self._inspect_want
        if want is None or self._res == (0, 0):
            return
        uid, view = want
        if view == "source":
            return
        layer = next((l for l in layers if l.uid == uid), None)
        if layer is None:
            return
        if view == "destination":
            if self._vector_view(self._tex, layer.destination) is not None:
                self._inspect_ready = (uid, view)
            return
        if layer.error:
            return
        src = self._inspect_src.get(uid)
        if src is None:
            return
        self._ensure_preview()
        self._preview_fbo.color_mask = (True, True, True, True)
        self._preview_fbo.clear()
        self.composite_one(src, layer, target=self._preview_fbo, alone=True)
        if self._vector_view(self._preview_tex, layer.destination) is not None:
            self._inspect_ready = (uid, view)

    def _vector_view(self, tex, destination: str):
        """Direction as hue, magnitude as brightness."""
        if tex is None:
            return None
        self._ensure_vector_view()
        tex.use(location=0)
        tryset(self._vview, "src", 0)
        tryset(self._vview, "pair", 1 if destination == "strafe" else 0)
        tryset(self._vview, "gain", 4.0)
        self._vview_fbo.color_mask = (True, True, True, True)
        self._vview_fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self._vview_vao.render(mode=moderngl.TRIANGLE_FAN, vertices=4)
        return self._vview_tex

    def _capture_thumbnail(self, uid, src_tex) -> None:
        if not self._thumbs_on:
            return
        w, h = src_tex.size
        scale = min(1.0, self.THUMB_MAX / max(w, h, 1))
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        entry = self._thumbs.get(uid)
        if entry is None or entry[0].size != size:
            if entry is not None:
                entry[1].release()
                entry[0].release()
            tex = self.ctx.texture(size, 4, dtype="f4")
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._thumbs[uid] = (tex, self.ctx.framebuffer(color_attachments=[tex]))
        tex, fbo = self._thumbs[uid]
        self._ensure_blit()
        previous_filter = src_tex.filter
        src_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        src_tex.use(location=0)
        tryset(self._blit, "src", 0)
        fbo.color_mask = (True, True, True, True)
        fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self._blit_vao.render(mode=moderngl.TRIANGLE_FAN, vertices=4)
        src_tex.filter = previous_filter

    def _ensure_blit(self) -> None:
        if self._blit is not None:
            return
        self._blit = self.ctx.program(
            vertex_shader=read_shader("shaders/canvas.vert"),
            fragment_shader=(
                "#version 430\n"
                "in vec2 texcoord;\nout vec4 fragColor;\n"
                "uniform sampler2D src;\n"
                "void main(){ fragColor = texture(src, texcoord); }\n"),
        )
        self._blit_vao = self.ctx.vertex_array(self._blit, [])

    def _source_for_layer(self, layer):
        """The layer's source object, rebuilt only if its KIND changed.

        Keyed by uid so reordering the stack neither recompiles a shader nor
        discards the brush's paint.
        """
        entry = self._sources.get(layer.uid)
        if entry is not None and entry[0] == layer.source:
            return entry[1]
        if entry is not None:
            entry[1].release()
        source = field_sources.make_source(layer.source, self.ctx)
        self._sources[layer.uid] = (layer.source, source)
        return source

    def _prune_sources(self, layers) -> None:
        live = {l.uid for l in layers}
        for uid in [u for u in self._sources if u not in live]:
            self._sources.pop(uid)[1].release()
            self._inspect_src.pop(uid, None)
            entry = self._thumbs.pop(uid, None)
            if entry is not None:
                entry[1].release()
                entry[0].release()

    def cleanup(self) -> None:
        self.set_thumbnails_enabled(False)
        if self._blit_vao is not None:
            self._blit_vao.release()
            self._blit_vao = None
        if self._blit is not None:
            self._blit.release()
            self._blit = None
        for _key, source in self._sources.values():
            source.release()
        self._sources.clear()
        self._inspect_src.clear()
        for vao in (self._vview_vao,):
            if vao is not None:
                vao.release()
        self._vview_vao = None
        if self._vview is not None:
            self._vview.release()
            self._vview = None
        for tex, fbo in ((self._preview_tex, self._preview_fbo),
                         (self._vview_tex, self._vview_fbo)):
            if fbo is not None:
                fbo.release()
            if tex is not None:
                tex.release()
        self._preview_tex = self._preview_fbo = None
        self._vview_tex = self._vview_fbo = None
        self._release_target()
        if self._composite_vao is not None:
            self._composite_vao.release()
            self._composite_vao = None
        if self._composite is not None:
            self._composite.release()
            self._composite = None

    # -- internal -------------------------------------------------------

    def _ensure_composite(self) -> None:
        if self._composite is not None:
            return
        self._composite = self.ctx.program(
            vertex_shader=read_shader("shaders/canvas.vert"),
            fragment_shader=read_shader("shaders/field/composite.frag"),
        )
        self._composite_vao = self.ctx.vertex_array(self._composite, [])

    def _preview_target(self, existing, res):
        """An off-stack RGBA32F buffer at the bus resolution."""
        tex, fbo = existing
        if tex is not None and tex.size == res:
            return tex, fbo
        if fbo is not None:
            fbo.release()
        if tex is not None:
            tex.release()
        tex = self.ctx.texture(res, 4, dtype="f4")
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex, self.ctx.framebuffer(color_attachments=[tex])

    def _ensure_preview(self) -> None:
        self._preview_tex, self._preview_fbo = self._preview_target(
            (self._preview_tex, self._preview_fbo), self._res)

    def _ensure_vector_view(self) -> None:
        self._vview_tex, self._vview_fbo = self._preview_target(
            (self._vview_tex, self._vview_fbo), self._res)
        if self._vview is not None:
            return
        self._vview = self.ctx.program(
            vertex_shader=read_shader("shaders/canvas.vert"),
            fragment_shader=read_shader("shaders/field/vector_view.frag"),
        )
        self._vview_vao = self.ctx.vertex_array(self._vview, [])

    def _ensure_scratch(self) -> None:
        if self._scratch is not None:
            return
        w, h = self._res
        tex = self.ctx.texture((w, h), 4, dtype="f4")
        # LINEAR at rest, never a mipmap filter: composite_one puts the filter
        # back the way it found it, so a mipmap resting state leaves the
        # scratch incomplete for every reader that is not blurring - which is
        # the thumbnail and the inspector, both of which then draw black.
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        self._scratch = tex
        self._scratch_fbo = self.ctx.framebuffer(color_attachments=[tex])

    def _release_target(self) -> None:
        if self._fbo is not None:
            self._fbo.release()
            self._fbo = None
        if self._tex is not None:
            self._tex.release()
            self._tex = None
        if self._scratch_fbo is not None:
            self._scratch_fbo.release()
            self._scratch_fbo = None
        if self._scratch is not None:
            self._scratch.release()
            self._scratch = None
        self._res = (0, 0)
