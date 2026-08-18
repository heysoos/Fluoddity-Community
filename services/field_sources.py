"""The field source registry.

A source produces an RGBA texture and returns a HANDLE. Procedural sources
render into the bus's shared scratch; sources that own a texture return their
own, so they cost no GPU pass at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from services.shader_params import ShaderParam, parse_shader_params
from utilities.gl_helpers import read_shader, tryset
from utilities.paths import get_app_dir, get_user_data_dir


@dataclass
class FrameContext:
    """Everything a source may read about the frame it is being drawn for."""
    time: float
    frame_count: int
    mouse: tuple[float, float]
    prev_mouse: tuple[float, float]
    canvas_texture: object | None


@dataclass
class SourceDescriptor:
    key: str
    label: str
    keeps_history: bool = False
    shader: str = ""              # bundled .frag, for the procedural sources
    builtin_params: list[ShaderParam] = field(default_factory=list)


def _params_of(shader_relpath: str) -> list[ShaderParam]:
    return parse_shader_params(read_shader(shader_relpath))


_REGISTRY: dict[str, SourceDescriptor] = {}


def _register(desc: SourceDescriptor) -> None:
    _REGISTRY[desc.key] = desc


def _build_registry() -> None:
    if _REGISTRY:
        return
    _register(SourceDescriptor("noise", "Noise", shader="shaders/field/noise.frag",
                               builtin_params=_params_of("shaders/field/noise.frag")))
    _register(SourceDescriptor("gradient", "Gradient",
                               shader="shaders/field/gradient.frag",
                               builtin_params=_params_of("shaders/field/gradient.frag")))
    _register(SourceDescriptor("image", "Image"))
    _register(SourceDescriptor("shader", "Shader", keeps_history=True))
    _register(SourceDescriptor("brush", "Brush"))
    _register(SourceDescriptor("feedback", "Feedback", keeps_history=True))


def descriptors() -> list[SourceDescriptor]:
    _build_registry()
    return list(_REGISTRY.values())


def get(key: str) -> SourceDescriptor:
    _build_registry()
    return _REGISTRY[key]


def _shader_folders():
    return (get_user_data_dir(), get_user_data_dir() / "shaders",
            get_app_dir() / "shaders" / "field_override")


def available_shader_files() -> list[str]:
    """User .frag files, then bundled ones. Filenames only."""
    names = []
    for folder in _shader_folders():
        if folder.exists():
            names += [f.name for f in sorted(folder.glob("*.frag"))]
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def resolve_shader_path(name: str) -> Path | None:
    for folder in _shader_folders():
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def params_for(layer) -> list[ShaderParam]:
    """The parameter list a layer's UI should draw."""
    desc = get(layer.source)
    if layer.source != "shader":
        return desc.builtin_params
    path = resolve_shader_path(layer.params.get("_file", ""))
    if path is None:
        return []
    try:
        return parse_shader_params(path.read_text())
    except OSError:
        return []


# -- sources ------------------------------------------------------------


class _ProgramSource:
    """A source backed by a fragment shader rendering into the bus scratch."""

    def __init__(self, ctx: moderngl.Context, frag_source: str):
        self.ctx = ctx
        self.error: str | None = None
        self._program = None
        self._vao = None
        self._compile(frag_source)

    def _compile(self, frag_source: str) -> None:
        try:
            program = self.ctx.program(
                vertex_shader=read_shader("shaders/canvas.vert"),
                fragment_shader=frag_source)
        except Exception as exc:
            text = str(exc).strip()
            self.error = text.splitlines()[0] if text else "compile failed"
            return
        self.release()
        self._program = program
        self._vao = self.ctx.vertex_array(program, [])
        self.error = None

    def evaluate(self, bus, layer, frame):
        if self._program is None:
            return None
        prog = self._program
        tryset(prog, "canvas_resolution", tuple(float(v) for v in bus.resolution))
        tryset(prog, "time", float(frame.time))
        tryset(prog, "frame_count", int(frame.frame_count))
        tryset(prog, "mouse", tuple(float(v) for v in frame.mouse))
        tryset(prog, "prev_mouse", tuple(float(v) for v in frame.prev_mouse))
        for param in params_for(layer):
            value = layer.params.get(param.name)
            if value is None:
                value = (param.default[0] if param.components == 1
                         else tuple(param.default))
            if param.kind == "int":
                value = int(value)
            elif param.kind == "bool":
                value = bool(value)
            elif param.components == 1:
                value = float(value)
            else:
                value = tuple(float(v) for v in value)
            tryset(prog, param.name, value)
        return bus.render_into_scratch(self._vao)

    def release(self) -> None:
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        if self._program is not None:
            self._program.release()
            self._program = None


class _BuiltinSource(_ProgramSource):
    def __init__(self, ctx, key):
        super().__init__(ctx, read_shader(get(key).shader))


class _FeedbackSource:
    """The sim's own canvas, borrowed. Costs no pass and no copy."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.error: str | None = None

    def evaluate(self, bus, layer, frame):
        self.error = None
        return frame.canvas_texture

    def release(self) -> None:
        pass


class _ImageSource:
    """A file on disk, uploaded once and reused until the path changes."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.error: str | None = None
        self._tex = None
        self._loaded = None

    def evaluate(self, bus, layer, frame):
        name = layer.params.get("_file", "")
        if name != self._loaded:
            self._load(name)
        return self._tex

    def _load(self, name: str) -> None:
        self.release()
        self._loaded = name
        if not name:
            self.error = "no image selected"
            return
        path = Path(name)
        if not path.is_absolute():
            path = get_user_data_dir() / name
        if not path.exists():
            self.error = f"image not found: {name}"
            return
        try:
            img = Image.open(path).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
        except OSError as exc:
            self.error = f"could not read {name}: {exc}"
            return
        data = (np.asarray(img, dtype=np.float32) / 255.0).astype("f4")
        tex = self.ctx.texture(img.size, 4, data.tobytes(), dtype="f4")
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        tex.build_mipmaps()
        self._tex = tex
        self.error = None

    def release(self) -> None:
        if self._tex is not None:
            self._tex.release()
            self._tex = None


class _UserShaderSource(_ProgramSource):
    """A user .frag, recompiled whenever the chosen file changes."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.error = "no shader selected"
        self._program = None
        self._vao = None
        self._loaded = None

    def evaluate(self, bus, layer, frame):
        name = layer.params.get("_file", "")
        if name != self._loaded:
            self._loaded = name
            path = resolve_shader_path(name) if name else None
            if path is None:
                self.release()
                self.error = (f"shader not found: {name}" if name
                              else "no shader selected")
            else:
                try:
                    self._compile(path.read_text())
                except OSError as exc:
                    self.release()
                    self.error = f"could not read {name}: {exc}"
        if self._program is None:
            return None
        return super().evaluate(bus, layer, frame)

    def reload(self) -> None:
        """Force a recompile on the next evaluate. Bound to the V key."""
        self._loaded = None


class _BrushSource:
    """The mouse brush's accumulation buffer.

    The one source with memory, which is what lets the bus stay stateless:
    persistence is a property of a source, never of the bus.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.error: str | None = None
        self._tex = None
        self._fbo = None

    def ensure(self, width: int, height: int):
        if self._tex is not None and self._tex.size == (width, height):
            return self._tex
        self.release()
        tex = self.ctx.texture((width, height), 4, dtype="f4")
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        self._tex = tex
        self._fbo = self.ctx.framebuffer(color_attachments=[tex])
        self._fbo.clear()
        return tex

    @property
    def framebuffer(self):
        return self._fbo

    def evaluate(self, bus, layer, frame):
        return self.ensure(*bus.resolution)

    def snapshot(self):
        if self._tex is None:
            return None
        h, w = self._tex.height, self._tex.width
        return np.frombuffer(self._tex.read(), dtype=np.float32).reshape(h, w, 4).copy()

    def write(self, data) -> None:
        tex = self.ensure(int(data.shape[1]), int(data.shape[0]))
        tex.write(np.ascontiguousarray(data, dtype="f4").tobytes())

    def clear(self) -> None:
        if self._fbo is not None:
            self._fbo.clear()

    def release(self) -> None:
        if self._fbo is not None:
            self._fbo.release()
            self._fbo = None
        if self._tex is not None:
            self._tex.release()
            self._tex = None


def make_source(key: str, ctx: moderngl.Context):
    """Instantiate the GPU-side object for a source kind."""
    _build_registry()
    if key in ("noise", "gradient"):
        return _BuiltinSource(ctx, key)
    if key == "feedback":
        return _FeedbackSource(ctx)
    if key == "image":
        return _ImageSource(ctx)
    if key == "shader":
        return _UserShaderSource(ctx)
    if key == "brush":
        return _BrushSource(ctx)
    raise KeyError(f"unknown source '{key}'")
