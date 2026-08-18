"""The field source registry.

A source produces an RGBA texture and returns a HANDLE. Procedural sources
render into the bus's shared scratch; sources that own a texture return their
own, so they cost no GPU pass at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import moderngl

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


def make_source(key: str, ctx: moderngl.Context):
    """Instantiate the GPU-side object for a source kind."""
    _build_registry()
    if key in ("noise", "gradient"):
        return _BuiltinSource(ctx, key)
    raise KeyError(f"source '{key}' is not implemented yet")
