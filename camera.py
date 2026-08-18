import glfw
import numpy as np
from utilities.gl_helpers import read_shader, tryset
import moderngl
from state import CameraState
from utilities.frame_assembler import FrameAssembler

class Camera:
    def __init__(self, ctx, sim, window):
        self.ctx = ctx
        self.sim = sim
        self.window = window
        self.BRIGHTNESS = 1
        self.trail_overlay_strength = 1.0  # view_mode 6 trail brightness (set by orchestrator)
        self.ink_weight = 1
        self.cam_brush_mode = True
        self.watercolor_mode = False

        # Camera state
        self.position = np.array([0.0, 0.0])  # 2D position
        self.zoom = 1.0

        self.setup_rendering()

    def setup_rendering(self):
        # Fullscreen quad vertices (position + texcoord)
        quad_vertices = np.array([
            -1.0, -1.0,  0.0, 0.0,  # bottom-left
             1.0, -1.0,  1.0, 0.0,  # bottom-right
             1.0,  1.0,  1.0, 1.0,  # top-right
            -1.0,  1.0,  0.0, 1.0   # top-left
        ], dtype=np.float32)

        quad_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        # Vertex shader
        self.vertex_shader = read_shader('shaders/camera.vert')

        # Fragment shader
        self.fragment_shader = read_shader('shaders/camera.frag')

        try:
            # Create shader program
            self.program = self.ctx.program(
                vertex_shader=self.vertex_shader,
                fragment_shader=self.fragment_shader
            )
        except Exception as e:
            print('Camera shader failed')
            print(e)

        # Create cam brush program
        self.cam_brush_vertex_shader = read_shader('shaders/cam_brush.vert')
        self.cam_brush_fragment_shader = read_shader('shaders/cam_brush.frag')

        try:
            self.cam_brush_program = self.ctx.program(
                vertex_shader=self.cam_brush_vertex_shader,
                fragment_shader=self.cam_brush_fragment_shader
            )
        except Exception as e:
            print('Cambrush shader failed')
            print(e)

        # Create vertex array
        vbo = self.ctx.buffer(quad_vertices.tobytes())
        ibo = self.ctx.buffer(quad_indices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(vbo, '2f 2f', 'in_position', 'in_texcoord')],
            ibo
        )
        self.cam_brush_vao = self.ctx.vertex_array(
            self.cam_brush_program,
            []
        )

        self.cam_brush_target = self.ctx.texture(glfw.get_framebuffer_size(self.window), 4, dtype='f4')
        self.cam_brush_fbo = self.ctx.framebuffer([self.cam_brush_target])

        # Frame assembler (temporal accumulation + gamma correction)
        self.frame_assembler = FrameAssembler(self.ctx, self.cam_brush_target)
        self.assembled_texture = None
        # The canvas rect that was valid when assembled_texture was produced.
        # Assigned together with it, never recomputed - see canvas_view_rect.
        self.assembled_view_rect = None

        # Bloom processor (lazily initialized on first use)
        self._bloom_processor = None

    def generate_view_texture(self, tiling_mode: bool = False):
        """Generate raw view texture (PRE-gamma correction) based on current mode.

        Args:
            tiling_mode: Whether tiling mode is enabled
        """

        if self.cam_brush_mode:
            # Render particles to cam_brush_target
            self.cam_brush_fbo.use()
            width, height = glfw.get_framebuffer_size(self.window)
            self.ctx.viewport = (0, 0, width, height)
            self.ctx.clear(0, 0, 0, 1)

            self.cam_brush_program['cam_pos'].value = tuple(self.position)
            self.cam_brush_program['cam_zoom'].value = self.zoom
            self.cam_brush_program['canvas_resolution'].value = self.sim.view_tex.size
            self.cam_brush_program['window_size'].value = (width, height)
            tryset(self.cam_brush_program, 'WATERCOLOR_MODE', self.watercolor_mode)

            # Tiling mode uniforms
            tryset(self.cam_brush_program, 'tiling_mode_enabled', tiling_mode)
            if tiling_mode:
                view_min, view_max = self.compute_tiling_view_bounds()
                tryset(self.cam_brush_program, 'view_min', tuple(view_min))
                tryset(self.cam_brush_program, 'view_max', tuple(view_max))

            # Particles need additive blending
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            self.ctx.blend_equation = moderngl.FUNC_ADD

            self.cam_brush_vao.render(mode=moderngl.TRIANGLE_FAN, instances=self.sim.entity_count, vertices=4)

            self.ctx.disable(moderngl.BLEND)

            # Return raw texture (NO gamma correction - that happens in FrameAssembler)
            return self.cam_brush_target
        else:
            return self.sim.view_tex

    def apply_state(self, state: CameraState) -> None:
        """Apply camera state from Orchestrator."""
        self.position = state.position.copy()
        self.zoom = state.zoom
        self.BRIGHTNESS = state.BRIGHTNESS
        self.cam_brush_mode = state.cam_brush_mode

    def compute_tiling_view_bounds(self):
        """Compute entity-space view bounds for tiling mode.

        Inverts the cam_brush vertex shader transform to find the rectangle
        of entity positions visible on screen. Accounts for both canvas and
        window aspect ratios with area-preserving entity space bounds.
        """
        import math
        width, height = glfw.get_framebuffer_size(self.window)
        window_aspect = width / max(height, 1)
        tex_w, tex_h = self.sim.view_tex.size
        tex_aspect = tex_w / max(tex_h, 1)
        x_edge = math.sqrt(tex_aspect)
        y_edge = 1.0 / math.sqrt(tex_aspect)

        if tex_aspect > window_aspect:
            scale_x = 1.0 / self.zoom
            scale_y = window_aspect / (tex_aspect * self.zoom)
        else:
            scale_x = tex_aspect / (window_aspect * self.zoom)
            scale_y = 1.0 / self.zoom

        cx, cy = self.position[0], self.position[1]
        view_min = np.array([
            (-1.0 + cx / self.zoom) * x_edge / scale_x,
            (-1.0 - cy / self.zoom) * y_edge / scale_y,
        ])
        view_max = np.array([
            (1.0 + cx / self.zoom) * x_edge / scale_x,
            (1.0 - cy / self.zoom) * y_edge / scale_y,
        ])
        return view_min, view_max

    def compute_tiling_scale(self):
        """Compute tiling_scale that converts frame_assembly world_pos to entity space.

        tiling_scale = (x_edge, y_edge) / (scale * zoom) where scale is the
        letterbox scaling factor from the vertex shader.
        """
        import math
        tex_w, tex_h = self.sim.view_tex.size
        tex_aspect = tex_w / max(tex_h, 1)
        width, height = glfw.get_framebuffer_size(self.window)
        window_aspect = max(width,1) / max(height, 1)
        x_edge = math.sqrt(tex_aspect)
        y_edge = 1.0 / math.sqrt(tex_aspect)

        if tex_aspect > window_aspect:
            # scale * zoom = (1, window_aspect / tex_aspect)
            return (x_edge, y_edge * tex_aspect / window_aspect)
        else:
            # scale * zoom = (tex_aspect / window_aspect, 1)
            return (x_edge * window_aspect / tex_aspect, y_edge)

    def apply_bloom(self, texture, threshold, intensity, radius,
                    tonemap_softness=3.0):
        """Apply bloom post-processing. Lazily initializes GPU resources."""
        if self._bloom_processor is None:
            from utilities.bloom import BloomProcessor
            self._bloom_processor = BloomProcessor(self.ctx)
        return self._bloom_processor.process(
            texture, threshold, intensity, radius,
            tonemap_softness=tonemap_softness,
        )

    def render(self, sim_going: bool = True, current_view_option: int = 2,
                sweep_mode: bool = False, sweep_reticle_pos: tuple = (0.5, 0.5),
                sweep_reticle_visible: bool = False, screen_aspect: float = 1.0,
                watercolor_mode: bool = False, ink_weight: float = 1.0,
                draw_trail_mode: bool = False, draw_size: float = 0.0,
                mouse_screen_coords: tuple = (0.5, 0.5), exposure: float = 0.0,
                tiling_mode: bool = False, tonemap_softness: float = 1.0,
                bloom_enabled: bool = False, bloom_threshold: float = 0.8,
                bloom_intensity: float = 0.5, bloom_radius: float = 1.0):
        self.watercolor_mode = watercolor_mode
        self.ink_weight = ink_weight

        # Compute view bounds for tiling mode
        view_min = (0.0, 0.0)
        view_max = (0.0, 0.0)
        if tiling_mode:
            view_min, view_max = self.compute_tiling_view_bounds()

        # ALWAYS use assembled texture when simulation is running
        # When paused, regenerate view to allow camera panning/zooming
        if sim_going and self.assembled_texture is not None:
            TEX_TO_VIEW = self.assembled_texture
        else:
            # When paused or no assembled texture yet, generate fresh frame and apply gamma
            raw_tex = self.generate_view_texture(tiling_mode=tiling_mode)
            # Apply gamma correction via frame assembler (single sample mode)
            # Pass draw_size only when in draw trail mode
            trail_draw_radius = draw_size if draw_trail_mode else 0.0
            TEX_TO_VIEW = self.frame_assembler.assemble_frame(
                raw_tex,
                total_samples=1,
                current_sample_index=0,
                view_mode=current_view_option,
                sweep_mode=sweep_mode,
                sweep_reticle_pos=sweep_reticle_pos,
                sweep_reticle_visible=sweep_reticle_visible,
                screen_aspect=screen_aspect,
                brightness=self.BRIGHTNESS,
                exposure=exposure,
                ink_weight=self.ink_weight,
                watercolor_mode=watercolor_mode,
                camera_position=tuple(self.position),
                camera_zoom=self.zoom,
                trail_draw_radius=trail_draw_radius,
                mouse_screen_coords=mouse_screen_coords,
                tiling_mode=tiling_mode,
                view_min=tuple(view_min),
                view_max=tuple(view_max),
                tiling_scale=self.compute_tiling_scale(),
                canvas_resolution=self.sim.get_canvas_dimensions(),
                tonemap_softness=tonemap_softness,
                trail_tex=(self.sim.can if current_view_option == 6 else None),
                trail_overlay_strength=self.trail_overlay_strength
            )
            # assemble_frame returns the texture immediately when total_samples=1
            if bloom_enabled and not watercolor_mode and TEX_TO_VIEW is not None:
                TEX_TO_VIEW = self.apply_bloom(
                    TEX_TO_VIEW, bloom_threshold, bloom_intensity, bloom_radius,
                    tonemap_softness=tonemap_softness,
                )

        # Render to screen
        self.ctx.screen.use()
        width, height = glfw.get_framebuffer_size(self.window)
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.program['cam_pos'].value = tuple(self.position)
        self.program['cam_zoom'].value = self.zoom
        self.program['tex_size'].value = TEX_TO_VIEW.size
        self.program['window_size'].value = (width, height)

        if self.cam_brush_mode:
            tryset(self.program, 'cam_pos', (0, 0))
            tryset(self.program, 'cam_zoom', 1)

        TEX_TO_VIEW.use(location=0)
        self.program['view_tex'].value = 0
        self.vao.render()

    def reload(self):
        winx, winy =glfw.get_framebuffer_size(self.window)
        if winx > 0 and winy > 0:
            self.setup_rendering()

    def canvas_view_rect(self, tex_size, fb_size=None, canvas_tex_size=None):
        """Where the canvas sits inside a screen-shaped texture, as a 0..1 rect.

        NOT used by the tournament capture any more - see services/capture_view.py.
        The capture renders the canvas directly at CLIP's resolution, so it has
        no crop rect to get wrong. If you are here because a capture looks
        misaligned, this is the wrong file.

        MUST be evaluated at the moment the texture is produced and stored
        alongside it - see assembled_view_rect. Recomputing it later crops one
        frame's pixels with another frame's camera: main.py applies camera state
        at step 5, captures at step 5.1.5 and renders at step 7, so the texture
        being cropped is always a frame old. Measured at grid 8, one frame of
        ordinary input displaces the crop by 4px (a small pan) to 43px (one
        scroll notch), and every tile picks up a strip of its neighbour.

        The divisor is the TEXTURE size rather than the framebuffer size, and
        the two are not assumed equal: cam_brush_target only follows a window
        resize on the debounced reload, so for ~150ms they differ.

        The result is a GL TEXTURE coordinate - v = 0 is the BOTTOM - because
        that is what capture_blit samples with. tex_to_screen returns top-down
        window coordinates, so y is flipped here. The two conventions cancel
        exactly when the canvas is centred in the window, which is why a
        centred camera looked perfect: any vertical pan mirrored the crop about
        the window centre and displaced it by twice the pan. Measured on a real
        4x4 run at pan 0.05, that put the top 20% of each tile's lower
        neighbour into its crop and pushed the whole bottom row of tiles below
        v = 0, where capture_blit paints black - so tiles 0-3 were rejected as
        non-viable every single generation.
        """
        w, h = tex_size
        src = canvas_tex_size if canvas_tex_size is not None else self.sim.view_tex.size
        x0, y0 = self.tex_to_screen((0.0, 0.0), src, fb_size)
        x1, y1 = self.tex_to_screen((1.0, 1.0), src, fb_size)
        # x needs no flip: window x and GL u both run left to right.
        return ((min(x0, x1) / w, 1.0 - max(y0, y1) / h),
                (max(x0, x1) / w, 1.0 - min(y0, y1) / h))

    def screen_to_tex(self, coord_tuple, tex_size: tuple = None):
        """
        Transform screen coordinates to texture coordinates.

        Args:
            coord_tuple: (x, y) screen coordinates where (0,0) is top-left
            tex_size: (width, height) of texture. If None, uses self.sim.view_tex.size

        Returns:
            (tex_x, tex_y) texture coordinates where (0,0) is top-left of texture
        """
        x_screen, y_screen = coord_tuple
        width, height = glfw.get_framebuffer_size(self.window)
        width = max(1,width)
        height = max(1,height)
        x_ndc = (x_screen / width) * 2 - 1
        y_ndc = (1 - y_screen / height) * 2 - 1

        if tex_size is None:
            tex_size = self.sim.view_tex.size
        tex_aspect = tex_size[0] / tex_size[1]
        window_aspect = width / height

        if tex_aspect > window_aspect:
            scale_x = 1.0
            scale_y = window_aspect / tex_aspect
        else:
            scale_x = tex_aspect / window_aspect
            scale_y = 1.0

        scale_x /= self.zoom
        scale_y /= self.zoom

        x_ndc += self.position[0] / self.zoom
        y_ndc -= self.position[1] / self.zoom

        in_pos_x = x_ndc / scale_x
        in_pos_y = y_ndc / scale_y

        tex_x = (in_pos_x + 1) / 2
        tex_y = (in_pos_y + 1) / 2

        return (tex_x, tex_y)

    def tex_to_screen(self, coord_tuple, tex_size: tuple = None, fb_size: tuple = None):
        """
        Transform texture coordinates to screen coordinates.

        Args:
            coord_tuple: (tex_x, tex_y) texture coordinates where (0,0) is top-left
            tex_size: (width, height) of texture. If None, uses self.sim.view_tex.size
            fb_size: (width, height) of the framebuffer. If None, queries GLFW.
                Passing it explicitly is what lets canvas_view_rect() be computed
                for a frame other than the current one, and tested without GL.

        Returns:
            (x, y) screen coordinates where (0,0) is top-left of screen
        """
        tex_x, tex_y = coord_tuple
        width, height = (fb_size if fb_size is not None
                         else glfw.get_framebuffer_size(self.window))
        # A minimised window reports 0x0. screen_to_tex has always clamped;
        # this did not, so the first frame after minimising divided by zero.
        width = max(1, width)
        height = max(1, height)

        in_pos_x = tex_x * 2 - 1
        in_pos_y = tex_y * 2 - 1

        if tex_size is None:
            tex_size = self.sim.view_tex.size
        tex_aspect = tex_size[0] / tex_size[1]
        window_aspect = width / height

        if tex_aspect > window_aspect:
            scale_x = 1.0
            scale_y = window_aspect / tex_aspect
        else:
            scale_x = tex_aspect / window_aspect
            scale_y = 1.0

        scale_x /= self.zoom
        scale_y /= self.zoom

        pos_x = in_pos_x * scale_x
        pos_y = in_pos_y * scale_y

        pos_x -= self.position[0] / self.zoom
        pos_y += self.position[1] / self.zoom

        x_screen = (pos_x + 1) / 2 * width
        y_screen = (1 - (pos_y + 1) / 2) * height

        return (x_screen, y_screen)
