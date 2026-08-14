import moderngl
import numpy as np
from PIL import Image
import os
from utilities.paths import get_screenshots_dir

def create_supersample_shader(ctx, supersample_k):
    """Create a shader program for spatial supersampling only (NO temporal, NO gamma)."""

    vertex_shader = """
    #version 330 core
    in vec2 position;
    out vec2 uv;

    void main() {
        uv = position * 0.5 + 0.5;  // Convert from [-1,1] to [0,1]
        // Sampled upside down, so reading the result back bottom-up (which is
        // the only way GL reads it) hands the encoder rows already in image
        // order. The alternative was np.flipud on the host every frame.
        uv.y = 1.0 - uv.y;
        gl_Position = vec4(position, 0.0, 1.0);
    }
    """

    # Generate the fragment shader with unrolled loops for better compatibility
    sample_code = ""
    for y in range(supersample_k):
        for x in range(supersample_k):
            sample_code += f"""
        // Sample {x},{y}
        sub_pixel_offset = vec2({x}.5, {y}.5) / {supersample_k}.0;
        frag_shape = 1./input_size;
        sample_pos = uv - .5*frag_shape + (vec2({x}.5, {y}.5) / {supersample_k}.0)*frag_shape;

        sampcol = texture(input_frame, sample_pos).rgb;
        // NO GAMMA CORRECTION - input is already gamma-corrected from FrameAssembler
        total_color += sampcol;
"""

    fragment_shader = f"""
    #version 330 core
    uniform sampler2D input_frame;

    in vec2 uv;
    out vec4 fragColor;

    void main() {{
        vec3 total_color = vec3(0.0);
        vec2 input_size = vec2(textureSize(input_frame, 0));

        vec2 sub_pixel_offset, sample_pos, frag_shape;

        // Sample {supersample_k}^2 points within this region
        vec3 sampcol;
{sample_code}

        // Average the spatial samples only
        total_color /= {supersample_k * supersample_k}.0;

        // NO temporal averaging - that happens in FrameAssembler
        // NO gamma correction - input is already gamma-corrected

        fragColor = vec4(total_color, 1.0);
    }}
    """

    return ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)


def setup_gpu_supersampling(ctx, input_width, input_height, supersample_k,
                            even_dimensions=False):
    """Set up GPU-based spatial supersampling system.

    `even_dimensions` rounds the output DOWN to even, which H.264 requires.
    Cropping at most one row and column is invisible and costs nothing; the
    alternative was rebuilding every frame into a padded host array.
    """

    # Calculate output dimensions
    output_width = input_width // supersample_k
    output_height = input_height // supersample_k
    if even_dimensions:
        output_width -= output_width % 2
        output_height -= output_height % 2

    # RGBA8, not float32: the readback is a quarter of the bytes and IS the
    # `rgba` ffmpeg wants, so nothing on the host has to touch it. RGB8 would
    # be smaller still and is NOT a required color-renderable format.
    output_texture = ctx.texture((output_width, output_height), 4, dtype='f1')
    output_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    output_fbo = ctx.framebuffer(color_attachments=[output_texture])

    # Create shader program
    shader = create_supersample_shader(ctx, supersample_k)

    # Create a fullscreen quad
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
         1.0,  1.0,
        -1.0,  1.0,
    ], dtype=np.float32)

    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    vbo = ctx.buffer(vertices.tobytes())
    ibo = ctx.buffer(indices.tobytes())
    vao = ctx.vertex_array(shader, [(vbo, '2f', 'position')], ibo)

    return {
        'output_fbo': output_fbo,
        'output_texture': output_texture,
        'shader': shader,
        'vao': vao,
        'supersample_k': supersample_k,
        'output_counter': 0,
        'output_width': output_width,
        'output_height': output_height,
        'input_width': input_width,
        'input_height': input_height,
        'even_dimensions': even_dimensions,
    }


def _render_supersampled(gpu_resources, frame_data):
    """Draw the supersampled frame into the resources' own framebuffer."""
    frame_data.use(location=0)
    gpu_resources['shader']['input_frame'] = 0
    gpu_resources['output_fbo'].use()
    gpu_resources['output_fbo'].clear()
    gpu_resources['vao'].render()


def _release(gpu_resources):
    for key in ('output_fbo', 'output_texture', 'shader', 'vao'):
        gpu_resources[key].release()
    for buf in gpu_resources.get('pbos', ()):
        buf.release()


class AsyncFrameReader:
    """Supersamples a frame and reads it back WITHOUT stalling the pipeline.

    `texture.read()` blocks until the GPU has finished everything queued ahead
    of it. Two buffers instead: the copy for this frame is started into one and
    the other, started a frame ago and long since complete, is mapped. Callers
    therefore get frame N-1 from submit(), and drain() collects the last one.
    """

    def __init__(self):
        self._res = None
        self._pbos = []
        self._slot = 0
        self._pending = False

    @property
    def size(self):
        """(width, height) of what is written, or None before the first frame."""
        if self._res is None:
            return None
        return self._res['output_width'], self._res['output_height']

    def _ensure(self, ctx, frame_data, supersample_k):
        w, h = frame_data.size
        res = self._res
        if res is not None and (res['supersample_k'] == supersample_k
                                and res['input_width'] == w
                                and res['input_height'] == h):
            return False
        self.release()
        self._res = setup_gpu_supersampling(ctx, w, h, supersample_k,
                                            even_dimensions=True)
        nbytes = self._res['output_width'] * self._res['output_height'] * 4
        self._pbos = [ctx.buffer(reserve=nbytes), ctx.buffer(reserve=nbytes)]
        self._slot = 0
        self._pending = False
        return True

    def submit(self, ctx, frame_data, supersample_k):
        """Returns (rgba_bytes, resized) - bytes is the PREVIOUS frame or None.

        `resized` says the output geometry changed, so the caller can restart
        whatever it was writing into.
        """
        resized = self._ensure(ctx, frame_data, supersample_k)
        _render_supersampled(self._res, frame_data)
        # alignment=1: a row is not required to be a multiple of four bytes,
        # and the default packing would insert padding this format does not use.
        self._res['output_texture'].read_into(self._pbos[self._slot],
                                              alignment=1)
        out = None
        if self._pending and not resized:
            out = self._pbos[1 - self._slot].read()
        self._pending = True
        self._slot = 1 - self._slot
        self._res['output_counter'] += 1
        return out, resized

    def drain(self):
        """The frame still in flight, so a recording keeps its last frame."""
        if not self._pending or self._res is None:
            return None
        self._pending = False
        return self._pbos[1 - self._slot].read()

    def release(self):
        if self._res is not None:
            self._res['pbos'] = self._pbos
            _release(self._res)
        self._res = None
        self._pbos = []
        self._pending = False


def save_frame_gpu(frame_data, ctx, supersample_k=1, return_array=False):
    """
    Save a moderngl texture using GPU-accelerated spatial supersampling.

    Synchronous, and for one-off captures only - a screenshot pays the stall
    once. The recorder uses AsyncFrameReader, which is the same render into a
    pair of buffers that are mapped a frame late.

    Args:
        frame_data: moderngl.Texture object (already gamma-corrected and temporally assembled)
        ctx: moderngl.Context
        supersample_k: Spatial supersampling factor (1 = no supersampling)
        return_array: If True, return numpy array instead of saving to file

    Returns:
        If return_array=True: numpy array (height, width, 3) of uint8 RGB data
        If return_array=False: str filename of saved image
    """

    # Get input dimensions
    input_width, input_height = frame_data.size

    # Initialize function attributes on first call or when settings change
    if not hasattr(save_frame_gpu, 'gpu_resources'):
        save_frame_gpu.gpu_resources = None

    # Check if we need to recreate resources
    recreate_resources = (
        save_frame_gpu.gpu_resources is None or
        save_frame_gpu.gpu_resources['supersample_k'] != supersample_k or
        save_frame_gpu.gpu_resources['input_width'] != input_width or
        save_frame_gpu.gpu_resources['input_height'] != input_height
    )

    if recreate_resources:
        # Clean up old resources if they exist
        if save_frame_gpu.gpu_resources is not None:
            _release(save_frame_gpu.gpu_resources)

        # Create new resources
        save_frame_gpu.gpu_resources = setup_gpu_supersampling(
            ctx, input_width, input_height, supersample_k
        )

    gpu_resources = save_frame_gpu.gpu_resources
    _render_supersampled(gpu_resources, frame_data)

    # The target is RGBA8 and the shader already flipped, so this is image-order
    # bytes; only the alpha column has to go.
    data = gpu_resources['output_texture'].read(alignment=1)
    width = gpu_resources['output_width']
    height = gpu_resources['output_height']
    pixels = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
    pixels = np.ascontiguousarray(pixels[:, :, :3])

    # Return array directly if requested
    if return_array:
        gpu_resources['output_counter'] += 1
        return pixels

    # Otherwise save as PNG (legacy behavior)
    img = Image.fromarray(pixels, 'RGB')

    # Create Screenshots directory if it doesn't exist
    screenshots_dir = get_screenshots_dir()
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    # Increment output counter and save
    gpu_resources['output_counter'] += 1
    filename = screenshots_dir / f"frame_{gpu_resources['output_counter']:04d}.png"
    img.save(filename)

    return str(filename)


def reset_gpu_frame_counter():
    """Reset the output frame counter."""
    if hasattr(save_frame_gpu, 'gpu_resources') and save_frame_gpu.gpu_resources is not None:
        save_frame_gpu.gpu_resources['output_counter'] = 0


def cleanup_gpu_supersampling():
    """Clean up GPU resources. Call this when completely done with supersampling."""
    if hasattr(save_frame_gpu, 'gpu_resources') and save_frame_gpu.gpu_resources is not None:
        _release(save_frame_gpu.gpu_resources)
        save_frame_gpu.gpu_resources = None


# Example usage:
"""
import moderngl

# Setup your context
ctx = moderngl.create_context()

# That's it! Just call this in your render loop:
for i in range(25):
    # Your rendering code here
    texture = render_your_scene(ctx)  # Creates a 1000x1000 texture (already gamma-corrected)

    # Simple interface - handles spatial supersampling only
    result = save_frame_gpu(texture, ctx, supersample_k=2)
    if result:
        print(f"Saved: {result}")

# This outputs 25 frames at 500x500 resolution
# Each frame has 4x spatial supersampling
# Temporal accumulation and gamma correction happen BEFORE this function

# Optional: Reset for new animation sequence
reset_gpu_frame_counter()

# Optional: Clean up when completely done (releases GPU memory)
cleanup_gpu_supersampling()
"""
