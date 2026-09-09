import math
import numpy as np
import moderngl
from PIL import Image
def create_grid_coords(N):
    W = math.ceil(math.sqrt(N))
    # Create indices 0, 1, 2, ..., N-1
    indices = np.arange(N)
    # Convert to x, y coordinates using row-major ordering
    y,x = indices % W,indices // W
    # Stack into N x 2 array
    return np.column_stack([x, y])/W

def read_shader(path:str):
    result=""
    with open(path, 'r') as file:
        
        result= file.read()
    return result
def prepend_defines(shader_source, entity_count):
    content_to_insert=f"#define ENTITY_COUNT {entity_count}\n"
    return shader_prepend(shader_source,content_to_insert)
def shader_prepend(shader_source, content_to_insert):
    first_newline = shader_source.find('\n')
    return shader_source[:first_newline+1] + content_to_insert + shader_source[first_newline+1:]

MUTED_TRYSET_WARNINGS={}
def restore_target(previous) -> None:
    """Put back whatever framebuffer was bound before an offscreen pass.

    Everything downstream - the sim's own passes, and imgui - inherits the
    target the last pass left. A pass that leaves its own bound sends the whole
    UI into an offscreen buffer, which reads as every window vanishing at once,
    or as the panels appearing on a projector. Restoring what was there beats
    binding the screen, which a standalone context does not have.
    """
    # A released framebuffer keeps its wrapper and swaps its `mglo` for an
    # InvalidObject, so the wrapper's own type says nothing. Binding one
    # raises; having nothing to put back is not an error.
    if previous is None:
        return
    if isinstance(getattr(previous, "mglo", None), moderngl.InvalidObject):
        return
    previous.use()


def tryset(program:moderngl.Program,uniform,value):
    """
    Gracefully handle a uniform that doesn't appear in program.
    Uniforms are frequently optimized out if they are not used in the current version of the shader.
    """
    if uniform in program:
        program[uniform]=value
    else:
        global MUTED_TRYSET_WARNINGS
        if uniform not in MUTED_TRYSET_WARNINGS:
            MUTED_TRYSET_WARNINGS[uniform]=0
        MUTED_TRYSET_WARNINGS[uniform]+=1
        if MUTED_TRYSET_WARNINGS[uniform]<10:
            print('Warning: ',uniform,' not present in ',program)

def tryset_mat3(program: moderngl.Program, uniform, mat):
    """Upload a 3x3 matrix uniform (column-major float32).

    Gracefully handles optimized-away uniforms like tryset.
    """
    if uniform in program:
        program[uniform].write(mat.astype("f4").T.tobytes())
    else:
        global MUTED_TRYSET_WARNINGS
        if uniform not in MUTED_TRYSET_WARNINGS:
            MUTED_TRYSET_WARNINGS[uniform] = 0
        MUTED_TRYSET_WARNINGS[uniform] += 1
        if MUTED_TRYSET_WARNINGS[uniform] < 10:
            print('Warning: ', uniform, ' not present in ', program)

def readback_rule(rule_buffer, layout=None):
    """
    Read the brain at the START of a buffer, as `layout`.

    Reads ONE brain of `layout.length` floats from offset 0. That is the whole
    of the per-particle readback buffer - click-to-adopt writes the one particle
    it was asked for and nothing else reads it - and slot 0 of the flat brain
    buffer. Fourier brains keep one row per centre so click-to-adopt hands the
    rest of the app what it has always expected.
    """
    from services.brains import default_layout, get

    layout = layout or default_layout()
    stride_bytes = layout.length * 4

    data = np.frombuffer(
        rule_buffer.read(size=stride_bytes, offset=0), dtype=np.float32)
    if layout.modality == "fourier":
        return data.reshape(layout.shape[0],
                            get("fourier").unit_floats(layout))
    return data.copy()


def set_brain_layout_uniforms(program, layout) -> None:
    """Push a layout's STRUCTURE: BRAIN_SHAPE plus whatever the modality adds.

    Shared by sim.py and services/brain_preview.py, so the Inspector runs the
    identical brain the particles do. BRAIN_LEN and BRAIN_PER_COHORT stay with
    sim.py on purpose: the preview never writes back and never reads a cohort
    slot, so GLSL drops those uniforms and setting them only prints a warning.
    """
    from services.brains import get

    shape = tuple(int(v) for v in layout.shape)
    tryset(program, 'BRAIN_SHAPE', (shape + (0, 0, 0, 0))[:4])
    tryset(program, 'BRAIN_AUDIO_IN', int(layout.audio_inputs))
    extra = getattr(get(layout.modality), "layout_uniforms", None)
    for name, value in (extra(layout) if extra is not None else {}).items():
        tryset(program, name, value)


def pack_brains(params_list, layout) -> bytes:
    """Pack brains into the flat SSBO, each zero-padded to MAX_BRAIN_FLOATS.

    std430 gives a float array a 4-byte stride with no padding, so this is a
    straight memcpy - there is no struct alignment to get wrong.
    """
    from services.brains import MAX_BRAIN_FLOATS

    out = np.zeros((len(params_list), MAX_BRAIN_FLOATS), dtype=np.float32)
    for i, p in enumerate(params_list):
        flat = np.asarray(p, dtype=np.float32).reshape(-1)
        out[i, : layout.length] = flat[: layout.length]
    return out.tobytes()

def load_image_as_texture(ctx, image_path):
    """
    Load an arbitrary image file and convert it to a ModernGL RGBA texture.
    
    Args:
        ctx: ModernGL context
        image_path: Path to the image file (JPEG, PNG, etc.)
    
    Returns:
        moderngl.Texture: RGBA texture object
    """
    # Load and convert image to RGBA
    img = Image.open(image_path)
    img = img.convert('RGBA')  # Ensure RGBA format
    
    # Get image dimensions
    width, height = img.size
    
    # Get raw pixel data as bytes
    # PIL uses top-left origin, ModernGL uses bottom-left, so flip vertically
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    pixel_data = img.tobytes()
    
    # Create ModernGL texture
    texture = ctx.texture((width, height), 4, pixel_data)
    
    return texture