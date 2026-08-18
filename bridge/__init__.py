"""fluobridge — Spout + OSC bridge between Python/ModernGL apps and vvvv.

Mirrors the transport already used by the VJ setup's gfx/Neuro modules:
GPU textures over Spout (zero-copy GL<->DX11 shared texture), parameters
over OSC. Self-contained: depends only on moderngl, SpoutGL and python-osc.

Typical use::

    from fluobridge import SpoutOut, SpoutIn, OscControl

    spout = SpoutOut(ctx, "fluoddity", (1920, 1080))
    ...
    spout.send(some_moderngl_texture)   # once per displayed frame
    spout.close()
"""
from .spout_out import SpoutOut
from .spout_in import SpoutIn
from .osc_control import OscControl
from .mod_matrix import ModMatrix

__all__ = ["SpoutOut", "SpoutIn", "OscControl", "ModMatrix"]
