"""Receive a Spout sender (e.g. a vvvv texture) into a ModernGL texture."""
import moderngl
import SpoutGL

from ._gl import GL_TEXTURE_2D


class SpoutIn:
    """Receives a named Spout sender into an owned RGBA8 ModernGL texture.

    The receiving texture is created lazily and recreated whenever the sender
    changes resolution, so the sender may start after us, restart, or resize
    without anything here needing to know.

    Args:
        ctx: the ModernGL context.
        name: Spout sender name to connect to.
        invert: flip vertically on receive — the mirror of ``SpoutOut.invert``.
    """

    def __init__(self, ctx: moderngl.Context, name: str, invert: bool = True):
        self.ctx = ctx
        self.name = name
        self.invert = invert

        self._receiver = SpoutGL.SpoutReceiver()
        self._receiver.setReceiverName(name)
        self._texture: moderngl.Texture | None = None
        self._closed = False

    @property
    def texture(self) -> moderngl.Texture | None:
        """Most recently received frame, or None if nothing has arrived yet."""
        return self._texture

    @property
    def connected(self) -> bool:
        return not self._closed and self._receiver.isConnected()

    def receive(self) -> moderngl.Texture | None:
        """Pull the current frame. Call once per frame on the GL thread.

        Returns the receiving texture, or None while no sender is present.
        The texture is reused across frames — hold the reference at your peril
        if you also call this again.
        """
        if self._closed:
            return None

        # receiveTexture() must be called before isUpdated()/getSenderWidth()
        # report anything: connecting to the sender is a side effect of it.
        ok = self._receiver.receiveTexture(
            self._texture.glo if self._texture is not None else 0,
            GL_TEXTURE_2D, self.invert, 0)

        if self._receiver.isUpdated():
            width = self._receiver.getSenderWidth()
            height = self._receiver.getSenderHeight()
            if width > 0 and height > 0:
                self._recreate((width, height))
            # The frame that triggered the resize landed in the old texture
            # (or nowhere); the next receive() fills the new one.
            return self._texture

        if not ok or self._texture is None:
            return None
        return self._texture

    def _recreate(self, size: tuple[int, int]) -> None:
        if self._texture is not None:
            if self._texture.size == size:
                return
            self._texture.release()
        self._texture = self.ctx.texture(size, 4, dtype="f1")
        self._texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._texture.repeat_x = False
        self._texture.repeat_y = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._receiver.releaseReceiver()
        except Exception:
            pass
        if self._texture is not None:
            self._texture.release()
            self._texture = None
