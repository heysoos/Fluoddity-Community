"""Send a ModernGL texture to vvvv (or anything else) as a Spout sender."""
import moderngl
import SpoutGL

from ._gl import GL_TEXTURE_2D, Blitter


class SpoutOut:
    """Publishes a ModernGL texture on a named Spout sender.

    The source texture is blitted into an owned RGBA8 target before sending.
    Two reasons: Spout's shared texture is 8-bit, while the interesting render
    targets in both Fluoddity apps are RGBA32F; and the blit is where output
    resolution can differ from render resolution.

    Must be constructed with the GL context current, and ``close()``-ed before
    the context goes away.

    Args:
        ctx: the ModernGL context.
        name: Spout sender name — what vvvv's ``Share Name`` pin must match.
        size: output resolution ``(width, height)``.
        invert: flip vertically on send. OpenGL is bottom-up and DX11 is
            top-down, so the default is almost certainly what you want; it is
            exposed because the only way to be sure is to look at the result.
        force_opaque: write alpha 1.0. A VJ source that turns out to be
            transparent looks like a broken feed, so this defaults on.
    """

    def __init__(self, ctx: moderngl.Context, name: str, size: tuple[int, int],
                 invert: bool = True, force_opaque: bool = True):
        self.ctx = ctx
        self.name = name
        self.size = (int(size[0]), int(size[1]))
        self.invert = invert
        self.force_opaque = force_opaque

        self._blitter = Blitter(ctx)
        self._texture = ctx.texture(self.size, 4, dtype="f1")
        self._texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._fbo = ctx.framebuffer(color_attachments=[self._texture])
        self._fbo.clear(0.0, 0.0, 0.0, 1.0)

        self._sender = SpoutGL.SpoutSender()
        self._sender.setSenderName(name)
        self._closed = False
        self._name_checked = False
        self.actual_name = name

        # The receiving app must be on the SAME GPU. A D3D11 shared handle
        # cannot be opened across adapters, so on a hybrid laptop a sender on
        # the integrated GPU is invisible to a receiver on the discrete one --
        # and the failure is silent: the name, handle and size all read back
        # correctly, but the texture resolves to nil.
        try:
            renderer = ctx.info["GL_RENDERER"]
        except Exception:
            renderer = "<unknown>"
        print(f"[fluobridge] Spout sender '{name}' on {renderer}")

    def _verify_name(self) -> None:
        """Warn if Spout silently renamed us because the name was taken.

        Spout appends _1, _2, ... when a sender with the requested name is
        already registered -- including a stale entry left by a process that
        was force-killed before it could release. The receiver keeps watching
        the original name and sees nothing.
        """
        self._name_checked = True
        try:
            names = SpoutGL.SpoutReceiver().getSenderList()
        except Exception:
            return
        if self.name in names:
            return
        taken = [n for n in names if n.startswith(self.name)]
        self.actual_name = taken[0] if taken else self.name
        print(f"[fluobridge] !! Spout renamed this sender to '{self.actual_name}' "
              f"-- '{self.name}' was already registered.")
        print(f"[fluobridge]    A receiver watching '{self.name}' will see nothing. "
              f"Usually a stale entry from a force-killed process;")
        print(f"[fluobridge]    close every other sender of that name and restart.")

    def resize(self, size: tuple[int, int]) -> None:
        """Change the output resolution, recreating the shared texture."""
        size = (int(size[0]), int(size[1]))
        if size == self.size:
            return
        self._fbo.release()
        self._texture.release()
        self.size = size
        self._texture = self.ctx.texture(size, 4, dtype="f1")
        self._texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._fbo = self.ctx.framebuffer(color_attachments=[self._texture])

    def send(self, texture: moderngl.Texture) -> bool:
        """Blit ``texture`` into the shared target and publish it.

        Call once per *displayed* frame. Calling it per intermediate render
        (for instance once per motion-blur accumulation sample) publishes
        partially accumulated frames.
        """
        if self._closed or texture is None:
            return False

        self._blitter.blit(texture, self._fbo, force_opaque=self.force_opaque)
        ok = self._sender.sendTexture(
            self._texture.glo, GL_TEXTURE_2D,
            self.size[0], self.size[1], self.invert, 0)

        # The registry entry only exists after the first successful send.
        if ok and not self._name_checked:
            self._verify_name()

        # Leave the default framebuffer bound so we don't surprise whatever
        # draws next.
        self.ctx.screen.use()
        return ok

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._sender.releaseSender()
        except Exception:
            pass
        self._fbo.release()
        self._texture.release()
        self._blitter.release()
